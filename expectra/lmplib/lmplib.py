"""ASE LAMMPS Calculator Library Version, modeified by xph for TSASE"""
""" Changes:
    1. Update atom positions instead of recreating atoms every time,
       so that the neighbor list don't need to be recalculated.
    2. Use a input file to initialize LAMMPS instead of lmp.command for easy debug.
    3. Update cell if changed.
"""
       

import os
import numpy as np
from numpy.linalg import norm
from lammps import lammps
from ase.calculators.calculator import Calculator
from ase.units import GPa
import ctypes, sys

# TODO
# 1. should we make a new lammps object each time ?
# 2. upper triangular test does not look good
# 3. lmp object is not closed
# 4. need a routine to get the model back from lammps
# 5. if we send a command to lmps directly then the calculator does
#    not know about it and the energy could be wrong.

# 6. do we need a subroutine generator that converts a lammps string
#   into a python function that can be called


class LAMMPSlib(Calculator):
    r"""
    LAMMPSlib Interface Documentation
    
Introduction
============

LAMMPSlib is an interface and calculator for LAMMPS_. LAMMPSlib uses
the python interface that comes with LAMMPS to solve an atoms model
for energy, atom forces and cell stress. This calculator creates a
'.lmp' object which is a running lammps program, so further commands
can be sent to this object executed until it is explicitly closed. Any
additional variables calculated by lammps can also be extracted. This
is still experimental code.
    
Arguments
=========

=================  ==========================================================
Keyword                               Description
=================  ==========================================================
``lmpcmds``        list of strings of LAMMPS commands. You need to supply
                   enough to define the potential to be used e.g.

                   ["pair_style eam/alloy",
                    "pair_coeff * * potentials/NiAlH_jea.eam.alloy Ni Al"]

``atom_types``     dictionary of "atomic_symbol":lammps_atom_type pairs,
                   e.g. {'Cu':1} to bind copper to lammps atom type 1.
                   Default method assigns lammps atom types in order that they
                   appear in the atoms model

``log_file``       string
                   path to the desired LAMMPS log file

``lammps_header``  string to use for lammps setup. Default is to use
                   metal units and simple atom simulation.

                   lammps_header=['units metal',
                                  'atom_style atomic',
                                  'atom_modify map array sort 0 0'])

``keep_alive``     Boolean
                   whether to keep the lammps routine alive for more commands

=================  ==========================================================


Requirements
============

To run this calculator you must have LAMMPS installed and compiled to
enable the python interface. See the LAMMPS manual.

If the following code runs then lammps is installed correctly.

   >>> from lammps import lammps
   >>> lmp = lammps()

The version of LAMMPS is also important. LAMMPSlib is suitable for
versions after approximately 2011. Prior to this the python interface
is slightly different from that used by LAMMPSlib. It is not difficult
to change to the earlier format.

LAMMPS and LAMMPSlib
====================

The LAMMPS calculator is another calculator that uses LAMMPS (the
program) to calculate the energy by generating input files and running
a separate LAMMPS job to perform the analysis. The output data is then
read back into python. LAMMPSlib makes direct use of the LAMMPS (the
program) python interface. As well as directly running any LAMMPS
comand line it allows the values of any of LAMMPS variables to be
extracted and returned to python.

Example
=======

1)
::

    from ase import Atom, Atoms
    #from lammpslib import LAMMPSlib
    from tsase.calculators.lmplib import LAMMPSlib

    cmds = ["pair_style eam/alloy",
            "pair_coeff * * NiAlH_jea.eam.alloy Al H"]
    
    a = 4.05
    al = Atoms([Atom('Al')], cell=(a, a, a), pbc=True)
    h = Atom([Atom('H')])
    alh = al + h

    #lammps = LAMMPSlib(lmpcmds = cmds, logfile='test.log')
    lammps = LAMMPSlib(lmpcmds = cmds, logfile='test.log', atoms=a1h)

    alh.set_calculator(lammps)
    print "Energy ", alh.get_potential_energy()

2)
:: 
    # comb3 potential, CuO 
    from tsase.calculators.lmplib import LAMMPSlib

    cmds = ["pair_style comb3 polar_off",
            "pair_coeff * * ffield.comb3 Cu O",
            "fix 1 all qeq/comb 1 0.003 file log_qeq"]
    

    p1 = read('POSCAR',format='vasp')
    calc=LAMMPSlib(lmpcmds=cmds, logfile='test.log', keep_alive=True,
                   atoms = p1,
                   atom_types={'Cu':1, 'O':2},
                   lammps_header=['units metal',
                                  'atom_style charge',
                                  'atom_modify map array sort 0 0.0'
                                 ])
    p1.set_calculator(calc)
    print p1.get_potential_energy()

    
Implementation
==============

LAMMPS provides a set of python functions to allow execution of the
underlying C++ LAMMPS code. The functions used by the LAMMPSlib
interface are::

    from lammps import lammps
    
    lmp = lammps(cmd_args) # initiate LAMMPS object with command line args

    lmp.scatter_atoms('x',1,3,positions) # atom coords to LAMMPS C array
    lmp.command(cmd) # executes a one line cmd string
    lmp.extract_variable(...) # extracts a per atom variable
    lmp.extract_global(...) # extracts a global variable
    lmp.close() # close the lammps object
    
For a single atom model the following lammps file commands would be run
by invoking the get_potential_energy() method::

    units metal
    atom_style atomic
    atom_modify map array sort 0 0
    
    region cell prism 0 xhi 0 yhi 0 zhi xy xz yz units box
    create_box 1 cell
    create_atoms 1 single 0 0 0 units box
    mass * 1.0

    ## user lmpcmds get executed here
    pair_style eam/alloy
    pair_coeff * * lammps/potentials/NiAlH_jea.eam.alloy Al
    ## end of user lmmpcmds

    run 0


Notes
=====

.. _LAMMPS: http://lammps.sandia.gov/

* Units: The default lammps_header sets the units to Angstrom and eV
  and for compatibility with ASE Stress is in GPa.

* The global energy is currently extracted from LAMMPS using
  extract_variable since lammps.lammps currently extract_global only
  accepts the following ['dt', 'boxxlo', 'boxxhi', 'boxylo', 'boxyhi',
  'boxzlo', 'boxzhi', 'natoms', 'nlocal'].

* If an error occurs while lammps is in control it will crash
  Python. Check the output of the log file to find the lammps error.

* If the are commands directly sent to the LAMMPS object this may
  change the energy value of the model. However the calculator will not
  know of it and still return the original energy value.

End LAMMPSlib Interface Documentation

    """

    implemented_properties = ['energy', 'forces', 'stress']

    default_parameters = dict(
        atom_types=None,
        log_file=None,
        in_file='in.lmp.initialize',
        data_file='data.lmp.initialize',
        keep_alive=True,
        lammps_header=['units metal',
                       'atom_style atomic',
                       'atom_modify map array sort 0 0'])

    # xph: set the lammps header and basic parameters through an input file
    # do not reset every time self.calculate is called 
    def __init__(self, restart=None, atoms=None, ignore_bad_restart_file=False, 
                 label=None, **kwargs):
        
        Calculator.__init__(self, restart=restart, atoms=atoms, ignore_bad_restart_file=ignore_bad_restart_file, 
                 label=label, **kwargs)

        # xph: cannot set pair_coeff before simulation box is defined. To avoid setting 
        #      pair_coeff repeatedly, an atoms object has to be provided for initialization
        if atoms == None: 
            print "Please assign an atoms object by setting 'atoms = ' for LAMMPSlib initialization"
            raise

        # xph: if the cell is not lower triangular, it is difficult to map the forces back.
        #      It is easier to rotate outside the calculator. 
        if not self.is_lower_triangular(atoms.get_cell()):
            print "Please rotate the cell following LAMMPS convention: 'A = (xhi-xlo,0,0); B = (xy,yhi-ylo,0); C = (xz,yz,zhi-zlo)'"
            print "A roation example for an arbitrary oriented cell: "
            print "'''"
            print "cell = atoms.get_cell()"
            print "atoms.rotate(cell[0], 'x', center=(0, 0, 0), rotate_cell=True)"
            print "cell = atoms.get_cell()"
            print "cell[1][0] = 0.0"
            print "atoms.rotate(cell[1], 'y', center=(0, 0, 0), rotate_cell=True)"
            print "'''"
            raise
        self.initialize_lammps(atoms)

        # xph: save screen output to out_screen. If comb3 used, "screen none" crushes.
        screenoption = 'none'
        for cmd in self.parameters.lmpcmds:
            if 'comb3' in cmd:
                screenoption = 'out_screen'
                break
        if self.parameters.log_file == None:
            cmd_args = ['-echo', 'log', '-log', 'none', '-screen', screenoption]
        else:
            cmd_args = ['-echo', 'log', '-log', self.parameters.log_file,
                        '-screen', screenoption]

        self.cmd_args = cmd_args

        if not hasattr(self, 'lmp'):
            self.lmp = lammps('', self.cmd_args)

        self.lmp.file(self.parameters.in_file)


    def calculate(self, atoms, properties, system_changes):
        """"atoms: Atoms object
            Contains positions, unit-cell, ...
        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces', 'stress', 'dipole', 'charges', 'magmom'
            and 'magmoms'.
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these five: 'positions', 'numbers', 'cell',
            'pbc', 'charges' and 'magmoms'.
        """
        # xph: self.atoms is saved in Calculator.calculate
        Calculator.calculate(self, atoms, properties, system_changes)
        if len(system_changes) == 0:
            return

        self.atom_types = None

        pos = atoms.get_positions()

        # xph: update cell if changed
        if "cell" in system_changes:
            cell = atoms.get_cell()
            cellupdate = ('change_box all x final 0.0 ' + str(cell[0][0]) +
                                        ' y final 0.0 ' + str(cell[1][1]) +
                                        ' z final 0.0 ' + str(cell[2][2]) +
                                        ' yz final '    + str(cell[2][1]) +
                                        ' xz final '    + str(cell[2][0]) +
                                        ' xy final '    + str(cell[1][0]) + 
                                        ' remap' +
                                        ' units box')
            self.lmp.command(cellupdate)

        # Convert ase position matrix to lammps-style position array
        lmp_positions = list(pos.ravel())

        # Convert that lammps-style array into a C object
        lmp_c_positions =\
            (ctypes.c_double * len(lmp_positions))(*lmp_positions)
#        self.lmp.put_coosrds(lmp_c_positions)
        self.lmp.scatter_atoms('x', 1, 3, lmp_c_positions)

        # Run for 0 time to calculate
        self.lmp.command('run 0')

        # Extract the forces and energy
#        if 'energy' in properties:
        self.results['energy'] = self.lmp.extract_variable('pe', None, 0)
#            self.results['energy'] = self.lmp.extract_global('pe', 0)
            
#        if 'stress' in properties:
        stress = np.empty(6)

        # xph: make the stress listed in the same order as in vasp.py and lammpsrun.py
        #stress_vars = ['pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz']
        stress_vars = ['pxx', 'pyy', 'pzz', 'pyz', 'pxz', 'pxy']

        for i, var in enumerate(stress_vars):
            stress[i] = self.lmp.extract_variable(var, None, 0)
            
        # 1 bar (used by lammps for metal units) = 1e-4 GPa
        self.results['stress'] = stress * -1e-4 * GPa

#        if 'forces' in properties:
        f = np.zeros((len(atoms), 3))
        force_vars = ['fx', 'fy', 'fz']
        for i, var in enumerate(force_vars):
            f[:, i] = np.asarray(self.lmp.extract_variable(
                    var, 'all', 1)[:len(atoms)])
            
        self.results['forces'] = f

        if not self.parameters.keep_alive:
            self.lmp.close()

    def is_lower_triangular(self, mat):
        """test if 3x3 matrix is upper triangular"""
        
        def near0(x):
            """Test if a float is within .00001 of 0"""
            return abs(x) < 0.00001
        
        return near0(mat[0, 1]) and near0(mat[0, 2]) and near0(mat[1, 2])

    def lammpsbc(self, pbc):
        if pbc:
            return 'p'
        else:
            return 's'

    def initialize_lammps(self, atoms):
        # Initialising commands
        # xph: write an input and data_file for initialization of lammps

        self.write_lammps_input(atoms)
        self.write_lammps_data(atoms)

    def write_lammps_input(self, atoms):

        # xph: open an input file
        fname = self.parameters.in_file
        f = open(fname, 'w')
        # Use metal units, Angstrom and eV
        for cmd in self.parameters.lammps_header:
            f.write(cmd+'\n')
            #self.lmp.command(cmd)


        # if the boundary command is in the supplied commands use that
        # otherwise use atoms pbc
        pbc = atoms.get_pbc()
        for cmd in self.parameters.lmpcmds:
            if 'boundary' in cmd:
                break
        else:
            #self.lmp.command(
            #    'boundary ' + ' '.join([self.lammpsbc(bc) for bc in pbc]))
            cmdt = 'boundary ' + ' '.join([self.lammpsbc(bc) for bc in pbc])
            f.write(cmdt+'\n')

        f.write('\n\n')

        # xph: read geometry from lammps data file
        f.write('read_data %s \n' % self.parameters.data_file)

        f.write('\n\n')

        # Set masses, even though they don't matter
        #self.lmp.command('mass * 1.0')
        f.write('mass * 1.0 \n')
        
        # execute the user commands
        for cmd in self.parameters.lmpcmds:
            #self.lmp.command(cmd)
            f.write(cmd+'\n')

        # xph: nve
        f.write('fix fix_nve all nve \n')

        # I am not sure why we need this next line but LAMMPS will
        # raise an error if it is not there. Perhaps it is needed to
        # ensure the cell stresses are calculated
        #self.lmp.command('thermo_style custom pe pxx')
        f.write('thermo_style custom pe pxx \n')
        
        # Define force & energy variables for extraction
        #self.lmp.command('variable pxx equal pxx')
        #self.lmp.command('variable pyy equal pyy')
        #self.lmp.command('variable pzz equal pzz')
        #self.lmp.command('variable pxy equal pxy')
        #self.lmp.command('variable pxz equal pxz')
        #self.lmp.command('variable pyz equal pyz')
        f.write('variable pxx equal pxx \n')
        f.write('variable pyy equal pyy \n')
        f.write('variable pzz equal pzz \n')
        f.write('variable pxy equal pxy \n')
        f.write('variable pxz equal pxz \n')
        f.write('variable pyz equal pyz \n')

        #self.lmp.command('variable fx atom fx')
        #self.lmp.command('variable fy atom fy')
        #self.lmp.command('variable fz atom fz')
        f.write('variable fx atom fx \n')
        f.write('variable fy atom fy \n')
        f.write('variable fz atom fz \n')

        # do we need this if we extract from a global ?
        #self.lmp.command('variable pe equal pe')
        f.write('variable pe equal pe \n')
        f.close()

    def write_lammps_data(self, atoms):
        
        # xph: open a data file for geometry setting
        # can switch to lmp.command by uncommenting those lines
        fname = self.parameters.data_file
        f = open(fname, 'w')
        
        # xph: the following part is from lammpsrun.py in ASE
        f.write(fname + ' (written by ASE) \n\n')

        symbols = atoms.get_chemical_symbols()
        n_atoms = len(symbols)
        f.write('%d \t atoms \n' % n_atoms)

        # This way it is assured that LAMMPS atom types are always
        # assigned predictively according to the alphabetic order 
        species = sorted(list(set(symbols)))
        n_atom_types = len(species)
        f.write('%d  atom types\n' % n_atom_types)

        # xph: check if cell is lower-triangle
        # Initialize cell
        cell = atoms.get_cell()
        xhi = cell[0, 0]
        yhi = cell[1, 1]
        zhi = cell[2, 2]
        xy = cell[1, 0]
        xz = cell[2, 0]
        yz = cell[2, 1]

        f.write('0.0 %s  xlo xhi\n' % xhi)
        f.write('0.0 %s  ylo yhi\n' % yhi)
        f.write('0.0 %s  zlo zhi\n' % zhi)
    
        f.write('%s %s %s  xy xz yz\n' % (xy, xz, yz))
        f.write('\n\n')

        f.write('Atoms \n\n')
         
        write_charge = False
        for cmd in self.parameters.lammps_header:
            if 'charge' in cmd:
                write_charge = True
                break

        # xph: add charge in data file 
        if write_charge:
            for i, r in enumerate(atoms.get_positions()):
                s = species.index(symbols[i]) + 1
                charge = atoms[i].charge
                f.write('%6d %3d %.4f %.4f %.4f %.4f\n' % ((i+1, s, charge)+tuple(r)))
        else:
            for i, r in enumerate(atoms.get_positions()):
                s = species.index(symbols[i]) + 1
                f.write('%6d %3d %.4f %.4f %.4f\n' % ((i+1, s)+tuple(r)))
        
        f.close()

        #cell_cmd = 'region cell prism 0 {} 0 {} 0 {} {} {} {} units box'\
        #    .format(xhi, yhi, zhi, xy, xz, yz)
        #self.lmp.command(cell_cmd)
        
        # The default atom_types has atom type in alphabetic order
        # by atomic symbol
        #symbols = np.asarray(atoms.get_chemical_symbols())

        # if the dictionary of types has not already been specified
        #if self.atom_types == None:
        #    self.atom_types = {}
        #    atom_types = np.sort(np.unique(symbols))

        #    for i, sym in enumerate(atom_types):
        #        self.atom_types[sym] = i + 1

        # Initialize box
        #n_types = len(self.atom_types)
        #types_command = 'create_box {} cell'.format(n_types)
        #self.lmp.command(types_command)

        # Initialize the atoms with their types
        # positions do not matter here
        #self.lmp.command('echo none') # don't echo the atom positions
        #for sym in symbols:
        #    cmd = 'create_atoms {} single 0.0 0.0 0.0  units box'.\
        #        format(self.atom_types[sym])
        #    self.lmp.command(cmd)


#print('done loading lammpslib')
