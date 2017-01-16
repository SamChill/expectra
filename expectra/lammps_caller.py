"""
Call and run PyLammps.
Lammps run in its own manner and output everything as standard setting.
Lammps inputs, including force filed and controlling command are required as run normal lammps.
The structure input file (file named as 'data_lammps') will be prepared based on the atoms provided.
"""
from mpi4py import MPI
from lammps import PyLammps
from ase.io import read, write
from ase import Atoms
#from expectra.io import read_lammps_trj, write_lammps_data
#from lammps_ext import write_lammps_data
from ase.calculators.lammpsrun import Prism
import os
import sys
import numpy

class lammps_caller:
      def __init__(self, atoms,
                   ncore=2,
                   specorder=None, #['Rh', 'Au']
                   data_lammps='data_lammps',
                   run_type='geo_opt',
                   ):
          self.atoms = atoms
          self.ncore = ncore
          self.specorder = specorder
          self.data_lammps = data_lammps
          self.run_type = run_type
          self.logfile = 'log.lammps'
          self.trajfile = 'trj_lammps'


      def get_energy(self):
          self.run(self.run_type)
          #Find and return optimized atoms and energy
          f = open(self.logfile, 'r')
          energy = None
          iteration = None
          while True:
               line = f.readline()
               if not line:
                   break
               if 'Energy initial, next-to-last, final' in line:
                   line = f.readline()
                   energy = float(line.split()[2])
                   print "energy:", energy
               if 'Iterations, force evaluations' in line:
                   iteration = int(line.split()[4])
                   print "iteration:", iteration
          f.close()
          atoms = read_lammps_trj(filename=self.trajfile, skip=iteration, specorder=self.specorder)
          write('lammps_opted.traj',images=atoms,format='traj')
          return energy, atoms[0]

      def run(self, run_type='geo_opt'):
          #prepare structure file 'data_lammps' for lammps
          write_lammps_data(fileobj=self.data_lammps, atoms=self.atoms, specorder=self.specorder, write_charge=True)
          
          if run_type == 'geo_opt':
             lammps_script = 'lp_opt.py'
             output = 'opt_lammps.out'
          elif run_type == 'md':
             print "MD with lammps is running"
             lammps_script = 'lp_md.py'
             output = 'md_lammps.out'

          self.rm_file(self.logfile)
          self.rm_file(self.trajfile)
          self.rm_file(output)
          print output
          run_para = ['mpirun -n', str(self.ncore),
                      'python', lammps_script,
                      '>', output]
          join_symbol = ' '
          run_lammps = join_symbol.join(run_para)
          os.system(run_lammps)
          print run_lammps

      def rm_file(self, filename):
          filename = os.getcwd()+'/'+filename
          try:
             os.remove(filename)
          except OSError:
             pass

def read_lammps_trj(filename=None, skip=0, every=1, specorder=None):
    """Method which reads a LAMMPS dump file."""
    if filename is None:
        print "No trajectory file is provided"

    if isinstance(specorder, str):
       specorder = specorder.split()
       
    atoms=[]
    f = open(filename, 'r')
    n_atoms = 0
    while True:
        line = f.readline()

        if not line:
            break

        #TODO: extend to proper dealing with multiple steps in one trajectory file
        if 'ITEM: TIMESTEP' in line:
            lo = [] ; hi = [] ; tilt = []
            id = [] ; type = []
            positions = [] ; velocities = [] ; forces = []
            # xph: add charges
            charges = []
            line = f.readline()
            itrj = int(line.split()[0])

        line = f.readline()
        if 'ITEM: NUMBER OF ATOMS' in line:
            line = f.readline()
            n_atoms = int(line.split()[0])
        
        #read geometries every 'every' step
        if itrj % every != 0 or itrj < skip:
#           print '%4d is jumped' % itrj
           for i in range(n_atoms + 5):
               line = f.readline()
        else:
           line = f.readline()
           if 'ITEM: BOX BOUNDS' in line:
               # save labels behind "ITEM: BOX BOUNDS" in triclinic case (>=lammps-7Jul09)
               tilt_items = line.split()[3:]
               for i in range(3):
                   line = f.readline()
                   fields = line.split()
                   lo.append(float(fields[0]))
                   hi.append(float(fields[1]))
                   if (len(fields) >= 3):
                       tilt.append(float(fields[2]))
           
           line = f.readline()
           if 'ITEM: ATOMS' in line:
               # (reliably) identify values by labels behind "ITEM: ATOMS" - requires >=lammps-7Jul09
               # create corresponding index dictionary before iterating over atoms to (hopefully) speed up lookups...
               atom_attributes = {}
               for (i, x) in enumerate(line.split()[2:]):
                   atom_attributes[x] = i
               for n in range(n_atoms):
                   line = f.readline()
                   fields = line.split()
                   id.append( int(fields[atom_attributes['id']]) )
                   type.append( specorder[int(fields[atom_attributes['type']])-1] )
                   positions.append( [ float(fields[atom_attributes[x]]) for x in ['x', 'y', 'z'] ] )
                   velocities.append( [ float(fields[atom_attributes[x]]) for x in ['vx', 'vy', 'vz'] ] )
                   forces.append( [ float(fields[atom_attributes[x]]) for x in ['fx', 'fy', 'fz'] ] )
#                   if hasattr('charges'):
#                        charges.append(  float(fields[atom_attributes['q']]) )

               xhilo = (hi[0] - lo[0])
               yhilo = (hi[1] - lo[1])
               zhilo = (hi[2] - lo[2])
           
               cell = [[xhilo,0,0],[0,yhilo,0],[0,0,zhilo]]
           
               cell_atoms = numpy.array(cell)
               type_atoms = type
               positions_atoms = numpy.array(positions)
               forces_atoms = numpy.array(forces)
           
#               positions_atoms = np.array( [np.dot(np.array(r), rotation_lammps2ase) for r in positions] )
#               velocities_atoms = np.array( [np.dot(np.array(v), rotation_lammps2ase) for v in velocities] )
#               forces_atoms = np.array( [np.dot(np.array(f), rotation_lammps2ase) for f in forces] )
           
               atoms.append(Atoms(type_atoms, positions=positions_atoms, cell=cell_atoms, pbc=True))

    f.close()
    return atoms

def write_lammps_data(fileobj, atoms, specorder=[], force_skew=False, write_charge=False):
    """Method which writes atomic structure data to a LAMMPS data file."""
    if isinstance(fileobj, str):
#        f = paropen(fileobj, 'w')
        f = open(fileobj, 'w')
        close_file = True
    else:
        # Presume fileobj acts like a fileobj
        f = fileobj
        close_file = False

    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError('Can only write one configuration to a lammps data file!')
        atoms = atoms[0]

    f.write(f.name + ' (written by ASE) \n\n')

    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    f.write('%d \t atoms \n' % n_atoms)

    if specorder is None:
        # This way it is assured that LAMMPS atom types are always
        # assigned predictively according to the alphabetic order 
        species = sorted(list(set(symbols)))
    else:
        # To index elements in the LAMMPS data file (indices must
        # correspond to order in the potential file)
        species = specorder
    n_atom_types = len(species)
    f.write('%d  atom types\n' % n_atom_types)

    p = Prism(atoms.get_cell())
    xhi, yhi, zhi, xy, xz, yz = p.get_lammps_prism_str()

    f.write('0.0 %s  xlo xhi\n' % xhi)
    f.write('0.0 %s  ylo yhi\n' % yhi)
    f.write('0.0 %s  zlo zhi\n' % zhi)
    
    if force_skew or p.is_skewed():
        f.write('%s %s %s  xy xz yz\n' % (xy, xz, yz))
    f.write('\n\n')

    f.write('Atoms \n\n')
    # xph: add charge in data file 
    if write_charge:
        for i, r in enumerate(map(p.pos_to_lammps_str,
                                  atoms.get_positions())):
            s = species.index(symbols[i]) + 1
            charge = atoms[i].charge
            f.write('%6d %3d %.4f %s %s %s\n' % ((i+1, s, charge)+tuple(r)))
    else:
        for i, r in enumerate(map(p.pos_to_lammps_str,
                                  atoms.get_positions())):
            s = species.index(symbols[i]) + 1
            f.write('%6d %3d %s %s %s\n' % ((i+1, s)+tuple(r)))
    
    if close_file:
        f.close()

