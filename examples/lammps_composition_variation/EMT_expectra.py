#!/usr/bin/env python
import sys
sys.path.append("../../")
import os

#from ase.test import NotAvailable
from ase.io import read
from ase.units import kB
from ase.optimize.lbfgs import LBFGS
from ase.calculators.emt import EMT
from ase.atoms import Atoms
from expectra.cal_exafs import Expectra
#from expectra.io import read_xdatcar, read_con, read_chi
from expectra.paretoOPT import ParetoLineOptimize
from tsase.calculators.lammps_ext import LAMMPS

def main():
    #read in geometries
#    p1 = read('CONTCAR',index=0,format='vasp')
    p1 = read('Rh80Au20.xyz',index=0,format='xyz')
    p1.set_cell([[80,0,0],[0,80,0],[0,0,80]],scale_atoms=False,fix=None)
    p1.set_pbc((True, True, True))

#    charges = [setcharge(a.symbol) for a in p1]
#    p1.set_initial_charges(charges)
#    potfile=['library.meam','Au-Rh.meam']
    #calculator for geometry optimization and MD
    """
    opt_calc = LAMMPS(parameters={
                                  'pair_style':'meam',
                                  'pair_coeff':['* * library.meam Au Rh Au-Rh.meam Au Rh'],
                                  'atom_style':'charge'},
                      files=potfile,
                      tmp_dir='/home/leili/tmp',
                      keep_tmp_files=True,
                      always_triclinic=False,
                      specorder=['Au','Rh']
                      )
    """
    #set up exafs calculator
    exafs_calc = []
    calc = Expectra(ncore = 24,
                    multiple_scattering =  '--multiple-scattering',
                    neighbor_cutoff = 6.0,
                    S02 = 0.91,
                    energy_shift = -0.11,
                    edge = 'L3',
                    absorber = 'Au',
                    skip = 9001, 
                    every = 1,
                    kmin = 3.0,
                    kmax = 12.0,
                    rmin = 1.0,
                    rmax = 6.0,
                    dk = 2,
                    kweight = 2,
                    exp_file = 'AuL3_Au50Rh50.chir'
                    )
    exafs_calc.append(calc)
    calc = Expectra(ncore = 24,
                    multiple_scattering =  '--multiple-scattering',
                    neighbor_cutoff = 6.0,
                    S02 = 0.77,
                    energy_shift = -9,
                    edge = 'K',
                    absorber = 'Rh',
                    skip = 9001, 
                    every = 1,
                    kmin = 3.0,
                    kmax = 14.0,
                    rmin = 1.5,
                    rmax = 6.0,
                    dk = 2,
                    kweight = 2,
                    exp_file ='RhK_Au50Rh50.chir'
                    )
    exafs_calc.append(calc)
     
    #pareto line optimization
    plo = ParetoLineOptimize(atoms=p1,
                             nnode = 5,
                             ncore=24,
                             opt_calculator = 'lammps',
                             exafs_calculator=exafs_calc,
                             md = True,
                             specorder = ['Rh','Au'],
                             bh_steps = 30,
                             bh_steps_0 = 30,
                             temperature = 300* kB,
                             #md_step = 2,
                             dr=0.5,
                             logfile='pot_log.dat',
#                             optimizer=LBFGS,
#                             fmax=0.05,
                             move_atoms = False,
                             scale=True,
                             switch = True,
                             jumpmax = 5,
                             switch_ratio = 0.05,
                             elements_lib = ['Rh', 'Au'])
    #run job
    plo.run(10)
if __name__ == '__main__':
    main()
    
