#!/usr/bin/env python
import sys
sys.path.append("../../")
import os

#from ase.test import NotAvailable
from ase.io import read
from ase.units import kB
from ase.atoms import Atoms
from expectra.cal_exafs import Expectra
from expectra.basin_surface import BasinHopping

def main():
    #read in geometries
    p1 = read('CONTCAR',index=0,format='vasp')

    #set up exafs calculator
    exafs_calc = []
    calc = Expectra(ncore = 24,
                    multiple_scattering =  '--multiple-scattering',
                    neighbor_cutoff = 6.0,
                    S02 = 0.89,
                    energy_shift = 3.4,
                    edge = 'L3',
                    absorber = 'Au',
                    specorder = "'Au'",
                    #skip = 999999, 
                    skip = 10001, 
                    every = 20,
                    kmin = 3.0,
                    kmax = 12.0,
                    rmin = 1.0,
                    rmax = 6.0,
                    dk = 2,
                    kweight = 2,
                    exp_file = 'exp_exafs.chir'
                    )
    exafs_calc.append(calc)
     
    #pareto line optimization
    bh = BasinHopping(atoms=p1,
                      ncore=24,
                      opt_calculator = 'lammps',
                      exafs_calculator=exafs_calc,
                      md = True,
                      specorder = ['Au'],
                      temperature = 300* kB,
                      logfile='pot_log',
                      elements_lib = ['Au'])
    #run job
    bh.run(0)
if __name__ == '__main__':
    main()
    
