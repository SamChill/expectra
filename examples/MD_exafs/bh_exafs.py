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
from expectra.basin_surface import BasinHopping

def main():
    #read in geometries
    p1 = read('au55.xyz',index=0,format='xyz')
    p1.set_cell([[20,0,0],[0,20,0],[0,0,20]],scale_atoms=False,fix=None)
    p1.set_pbc((True, True, True))

    #calculator for geometry optimization and MD
    opt_calc = EMT()
    #set up exafs calculator
    exafs_calc = []
    calc = Expectra(ncore = 24,
                    multiple_scattering =  '--multiple-scattering',
                    neighbor_cutoff = 6.0,
                    S02 = 0.91,
                    energy_shift = -0.11,
                    edge = 'L3',
                    absorber = 'Au',
                    skip = 0, 
                    every = 1,
                    kmin = 3.0,
                    kmax = 12.0,
                    rmin = 1.0,
                    rmax = 6.0,
                    dk = 2,
                    kweight = 2,
                    exp_file = 'AuL3_Au50.chir'
                    )
    exafs_calc.append(calc)
     
    #pareto line optimization
    bh = BasinHopping(atoms=p1,
                      opt_calculator = opt_calc,
                      exafs_calculator=exafs_calc,
                      optimizer=LBFGS,
                      md = True,
                      fmax=0.05
                      )
    #run job
    bh.run(20)
if __name__ == '__main__':
    main()
    
