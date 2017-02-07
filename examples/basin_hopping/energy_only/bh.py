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
#from expectra.cal_exafs import Expectra
from expectra.basin_surface import BasinHopping

def main():
    #read in geometries
#    p1 = read('CONTCAR',index=0,format='vasp')
    p1 = read('au55.xyz',index=0,format='xyz')
    p1.set_cell([[20,0,0],[0,20,0],[0,0,20]],scale_atoms=False,fix=None)
    p1.set_pbc((True, True, True))

    #calculator for geometry optimization and MD
    opt_calc = EMT()
    bh = BasinHopping(atoms=p1,
                      opt_calculator = opt_calc,
                      temperature = 600* kB,
                      dr=0.5,
                      logfile='pot_log.dat',
                      optimizer=LBFGS,
                      fmax=0.05,
                      move_atoms = True,
                      switch = False,
                      adjust_step = True,
                      adjust_every = 5,
                      temp_adjust_fraction = 0.02,
                      jumpmax = 10
                      )
    #run job
    bh.run(20)
if __name__ == '__main__':
    main()
    
