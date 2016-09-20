#!/usr/bin/env python
import sys
sys.path.append("../../")
import os

#from ase.test import NotAvailable
from ase.io import read
from ase.optimize.lbfgs import LBFGS
from ase.calculators.emt import EMT
from expectra.aselite import Atoms
from expectra.cal_exafs import Expectra
#from expectra.io import read_xdatcar, read_con, read_chi
from expectra.basin import BasinHopping

def main():
    p1 = read('geometry.xyz',index=0,format='xyz')
    p1.set_cell([[20,0,0],[0,20,0],[0,0,20]],scale_atoms=False,fix=None)
    p1.set_pbc((True, True, True))

    #set up exafs calculator
    exafs_calc = Expectra(kmax = 14.0,
                          kmin = 2.0,
                          ncore = 24,
                          multiple_scattering =  '--multiple-scattering')

    bh = BasinHopping(atoms=p1,
                      opt_calculator = EMT(),
                      exafs_calculator=exafs_calc,
                      ratio = 0.3,
                      dr=0.5,
                      logfile='pot_log.dat',
                      optimizer=LBFGS,
                      fmax=0.05)
    #run job
    bh.run(500)
if __name__ == '__main__':
    main()
    
