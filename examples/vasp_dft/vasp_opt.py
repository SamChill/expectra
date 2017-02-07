#!/usr/bin/env python
import sys
sys.path.append("../../")
import os

#from ase.test import NotAvailable
from ase.io import read
from ase.units import kB
from ase.calculators.vasp import Vasp
from ase.atoms import Atoms
from ase.optimize.fire import FIRE
from expectra.basin_surface import BasinHopping

def main():
    #read in geometries
    p1 = read('au20.xyz',index=0,format='xyz')
    p1.set_cell([[20,0,0],[0,20,0],[0,0,20]],scale_atoms=False,fix=None)
    p1.set_pbc((True, True, True))

    opt_calc = Vasp(prec = 'Low',
                    ediff = 1e-4,
                    sigma = 0.10,
                    nelm = 200,
                    nelmin = 8,
                    kpts = (1,1,1),
                    lcharg = False,
                    algo = 'Fast',
                    lreal= False,
                    lplane=True
                    )
    bh = BasinHopping(atoms = p1,
                      opt_calculator = opt_calc,
                      #uncomment them if want to run md after geometry optimization
                      #md_step = 10,
                      #md = True,
                      optimizer = FIRE
                      )

    print(bh.get_energy(p1.get_positions(),p1.get_chemical_symbols(),step=0))
if __name__ == '__main__':
    main()
    
