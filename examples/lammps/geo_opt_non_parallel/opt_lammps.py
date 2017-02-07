#!/usr/bin/env python
import sys
import os

#from ase.test import NotAvailable
from ase.io import read
from ase.optimize.fire import FIRE
from ase.units import kB
from ase.atoms import Atoms
from tsase.calculators.lmplib import LAMMPSlib
#from expectra.lmplib import LAMMPSlib
from expectra.basin_surface import BasinHopping

def main():
    #read in geometries
    p1 = read('CONTCAR',index=0,format='vasp')

    cmds = ["pair_style eam",
            "pair_coeff * * Au_u3.eam"]
    lammps = LAMMPSlib(lmpcmds = cmds, 
                       #logfile='test.log', 
                       atoms=p1,
                       lammps_header=['units metal',
                                      'atom_style charge',
                                      'atom_modify map array sort 0 0.0'])
    bh = BasinHopping(atoms = p1,
                      opt_calculator = lammps,
                      #uncomment them if want to run md after geometry optimization
                      #md_step = 10000,
                      #md = True,
                      optimizer = FIRE
                      )
    print(bh.get_energy(p1.get_positions(),p1.get_chemical_symbols(),step=0))
if __name__ == '__main__':
    main()
    
