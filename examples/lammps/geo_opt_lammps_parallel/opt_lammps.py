#!/usr/bin/env python
from ase.io import read
from ase.units import kB
from ase.atoms import Atoms
from expectra.basin_surface import BasinHopping

def main():
    #read in geometries
    p1 = read('CONTCAR',index=0,format='vasp')

    #pareto line optimization
    bh = BasinHopping(atoms=p1,
                      ncore=24,
                      opt_calculator = 'lammps',
                      #exafs_calculator=exafs_calc,
                      #md = True,
                      specorder = ['Au'],
                      temperature = 300* kB,
                      local_minima_trajectory='local_minima.xyz',
                      elements_lib = ['Au'])
    #run job
    bh.run(0)
if __name__ == '__main__':
    main()
    
