#!/usr/bin/env python

import time
import copy
import numpy as np
import sys
import os

from math import sqrt, atan, cos, sin, tan, pi
from ase.io import read
from tsase.neb.util import vunit, vmag, vrand, sPBC
from tsase.calculators import morse
import sys


def Hessmatrix(p, displacement = 0.0005):
        dim    = 3
        Natoms = len(p)
        p.totalf = p.get_forces()
        ptmp = p.copy()
        ptmp.set_calculator(p.get_calculator())

        Hdim = dim * Natoms
        Hess = np.zeros((Hdim,Hdim),'d')
        nk=0
        for i in range(Natoms):
           for j in range(dim):
              rtmp = p.get_positions()
              rtmp[i][j] += displacement
              ptmp.set_positions(rtmp)
        
              # call force
              ptmp.totalf = ptmp.get_forces()
              ni = 0
              nj = 0
              for k in range(Hdim):
                  Hess[nk][k]=(p.totalf[ni][nj]-ptmp.totalf[ni][nj])/displacement
                  nj+=1

                  if (nj>dim-1):
                      ni+=1
                      nj=0
              nk+=1
              ptmp.set_positions(p.get_positions())
        return Hess

p=read('POSCAR', format='vasp')
calc = morse()
p.set_calculator(calc)
Hessian = Hessmatrix(p)
print Hessian



