from mpi4py import MPI
from lammps import PyLammps
from ase.io import read
#from lammps_ext import write_lammps_data
#from expectra.io import read_lammps_trj
#from lammps_caller import lammps_caller
import ase

L = PyLammps()
L.file("in.md")
#print "lammps run"


if MPI.COMM_WORLD.rank == 0:
   print("Potential energy: ", L.eval("pe"))

MPI.Finalize()
