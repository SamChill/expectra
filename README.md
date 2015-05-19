Expectra is program to simulate EXAFS from the outputs of molecular
dynamics simulations. It also has the ability to to sample structures
based on a harmonic potential generated from a normal modes calculation.

The EXAFS multiple scattering calculations are performed using [FEFF6-lite][feff],
which was written at the University of Washington by J.J. Rehr and co-workers.<sup>1</sup>

1. J.J. Rehr, S.I. Zabinsky and R.C. Albers,
"High-order multiple scattering calculations of x-ray-absorption
fine structure", *Phys. Rev. Lett.* **69**, 3397 (1992).


[feff]: http://www.feffproject.org/

### Installation

This program is distributed as a Python package. It makes use the Python libraries numpy and mpi4py. It will also require a Fortran compiler to build the FEFF6-lite program that is redistributed with the code.

First build FEFF6-lite by running `make` in the `bin/` directory. If gfortran is installed then the executable `feff` will be produced. Next install an MPI library such as OpenMPI or MPICH2 and then you can install mpi4py and numpy. On Ubuntu the following commands should get all the dependencies installed:

```
$ sudo apt-get install build-essential gfortran python-mpi4py python-numpy
```

Once the dependencies are installed the Python package needs to be added to the `PYTHONPATH` environment variable and the path to the `bin/` folder needs to be added to the `PATH` environment variable.

### Usage

```
usage: expectra [-h] [--first-shell] [--neighbor-cutoff DISTANCE]
                [--multiple-scattering] [--rmax DISTANCE] [--S02 FACTOR]
                [--energy-shift ENERGY] [--absorber ELEMENT]
                [--ignore-elements ELEMENTS] [--edge EDGE] [--skip SKIP]
                [--every EVERY]
                TRAJ [TRAJ ...]

positional arguments:
  TRAJ                  trajectory file (POSCAR, con, xyz)

optional arguments:
  -h, --help            show this help message and exit
  --first-shell         a single scattering calculation that uses an
                        automatically calculated reference path (default:
                        True)
  --neighbor-cutoff DISTANCE
                        1st neighbor cutoff distance (default: 3.4)
  --multiple-scattering
  --rmax DISTANCE       maximum scattering half-path length
  --S02 FACTOR          amplitude reduction factor
  --energy-shift ENERGY
                        energy shift to apply in eV
  --absorber ELEMENT    atomic symbol of the xray absorber
  --ignore-elements ELEMENTS
                        comma delimited list of elements to ignore in the
                        scattering calculation
  --edge EDGE           one of K, L1, L2, L3
  --skip SKIP           number of frames to skip at the beginning
  --every EVERY         number of frames to between each step
```
