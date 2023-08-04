# Expectra

## Introduction

Expectra is program to simulate EXAFS from the outputs of molecular
dynamics simulations. It also has the ability to to sample structures
based on a harmonic potential generated from a normal modes calculation.

The EXAFS multiple scattering calculations are performed using
[FEFF6-lite][feff], which was written at the University of Washington by J.J.
Rehr and co-workers.<sup>1</sup>

1. J.J. Rehr, S.I. Zabinsky and R.C. Albers,
"High-order multiple scattering calculations of x-ray-absorption
fine structure", *Phys. Rev. Lett.* **69**, 3397 (1992).

[feff]: http://www.feffproject.org/

## Requirements

* gfortran
* MPI (e.g. OpenMPI or MPICH)
* Python
  * numpy
  * mpi4py
  * [ASE](https://wiki.fysik.dtu.dk/ase/)

## Installation

This program is distributed as a Python package. It requires a Fortran
compiler (e.g. gfortran) to build the FEFF6-lite program that is redistributed with the code. A MPI library (e.g. OpenMPI) is also required.

The first step is to install Python, GFortran, and MPICH. On Ubuntu this can be accomplished like so:

```
$ sudo apt-get install build-essential gfortran mpich python python-pip
```

Once the dependencies are installed expectra (and the Python packages it
depends on) can be installed using pip:

```
$ pip install --user expectra
```

## Usage

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

## Installing from source

If installing Expectra from source instead of `pip`, clone the source code
from [GitHub](https://github.com/SamChill/expectra):

```
git clone https://github.com/SamChill/expectra.git
```

After all the dependency requirements have been satisfied, build the `feff` binary,
and move it to `expectra/expectra/`:

```
cd expectra/feff

# To use ifort instead of gfortran, set FC=ifort in ./Makefile

make clean
make  # should create the feff binary

mv ./feff ../expectra
```

Finally, set relevant environmental variables (assuming Expectra was installed at `$HOME`):

```
# consider including these in ~/.bash_profile, ~/.profile, or ~/.bashrc
export PATH=$HOME/expectra/bin:$PATH
export PYTHONPATH=$HOME/expectra:$PYTHONPATH
```

## Note from developers

Expectra was recently migrated from Python2 to Python3.
Please report any bugs to [sunghjung3@utexas.edu](sunghjung3@utexas.edu).
