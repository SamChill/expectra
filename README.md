Expectra is program to simulate EXAFS from the outputs of molecular
dynamics simulations. It also has the ability to to sample structures
based on a harmonic potential generated from a normal modes calculation.

The EXAFS multiple scattering calculations are performed using [FEFF6-lite][feff],
which was written at the University of Washington by J.J. Rehr and co-workers.<sup>1</sup>

1. J.J. Rehr, S.I. Zabinsky and R.C. Albers,
"High-order multiple scattering calculations of x-ray-absorption
fine structure", *Phys. Rev. Lett.* **69**, 3397 (1992).


[feff]: http://www.feffproject.org/

### Usage

```
usage: expectra [-h] [--first-shell] [--neighbor-cutoff DISTANCE]
                [--multiple-scattering] [--rmax DISTANCE] [--S02 FACTOR]
                [--energy-shift ENERGY] [--absorber ELEMENT]
                [--ignore-elements ELEMENTS] [--edge EDGE]
                {snapshots,modes} ...

positional arguments:
  {snapshots,modes}     sub-command help
    snapshots           average EXAFS signal from snapshots of a trajectory
    modes               average EXAFS signal from harmonic approximation using
                        normal modes

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
```
