# Au147 Example

This example will run a parallel EXAFS simulation using 4 processes,
where the L3-edge EXAFS spectrum will be calculated by running FEFF will be run
once for every 10 configurations contained in the `XDATCAR` file. The absorbing
atom is set to Au and all scattering interactions up to 6 Angstrom are
included. Additionally, an experimentally determined energy shift (E0) is
specified to be 4.11 eV. The first 100 configurations in the trajectory are
skipped in order to let the system reach local thermal equilibrium.

The file `run.sh` contains the following script that will run the calculation:
```bash
mpirun -n 4 expectra \
    --multiple-scattering \
    --absorber Au \
    --rmax 6.0 \
    --S02 0.837 \
    --energy-shift 4.11 \
    --edge L3 \
    --skip 1000 \
    --every 10 \
    XDATCAR
```

The output file `chi.dat` contains is the averaged chi(k) signal. It can be
plotted with gnuplot like so:
```
plot "chi.dat" w l
```
