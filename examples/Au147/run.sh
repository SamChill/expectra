#!/bin/bash
mpirun -n 4 expectra \
    --multiple-scattering \
    --absorber Au \
    --rmax 6.0 \
    --S02 0.837 \
    --energy-shift 4.11 \
    --edge L3 \
    --skip 100 \
    --every 10 \
    XDATCAR
