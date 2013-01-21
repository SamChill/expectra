#for intel fortran compiler
FC=ifort
FFLAGS=-O2

#for gfortran, note this needs a somewhat recent version of gfortran
#version 4.1 is known not to work
#FC=gfortran
#FFLAGS=-O2

feff: feff6l.f
	$(FC) $(FFLAGS) -o feff feff6l.f
.PHONY: clean
clean:
	rm feff
