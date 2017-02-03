# (written by ASE)
clear
#variable dump_file string "/home/leili/tmp/trj_lammps000001ngNear"
#variable data_file string "/home/leili/tmp/data_lammps000001vZOftm"
units metal 
boundary p p p 
atom_modify sort 0 0.0 
atom_style charge 

#geometry information
read_data ./data_lammps

### interactions 
pair_style meam 
pair_coeff * * library.meam Rh Au Au-Rh.meam Rh Au 

### run
neighbor	0.3 bin
neigh_modify  every 20 delay 0 check no
fix 1 all nvt temp 300.0 300.0 0.01
#write output every N timesteps in one of several styles to one or more miles
dump dump_all all custom 100 trj_lammps id type x y z vx vy vz fx fy fz q

#Set the style and content for printing thermodynamic data to the screen and log file.
thermo_style custom step temp press cpu pxx pyy pzz pxy pxz pyz ke pe etotal vol lx ly lz atoms
thermo_modify flush yes
#output thermodynamics every N timesteps
thermo 1
run 10000