import os
from ase.units import kB, fs
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS
#dictionary to store parameters for basin hopping
default_parameters=dict(
             #Switch or modify elements in structures
             move_atoms = True,
             dr_min = 0.1,  #reset dr if dr becomes smaller than dr_min
             switch = False,
             active_ratio = None, #percentage of atoms will be used to switch or modified
             cutoff=3.0,
             elements_lib = None, #elements used to replace the atoms
             #Structure optimization
             optimizer=FIRE,
             max_optsteps=1000,
             fmax=0.001,
             mss=0.2,
             #MD parameters
             md = False,
             md_temperature = 300 * kB,
             md_step_size = 2.0 * fs,
             md_steps = 4000,
             max_md_cycle = 10,
             md_ttime = 25.0*fs,
             md_trajectory = 'md.traj',
             md_interval = 20,
             in_memory_mode = True,
             specorder = None, #for 'lammps', specify the order of species which should be same to that in potential file
             #Basin Hopping parameters
             temperature = 300 * kB,
             distribution='uniform',
             adjust_method = 'local', #method used to adjust dr, available selection: global, local, linear
             adjust_step_size = None,
             target_ratio = 0.5,
             adjust_fraction = 0.05,
             adjust_temperature = False,
             temp_adjust_fraction = 0.05,
             significant_structure = False,  # displace from minimum at each move
             pushapart = 0.4,
             jumpmax=None,
             substrate = None,
             absorbate = None,
             z_min=14.0,
             adjust_cm=True,
             minenergy=None,
             #Structure Comparison
             indistinguishable = True,
             match_structure = False,
             comp_eps_e = 2.e-3, #criterion to determine if two configurations are identtical in energy 
             comp_eps_r = 0.2, #criterion to determine if two configurations are identical in geometry
             #files to log data
             logfile='basin_log', 
             trajectory='lowest.traj',
             optimizer_logfile='geo_opt.log',
             #directories to create for data logging
             configs_dir = os.getcwd()+'/configs',
             exafs_dir = os.getcwd()+'/exafs',
             pot_dir = os.getcwd()+'/pot'
             )

