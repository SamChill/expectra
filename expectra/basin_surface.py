import mpi4py.MPI
import numpy as np

import time
#from time import strftime
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md import MDLogger
from ase.optimize.optimize import Dynamics
from ase.optimize.fire import FIRE
from ase.units import kB, fs
from ase.parallel import world
from ase.io.trajectory import PickleTrajectory
from ase.io.trajectory import Trajectory
from ase.io import write
from ase.utils.geometry import sort
from expectra.md import run_md
from expectra.io import read_atoms
#from expectra.switch_elements import switch_elements
from expectra.atoms_operator import match
from expectra.lammps_caller import lammps_caller
import random
import sys
import os

#Flush buffer after each 'print' statement so that to see redirected output imediately 
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

COMM_WORLD = mpi4py.MPI.COMM_WORLD


class BasinHopping(Dynamics):
    """Basin hopping algorithm.

    After Wales and Doye, J. Phys. Chem. A, vol 101 (1997) 5111-5116

    and 

    David J. Wales and Harold A. Scheraga, Science, Vol. 285, 1368 (1999)
    """

    def __init__(self, atoms,
                 alpha = 0,
                 scale_ratio = 1.0,
                 pareto_step = None,
                 node_numb = None,
                 ncore = 2,
                 opt_calculator = None,
                 exafs_calculator = None,
                 #Switch or modify elements in structures
                 move_atoms = True,
                 switch = False,
                 active_ratio = None, #percentage of atoms will be used to switch or modified
                 cutoff=None,
                 elements_lib = None, #elements used to replace the atoms
                 #MD parameters
                 md = False,
                 md_temperature = 300 * kB,
                 md_step_size = 1 * fs,
                 md_steps = 1000,
                 max_md_cycle = 10,
                 md_trajectory = 'md',
                 specorder = None, #for 'lammps', specify the order of species which should be same to that in potential file
                 #Basin Hopping parameters
                 optimizer=FIRE,
                 max_optsteps=1000,
                 adjust_temperature = False,
                 temp_adjust_fraction = 0.05,
                 temperature=100 * kB,
                 fmax=0.1,
                 dr=0.1,
                 z_min=14.0,
                 substrate = None,
                 absorbate = None,
                 indistinguishable = True,
                 match_structure = False,
                 visited_configs = {}, # {'state_number': [energy, chi, repeats], ...}
                 comp_eps_e = 1.e-4, #criterion to determine if two configurations are identtical in energy 
                 comp_eps_r = 0.2, #criterion to determine if two configurations are identical in geometry
                 logfile='basin_log', 
                 trajectory='lowest.xyz',
                 optimizer_logfile='-',
                 local_minima_trajectory='localminima.xyz',
                 exafs_logfile = 'exafs.dat',
                 adjust_cm=True,
                 mss=0.2,
                 minenergy=None,
                 distribution='uniform',
                 adjust_step_size = None,
                 adjust_method = 'local', #method used to adjust dr, available selection: global, local, linear
                 target_ratio = 0.5,
                 adjust_fraction = 0.05,
                 significant_structure = False,  # displace from minimum at each move
                 significant_structure2 = False, # displace from global minimum found so far at each move
                 pushapart = 0.4,
                 jumpmax=None
                 ):

        self.pareto_step = str(pareto_step)
        self.node_numb = str(node_numb)
        self.ncore = ncore
        self.trajectory = trajectory
        self.opt_calculator = opt_calculator
        self.exafs_calculator = exafs_calculator

        if self.opt_calculator == 'lammps':
           print "lammps is true"
           self.lammps = True
        else:
           self.lammps = False

        self.move_atoms = move_atoms   
        self.switch = switch
        self.active_ratio = active_ratio
        self.cutoff = cutoff
        if self.active_ratio is not None:
           self.active_space = int(active_ratio * len(atoms))
        self.elements_lib = elements_lib
        self.indistinguishable = indistinguishable
        self.match_structure = match_structure
        self.visited_configs = visited_configs # list element: [step, energy, chi_diff, atoms]
        self.comp_eps_e = comp_eps_e
        self.comp_eps_r = comp_eps_r

        self.md = md
        self.md_temperature = md_temperature
        self.md_step_size = md_step_size
        self.md_steps = md_steps
        self.max_md_cycle = max_md_cycle
        self.md_trajectory = md_trajectory
        self.specorder = specorder

        self.optimizer = optimizer
        self.max_optsteps = max_optsteps
        self.adjust_temperature = adjust_temperature
        self.temp_adjust_fraction = temp_adjust_fraction
        self.kT = temperature
        self.fmax = fmax
        self.dr = dr
        self.z_min = z_min
        self.alpha = alpha
        self.scale_ratio = scale_ratio
        self.substrate = substrate
        self.absorbate = absorbate

        self.exafs_logfile = exafs_logfile 
        self.logfile = logfile 

        if adjust_cm:
            self.cm = atoms.get_center_of_mass()
        else:
            self.cm = None

        self.optimizer_logfile = optimizer_logfile 
        self.lm_traject = local_minima_trajectory
        self.log_trajectory = self.lm_traject
        if isinstance(local_minima_trajectory, str):
           self.log_trajectory = open(self.log_trajectory, 'w')

        self.atoms_debug = open("atoms_debug.xyz", 'w')
       # if isinstance(local_minima_trajectory, str):
       #     self.lm_trajectory = Trajectory(self.lm_traject,
       #                                           'w', atoms)
       # if isinstance(local_minima_trajectory, str):
       #     tsase.io.write_con(self.lm_trajectory,atoms,w='w')
       # self.all_local = Trajectory("all_opted_local.traj",
       #                                           'w', atoms)
        self.minenergy = minenergy
        self.distribution = distribution
        self.adjust_step_size = adjust_step_size
        self.adjust_method = adjust_method
        self.target_ratio = target_ratio
        self.adjust_fraction = adjust_fraction
        self.significant_structure = significant_structure
        self.significant_structure2 = significant_structure2
        self.pushapart = pushapart
        self.jumpmax = jumpmax
        self.mss = mss

        Dynamics.__init__(self, atoms, logfile, self.trajectory)
        self.initialize()

    #needed for ase 3.12.0 which includes 'todict' functions in Dynamics class
    def todict(self):
        d = {'type': 'optimization',
             'optimizer': self.__class__.__name__,
#             'local-minima-optimizer': self.optimizer.__name__,
             'temperature': self.kT,
             'max-force': self.fmax,
             'maximal-step-width': self.dr}
        return d

    def initialize(self):
        self.atoms_recording=None
        self.k = None
        self.chi = None
        self.chi_deviation = 100.00
        self.chi_differ = []
        self.positions = 0.0 * self.atoms.get_positions()
        self.Umin = 1.e32 
        self.energy = 1.e32
        self.rmin = self.atoms.get_positions()
        self.local_min_pos = self.atoms.get_positions()
        self.positions = self.atoms.get_positions()
        self.call_observers()
        self.time_stamp=None
        self.ratio = 0.0001
#        self.log(-1, self.Emin, self.Emin,self.dr)
                
        #'logfile' is defined in the superclass Dynamics in 'optimize.py'
#        self.logfile.write('#%12s: %s %s %21s %21s %21s %21s %21s\n'
#                            % ("name", "step", "accept", "alpha", "energy",
#                               "chi_deviation", "pseudoPot", "Umin"))

    def run(self, steps):
        """Hop the basins for defined number of steps."""
        self.steps = 0
        alpha = self.alpha
        scale_ratio = self.scale_ratio
        atom_min = None

        print 'Scale ratio: ', scale_ratio

        ro = self.atoms.get_positions()
        symbol_o = self.atoms.get_chemical_symbols()
        #self.time_stamp = strftime("%F %T")
        start_time = time.time()

        self.exafs_log = open(self.exafs_logfile, 'w')
        Eo = self.get_energy(ro, symbol_o, -1)

        if self.exafs_calculator is not None:
           chi_o = self.get_chi_deviation(ro, -1)
           Uo = (1.0 - alpha ) * Eo + alpha * scale_ratio * chi_o
        else:
           Uo = Eo
           chi_o = 0.0
           self.chi_differ.append(0.0)
        if self.match_structure:
           repeated, state = self.config_memo(-1)
           self.visited_configs[state][1] = chi_o
           self.visited_configs[state][2] = self.chi_differ
        print 'Energy: ', Eo, 'chi_differ: ', chi_o
        print '====================================================================='

        self.time_stamp = time.time() - start_time
        self.log_atoms(-1, Uo, chi_o)
        self.log(-1, True, alpha, Eo, self.chi_differ, Uo, Uo)

        acceptnum = 0
        recentaccept = 0
        rejectnum = 0
        for step in range(steps):
            Un = None
            self.steps += 1
            self.chi_differ = []
            while Un is None:
                if self.switch:
                   #symbol_n = switch_elements(self.atoms, symbol_o, self.cutoff)
                   symbol_n = self.random_swap(symbol_o)
                   rn = ro
                if self.move_atoms:
                   rn = self.move(ro)
                   symbol_n = symbol_o
                #self.time_stamp = strftime("%F %T")
                start_time = time.time()
                En = self.get_energy(rn, symbol_n, step)
                #check if the new configuration was visited
                if self.match_structure:
                   repeated, state = self.config_memo(step)
                else:
                   repeated = False
                print "repeated:", repeated
                if self.exafs_calculator is not None:
                   if not repeated:
                      #Calculate exafs for new structure
                      chi_n, stabilize= self.get_chi_deviation(self.atoms.get_positions(), step)
                      if not stabilize:
                         #The new structure can not be stabilized via MD simulation
                         self.visited_configs[state][1] = None
                         self.visited_configs[state][2] = None
                         #self.visited_configs.pop(state)
                         continue
                      self.visited_configs[state][1] = chi_n
                      self.visited_configs[state][2] = self.chi_differ
                      Un =(1 - alpha) * En + alpha * scale_ratio * chi_n
                   else:
                      chi_n = self.chi_deviation
                      if chi_n is None:
                         continue
                      Un =(1 - alpha) * En + alpha * scale_ratio * chi_n
                else:
                   Un = En
                   chi_n = 0.0

                self.time_stamp = time.time() - start_time
                self.log_atoms(step, Un, chi_n)
                print 'Energy: ', En, 'chi_differ: ', chi_n
                print '====================================================================='
            if Un < self.Umin:
                self.Umin = Un
                self.rmin = self.atoms.get_positions()
                #if not self.switch:
                self.call_observers()
                #record the atoms with minimum U
                atom_min = self.atoms

            #accept or reject?
            #take care of overflow problem for exp function
            if Un < Uo:
               accept = True
            else:
               accept = np.exp((Uo - Un) / self.kT) > np.random.uniform()

            if rejectnum > self.jumpmax and self.jumpmax is not None:
                accept = True
                rejectnum = 0
            if accept:
                print "accepted"
                acceptnum += 1.
                recentaccept += 1.
                rejectnum = 0
                if self.significant_structure == True:
                    #ro = self.local_min_pos.copy()
                    ro = self.atoms.get_positions()
                else:
                    ro = rn.copy()
                    if self.switch:
                       symbol_o = symbol_n
                Uo = Un
               # if self.lm_trajectory is not None:
               #     tsase.io.write_con(self.lm_trajectory,self.atoms,w='a')
            else:
                rejectnum += 1
            self.ratio = float(acceptnum)/float(self.steps)
            ratio = None
            if self.adjust_step_size is not None:
                if step % self.adjust_step_size == 0:
                    if self.adjust_method == 'global':
                       ratio = self.ratio
                    if self.adjust_method == 'local':
                       ratio = float(recentaccept)/float(self.adjust_step_size)
                       recentaccept = 0.

                    if self.adjust_method == 'linear':
                       self.dr = self.dr * self.ratio/self.target_ratio

                    if ratio is not None:
                       if ratio > self.target_ratio:
                          self.dr = self.dr * (1+self.adjust_fraction)
                          if self.adjust_temperature:
                             self.kT = self.kT * (1 - self.temp_adjust_fraction)
                          if self.switch:
                             self.active_ratio = self.active_ratio * (1-self.adjust_fraction)
                       elif ratio < self.target_ratio:
                           self.dr = self.dr * (1-self.adjust_fraction)
                           if self.adjust_temperature:
                              self.kT = self.kT * (1 + self.temp_adjust_fraction)
                           if self.switch:
                              self.active_ratio = self.active_ratio * (1+self.adjust_fraction)
            self.log(step, accept, alpha, En, self.chi_differ, Un, self.Umin)

            if self.minenergy != None:
                if Uo < self.minenergy:
                    break

        return atom_min

    def log(self, step, accept, alpha, En, chi_differ, Un, Umin):
        if self.logfile is None:
            return
        if step == -1:
           self.logfile.write('#%12s: %s %s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s\n'
                               % ("name", "step", "accept", "temperature", "active_ratio", "alpha", 
                                  "energy","chi_deviation", "chi", "pseudoPot", "Umin", "acc_ratio", "time"))
           
        name = self.__class__.__name__
        temp_chi = '   '.join(map(str, chi_differ))
        self.logfile.write('%s: %d  %d  %10.2f %10.2f  %15.6f  %15.6f  %15.6f  %s  %15.6f  %15.6f  %8.4f  %8.4f  %8.4f\n'
                           % (name, step, accept, self.kT/kB, self.active_ratio, alpha, 
                           En, self.chi_deviation, temp_chi, Un, Umin, self.dr, self.ratio, self.time_stamp))
        self.logfile.flush()

    def log_exafs(self, step, absorber):
        self.exafs_log.write("step: %d absorber: %s\n" % (step, absorber))
        k = self.k
        chi = self.chi
        for i in xrange(len(k)):
            self.exafs_log.write("%6.3f %16.8e\n" % (k[i], chi[i]))
        self.exafs_log.flush()

    def log_atoms(self,step, Un, chi_differ):
        self.log_trajectory.write("%d\n" % (len(self.atoms)))
        print type(Un), type(chi_differ)
        self.log_trajectory.write("node: %s  step: %d potential: %15.6f chi_differ: %15.6f \n"
                                  %(self.node_numb, step, Un, chi_differ))
        for atom in self.atoms:
            self.log_trajectory.write("%s  %15.6f  %15.6f  %15.6f\n" % (atom.symbol,
                                     atom.x, atom.y, atom.z))
        self.log_trajectory.flush()
    
    #log atoms, used for debug
    def dump_atoms(self,atoms):
        self.atoms_debug.write("%d\n" % (len(atoms)))
        self.atoms_debug.write(" \n")
        for atom in atoms:
            self.atoms_debug.write("%s  %15.6f  %15.6f  %15.6f\n" % (atom.symbol,
                                     atom.x, atom.y, atom.z))
        self.atoms_debug.flush()

    def move(self, ro):
        """Move atoms by a random step."""
        atoms = self.atoms
        disp = np.zeros(np.shape(atoms.get_positions()))
        while np.alltrue(disp == np.zeros(np.shape(atoms.get_positions()))):
           if self.distribution == 'uniform':
               disp = np.random.uniform(-self.dr, self.dr, (len(atoms), 3))
           elif self.distribution == 'gaussian':
               disp = np.random.normal(0,self.dr,size=(len(atoms), 3))
           elif self.distribution == 'linear':
               distgeo = self.get_dist_geo_center()
               disp = np.zeros(np.shape(atoms.get_positions()))
               for i in range(len(disp)):
                   maxdist = self.dr*distgeo[i]
               #    disp[i] = np.random.normal(0,maxdist,3)
                   disp[i] = np.random.uniform(-maxdist,maxdist,3)
           elif self.distribution == 'quadratic':
               distgeo = self.get_dist_geo_center()
               disp = np.zeros(np.shape(atoms.get_positions()))
               for i in range(len(disp)):
                   maxdist = self.dr*distgeo[i]*distgeo[i]
               #    disp[i] = np.random.normal(0,maxdist,3)
                   disp[i] = np.random.uniform(-maxdist,maxdist,3)
           else:
               disp = np.random.uniform(-1*self.dr, self.dr, (len(atoms), 3))
           
           #set all other disp to zero except a certain number of disp
           if self.active_ratio is not None:
              fix_space = len(atoms) - int(self.active_ratio * len(atoms))
              fix_atoms = random.sample(range(len(atoms)), fix_space)
              for i in range(len(fix_atoms)):
                  disp[fix_atoms[i]] = (0.0, 0.0, 0.0)
           #donot move substrate
           if self.substrate is not None:
              for i in range(len(atoms)):
                  for j in range(len(self.substrate)):
                      if i==self.substrate[j]:
                         disp[i] = (0.0,0.0,0.0)

        if self.significant_structure2 == True:
            ro,reng = self.get_minimum()
            rn = ro + disp
        else:
            rn = ro + disp
        atoms.set_positions(rn)
        if self.absorbate is not None:
           rn = self.push_to_surface(rn)
        rn = self.push_apart(rn)
        atoms.set_positions(rn)
        if self.cm is not None:
            cm = atoms.get_center_of_mass()
            atoms.translate(self.cm - cm)
        rn = atoms.get_positions()
        world.broadcast(rn, 0)
        atoms.set_positions(rn)
        return atoms.get_positions()

    def random_swap(self, symbols):
        atoms = self.atoms
        elements_lib = self.elements_lib
        switch_space = int(self.active_ratio * len(atoms))
        atoms.set_chemical_symbols(symbols)
        chemical_symbols = atoms.get_chemical_symbols()
        spec_index=[]
        elements_numb=[]
        for i in range(len(elements_lib)):
            spec_index.append([])
            elements_numb.append(0)
        for i in xrange(len(atoms)):
            for j in range (len(elements_lib)):
                if chemical_symbols[i] == elements_lib[j]:
                   spec_index[j].append(i)
                   elements_numb[j] += 1

        print "Elements_numb:", elements_numb
        if switch_space > min(elements_numb):
           switch_space = min(elements_numb)
           
        index_zero=random.sample(spec_index[0], switch_space)
        index_one=random.sample(spec_index[1], switch_space)
        for i in xrange(switch_space):
            chemical_symbols[index_zero[i]]=elements_lib[1]
            chemical_symbols[index_one[i]]=elements_lib[0]

        #atoms.set_chemical_symbols(chemical_symbols)
        #print "Elements_numb after switch:", atoms.symbols
        #counter = 0
        #for i in range(len(chemical_symbols)):
        #    if chemical_symbols[i] != symbols[i]:
        #      counter += 1
        #print "different atoms:", counter
        #self.dump_atoms(sort(atoms))

#        while (chemical_symbols == symbols):
#              print "Switching elements"
#              for i in range (switch_space):
#                  chemical_symbols[index[i]] = random.choice(elements_lib)

        #self.atoms.set_chemical_symbols(chemical_symbols)
        #write('switched.traj',images=self.atoms,format='traj')
        return chemical_symbols

    def get_minimum(self):
        """Return minimal energy and configuration."""
        atoms = self.atoms.copy()
        atoms.set_positions(self.rmin)
        print 'get_minimum',self.Umin
        return self.Umin, atoms
  
    def config_memo(self, step):
        """Add the new config if it is not visited before:
           Compare energies first, then compare geometries
        """
        lm_trajectory = self.lm_traject.split('_')
        repeated = False
        #print "visited_configs:",self.visited_configs
        if self.visited_configs:
           count = 0
           for state in self.visited_configs:
               if abs(self.energy - self.visited_configs[state][0]) < self.comp_eps_e:
                  state_list = state.split('_')
                  if len(state_list) == 1:
                     traj_file = self.lm_traject
                     #traj_file = 'geolog.xyz'
                     config_number = int(state) 
                  else:
                     traj_file = lm_trajectory[0] +'_'+ state_list[0] + '_'+ state_list[1]
                     config_number = int(state_list[2])
                  starttime = time.time()
                  config_o = read_atoms(traj_file, config_number)
                  donetime = time.time()
                  readtime += (donetime - starttime)
                  config_o.set_cell(self.atoms.get_cell())
               
                  #print "rot match or not:", rot_match(config_o, self.atoms, self.comp_eps_r)
                  #If the new structure has been visited, read chi data from library
                  if match(config_o, self.atoms, self.comp_eps_r, self.indistinguishable):
                     print "Found a repeat of state:", state
                     repeated = True
                     self.visited_configs[state][3] += 1
                     self.chi_deviation = self.visited_configs[state][1]
                     self.chi_differ = self.visited_configs[state][2]
                     return repeated, state
                  matchtime += time.time() - donetime 
                  count += 1
           print "readtime:", readtime, "matchtime:", matchtime, "number compared:", count

        #a new state is found or visited_configs is empty
        #Note: chi_deviation is not calculated yet
        if not repeated:
           if len(lm_trajectory) == 1:
              new_state = str(step)
           else:
              new_state = lm_trajectory[1] + '_' + lm_trajectory[2] + '_' + str(step)
           self.visited_configs[new_state] = [self.energy, 0.0, [0.0], 1]
        return repeated, new_state
    
    '''Run md simulation and self.atoms will be automatically updated after each run
       Trajectory will be stored in the file defined by 'self.md_trajectory'
    '''
    def run_md(self, md_steps=100):
        print "Running MD simulation:"
        # Set the momenta corresponding to md_temperature
        MaxwellBoltzmannDistribution(atoms=self.atoms, temp=self.md_temperature)
        # We want to run MD with constant temperature using the Langevin algorithm
        #dyn = VelocityVerlet(atoms, step_size, 
        #                     trajectory=trajectory)
        traj = Trajectory(self.md_trajectory, 'w',
                             self.atoms)
        dyn = Langevin(self.atoms, self.step_size, 
                       self.md_temperature, 0.002)
        log = MDLogger(dyn, self.atoms, 'md.log',
                       header=True, stress=False, peratom=False,
                       mode='w')
        dyn.attach(log, interval=1)
        dyn.attach(traj.write, interval=1)
        #for count in range (md_steps):
        dyn.run(md_steps)
        #    print count, self.atoms.get_potential_energy()
        #write('last.traj', self.atoms, format='traj')
        #print self.atoms.get_potential_energy()
        # Reset atoms to minimum point.

    '''
      run MD simulation until the structure is stabilized
    '''
    def stabilize_structure(self, max_md_cycle=10):
        atoms_initial = self.atoms
        stabilized = False
        md_cycle = 0
        while not stabilized or md_cycle < max_md_cycle:
           #TODO: run MD with lammps included in lammps_caller but not ase
           if self.lammps:
              md_trajectory = 'trj_lammps'
              lp =lammps_caller(atoms=self.atoms,
                                ncore = self.ncore,
                                specorder = self.specorder)
              lp.run('md')
           else:
              run_md(self.md_steps)
           md_cycle += 1
           if match(self.atoms, atoms_initial, self.comp_eps_r, self.indistinguishable):
              stabilized = True
        return stabilized, md_cycle 

    def get_energy(self, positions, symbols, step):
        """Return the energy of the nearest local minimum."""
#        if np.sometrue(self.positions != positions):
        self.positions = positions
        self.atoms.set_positions(positions)
        self.atoms.set_chemical_symbols(symbols)
        atoms = self.atoms
        #self.dump_atoms(sort(atoms))
        try:
            if self.lammps:
               print "lammps is running to optimize geometry"
               lp =lammps_caller(atoms=self.atoms,
                                 ncore = self.ncore,
                                 specorder = self.specorder)
               self.energy, self.atoms = lp.get_energy()
            else:
               self.atoms.set_calculator(self.opt_calculator)
               if self.optimizer.__name__ == "FIRE":
                  opt = self.optimizer(self.atoms,
                                       maxmove = 1.0,
                                       dt = 0.2, dtmax = 1.0,
                                           logfile=self.optimizer_logfile)
               else:
                  opt = self.optimizer(self.atoms,
                                           logfile=self.optimizer_logfile,
                                           maxstep=self.mss)
               print "geometry optimization is running"
               opt.run(fmax=self.fmax, steps=self.max_optsteps)
               self.energy = self.atoms.get_potential_energy()
            self.local_min_pos = self.atoms.get_positions()
            #write('opted.traj',images=self.atoms,format='traj')

            #if self.lm_trajectory is not None:
            #    self.lm_trajectory.write(self.atoms)
               

            print 'Total energy: ', self.energy
            print '--------------------------------------------'

            
#            if self.all_local is not None:
#               self.all_local.write(self.atoms)
        except:
                # Something went wrong.
                # In GPAW the atoms are probably to near to each other.
            return "Optimizer error"
        
        return self.energy

    def get_chi_deviation(self, positions, step):
        """Return the standard deviation of chi between calculated and
        experimental."""
        self.positions = positions
        self.atoms.set_positions(positions)

        if self.exafs_calculator is None:
           print "Warning: exafs is not calculated. Only energy used for basin hopping"
           self.chi_deviation = 0.0
           return 0.0

        try:
            if self.md:
               stabilize, md_cycle = self.stabilize_structure(self.max_md_cycle)
               if not stabilize:
                  return None, stabilize
            '''
               if self.lammps:
                  md_trajectory = 'trj_lammps'
                  lp =lammps_caller(atoms=self.atoms,
                                    ncore = self.ncore,
                                    specorder = self.specorder)
                  lp.run('md')
               else:
                  md_trajectory = self.md_trajectory
                  run_md(atoms=self.atoms, 
                         md_step = self.md_step,
                         temperature = self.md_temperature,
                         step_size = self.md_step_size,
                         trajectory = md_trajectory)
            
            print '--------------------------------------------'
            '''
        except:
            return "MD Error During EXAFS calculation", False

        try: 
            if isinstance(self.exafs_calculator, list):
               i = 0
               for calc in self.exafs_calculator:
                   if self.md:
                      print 'MD trajectories are used'
                      chi_deviation, self.k, self.chi = calc.get_chi_differ(filename=md_trajectory)
                   else:
                      print 'calculate exafs with optimized structure'
                      chi_deviation, self.k, self.chi = calc.get_chi_differ(atoms=self.atoms)
                   cp_cmd = 'cp chi.dat '+'exafs_' + self.specorder[i] +'.chi'
                   os.system(cp_cmd)
                   cp_cmd = 'cp exafs.chir '+'pdf_' + self.specorder[i] +'.chir'
                   os.system(cp_cmd)
                   i+=1
                   self.log_exafs(step, calc.get_absorber())
                   self.chi_differ.append(chi_deviation)
            else:
               if self.md:
                  print 'MD trajectories are used'
                  chi_deviation, self.k, self.chi = self.exafs_calculator.get_chi_differ(filename=md_trajectory)
               else:
                  print 'calculate exafs with optimized structure'
                  chi_deviation, self.k, self.chi = self.exafs_calculator.get_chi_differ(atoms = self.atoms)
               self.log_exafs(step, self.exafs_calculator.get_absorber())
               self.chi_differ.append(chi_deviation)

            self.chi_deviation = sum(x for x in self.chi_differ)
               
        except:
            # Something went wrong.
            # In GPAW the atoms are probably to near to each other.
            return None, False
   
        return self.chi_deviation, True


    def push_apart(self,positions):
        movea = np.zeros(np.shape(positions))
        alpha = 0.025
        for w in range(500):
            moved = 0
            movea = np.zeros(np.shape(positions))
            for i in range(len(positions)):
                for j in range(i+1,len(positions)):
                    d = positions[i] - positions[j]
                    magd = np.sqrt(np.vdot(d,d))
                    if magd < self.pushapart:
                        moved += 1
                        vec = d/magd
                        movea[i] += alpha *vec
                        movea[j] -= alpha *vec
            positions += movea
            if moved == 0:
                break
        self.atoms
        return positions

    #push absorbers up to surface
    def push_to_surface(self,positions):
        atoms = self.atoms
        for i in range(len(atoms)):
            for j in range(len(self.absorbate)):
                if i == self.absorbate[j]:
                   if positions[i][2] < self.z_min:
                      moved = (0.0, 0.0, self.z_min-positions[i][2])
                      positions[i] = positions[i] + moved
        return positions
            
    def get_dist_geo_center(self):
        position = self.atoms.get_positions()
        geocenter = np.sum(position,axis=0)/float(len(position))
        distance = np.zeros(len(position))
        for i in range(len(distance)):
            vec = position[i]-geocenter
            distance[i] = np.sqrt(np.vdot(vec,vec))
        distance /= np.max(distance)  #np.sqrt(np.vdot(distance,distance))
        return distance 
