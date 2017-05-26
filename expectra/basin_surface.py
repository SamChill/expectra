import mpi4py.MPI
import numpy as np

import time
import errno
import copy
#from time import strftime
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md import MDLogger
from ase.optimize.optimize import Dynamics
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS
from ase.units import kB, fs
from ase.parallel import world
from ase.io.trajectory import PickleTrajectory
from ase.io.trajectory import Trajectory
from ase.io import write
from ase.utils.geometry import sort
from expectra.io import read_atoms
from ase.md.npt import NPT
#from expectra.switch_elements import switch_elements
from expectra.atoms_operator import match, single_atom
from expectra.lammps_caller import lammps_caller, read_lammps_trj
from expectra import default_parameters
import random
import sys
import os

#Flush buffer after each 'print' statement so that to see redirected output imediately 
#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

COMM_WORLD = mpi4py.MPI.COMM_WORLD

default_parameters=default_parameters.default_parameters

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
           raise


class BasinHopping(Dynamics):
    """Basin hopping algorithm.

    After Wales and Doye, J. Phys. Chem. A, vol 101 (1997) 5111-5116

    and 

    David J. Wales and Harold A. Scheraga, Science, Vol. 285, 1368 (1999)
    """
    def __init__(self, atoms,
                 atoms_state = None, #the state_number of atoms passed from paretoOPT.py
                 dr = 0.5,
                 alpha = 0.0,
                 scale_ratio = 1.0,
                 pareto_step = None,
                 node_numb = None,
                 ncore = 2,
                 opt_calculator = None,
                 exafs_calculator = None,
                 visited_configs = {}, # {'state_number': [energy, chi, repeats], ...}
                 Umin = 1.0e32,
                 **kwargs
                 ):
        self.atoms_state = atoms_state
        self.dr = dr
        self.alpha = alpha
        self.scale_ratio = scale_ratio
        self.pareto_step = pareto_step
        self.node_numb = node_numb
        self.ncore = ncore
        self.opt_calculator = opt_calculator
        self.exafs_calculator = exafs_calculator
        self.visited_configs = visited_configs
        self.Umin = Umin

        if self.opt_calculator == 'lammps':
           print "lammps is true"
           self.lammps = True
        else:
           self.lammps = False

        for parameter in kwargs:
            if parameter not in default_parameters:
               print parameter, 'is not in the keywords included'
               break
        for (parameter, default) in default_parameters.iteritems():
            setattr(self, parameter, kwargs.get(parameter, default))

        if self.active_ratio is not None:
           self.active_space = int(self.active_ratio * len(atoms))

        self.kT = self.temperature
        self.time_logger = open('time_log.dat', 'a')

        if self.adjust_cm:
            self.cm = atoms.get_center_of_mass()
        else:
            self.cm = None

        #Store the optimized local minima
        #self.lm_traject = self.local_minima_trajectory
        #self.log_trajectory = self.lm_traject
        #if isinstance(self.local_minima_trajectory, str):
        #   self.log_trajectory = open(self.log_trajectory, 'w')

       # if isinstance(local_minima_trajectory, str):
       #     self.lm_trajectory = Trajectory(self.lm_traject,
       #                                           'w', atoms)
       # if isinstance(local_minima_trajectory, str):
       #     tsase.io.write_con(self.lm_trajectory,atoms,w='w')
       # self.all_local = Trajectory("all_opted_local.traj",
       #                                           'w', atoms)
        make_dir(self.configs_dir)
        make_dir(self.exafs_dir)
        make_dir(self.pot_dir)

        if self.pareto_step is not None:
           self.logfile = ''.join(filter(None,[self.pot_dir,'/',self.logfile,'_',self.pareto_step,'_',self.node_numb]))
        else:
           self.logfile = ''.join(filter(None,[self.pot_dir,'/',self.logfile]))

        Dynamics.__init__(self, atoms, self.logfile, self.trajectory)
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
        self.energy = 1.e32
        self.rmin = self.atoms.get_positions()
        self.local_min_pos = self.atoms.get_positions()
        self.positions = self.atoms.get_positions()
        self.call_observers()
        self.time_stamp=None
        self.ratio = 0.0001
        self.md_run_time = 0.0
        self.md_cycle = 0
        self.move_step = 0
        self.acceptnumb = 0
        self.recentaccept = 0
        self.repeated = False
        self.state = None
        self.time_logger.write("{:>10}  {}  {}  {}\n".format('state','FileReading','StructureMatching','#OfStructureCompared'))
        #'logfile' is defined in the superclass Dynamics in 'optimize.py'

    def run(self, steps):
        """Hop the basins for defined number of steps."""
        alpha = self.alpha
        scale_ratio = self.scale_ratio
        atom_min = None
        
        print 'Scale ratio: ', scale_ratio

        ro = self.atoms.get_positions()
        symbol_o = self.atoms.get_chemical_symbols()
        start_time = time.time()
        #Calculate energy and chi for the initial structure
        #If atoms_state given, retrieve energy and chi data from visited_configs
        if self.atoms_state is not None:
           self.state =self.atoms_state
           self.energy = self.visited_configs[self.atoms_state][0]
           chi_o = self.visited_configs[self.atoms_state][1]
           self.chi_differ = copy.deepcopy(self.visited_configs[self.atoms_state][2])
           self.chi_deviation =chi_o
           Eo = self.energy
           Uo = (1.0 - alpha ) * Eo + alpha * scale_ratio * chi_o
        #Atoms_state not given: calculate energy and chi
        else:
           Eo = self.get_energy(ro, symbol_o)
           state = '_'.join(filter(None,[self.pareto_step, self.node_numb, str(-1)]))
           if self.match_structure:
              self.repeated, self.state = self.config_memo(state)
           if self.exafs_calculator is not None:
              if self.repeated:
                 chi_o = self.visited_configs[self.state][1]
                 self.chi_differ = copy.deepcopy(self.visited_configs[self.state][2])
              else:
                 chi_o, stabilize = self.get_chi_deviation(self.atoms.get_positions(), state)
                 self.visited_configs[self.state][1] = chi_o
                 self.visited_configs[self.state][2] = copy.deepcopy(self.chi_differ)
              print chi_o
              Uo = (1.0 - alpha ) * Eo + alpha * scale_ratio * chi_o
           else:
              Uo = Eo
              chi_o = 0.0
              self.chi_differ.append(0.0)
           self.log_atoms(state, Uo, chi_o)
           self.Umin =Uo
        print 'Energy: ', Eo, 'chi_differ: ', chi_o
        print '====================================================================='

        self.time_stamp = time.time() - start_time

        self.log(-1, True, self.state, alpha, Eo, self.chi_differ, Uo, self.Umin)

        acceptnum = 0
        recentaccept = 0
        rejectnum = 0
        for step in range(steps):
            #bad_numb = 0
            #bad_configs = []
            self.md_run_time = 0.0
            self.md_cycle = 0
            self.move_step = 0
            Un = None
            self.chi_differ = []
            curr_state = '_'.join(filter(None,[self.pareto_step, self.node_numb, str(step)]))

            start_time = time.time()
            while Un is None:
                self.move_step += 1
                if self.switch:
                   #symbol_n = switch_elements(self.atoms, symbol_o, self.cutoff)
                   symbol_n = self.random_swap(symbol_o)
                   rn = ro
                if self.move_atoms:
                   print '==========='
                   print 'move atoms'
                   rn = self.move(ro)
                   symbol_n = symbol_o
                En = self.get_energy(rn, symbol_n)
                #if single_atom(self.atoms, self.atoms.get_positions(), 5.0):
                #   bad_numb += 1
                #   bad_configs.append(self.atoms.copy())
                #   if bad_numb > 20:
                #      self.dump_atoms(atoms=bad_configs, filename='config_single.xyz')
                #      sys.exit()
                #   continue
                #check if the new configuration was visited
                if self.match_structure:
                   self.repeated, self.state = self.config_memo(curr_state)
                else:
                   self.repeated = False
                print "repeated:", self.repeated
                if self.exafs_calculator is not None:
                   if not self.repeated:
                      #Calculate exafs for new structure
                      chi_n, stabilize= self.get_chi_deviation(self.atoms.get_positions(), curr_state)
                      if not stabilize:
                         #The new structure can not be stabilized via MD simulation, go back to while loop
                         print 'the structure is not stabilized. move atoms again'
                         continue
                      #update Energy and Chi data
                      En = self.energy
                      self.visited_configs[self.state][1] = chi_n
                      self.visited_configs[self.state][2] = copy.deepcopy(self.chi_differ)
                      Un =(1 - alpha) * En + alpha * scale_ratio * chi_n
                   else:
                      chi_n = self.chi_deviation
                      if chi_n is None:
                         print 'none chi_n'
                         continue
                      Un =(1 - alpha) * En + alpha * scale_ratio * chi_n
                else:
                   Un = En
                   chi_n = 0.0

                self.time_stamp = time.time() - start_time
                if not self.repeated:
                   self.log_atoms(curr_state, Un, chi_n)
                print 'Energy: ', En, 'chi_differ: ', chi_n
                print '====================================================================='
            if Un < self.Umin:
                self.Umin = Un
                self.rmin = self.atoms.get_positions()
                #if not self.switch:
                self.call_observers()
                #record the atoms with minimum U
                atom_min = self.atoms.copy()

            #accept or reject?
            #take care of overflow problem for exp function
            if Un < Uo:
               accept = True
            #elif self.repeated:
            #   accept = True
            #   self.acceptnumb += 1.
            #   self.recentaccept += 1.
            #   rejectnum = 0
            #   self.adjust_step(step)
            #   self.log(step, accept, self.state, alpha, En, self.chi_differ, Un, self.Umin)
            #   continue
            else:
               accept = np.exp((Uo - Un) / self.kT) > np.random.uniform()

            if rejectnum > self.jumpmax and self.jumpmax is not None:
                accept = True
                rejectnum = 0
            if accept:
                print "accepted"
                self.acceptnumb += 1.
                self.recentaccept += 1.
                rejectnum = 0
                if self.significant_structure == True:
                    #get_positions() return a numpy array
                    ro = self.atoms.get_positions()
                else:
                    ro = rn.copy()
                    if self.switch:
                       symbol_o = symbol_n
                Uo = Un
            else:
                rejectnum += 1
            self.adjust_step(step)
            self.log(step, accept, self.state, alpha, En, self.chi_differ, Un, self.Umin)

            if self.minenergy != None:
                if Uo < self.minenergy:
                    break
        print "Basin Hopping completed successfully!"
        return copy.deepcopy(self.visited_configs), self.dr, self.Umin

    def adjust_step(self, step):
        self.ratio = float(self.acceptnumb)/float(step+2)
        ratio = None
        if self.adjust_step_size is not None:
            if step % self.adjust_step_size == 0:
                if self.adjust_method == 'global':
                   ratio = self.ratio
                if self.adjust_method == 'local':
                   ratio = float(self.recentaccept)/float(self.adjust_step_size)
                   self.recentaccept = 0

                if self.adjust_method == 'linear':
                   self.dr = self.dr * self.ratio/self.target_ratio

                if ratio is not None:
                   if ratio > self.target_ratio:
                      self.dr = self.dr * (1+self.adjust_fraction)
                      if self.adjust_temperature:
                         self.kT = self.kT * (1 - self.temp_adjust_fraction)
                      #if self.switch:
                      #   self.active_ratio = self.active_ratio * (1-self.adjust_fraction)
                   elif ratio < self.target_ratio:
                       self.dr = self.dr * (1-self.adjust_fraction)
                       if self.adjust_temperature:
                          self.kT = self.kT * (1 + self.temp_adjust_fraction)
                       #if self.switch:
                       #   self.active_ratio = self.active_ratio * (1+self.adjust_fraction)
        
    def log(self, step, accept, state, alpha, En, chi_differ, Un, Umin):
        if self.logfile is None:
            return
        temp_chi = '   '.join(map(str, chi_differ))
        if step == -1:
           self.logfile.write('#{:11s}: {:4s} {:5s} {:12s} {:8s} {:5s} {:15s} {:15s} {:15s} {:15s} {:15s} {:5s} {:8s} {:8s}\n'.format(
                                  "name", "step", "accept", "state", "accRatio", "alpha", 
                                  "energy","chi_deviation", "chi", "pseudoPot", "Umin","dr","MD/opt", "Tottime"))
           
        name = self.__class__.__name__
        #keep En and self.chi_deviation at Column 6 and Column 7 (start from 0).
        #otherwise, need to change read_dots in io.py
        if self.exafs_calculator is None:
           self.logfile.write('{:12s}: {:4d} {:5d} {:12s} {:8.6f} {:5.4f} {:15.6f} {:15.6f} {:15.6f} {:5.4f} {:8.4f}\n'.format(
                              name, step, accept, state, self.ratio, alpha, 
                              En, Un, Umin, self.dr, self.time_stamp))
        else:
           self.logfile.write('{:12s}: {:4d} {:5d} {:12s} {:8.6f} {:5.4f} {:15.6f} {:15.6f} {:15s} {:15.6f} {:15.6f} {:5.4f} {:2d} {:4d} {:8.4f} {:8.4f}\n'.format(
                              name, step, accept, state, self.ratio, alpha, 
                              En, self.chi_deviation, temp_chi, Un, Umin, 
                              self.dr, self.md_cycle, self.move_step, self.md_run_time, self.time_stamp))
        self.logfile.flush()

    def log_exafs(self, state, absorber, chi_deviation):
        #in_memory_mode for bimetallic will not store exafs but output it
        if self.in_memory_mode and len(self.specorder)==1:
           if len(self.visited_configs[self.state]) > 5:
              print "something wrong with config_memo"
              sys.exit()
           self.visited_configs[self.state].append(self.k)
           self.visited_configs[self.state].append(self.chi)
           return
        exafs_log = open(self.exafs_dir+'/'+state+'_'+absorber,'w')
        exafs_log.write("state: %s absorber: %s energy: %15.6f chi_differ: %15.6f\n" % 
                        (state, absorber, self.energy, chi_deviation))
        k = self.k
        chi = self.chi
        for i in xrange(len(k)):
            exafs_log.write("%6.3f %16.8e\n" % (k[i], chi[i]))
        exafs_log.close()

    def log_time(self, state, readtime, matchtime, count):
        if count == 0:
           self.time_logger.write("{:>10}{:12.6f}{:12.6f}{:>10d}{:12.6f}\n".format
                                   (state, readtime, matchtime, count, matchtime))
        else:
           self.time_logger.write("{:>10}{:12.6f}{:12.6f}{:>10d}{:12.6f}\n".format
                                   (state, readtime, matchtime, count, matchtime/float(count)))
        #self.time_logger.write(" %s %15.6f %15.6f %d\n" %
        #                        (state, readtime, matchtime, count))
        self.time_logger.flush() 

    def log_atoms(self, state, Un, chi_differ):
        #in_memory_mode for bimetallic will store and output atoms
        if self.in_memory_mode and len(self.specorder)==1:
           self.visited_configs[self.state][4] = self.atoms.copy()
           return
        output_atoms = open(self.configs_dir+'/'+state,'w')
        output_atoms.write("%d\n" % (len(self.atoms)))
        output_atoms.write("state: %s  potential: %15.6f chi_differ: %15.6f \n"
                                  %(state, Un, chi_differ))
        for atom in self.atoms:
            output_atoms.write("%s  %15.6f  %15.6f  %15.6f\n" % (atom.symbol,
                                     atom.x, atom.y, atom.z))
        output_atoms.close()

    #log atoms, used for debug
    def dump_atoms(self, atoms, filename, mode='w'):
        atoms_debug = open(filename, mode)
        for config in atoms:
            atoms_debug.write("%d\n" % (len(config)))
            atoms_debug.write(" \n")
            for atom in config:
                atoms_debug.write("%s  %15.6f  %15.6f  %15.6f\n" % (atom.symbol,
                                         atom.x, atom.y, atom.z))
        atoms_debug.close()

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
           #if self.active_ratio is not None:
           #   fix_space = len(atoms) - int(self.active_ratio * len(atoms))
           #   fix_atoms = random.sample(range(len(atoms)), fix_space)
           #   for i in range(len(fix_atoms)):
           #       disp[fix_atoms[i]] = (0.0, 0.0, 0.0)
           #donot move substrate
           if self.substrate is not None:
              for i in range(len(atoms)):
                  for j in range(len(self.substrate)):
                      if i==self.substrate[j]:
                         disp[i] = (0.0,0.0,0.0)

        #if self.significant_structure2 == True:
        #    ro,reng = self.get_minimum()
        #    rn = ro + disp
        #else:
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

        return chemical_symbols

    def get_minimum(self):
        """Return minimal energy and configuration."""
        atoms = self.atoms.copy()
        atoms.set_positions(self.rmin)
        print 'get_minimum',self.Umin
        return self.Umin, atoms
  
    def config_memo(self, new_state, md_opt_cycle = False):
        """Add the new config if it is not visited before:
           Compare energies first, then compare geometries
        """
        repeated = False
        count = 0
        readtime = 0.0
        matchtime = 0.0
        if self.visited_configs:
           for state in self.visited_configs:
               #For a new state, the structure is not stored yet and also no need to compare
               if state == new_state:
                  continue
               if abs(self.energy - self.visited_configs[state][0]) < self.comp_eps_e:
                  starttime = time.time()
                  if self.in_memory_mode:
                     config_o = self.visited_configs[state][4]
                  else:
                     traj_file = self.configs_dir + '/' +state
                     configs = read_atoms(filename=traj_file)
                     config_o = configs[0]
                     config_o.set_cell(self.atoms.get_cell())
                     config_o.set_pbc(self.atoms.get_pbc())
                  donetime = time.time()
                  readtime += (donetime - starttime)
               
                  #If the new structure has been visited, read chi data from library
                  if match(config_o, self.atoms, self.comp_eps_r, self.cutoff, self.indistinguishable):
                     print "Found a repeat of state:", state
                     repeated = True
                     self.visited_configs[state][3] += 1
                     self.chi_deviation = self.visited_configs[state][1]
                     self.chi_differ = copy.deepcopy(self.visited_configs[state][2])
                     matchtime += time.time() - donetime 
                     count += 1

                     self.log_time(new_state, readtime, matchtime, count)

                     if len(state.split('_'))==1:
                        self.visited_configs[new_state]=self.visited_configs.pop(state)
                        return repeated, new_state

                     return repeated, state
                  matchtime += time.time() - donetime 
                  count += 1

        #A new state is found or visited_configs is empty
        #Note: chi_deviation is not calculated yet
        if not repeated:
           self.log_time(new_state, readtime, matchtime, count)
           if not md_opt_cycle:
              return repeated, new_state
           #if new_state in self.visited_configs:
           #   return repeated, new_state
           if self.in_memory_mode:
              self.visited_configs[new_state] = [self.energy, 0.0, [0.0], 1, self.atoms.copy()]
           else:
              self.visited_configs[new_state] = [self.energy, 0.0, [0.0], 1]
        return repeated, new_state
    
    '''Run md simulation and self.atoms will be automatically updated after each run
       Trajectory will be stored in the file defined by 'self.md_trajectory'
    '''
    def run_md(self, md_steps=100):
        print "Running MD simulation:"
        natoms=self.atoms.get_number_of_atoms()
        # Set the momenta corresponding to md_temperature
        #self.atoms.set_calculator(self.opt_calculator)
        atoms_md = []
        e_log = []
        atoms_md.append(self.atoms.copy())
        MaxwellBoltzmannDistribution(atoms=self.atoms, temp=self.md_temperature)
        # We want to run MD with constant temperature with Nose thermostat
        #dyn = NPT(self.atoms, timestep=self.md_step_size,temperature=self.md_temperature,
        #          externalstress=np.array((0,0,0,0,0,0)),ttime=self.md_ttime,pfactor=None,
        #          mask=(0,0,0))
        dyn = Langevin(atoms, self.md_step_size,self.md_temperature,0.05)
        if self.in_memory_mode:
           starttime = time.time()
           def md_log(atoms=self.atoms):
               atoms_md.append(atoms.copy())
               epot=atoms.get_potential_energy()
               ekin=atoms.get_kinetic_energy()
               temp = ekin / (1.5 * kB * natoms)
               e_log.append([epot, ekin, temp])
           dyn.attach(md_log, interval=self.md_interval)
           dyn.run(md_steps)
           print 'time used:',time.time()-starttime
           return atoms_md, e_log
           #for debug
           self.dump_atoms(atoms_md, 'md.xyz')
           log_e = open('md.log', 'w')
           i = 0
           for e in e_log:
               log_e.write("%d %15.6f %15.6f\n" %(i, e[0], e[1]))
               i+=1
           log_e.close()

        else:
           starttime=time.time()
           traj = Trajectory(self.md_trajectory, 'w',
                               self.atoms)
           log = MDLogger(dyn, self.atoms, 'md.log',
                          header=True, stress=False, peratom=False,
                          mode='w')
           dyn.attach(log, interval=self.md_interval)
           dyn.attach(traj.write, interval=self.md_interval)
           dyn.run(md_steps)
           print 'time used:',time.time()-starttime

    '''
      run MD simulation until the structure is stabilized.
      Geometry optimization is followed after each MD simulation, and self.energy and self.atoms updated each MD/OPT cycle
    '''
    def stabilize_structure(self, max_md_cycle=10):
        stabilized = False
        md_cycle = 0
        md_atoms = []
        print 'stabilizing structure'
        while not stabilized and md_cycle < max_md_cycle:
           pot_energy_old = self.energy
           atoms_initial = self.atoms.copy()
           md_atoms.append(self.atoms.copy())
           #TODO: run MD with lammps included in lammps_caller but not ase
           if self.lammps:
              md_trajectory = 'trj_lammps'
              lp =lammps_caller(atoms=self.atoms,
                                ncore = self.ncore,
                                specorder = self.specorder)
              lp.run('md')
              traj = read_lammps_trj(filename=md_trajectory, skip=0, specorder=self.specorder)
              self.atoms=traj[-1]
              print "------------"
              self.get_energy()
              pot_energy=self.energy
              print pot_energy
           else:
              try:
                 md_traj, e_log = self.run_md(md_steps=self.md_steps)
                 pot_energy=self.get_energy()
                 print pot_energy
              except:
                 print "MD Error"
                 sys.exit()
           md_cycle += 1
           #TODO: stabilization for large system 
           if len(self.atoms)>500:
              stabilized = True
           if abs(pot_energy - pot_energy_old)<self.comp_eps_e:
              match_results = match(self.atoms, atoms_initial, self.comp_eps_r, 3.0, self.indistinguishable)
              if match_results:
                 stabilized = True
                 if not self.lammps:
                     write(self.md_trajectory, images=md_traj, format='traj')
                     i = 0
                     log_e = open('md.log', 'w')
                     for e in e_log:
                         log_e.write("%d %15.6f %15.6f %15.6f\n" %(i, e[0], e[1], e[3]))
                         i+=1
                     log_e.close()
           print 'Stable:', stabilized
        #self.dump_atoms(md_atoms,'stabilize_stru.xyz')
        return stabilized, md_cycle 

    def get_energy(self, positions=None, symbols=None):
        """Return the energy of the nearest local minimum."""
#        if np.sometrue(self.positions != positions):
        if positions is not None:
           self.positions = positions
           self.atoms.set_positions(positions)
        if symbols is not None:
           self.atoms.set_chemical_symbols(symbols)
        atoms = self.atoms
        try:
            if self.lammps:
               print "lammps is running to optimize geometry"
               lp =lammps_caller(atoms=self.atoms,
                                 ncore = self.ncore,
                                 trajfile='opt_lammps',
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

        except:
                # Something went wrong.
                # In GPAW the atoms are probably to near to each other.
            return "Optimizer error"
        
        return self.energy

    def get_chi_deviation(self, positions, state):
        """Return the standard deviation of chi between calculated and
        experimental."""
        self.positions = positions
        self.atoms.set_positions(positions)

        if self.exafs_calculator is None:
           print "Warning: exafs is not calculated. Only energy used for basin hopping"
           self.chi_deviation = 0.0
           return 0.0
        #Stabilize Structure before exafs calculation
        if self.md:
           starttime = time.time()
           stabilize, md_cycle = self.stabilize_structure(self.max_md_cycle)
           self.md_cycle = md_cycle
           self.md_run_time = time.time()-starttime
           if not stabilize:
              return None, stabilize
           #Corner situation: structure is stabilized in 1 cycle. Same to initial structure
           #No need to check if visited. Structure was not recorded after geo_opt
           #For single-element composition
           if md_cycle == 1 and len(self.specorder)==1:
              if self.in_memory_mode:
                 self.visited_configs[state] = [self.energy, 0.0, [0.0], 1, self.atoms.copy()]
              else:
                 self.visited_configs[state] = [self.energy, 0.0, [0.0], 1]
              
           #The structure converts to a new structure during MD simulation OR for two-element systems
           #Check if it was visited or not. If visited, return chi_deviations
           if md_cycle > 1 or len(self.specorder)>1:
              self.repeated, self.state = self.config_memo(state, md_opt_cycle=True)
              #for the situation where visited_configs was read from a database
              #if state not in self.visited_configs:
              #   print "state not found in visited configs"
              #   return self.chi_deviation, True
              if self.repeated:
                 #self.visited_configs.pop(state)
                 print self.state, 'is repeated'
                 return self.chi_deviation, True

           if self.lammps:
              md_trajectory = 'trj_lammps'
           else:
              md_trajectory = self.md_trajectory
        
        print '--------------------------------------------'
        #except:
        #    return "MD Error During EXAFS calculation", False

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
                   #cp_cmd = 'cp chi.dat '+'exafs_' + self.specorder[i] +'.chi'
                   #os.system(cp_cmd)
                   #cp_cmd = 'cp exafs.chir '+'pdf_' + self.specorder[i] +'.chir'
                   #os.system(cp_cmd)
                   i+=1
                   self.log_exafs(state, calc.get_absorber(), chi_deviation)
                   self.chi_differ.append(round(chi_deviation, 6))
            else:
               if self.md:
                  print 'MD trajectories are used'
                  chi_deviation, self.k, self.chi = self.exafs_calculator.get_chi_differ(filename=md_trajectory)
               else:
                  print 'calculate exafs with optimized structure'
                  chi_deviation, self.k, self.chi = self.exafs_calculator.get_chi_differ(atoms = self.atoms)
               self.log_exafs(state, self.exafs_calculator.get_absorber(), chi_deviation)
               self.chi_differ.append(chi_deviation)

            self.chi_deviation = sum(x for x in self.chi_differ)
               
        except:
            # Something went wrong.
            # In GPAW the atoms are probably to near to each other.
            print "Something wrong with exafs calculation"
            sys.exit()
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
