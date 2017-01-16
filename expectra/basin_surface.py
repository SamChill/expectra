import mpi4py.MPI
import numpy as np

from time import strftime
from ase.optimize.optimize import Dynamics
from ase.optimize.fire import FIRE
from ase.units import kB, fs
from ase.parallel import world
from ase.io.trajectory import PickleTrajectory
from ase.io.trajectory import Trajectory
from ase.io import write
from expectra.md import run_md
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
                 switch_ratio = 0.1, #percentage of atoms will be used to switch or modified
                 elements_lib = None, #elements used to replace the atoms
                 #MD parameters
                 md = True,
                 md_temperature = 300 * kB,
                 md_step_size = 1 * fs,
                 md_step = 1000,
                 md_trajectory = 'md',
                 specorder = None, #for 'lammps', specify the order of species which should be same to that in potential file
                 #Basin Hopping parameters
                 optimizer=FIRE,
                 adjust_temperature = True,
                 temp_adjust_fraction = 0.01,
                 temperature=100 * kB,
                 fmax=0.1,
                 dr=0.1,
                 z_min=14.0,
                 substrate = None,
                 absorbate = None,
                 logfile='basin_log', 
                 trajectory='lowest',
                 optimizer_logfile='-',
                 local_minima_trajectory='local_minima',
                 exafs_logfile = 'exafs',
                 adjust_cm=True,
                 mss=0.2,
                 minenergy=None,
                 distribution='uniform',
                 adjust_step= True,
                 adjust_every = None,
                 target_ratio = 0.5,
                 adjust_fraction = 0.005,
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
        self.switch_ratio = switch_ratio
        self.switch_space = int(switch_ratio * len(atoms))
        self.elements_lib = elements_lib

        self.md = md
        self.md_temperature = md_temperature
        self.md_step_size = md_step_size
        self.md_step = md_step
        self.md_trajectory = md_trajectory
        self.specorder = specorder

        self.optimizer = optimizer
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
       # if isinstance(local_minima_trajectory, str):
       #     self.lm_trajectory = Trajectory(self.lm_traject,
       #                                           'w', atoms)
       # if isinstance(local_minima_trajectory, str):
       #     tsase.io.write_con(self.lm_trajectory,atoms,w='w')
        self.all_local = Trajectory("all_opted_local.traj",
                                                  'w', atoms)
        self.minenergy = minenergy
        self.distribution = distribution
        self.adjust_step = adjust_step
        self.adjust_every = adjust_every
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
        self.positions = self.atoms.get_positions()
        self.call_observers()
        self.time_stamp=None
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
        self.time_stamp = strftime("%F %T")

        self.exafs_log = open(self.exafs_logfile, 'w')
        Eo = self.get_energy(ro, symbol_o, -1)
        chi_o = self.get_chi_deviation(ro, -1)
        print 'Energy: ', Eo, 'chi_differ: ', chi_o
        print '====================================================================='
        Uo = (1.0 - alpha ) * Eo + alpha * scale_ratio * chi_o
        self.log_atoms(-1)
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
                   symbol_n = self.switch_elements(symbol_o)
                   rn = ro
                if self.move_atoms:
                   rn = self.move(ro)
                   symbol_n = symbol_o
                self.time_stamp = strftime("%F %T")
                En = self.get_energy(rn, symbol_n, step)
                self.log_atoms(step)
                chi_n = self.get_chi_deviation(self.atoms.get_positions(), step)
                print 'Energy: ', En, 'chi_differ: ', chi_n
                print '====================================================================='
                Un =(1 - alpha) * En + alpha * scale_ratio * chi_n
            if Un < self.Umin:
                self.Umin = Un
                self.rmin = self.atoms.get_positions()
                if not self.switch:
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
            print "accept: ", accept
            if accept:
                print "accepted"
                acceptnum += 1.
                recentaccept += 1.
                rejectnum = 0
                if self.significant_structure2 == True:
                    ro = self.local_min_pos.copy()
                else:
                    ro = rn.copy()
                    if self.switch:
                       symbol_o = symbol_n
                Uo = Un
               # if self.lm_trajectory is not None:
               #     tsase.io.write_con(self.lm_trajectory,self.atoms,w='a')
            else:
                rejectnum += 1
            if self.minenergy != None:
                if Uo < self.minenergy:
                    break
            if self.adjust_step == True:
                if step % self.adjust_every == 0:
                    ratio = float(acceptnum)/float(self.steps)
                    ratio = float(recentaccept)/float(self.adjust_every)
                    recentaccept = 0.
                    if ratio > self.target_ratio:
                       self.dr = self.dr * (1+self.adjust_fraction)
                       if self.adjust_temperature:
                          self.kT = self.kT * (1 - self.temp_adjust_fraction)
                       if self.switch:
                          self.switch_ratio = self.switch_ratio * (1-self.adjust_fraction)
                    elif ratio < self.target_ratio:
                        self.dr = self.dr * (1-self.adjust_fraction)
                        if self.adjust_temperature:
                           self.kT = self.kT * (1 + self.temp_adjust_fraction)
                        if self.switch:
                           self.switch_ratio = self.switch_ratio * (1+self.adjust_fraction)
            self.log(step, accept, alpha, En, self.chi_differ, Un, self.Umin)

        return atom_min

    def log(self, step, accept, alpha, En, chi_differ, Un, Umin):
        if self.logfile is None:
            return
        name = self.__class__.__name__
        temp_chi = '   '.join(map(str, chi_differ))
        self.logfile.write('%s: %d  %d  %10.2f %10.2f  %15.6f  %15.6f  %15.6f  %s  %15.6f  %15.6f  %s\n'
                           % (name, step, accept, self.kT/kB, self.switch_ratio, alpha, 
                           En, self.chi_deviation, temp_chi, Un, Umin, self.time_stamp))
        self.logfile.flush()

    def log_exafs(self, step, absorber):
        self.exafs_log.write("step: %d absorber: %s\n" % (step, absorber))
        k = self.k
        chi = self.chi
        for i in xrange(len(k)):
            self.exafs_log.write("%6.3f %16.8e\n" % (k[i], chi[i]))
        self.exafs_log.flush()

    def log_atoms(self,step):
        self.log_trajectory.write("%d\n" % (len(self.atoms)))
        self.log_trajectory.write("node: %s  step: %d\n" %(self.node_numb, step))
        for atom in self.atoms:
            self.log_trajectory.write("%s  %15.6f  %15.6f  %15.6f\n" % (atom.symbol,
                                     atom.x, atom.y, atom.z))
        self.log_trajectory.flush()

    def move(self, ro):
        """Move atoms by a random step."""
        atoms = self.atoms
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
        #donot move substrate
        if self.substrate is not None:
           for i in range(len(atoms)):
               for j in range(len(self.substrate)):
                   if i==self.substrate[j]:
                      disp[i] = (0.0,0.0,0.0)

        if self.significant_structure == True:
            rn = self.local_min_pos + disp
        elif self.significant_structure2 == True:
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

    def switch_elements(self, symbols):
        atoms = self.atoms
        atoms.set_chemical_symbols(symbols)
        elements_lib = self.elements_lib
        switch_space = int(self.switch_ratio * len(atoms))
        chemical_symbols = atoms.get_chemical_symbols()
        index=random.sample(xrange(len(atoms)), switch_space)
        while (chemical_symbols == symbols):
              print "Switching elements"
              for i in range (switch_space):
                  chemical_symbols[index[i]] = random.choice(elements_lib)

        #self.atoms.set_chemical_symbols(chemical_symbols)
        #write('switched.traj',images=self.atoms,format='traj')
        return chemical_symbols

    def get_minimum(self):
        """Return minimal energy and configuration."""
        atoms = self.atoms.copy()
        atoms.set_positions(self.rmin)
        print 'get_minimum',self.Umin
        return self.Umin, atoms
  
    def get_energy(self, positions, symbols, step):
        """Return the energy of the nearest local minimum."""
#        if np.sometrue(self.positions != positions):
        self.positions = positions
        self.atoms.set_positions(positions)
        self.atoms.set_chemical_symbols(symbols)
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
                                           logfile=self.optimizer_logfile)
               else:
                  opt = self.optimizer(self.atoms,
                                           logfile=self.optimizer_logfile,
                                           maxstep=self.mss)
               opt.run(fmax=self.fmax)
               self.energy = self.atoms.get_potential_energy()

#            write('opted.traj',images=self.atoms,format='traj')
#            write('opted.xyz',images=self.atoms_recording,format='xyz')

#            if self.lm_trajectory is not None:
#                self.lm_trajectory.write(self.atoms)
               

            print 'Total energy: ', self.energy
            print '--------------------------------------------'

            if self.md:
               if self.lammps:
                  lp =lammps_caller(atoms=self.atoms,
                                    ncore = self.ncore,
                                    specorder = ['Rh', 'Au'])
                  lp.run('md')
               else:
                  md_trajectory = self.md_trajectory+"_"+str(step)+".traj"
                  run_md(atoms=self.atoms, 
                         md_step = self.md_step,
                         temperature = self.md_temperature,
                         step_size = self.md_step_size,
                         trajectory = md_trajectory)

            print '--------------------------------------------'
            
#            if self.all_local is not None:
#               self.all_local.write(self.atoms)
#               self.local_min_pos = self.atoms.get_positions()
        except:
                # Something went wrong.
                # In GPAW the atoms are probably to near to each other.
            print "Optimizer or MD error"
            return "Optimizer or MD error"
        
        return self.energy

    def get_chi_deviation(self, positions, step):
        """Return the standard deviation of chi between calculated and
        experimental."""
        self.positions = positions
        self.atoms.set_positions(positions)

        if self.lammps:
           md_trajectory = 'trj_lammps'
        else:
           md_trajectory = self.md_trajectory+"_"+str(step)+".traj"

        try: 
            if isinstance(self.exafs_calculator, list):
               for calc in self.exafs_calculator:
                   if self.md:
                      print 'MD trajectories are used'
                      chi_deviation, self.k, self.chi = calc.get_chi_differ(filename=md_trajectory)
                   else:
                      print 'calculate exafs with optimized structure'
                      chi_deviation, self.k, self.chi = calc.get_chi_differ(atoms=self.atoms)
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
            return None
   
        return self.chi_deviation


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
