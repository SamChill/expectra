import numpy as np

from ase.optimize.optimize import Dynamics
from tsase.optimize.sdlbfgs import SDLBFGS
from ase.units import kB
from ase.parallel import world
from ase.io.trajectory import PickleTrajectory
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
import tsase
import sys

class BasinHopping(Dynamics):
    """Basin hopping algorithm.

    After Wales and Doye, J. Phys. Chem. A, vol 101 (1997) 5111-5116

    and 

    David J. Wales and Harold A. Scheraga, Science, Vol. 285, 1368 (1999)
    """

    def __init__(self, atoms,
                 opt_calculator = None,
                 exafs_calculator = None,
                 temperature=100 * kB,
                 optimizer=SDLBFGS,
                 alpha = 0,
                 fmax=0.1,
                 dr=0.1,
                 z_min=14.0,
                 substrate = None,
                 absorbate = None,
                 pareto_step = None,
                 node_numb = None,
                 logfile='basin_log', 
                 trajectory='lowest.traj',
                 optimizer_logfile='-',
                 local_minima_trajectory='local_minima.traj',
                 exafs_logfile = 'exafs',
                 adjust_cm=True,
                 mss=0.2,
                 minenergy=None,
                 distribution='uniform',
                 adjust_step_size=None,
                 adjust_every = None,
                 target_ratio = 0.5,
                 adjust_fraction = 0.05,
                 significant_structure = False,  # displace from minimum at each move
                 significant_structure2 = False, # displace from global minimum found so far at each move
                 pushapart = 0.4,
                 jumpmax=None
                 ):
        Dynamics.__init__(self, atoms, logfile, trajectory)
        self.opt_calculator = opt_calculator
        self.exafs_calculator = exafs_calculator
        self.kT = temperature
        self.optimizer = optimizer
        self.fmax = fmax
        self.dr = dr
        self.z_min = z_min
        self.alpha = alpha
        self.substrate = substrate
        self.absorbate = absorbate
        self.pareto_step = str(pareto_step)
        self.node_numb = str(node_numb)
        self.exafs_logfile = exafs_logfile + "_" + self.pareto_step + "_" + self.node_numb
        self.logfile = logfile + "_" + self.pareto_step + "_" + self.node_numb

        if adjust_cm:
            self.cm = atoms.get_center_of_mass()
        else:
            self.cm = None

        self.optimizer_logfile = optimizer_logfile + "_" + self.pareto_step + "_" + self.node_numb
        self.lm_traject = local_minima_trajectory + "_" + self.pareto_step + "_" + self.node_numb
        if isinstance(local_minima_trajectory, str):
            self.lm_trajectory = Trajectory(self.lm_traject,
                                                  'w', atoms)
       # if isinstance(local_minima_trajectory, str):
       #     tsase.io.write_con(self.lm_trajectory,atoms,w='w')
        self.traj_nonOPT = Trajectory("traj_before_push.traj",
                                                  'w', atoms)
        self.cst_nonOPT = Trajectory("cst_after_push.traj",
                                                  'w', atoms)
        self.all_local = Trajectory("all_opted_local.traj",
                                                  'w', atoms)
        self.minenergy = minenergy
        self.distribution = distribution
        self.adjust_step = adjust_step_size
        self.adjust_every = adjust_every
        self.target_ratio = target_ratio
        self.adjust_fraction = adjust_fraction
        self.significant_structure = significant_structure
        self.significant_structure2 = significant_structure2
        self.pushapart = pushapart
        self.jumpmax = jumpmax
        self.mss = mss
        self.initialize()

    def initialize(self):
        self.k = None
        self.chi = None
        self.chi_deviation = 100.00
        self.positions = 0.0 * self.atoms.get_positions()
        self.Umin = self.get_energy(self.atoms.get_positions()) or 1.e32 
        self.rmin = self.atoms.get_positions()
        self.positions = self.atoms.get_positions()
        self.call_observers()
#        self.log(-1, self.Emin, self.Emin,self.dr)
                
        #'logfile' is defined in the superclass Dynamics in 'optimize.py'
        self.logfile.write('%12s: %s %s %21s %21s %21s %21s %21s\n'
                            % ("name", "step", "accept", "alpha", "energy",
                               "chi_deviation", "pseudoPot", "Umin"))

    def run(self, steps):
        """Hop the basins for defined number of steps."""
        self.steps = 0
        alpha = self.alpha

        ro = self.positions
        Eo = self.get_energy(ro)
        chi_o = self.get_chi_deviation(self.atoms.get_positions())
        Uo = (1.0 - alpha ) * Eo + alpha * chi_o

        self.exafs_log = open(self.exafs_logfile, 'w')
        self.log(-1,'Yes', alpha, Eo, chi_o, Uo, self.Umin)

        acceptnum = 0
        recentaccept = 0
        rejectnum = 0
        for step in range(steps):
            Un = None
            self.steps += 1
            while Un is None:
                rn = self.move(ro)
                En = self.get_energy(rn)
                chi_n = self.get_chi_deviation(self.atoms.get_positions())
                Un = En + alpha * chi_n
            if Un < self.Umin:
                self.Umin = Un
                self.rmin = self.atoms.get_positions()
                self.call_observers()

            self.log(step, En, self.Emin,self.dr)
            
            #accept or reject?
            accept = np.exp((Uo - Un) / self.kT) > np.random.uniform()
            if rejectnum > self.jumpmax:
                accept = True
                rejectnum = 0
            if accept:
                acceptnum += 1.
                recentaccept += 1.
                rejectnum = 0
                if self.significant_structure2 == True:
                    ro = self.local_min_pos.copy()
                else:
                    ro = rn.copy()
                Uo = Un
                if self.lm_trajectory is not None:
                    self.lm_trajectory.write(self.atoms)
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
                    elif ratio < self.target_ratio:
                        self.dr = self.dr * (1-self.adjust_fraction)
            self.log(step, accept, alpha, En, chi_devi_n, Un, self.Umin)

    def log(self, step, accept, alpha, En, chi_devi_n, Un, Umin):
        if self.logfile is None:
            return
        name = self.__class__.__name__
        self.logfile.write('%s: %d  %s  %15.6f  %15.6f  %15.8f  %15.6f  %15.6f\n'
                           % (name, step, accept, alpha, En, chi_devi_n, Un, Umin))
        self.logfile.flush()

    def log_exafs(self, step, absorber):
        self.exafs_log.write("step: %d absorber: %s\n" % (step, absorber))
        k = self.k
        chi = self.chi
        for i in xrange(len(k)):
            self.exafs_log.write("%6.3f %16.8e\n" % (k[i], chi[i]))
        self.chi_log.flush()

    def move(self, ro):
        """Move atoms by a random step."""
        atoms = self.atoms
        # displace coordinates
        disp = np.random.uniform(-1., 1., (len(atoms), 3))
        rn = ro + self.dr * disp
        atoms.set_positions(rn)
        if self.cm is not None:
            cm = atoms.get_center_of_mass()
            atoms.translate(self.cm - cm)
        rn = atoms.get_positions()
        world.broadcast(rn, 0)
        atoms.set_positions(rn)

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
        print(disp)
        if self.substrate is not None:
           for i in range(len(atoms)):
               for j in range(len(self.substrate)):
                   if i==self.substrate[j]:
                      disp[i] = (0.0,0.0,0.0)
        print "new disp"
        print(disp)

        if self.significant_structure == True:
            rn = self.local_min_pos + disp
        elif self.significant_structure2 == True:
            ro,reng = self.get_minimum()
            rn = ro + disp
        else:
            rn = ro + disp
        atoms.set_positions(rn)
        if self.traj_nonOPT is not None:
            self.traj_nonOPT.write(atoms)
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

    def get_minimum(self):
        """Return minimal energy and configuration."""
        atoms = self.atoms.copy()
        atoms.set_positions(self.rmin)
        print 'get_minimum',self.Emin
        return self.Emin, atoms
  
    def run_md(self, positions):
        # Describe the interatomic interactions with the Effective Medium Theory
        atoms.set_calculator(self.opt_calculator)

        # Set the momenta corresponding to T=300K
        MaxwellBoltzmannDistribution(atoms, 300 * units.kB)

        # We want to run MD with constant energy using the VelocityVerlet algorithm.
        dyn = VelocityVerlet(atoms, 5 * units.fs)  # 5 fs time step.
        dyn.run(200)

    def get_energy(self, positions):
        """Return the energy of the nearest local minimum."""
        if np.sometrue(self.positions != positions):
            self.positions = positions
            self.atoms.set_positions(positions)
            if self.cst_nonOPT is not None:
               self.cst_nonOPT.write(self.atoms)
 
            try:
                self.atoms.set_calculator(self.opt_calculator)
                if self.optimizer.__name__ == "FIRE":
                   opt = self.optimizer(self.atoms,
                                            logfile=self.optimizer_logfile)
                else:
                   opt = self.optimizer(self.atoms,
                                            logfile=self.optimizer_logfile,
                                            maxstep=self.mss)
                #    opt = self.optimizer(self.atoms, 
                #                     logfile=self.optimizer_logfile,
                #                     maxstep=self.mss)
                opt.run(fmax=self.fmax)
                self.energy = self.atoms.get_potential_energy()
                
                if self.all_local is not None:
                    self.all_local.write(self.atoms)
                self.local_min_pos = self.atoms.get_positions()
            except:
                # Something went wrong.
                # In GPAW the atoms are probably to near to each other.
                return None
        
        return self.energy

    def get_chi_deviation(self, positions, step):
        """Return the standard deviation of chi between calculated and
        experimental."""
        self.positions = positions
        self.atoms.set_positions(positions)

        try: 
            for calc in self.exafs_calculator:
                self.atoms.set_calculator(self.calc)
                chi_deviation, self.k, self.chi = self.atoms.get_potential_energy()
                self.log_exafs(step, self.calc.get_absorber)
                self.chi_deviation = chi_deviation + self.chi_deviation
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
            print "moved atoms:"
            print(moved)
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
