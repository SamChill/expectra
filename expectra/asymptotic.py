import sys
import numpy as np

from ase.optimize.optimize import Dynamics
from ase.optimize.fire import FIRE
#from ase.optimize.LBFGS import LBFGS
from ase.units import kB
from ase.parallel import world
from ase.io.trajectory import Trajectory

class Asymptotic(Dynamics):
    """Basin hopping algorithm.

    After Wales and Doye, J. Phys. Chem. A, vol 101 (1997) 5111-5116

    and 

    David J. Wales and Harold A. Scheraga, Science, Vol. 285, 1368 (1999)
    """

    def __init__(self, atoms,
                 opt_calculator = None,
                 exafs_calculator = None,
                 ratio = 0.01,
                 temperature=100 * kB,
                 optimizer=FIRE,
                 fmax=0.1,
                 dr=0.1,
                 logfile='-',
                 chi_logfile='chi_log.dat',
                 trajectory='lowest.traj',
                 optimizer_logfile='-',
                 local_minima_trajectory='local_minima.traj',
                 adjust_cm=True):
        """Parameters:

        atoms: Atoms object
            The Atoms object to operate on.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.
        pseudo_pot: pseudo potential defined
        """
        Dynamics.__init__(self, atoms, logfile, trajectory)
        self.kT = temperature
        self.optimizer = optimizer
        self.fmax = fmax
        self.dr = dr
        self.opt_calculator = opt_calculator
        self.exafs_calculator = exafs_calculator
        self.ratio = ratio
        self.chi_logfile = chi_logfile
        self.k = None
        self.chi = None

        if adjust_cm:
            self.cm = atoms.get_center_of_mass()
        else:
            self.cm = None

        self.optimizer_logfile = optimizer_logfile
        self.lm_trajectory = local_minima_trajectory
        if isinstance(local_minima_trajectory, str):
            self.lm_trajectory = Trajectory(local_minima_trajectory,
                                                  'w', atoms)

        self.initialize()

    def initialize(self):
        self.positions = 0.0 * self.atoms.get_positions()
        self.energy = 1.e32
        self.Umin = self.energy
        self.chi_deviation = 100.00
        self.rmin = self.atoms.get_positions()
        self.positions = self.atoms.get_positions()
        self.call_observers()

        #'logfile' is defined in the superclass Dynamics in 'optimize.py'
        self.logfile.write('   name      step accept     energy         chi_difference\n')
                
    def run(self, steps):
        """Hop the basins for defined number of steps."""

        ro = self.positions
        conf_o = self.get_energy_chi_diff(ro)

        #Construct a base line which take a negative slope in E_S space.
        slope = -1.0
        numb_cycle = 0
        while slope > 0:
              numb_cycle +=1
              rn = self.move(ro)
              if np.sometrue(rn != ro):
                 conf_n = self.get_energy_chi_diff(rn)
                 slope = (conf_n[2] - conf_o[2])/(conf_n[1]-conf_o[1])
        #conf_e and conf_s are two dots with smaller E and smaller S, respectively.
        #The line is determined by these two dots.
        if conf_n[1] > conf_o[1]:
           conf_e = conf_o
           conf_s = conf_n

        else:
           conf_e = conf_n
           conf_s = conf_o

        print "number of cycle used to construct a base line:"
        print(numb_cycle)

        #set a pseud-dot below the line but with small area
        ro = rn.copy()
        sn = conf_e[2] - (conf_e[2] - conf_s[2])/20
        conf_o = [0.0, conf_e[1], sn]
        #calculate the area of the triangle formed by the pseudo-dot and the base line and use it as an initial
        cross_vect_o = self.get_area(conf_s, conf_e, conf_o)
        ao = np.absolute(cross_vect_o)

        self.chi_log = open(self.chi_logfile, 'w')
        self.log_chi(-1)

        self.log(-1, 1, alpha, Eo, chi_devi_o, Uo, self.Umin)
        #initialize variables
        cross_vect_n = -1.0 
        step = 0

        while (step < steps):
              step += 1

              rn = self.move(ro)
              conf_n = self.get_energy_chi_diff(rn)

              """
              Deal with corner situations. Update the base line accordingly
              """
              vect = None
              area = self.get_area(conf_e, conf_s, vect)
              if conf_n[1] < conf_e[1] and conf_n[2] > conf_s[2]:
                 area_n = self.get_area(conf_n, conf_s, vect)
                 if area_n > area:
                    #update conf_n
                    conf_e = conf_n
                    re = rn.copy()
                    step = step - 1
                 continue

              elif conf_n[1] > conf_e[1] and conf_n[2] < conf_s[2]:
                 area_n = self.get_area(conf_n, conf_e, vect)
                 if area_n > area:
                    #update conf_s
                    conf_s = conf_n
                    rs = rn.copy()
                    step = step - 1
                 continue

              elif conf_n[1] < conf_e[1] and conf_n[2] < conf_s[2]:
                    step = step - 1
                 continue
              
              """
              cross_vect_n > 0 indicates the new dot below the base line. 
              All other corner situations have been ruled out. This dot should be in the
              triangle defined by the conf_o and the base line.
              """
              cross_vect_n = self.get_area(conf_s, conf_e, conf_n)
              if cross_vect_n > 0:
                 an = np.absolute(cross_vect_n) 
              else:
                 continue

              accept = np.exp(an - ao) > np.random.uniform()
              if accept:
                 ao = an
                 ro = rn.copy()
              self.log(step, accept, conf_n[1], conf_n[2])

    def log(self, step, accept, En, chi_diff_n):
        if self.logfile is None:
            return
        name = self.__class__.__name__
        self.logfile.write('%s: %d  %d  %15.6f  %15.8f\n'
                           % (name, step, accept, En, chi_diff_n))
        self.logfile.flush()

    def log_chi(self, step):
        self.chi_log.write("step: %d\n" % (step))
        k = self.k
        chi = self.chi
        for i in xrange(len(k)):
            self.chi_log.write("%6.3f %16.8e\n" % (k[i], chi[i]))
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
        return atoms.get_positions()

    def get_area(self, conf1, conf2, conf3=None):
         #calculate the area the defined rectangle trangle interested
        if conf3 is not None:
           vect1 = conf_1 - conf_3
           vect2 = conf_2 - conf_3
           cross_vect = np.cross(vect1, vect2)
           return cross_vect[0]
        area = np.absolute((conf1[1] - conf2[1]) * (conf1[2] - conf2[2]))
        return area

    def get_minimum(self):
        """Return minimal energy and configuration."""
        atoms = self.atoms.copy()
        atoms.set_positions(self.rmin)
        return self.Umin, atoms
    
    """
    constructure dots in x_E_S space.
    x is set to 0 so that all the dots in E_S plane
    """
    def get_energy_chi_diff(self, positions):
        self.positions = positions
        self.atoms.set_positions(positions)

        En = self.get_energy()
        chi_devi_n = self.get_chi_deviation()

        config = [0, En, chi_devi_n]

        return config
        
    def get_energy(self):
        """Return the energy of the nearest local minimum."""
        #opt_calculator can be any calculator compatible with ASE
        try:
            self.atoms.set_calculator(self.opt_calculator)
            opt = self.optimizer(self.atoms, 
                                 logfile=self.optimizer_logfile)
            opt.run(fmax=self.fmax)
            if self.lm_trajectory is not None:
                self.lm_trajectory.write(self.atoms)

            self.energy = self.atoms.get_potential_energy()
        except:
            # Something went wrong.
            # In GPAW the atoms are probably to near to each other.
            return None

        return self.energy

    def get_chi_deviation(self):
        """Return the standard deviation of chi between calculated and
        experimental."""
        try:
            self.atoms.set_calculator(self.exafs_calculator)
            self.chi_deviation, self.k, self.chi = self.atoms.get_potential_energy()
        except:
            # Something went wrong.
            # In GPAW the atoms are probably to near to each other.
            return None
   
        return self.chi_deviation

