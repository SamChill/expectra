import sys
import numpy as np

from ase.optimize.optimize import Dynamics
from ase.optimize.fire import FIRE
#from ase.optimize.LBFGS import LBFGS
from ase.units import kB
from ase.parallel import world
from ase.io.trajectory import Trajectory

class BasinHopping(Dynamics):
    """Basin hopping algorithm.

    After Wales and Doye, J. Phys. Chem. A, vol 101 (1997) 5111-5116

    and 

    David J. Wales and Harold A. Scheraga, Science, Vol. 285, 1368 (1999)
    """

    def __init__(self, atoms,
                 opt_calculator = None,
                 exafs_calculator = None,
                 basin = True,
                 parabolic = True,
                 optimizer=FIRE,
                 fmax=0.1,
                 dr=0.1,
                 ratio=0.0,
                 temperature=100 * kB,
                 logfile='-',
                 chi_logfile='chi_log.dat',
                 parabola_log = 'parabola.dat',
                 trajectory='lowest.traj',
                 optimizer_logfile='-',
                 local_minima_trajectory='local_minima.traj',
                 adjust_cm=True):
        """Parameters:

        atoms: Atoms object
            The Atoms object to operate on.
        basin: switch on basin hopping when True
        parabolic: switch on parabolic_push method when True

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.
        pseudo_pot: pseudo potential defined
        """
        Dynamics.__init__(self, atoms, logfile, trajectory)
        self.kT = temperature
        self.basin = basin
        self.parabolic = parabolic
        self.optimizer = optimizer
        self.fmax = fmax
        self.dr = dr
        self.ratio = ratio
        self.opt_calculator = opt_calculator
        self.exafs_calculator = exafs_calculator
        self.chi_logfile = chi_logfile
        self.parabola_log = parabola_log
        self.k = None
        self.chi = None

        if basin:
           print "Basin Hopping switched on"
        elif parabolic:
           print "Parabolic pushing switched on"
        else:
           print "basin and parabolic, at least one must be true"

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
        #self.energy = 1.e32
        self.Umin = 1.e32
        #self.chi_deviation = 100.00
        self.rmin = self.atoms.get_positions()
        self.positions = self.atoms.get_positions()
        self.call_observers()
        self.debug = open('debug.dat', 'w')

        #'logfile' is defined in the superclass Dynamics in 'optimize.py'
#        self.logfile.write('   name      step accept     energy         chi_difference\n')
                
    def run(self, steps):
        """Hop the basins for defined number of steps."""
        ro = self.positions
        dot_o = self.get_ES_dot(ro)
        
        ratio = self.ratio
        alpha = dot_o[0]*ratio/dot_o[1]
        Uo = dot_o[0] + alpha * dot_o[1]

        parabola = []
        parabola.append(dot_o)

        #log data
        self.log_parabola = open(self.parabola_log, 'w')
#        self.chi_log = open(self.chi_logfile, 'w')
#        self.log_chi(-1)
#        self.log(-1, 1, conf_o[1], conf_o[2])

        #Basin Hopping methods
        for step in range(steps):
              Un = None
              while Un is None:
                  rn = self.move(ro)
                  if np.sometrue(rn != ro):
                      dot_n = self.get_ES_dot(rn)

                      if basin:
                         alpha = ratio * np.absolute(dot_n[0]-dot_o[0]) / np.absolute(dot_n[1] - dot_o[1])
                         Un = dot_n[0] + alpha * dot_n[1]
                         accept = np.exp((Uo - Un) / self.kT) > np.random.uniform()
                      else:
                         accept = False
#                      self.log_chi(step)
              if Un < self.Umin:
                  # new minimum found
                  self.Umin = Un
                  self.rmin = self.atoms.get_positions()

              if self.lm_trajectory is not None:
                  self.lm_trajectory.write(self.atoms)

              #accept or reject?
              self.debug.write('%s %d\n' % ('step', step))
              
              if parabolic:
                 parabola_n = self.parabolic_push(step, parabola, dot_n)
                 if cmp(parabola, parabola_n) != 0:
                    parabola = parabola_n
                    para_accept = True
                 else:
                    para_accept = False
                 
              if accept or para_accept:
                  ro = rn.copy()
                  Uo = Un

              self.log(step, accept, para_accept, alpha, dot_n[0], dot_n[1], Un, self.Umin)
              
#              self.log(step, accept, conf_n[1], conf_n[2])
#              self.log_chi(step)

    def parabolic_push(self, step, parabola, dot_n): 
        """
        Parabolic method to determine if accept rn or not
        """
        #only one dot in parabola
        temp = parabola
        if len(temp)==1:
           if dot_n[0] < temp[0][0] and dot_n[1] > temp[0][1]:
              parabola[0] = dot_n
              parabola.append(temp[0])
           elif dot_n[0] > temp[0][0] and dot_n[1] < temp[0][1]:
              parabola.append(dot_n)
           elif dot_n[0] < temp[0][0] and dot_n[1] < temp[0][1]:
              parabola[0] = dot_n
           
           self.logParabola(step, parabola)
           return parabola
        
        #more than one dot in parabola
        replace = False
        for i in range(0, len(temp)):
            #parabola may change after every cycle. Need to find new index for element temp[i]
            index = self.find_index(parabola, temp[i])

            #corner situation
            if i == 0:
               #start point
               dot_b1 = np.sum([temp[0], [0, 1]], axis=0)
               dot_c  = temp[0]
               dot_f1 = temp[i+1]
               if len(temp) == 2:
                  dot_f2 = np.sum([temp[1], [1, 0]], axis=0)
               else:
                  dot_f2 = temp[i+2]

               if dot_n[0] < dot_b1[0]:
                  parabola[index] = dot_n
                  replace = True

            elif i == len(temp)-2 and i != 0:
               #next to last
               dot_b1 = temp[i-1]
               dot_c  = temp[i]
               dot_f1 = temp[i+1]
               #creat a pseudo dot
               dot_f2 = np.sum([temp[len(temp)-1], [1, 0]], axis=0)

            elif i == len(temp)-1:
               #end point
               if dot_n[1] < temp[i][1]:
                  if replace:
                     parabola.pop(index)
                  else:
                     parabola[index] = dot_n
               continue
            else:
               dot_b1 = temp[i-1]
               dot_c  = temp[i]
               dot_f1 = temp[i+1]
               dot_f2 = temp[i+2]
            
            cross_b1 = self.get_cross(dot_b1, dot_c, dot_n)
            cross_f1 = self.get_cross(dot_c, dot_f1, dot_n)
            cross_f2 = self.get_cross(dot_f1, dot_f2, dot_n)

            if cross_b1 > 0 and cross_f1 > 0:
               self.debug.write('%s\n' % (replace))
               if i == 0:
                  #'replace' has already been done in corner situation
                  self.debug.write('%d\n' % (i))
                  continue
               if replace:
                  #A previous dot has already been replaced. Discard the current one
                  #'poped' is to record the number of dots poped before current one 
                  #which will change the location of current dot
                  parabola.pop(index)
                  self.debug.write('%d  %s  %d\n' % (i, 'poped',  index))
               else:
                  parabola[index] = dot_n
                  self.debug.write('%d  %s  %d\n' % (i, 'replaced',  index))
                  replace = True
               continue
            elif cross_b1 < 0  and cross_f1 > 0 and cross_f2 <0:
               #If 'insert' happens, no further action to other dots is needed
               parabola.insert(index+1, dot_n)
               self.debug.write('%d  %s  %d\n' % (i, 'inserted',  index+1))
               return parabola
            else:
               #For other situations, no acition is needed
               self.debug.write('%d  %s  %d\n' % (i, 'nothing done',  index))
               continue
        self.logParabola(step, parabola)
        return parabola

    def logParabola(self, step, parabola=[]):
        if self.log_parabola is None:
            return
        self.log_parabola.write('%s = %d  %s = %d\n' % ("step", step, "dot_number", len(parabola)))
        for i in range(0, len(parabola)):
             self.log_parabola.write('%15.6f  %15.6f\n' % (parabola[i][0], parabola[i][1]))
        self.log_parabola.flush()
        

    def log(self, step, accept, para_accept, alpha, En, Sn, Un, Umin):
        if self.logfile is None:
            return
        name = self.__class__.__name__
        self.logfile.write('%s: %d  %d %d %15.6f  %15.6f  %15.8f  %15.6f  %15.6f\n'
                           % (name, step, accept, para_accept, alpha, En, Sn, Un, Umin))
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

    def find_index(self, parabola, temp):
        numb = 0
        for i in range (len(parabola)):
            if parabola[i] == temp:
               index = i
               numb +=1
        if numb > 1:
           print "duplicated dots found in parabola"
           return None
        elif numb == 0:
           print "no corresponding dot is found in parabola"
           return None
        return index

    def get_cross(self, config_1, config_2, config_3):
        #calculate the cross vector
        vect1 = np.subtract(config_1, config_3)
        vect2 = np.subtract(config_2, config_3)
        cross_vect = np.cross(vect2, vect1)
        return cross_vect

    def get_minimum(self):
        """Return minimal energy and configuration."""
        atoms = self.atoms.copy()
        atoms.set_positions(self.rmin)
        return self.Umin, atoms
    
    """
    constructure dots in x_E_S space.
    x is set to 0 so that all the dots in E_S plane
    """
    def get_ES_dot(self, positions):
        self.positions = positions
        self.atoms.set_positions(positions)

        En = self.get_energy()
        chi_devi_n = self.get_chi_deviation()

        config = [En, chi_devi_n]

        return config
        
    def get_energy(self):
        """Return the energy of the nearest local minimum."""
        #opt_calculator can be any calculator compatible with ASE
        try:
            self.atoms.set_calculator(self.opt_calculator)
            opt = self.optimizer(self.atoms, 
                                 logfile=self.optimizer_logfile)
            opt.run(fmax=self.fmax)
           # if self.lm_trajectory is not None:
           #     self.lm_trajectory.write(self.atoms)

            energy = self.atoms.get_potential_energy()
        except:
            # Something went wrong.
            # In GPAW the atoms are probably to near to each other.
            return None

        return energy

    def get_chi_deviation(self):
        """Return the standard deviation of chi between calculated and
        experimental."""
        try:
            self.atoms.set_calculator(self.exafs_calculator)
            chi_deviation, self.k, self.chi = self.atoms.get_potential_energy()
        except:
            # Something went wrong.
            # In GPAW the atoms are probably to near to each other.
            return None
   
        return chi_deviation

