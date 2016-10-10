import sys
import numpy as np
import copy

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
                 cutoff=3.20,
                 alpha = 0.5,
                 beta = 1.0,
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
        self.cutoff = cutoff
        self.alpha = alpha
        self.beta = beta
        self.opt_calculator = opt_calculator
        self.exafs_calculator = exafs_calculator
        self.chi_logfile = chi_logfile
        self.parabola_log = parabola_log

        #variables used to store g_r function(x: distance) or k_chi data(x: k, y: chi)
        self.x_thy = None
        self.y_thy = None

        if self.basin:
           print "Basin Hopping switched on"
        if self.parabolic:
           print "Parabolic pushing switched on"
        if not self.basin and not self.parabolic:
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
        self.rmin = self.atoms.get_positions()
        self.positions = self.atoms.get_positions()
        self.call_observers()
        self.debug = None

        #'logfile' is defined in the superclass Dynamics in 'optimize.py'
#        self.logfile.write('   name      step accept     energy         chi_difference\n')
                
    def run(self, steps):
        """Hop the basins for defined number of steps."""

        ro = self.positions
        dot_o = self.get_ES_dot(ro)
        print(type(dot_o))
        print(dot_o)
        beta = self.beta
        alpha = self.alpha
        parabola = []
        parabola.append(dot_o)

        #log data
        self.log_parabola = open(self.parabola_log, 'w')
        self.debug = open('debug.dat', 'w')
#        self.chi_log = open(self.chi_logfile, 'w')
#        self.log_chi(-1)
#        self.log(-1, 1, conf_o[1], conf_o[2])

        #Basin Hopping methods
        for step in range(steps):
              dot_n = None
              while dot_n is None:
                  rn = self.move(ro)
                  if np.sometrue(rn != ro) and not self.single_atom(rn):
                      print('%s = %d: %s' % ("Step", step, "Optimize the new structure and find area_diff"))
                      dot_n = self.get_ES_dot(rn)

                      if dot_n is None:
                         print "One bad structure with single atom is found"
                         continue

              self.debug.write('%s %d\n' % ("step", step))
              old_p = copy.deepcopy(parabola)
              accept = False
              if self.parabolic:
                 self.parabolic_push(step, parabola, dot_n)
                 print "old parabola:"
                 print(old_p)
                 print "new parabola:"
                 print(parabola)
                 if cmp(old_p, parabola) != 0:
                    self.logParabola(step, parabola)
                    ro = rn.copy()
                    dot_o = dot_n
                    para_accept = True
                 else:
                    para_accept = False
                    if self.basin:
                       alpha = self.get_alpha(parabola, dot_n)
                       print('%s: %15.6f' % ("alpha", alpha))
                       Uo = (1 - alpha) * dot_o[0] + alpha * beta * dot_o[1]
                       Un = (1 - alpha) * dot_n[0] + alpha * beta * dot_n[1]
                       accept = np.exp((Uo - Un) / self.kT) > np.random.uniform()
                       if accept:
                          ro = rn.copy()
                          dot_o = dot_n
                   
              self.log(step, accept, para_accept, alpha, dot_n[0], dot_n[1])
              
    def parabolic_push(self, step, parabola, dot_n): 
        """
        Parabolic method to determine if accept rn or not
        """
        #only one dot in parabola
        temp = copy.deepcopy(parabola)
        if len(temp)==1:
           if dot_n[0] < temp[0][0] and dot_n[1] > temp[0][1]:
              parabola[0] = dot_n
              parabola.append(temp[0])
           elif dot_n[0] > temp[0][0] and dot_n[1] < temp[0][1]:
              parabola.append(dot_n)
           elif dot_n[0] < temp[0][0] and dot_n[1] < temp[0][1]:
              parabola[0] = dot_n
           
           return parabola
        
        #more than one dot in parabola
        replace = False
        for i in range(0, len(temp)):
            #parabola may be changed after every cycle. Need to find new index for element temp[i]
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
#        self.logParabola(step, parabola)
        return parabola
    
    def logParabola(self, step, parabola=[]):
        if self.log_parabola is None:
            return
        self.log_parabola.write('%s = %d  %s = %d\n' % ("step", step, "dot_number", len(parabola)))
        for i in range(0, len(parabola)):
             self.log_parabola.write('%15.6f  %15.6f\n' % (parabola[i][0], parabola[i][1]))
        self.log_parabola.flush()
        

    def log(self, step, accept, para_accept, alpha, En, Sn):
        if self.logfile is None:
            return
        name = self.__class__.__name__
        self.logfile.write('%s: %d  %d %d %15.6f  %15.6f  %15.8f\n'
                           % (name, step, accept, para_accept, alpha, En, Sn))
        self.logfile.flush()

    def log_chi(self, step):
        self.chi_log.write("step: %d\n" % (step))
        x = self.x_thy
        y = self.y_thy
        for i in xrange(len(x)):
            self.chi_log.write("%6.3f %16.8e\n" % (x[i], y[i]))
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

    def single_atom(self, rn):
        """
        Check if there is single atom in the structure
        """
        atoms = self.atoms
        atoms.set_positions(rn)
        for i in range (len(atoms)):
            coordination = 0
            for j in range (len(atoms)):
                if j != i:
                   if atoms.get_distance(i,j) < self.cutoff:
                      coordination +=1
            if coordination == 0:
               return True
        return False

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

    def get_alpha(self, parabola, dot):
        """
        If a new dot cannot help push the parabola close to the origin, 
        find out which line on the parabola it is closest to.
        Then calculate alpha based on the slope of the line perpendicular to
        the closest line
        """
        temp_dot = copy.deepcopy(parabola)

        #Extend the parabola along two ending direction. The extended dots used
        #to construct two lines for the starting and the ending dots, respectively.
        temp_dot.insert(0, np.sum([parabola[0], [0, 1]], axis=0))
        temp_dot.append(np.sum([parabola[len(parabola)-1], [1, 0]], axis=0))
        #find out the line cloest to the dot
        for i in range (len(temp_dot)-1):
            cross_norm = np.absolute(self.get_cross(temp_dot[i+1], dot, temp_dot[i]))
            line_norm = np.linalg.norm(np.asarray(temp_dot[i]) - np.asarray(temp_dot[i+1]))
            distance = cross_norm / line_norm
            if i == 0:
               min_dist = distance
               index = 0
            print('%s %15.6f' % ("min_dist", min_dist))
            if distance < min_dist:
               min_dist = distance
               index = i
               print('%s %d' % ("new index", i))
        if index == 0:
           print "left end"
           alpha = 0.0
        elif index == len(temp_dot)-1:
           print "right end"
           alpha = 1.0
        else:
           slope = (temp_dot[index][1] - temp_dot[index+1][1]) / (temp_dot[index][0] - temp_dot[index+1][0])
           print('%s: %15.6f' % ("slope", slope))
           alpha = 1.0 / (1.0 - self.beta * slope)
        return alpha

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
        if self.single_atom(self.atoms.get_positions()):
           return None
        area_diff_n = self.get_area_diff()

        config = [En, area_diff_n]

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

            energy = self.atoms.get_potential_energy()
        except:
            # Something went wrong.
            # In GPAW the atoms are probably to near to each other.
            return None

        return energy

    def get_area_diff(self):
        """Return the area circulated by calculated and
        experimental curve."""
        try:
            self.atoms.set_calculator(self.exafs_calculator)
            area_diff, self.x_thy, self.y_thy = self.atoms.get_potential_energy()
        except:
            # Something went wrong.
            # In GPAW the atoms are probably to near to each other.
            return None
   
        return area_diff

