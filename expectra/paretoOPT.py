import sys
import numpy as np
import copy

from ase.optimize.optimize import Dynamics
from ase.optimize.fire import FIRE
#from ase.optimize.LBFGS import LBFGS
from ase.units import kB
from ase.parallel import world
from ase.io.trajectory import Trajectory

class ParetoLineOptimize(Dynamics):
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
        self.cutoff = cutoff
        self.ratio = ratio
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

        #initialize alpha for each basin hopping jobs
        interval = 1 / ncore
        target_ratio = interval
        alpha_list = []
        for i in range (ncore + 1):
            alpha_list.append(i * interval)
            prob[i] = interval

        total_prob = 0.0
        for i in range (ncore):
            alpha[i] = np.random(alpha_list[i], alpha_list[i+1])
            opt = BasinHopping(nsteps, alpha=alpha[i])
            opt.run(fmax)

            pareto_base
            for dot in dots:
                pareto_line = self.pareto_push(step, pareto_base, dot)
                accept_numb += 1
            prob[i] = prob[i] + accept_numb / len(dots)
            total_prob = total_prob + prob[i]

        #normalize the total probablility to 1
        for i in range (ncore):
            prob[i] = prob[i] / total_prob 
            temp = temp + prob[i]
            biased_prob[i] = temp


        for step in range(steps):

            #run BasinHopping
            opt = BasinHopping(nsteps)
            opt.run(fmax)
            
            #determine pareto line
            old_pareto = copy.deepcopy(pareto_line)
            for dot in dots:
                pareto_line = self.pareto_push(step, parabola, dot)
            
            alpha_dots, mid_alpha = self.sort_dots(pareto_line, dots)


         if alteration = "even_distribution":
            for i in range(ncore):
                weight_sum = 0
                numb_dots  = 0
                for alpha_dot in alpha_dots:
                    if alpha_dot[0] > alpha_list[i] and alpha_dot[0] < alpha_list[i+1]:
                       numb_dots = alpha_dot[1] + numb_dots
                       weight_sum = alpha_dot[1] * alpha_dot[0] + weight_sum

                #alter alpha and try to get an even distribution of dots in the effective area
                avg_alpha = weight_sum / numb_dots
                curr_ratio = numb_dots / float(len(dots))
                if avg_alpha < mid_alpha:
                   if curr_ratio < target_ratio:
                      alpha[i] = np.random(alpha_list[i], avg_alpha)
                   else:
                      alpha[i] = np.random(avg_alpha, alpha_list[i+1])
                else:
                   if curr_ratio < target_ratio:
                      alpha[i] = np.random(avg_alpha, alpha_list[i+1])
                   else:
                      alpha[i] = np.random(alpha_list[i], avg_alpha)

         if alteration = "kmc_like":
            


    def pareto_push(self, step, parabola, dot_n): 
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
        

    def log(self, step, accept, para_accept, alpha, En, Sn, Un, Umin):
        if self.logfile is None:
            return
        name = self.__class__.__name__
        self.logfile.write('%s: %d  %d %d %15.6f  %15.6f  %15.8f  %15.6f  %15.6f\n'
                           % (name, step, accept, para_accept, alpha, En, Sn, Un, Umin))
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
    
    def get_alpha(self, config_1, config_2, config_3):
        slope = (config_1[1] - config_2[1])/(config_1[0] - config_2[0])
        alpha = 1 / (1+self.beta/slope)
        return alpha

    def sort_dots(self, parabola, dots):
        """
        Sort the dots in the scaled ES space: seperate dots to different groups
        each group belongs to one alpha based on the following rule:
        If a new dot cannot help push the parabola close to the origin, 
        find out which line on the parabola it is closest to.
        Then calculate alpha based on the slope of the line perpendicular to
        the closest line
        """
        temp_dot = copy.deepcopy(parabola)
        dist_cutoff =  self.dist_cutoff
        sorted_dots = []

        #transfer ES space to scaled ES space
        parabola, dots, scale_ratio = self.scale_dots(parabola, dots)

        #Extend the parabola along two ending direction. The extended dots used
        #to construct two lines for the starting and the ending dots, respectively.
        temp_dot.insert(0, np.sum([parabola[0], [0, 1]], axis=0))
        temp_dot.append(np.sum([parabola[len(parabola)-1], [1, 0]], axis=0))
        
        #calculate the corresponding alpha and effective area which is defined by the
        #corresponding distance cutoff for each line
        alpha_mid = 1 / ( 1 - scale_ratio * ( 1 - 1 / 0.5))
        for i in range (len(temp_dot)-1):
            if i == 0:
               print "left end"
               alpha = 0.0
            elif i == len(temp_dot)-1:
               print "right end"
               alpha = 1.0/self.beta
            else:
               #calculate the alpha in scaled space
               slope = (temp_dot[i][1] - temp_dot[i+1][1]) / (temp_dot[i][0] - temp_dot[i+1][0])
               print('%s: %15.6f' % ("slope", slope))
               alpha = 1.0 / (1.0 - self.beta * slope)
               #convert the aplpha to the one in the non-scaled ES space use following eqation:
               # alpha = 1 / ( 1 - (S_factor / E_factor)*(1 - 1/scaled_alpha))
               alpha = 1 / ( 1 - scale_ratio * ( 1 - 1 / alpha))

            sorted_dots.append(alpha, 0.0)

        for dot in dots:
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
            if min_dist < dist_cutoff:
               sorted_dots[index][1] += 1

        return sorted_dots, alpha_mid

    def scale_dots(self, parabola, dots)
        
        S_factor = parabola[0][1] - parabola[len(parabola)-1][1]
        E_factor = parabola[len(parabola)-1][0] - parabola[0][0]
        scale_ratio = S_factor / E_factor
        
        for i in range(len(parabola)) 
            E_scaled = parabola[i][0]/E_factor
            S_scaled = parabola[i][1]/S_factor
            parabola[i] = ([E_scaled, S_scaled])

        for dot in dots: 
            E_scaled = dot[0]/E_factor
            S_scaled = dot[1]/S_factor
            dot = ([E_scaled, S_scaled])

     return parabola, dots, scale_ratio

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

