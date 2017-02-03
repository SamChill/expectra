import sys
import numpy as np
import copy

from ase.optimize.optimize import Dynamics
from ase.optimize.fire import FIRE
#from ase.optimize.LBFGS import LBFGS
from ase.units import kB, fs
from ase.parallel import world
from ase.io import write
from ase.io.trajectory import Trajectory
from expectra.basin_surface import BasinHopping
from expectra.io import read_dots, read_atoms, write_atoms

class ParetoLineOptimize(Dynamics):

    def __init__(self, atoms,
                 opt_calculator = None,
                 exafs_calculator = None,
                 nnode = 5,
                 ncore = 2,
                 bh_steps = 10,
                 bh_steps_0 = 10,
                 scale = False,
                 mini_output = True, #minima output
                 #Modify composition by randomly switching chemical symbol in structures
                 move_atoms = True, #if true, the position of atoms will be modified as normal basin hopping
                 switch = False,    #if true, the chemical symbols of atoms in the active space (defined by switch_ratio) is switchable
                 switch_ratio = 0.1, #percentage of atoms that used to modify composition
                 cutoff = None,
                 elements_lib = None, #library of elements where the chemical symbols are picked to modify composition
                 #MD parameters
                 md = True,
                 md_temperature = 300 * kB,
                 md_step_size = 1 * fs,
                 md_step = 1000,
                 md_trajectory = 'md',
                 specorder=None,
                 #Basin Hopping parameters
                 alpha = None,
                 optimizer = FIRE,
                 temperature=100 * kB,
                 fmax = 0.1,
                 dr = 0.1,
                 z_min = 14.0,
                 substrate = None,
                 absorbate = None,
                 logfile='basin_log', 
                 trajectory='lowest',
                 optimizer_logfile='-',
                 local_minima_trajectory='local_minima.traj',
                 exafs_logfile = 'exafs',
                 log_paretoLine = 'paretoLine.dat',
                 log_paretoAtoms = 'paretoAtoms.traj', 
                 adjust_cm=True,
                 mss=0.2,
                 minenergy=None,
                 distribution='uniform',
                 adjust_step=True,
                 adjust_every = 5,
                 target_ratio = 0.5,
                 adjust_fraction = 0.05,
                 significant_structure = False,  # displace from minimum at each move
                 significant_structure2 = False, # displace from global minimum found so far at each move
                 pushapart = 0.4,
                 jumpmax=None
                 ):
        """Parameters:

        atoms: Atoms object
            The Atoms object to operate on.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.
        """
#       Dynamics.__init__(self, atoms, logfile, trajectory)
        self.atoms = atoms
        self.opt_calculator = opt_calculator
        self.exafs_calculator = exafs_calculator
        self.nnode = nnode
        self.ncore = ncore
        self.bh_steps = bh_steps
        self.bh_steps_0 = bh_steps_0
        self.scale = scale

        self.move_atoms = move_atoms
        self.switch = switch
        self.switch_ratio = switch_ratio
        self.cutoff = cutoff
        self.elements_lib = elements_lib
 
        self.md = md
        self.md_temperature = md_temperature
        self.md_step_size = md_step_size
        self.md_step = md_step
        self.md_trajectory = md_trajectory
        self.specorder = specorder

        self.alpha = alpha
        self.optimizer = optimizer
        self.temperature = temperature
        self.fmax = fmax
        self.dr = dr
        self.z_min = z_min
        self.substrate = substrate
        self.absorbate = absorbate
        self.exafs_logfile = exafs_logfile 
        self.logfile = logfile 

        self.trajectory = trajectory
        self.optimizer_logfile = optimizer_logfile
        self.local_minima_trajectory = local_minima_trajectory
        self.log_paretoLine = log_paretoLine
        self.log_paretoAtoms = log_paretoAtoms

        if isinstance(log_paretoLine, str):
            self.log_paretoLine = open(self.log_paretoLine, 'w')

        #if isinstance(log_paretoAtoms, str):
        #    self.log_paretoAtoms = Trajectory(self.log_paretoAtoms,
        #                                          'w', atoms)
        self.minenergy = minenergy
        self.distribution = distribution
        self.adjust_step = adjust_step
        self.adjust_every = adjust_every
        self.adjust_cm = adjust_cm
        self.target_ratio = target_ratio
        self.adjust_fraction = adjust_fraction
        self.significant_structure = significant_structure
        self.significant_structure2 = significant_structure2
        self.pushapart = pushapart
        self.jumpmax = jumpmax
        self.mss = mss

        self.initialize()

    def initialize(self):
        #self.energy = 1.e32
        self.Umin = 1.e32
        self.rmin = self.atoms.get_positions()
        self.debug = open('debug', 'w')

#        self.logfile.write('   name      step accept     energy         chi_difference\n')
                
    def run(self, steps):
        """Hop the basins for defined number of steps."""

        #initialize alpha and probablility for each basin hopping jobs
        nnode = self.nnode
        prob = []
        accept_ratio = []
        pareto_base = []
        pareto_line = []
        pareto_atoms = []
        images = []
        E_factor = None
        S_factor = None
        for i in range (nnode):
            accept_ratio.append(0.0)
            images.append(None)

        for step in range(steps):
            total_prob = 0.0
            if step == 0:
               bh_steps = self.bh_steps_0
            else:
               bh_steps = self.bh_steps

            print "====================================================================="
            for i in range (nnode):
                print "====================================================================="
                #select alpha
                if step == 0:
                   atoms = self.atoms
                   scale_ratio = 1.0
                else:
                   if not self.scale:
                      scale_ratio = 1.0
                   else:
                      print "Scale ratio is calculated based on the current paretoLine for each paretoCycle"
                      if E_factor is None or S_factor is None:
                         print "E_factor or S_factor is not calculated correctly"
                         break
                      scale_ratio = E_factor/S_factor

                   
                   atoms = images
                   #atoms from pareto_atoms has no pbc. This bug is to be fixed
                   #atoms.set_cell([[80,0,0],[0,80,0],[0,0,80]],scale_atoms=False,fix=None)
                  # write("pl_atoms.trj",images=atoms,format="traj")
                    
                #run BasinHopping
                #define file names used to store data
                pareto_step = str(step)
                node_numb = str(i)
                trajectory = self.trajectory+'_'+ pareto_step + "_" + node_numb + ".traj"
                md_trajectory = self.md_trajectory+"_"+pareto_step + "_" + node_numb
                exafs_logfile = self.exafs_logfile + "_" + pareto_step + "_" + node_numb
                logfile = self.logfile + "_" + pareto_step + "_" + node_numb
                lm_trajectory = self.local_minima_trajectory + "_" + pareto_step + "_" + node_numb
                opt = BasinHopping(atoms,
                                   alpha = self.alpha,
                                   scale_ratio = scale_ratio,
                                   opt_calculator = self.opt_calculator,
                                   exafs_calculator = self.exafs_calculator,
                                   ncore = self.ncore,
                                   #Switch or modify elements in structures
                                   move_atoms = self.move_atoms,
                                   switch = self.switch,
                                   switch_ratio = self.switch_ratio, #How many atoms will be switched or modified
                                   cutoff = self.cutoff,
                                   elements_lib = self.elements_lib, #elements used to replace the atoms
                                   #MD parameters
                                   md = self.md,
                                   md_temperature = self.md_temperature,
                                   md_step_size = self.md_step_size,
                                   md_step = self.md_step,
                                   md_trajectory = md_trajectory,
                                   specorder = self.specorder,
                                   #Basin Hopping parameters
                                   optimizer = self.optimizer,
                                   temperature = self.temperature,
                                   fmax = self.fmax,
                                   dr = self.dr,
                                   z_min = self.z_min,
                                   substrate = self.substrate,
                                   absorbate = self.absorbate,
                                   logfile = logfile, 
                                   trajectory = trajectory,
                                   optimizer_logfile = self.optimizer_logfile,
                                   local_minima_trajectory = lm_trajectory,
                                   exafs_logfile = exafs_logfile,
                                   adjust_cm = self.adjust_cm,
                                   mss = self.mss,
                                   minenergy = self.minenergy,
                                   distribution = self.distribution,
                                   adjust_step = self.adjust_step,
                                   adjust_every = self.adjust_every,
                                   target_ratio = self.target_ratio,
                                   adjust_fraction = self.adjust_fraction,
                                   significant_structure = self.significant_structure,  
                                   significant_structure2 = self.significant_structure2, 
                                   pushapart = self.pushapart,
                                   jumpmax = self.jumpmax,
                                   )

                images = opt.run(bh_steps)

                #read the dots and geometries obtained from basin hopping
                dots = read_dots(logfile)
                #lowest_traj = Trajectory(trajectory)
                #images[i] = lowest_traj[-1]
                traj = read_atoms(lm_trajectory)

                #initialize pareto line
                if step == 0 and i == 0:
                   pareto_base.append(dots[0])
                   pareto_line = copy.deepcopy(pareto_base)
                   pareto_atoms.append(traj[0])

                #pick out the dots which can push pareto line
                accepted_numb = 0
                for j in xrange (len(dots)):
                    promoter = self.dots_filter(pareto_base, dots[j])
                    if promoter:
                       pareto_line = self.pareto_push(step, i, pareto_line, pareto_atoms, dots[j], traj[j])
                       accepted_numb += 1
                print "Accepted number", accepted_numb, "dots number", len(dots)
            
            pareto_base = copy.deepcopy(pareto_line)
            S_factor, E_factor = self.find_scale_factor(pareto_base)

        if pareto_atoms is None:
           print "Something wrong on pareto_atoms", type(pareto_atoms)
        else:
           print "tyep of pareto_atoms", type(pareto_atoms[0])
           print(pareto_atoms[0])

           write_atoms(self.log_paretoAtoms, pareto_atoms)
#        for atoms in pareto_atoms:
#            self.log_paretoAtoms.write(atoms)
     
    """
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
    """
            
    def dots_filter(self, pareto_line, dot):
        #To determine if the dot can push the pareto line.
        temp = copy.deepcopy(pareto_line)

        #Extend the parabola along two ending direction. The extended dots used
        #to construct two lines for the starting and the ending dots, respectively.
        temp.insert(0, np.sum([pareto_line[0], [0, 1]], axis=0))
        temp.append(np.sum([pareto_line[len(pareto_line)-1], [1, 0]], axis=0))
        
        for i in range(len(temp)-1):
            if self.get_cross(temp[i], temp[i+1], dot) > 0:
               promoter = True
               return promoter
            else:
               promoter = False
               continue
        return promoter

    def pareto_push(self, step, node_numb, pareto_line, pareto_atoms, dot_n, atom_n): 
        """
        Parabolic method to determine if accept rn or not
        """
        #only one dot in pareto_line
        temp = copy.deepcopy(pareto_line)
        if len(temp)==1:
           if dot_n[0] < temp[0][0] and dot_n[1] > temp[0][1]:
              pareto_line[0] = dot_n
              pareto_line.append(temp[0])
              pareto_atoms.append(pareto_atoms[0])
              pareto_atoms[0] = atom_n
           elif dot_n[0] > temp[0][0] and dot_n[1] < temp[0][1]:
              pareto_line.append(dot_n)
              pareto_atoms.append(atom_n)
           elif dot_n[0] < temp[0][0] and dot_n[1] < temp[0][1]:
              pareto_line[0] = dot_n
              pareto_atoms[0] = atom_n
           
           if pareto_line != temp:
              self.logParetoLine(step, node_numb, pareto_line)
           return pareto_line
        
        #more than one dot in pareto_line
        replace = False
        for i in range(0, len(temp)):
            #pareto_line may be changed after each cycle. Need to find new index for element temp[i]
            index = self.find_index(pareto_line, temp[i])

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

               cross_f1 = self.get_cross(dot_c, dot_f1, dot_n)
               if dot_n[0] < dot_b1[0] and cross_f1 < 0:
                  pareto_line.insert(0, dot_n)
                  pareto_atoms.insert(0, atom_n)
                  continue

            elif i == len(temp)-2 and i != 0:
               #next to last
               dot_b1 = temp[i-1]
               dot_c  = temp[i]
               dot_f1 = temp[i+1]
               #creat a pseudo dot
               dot_f2 = np.sum([temp[len(temp)-1], [1, 0]], axis=0)

            elif i == len(temp)-1:
               #end point
               cross = self.get_cross(temp[i-1], temp[i], dot_n)
               if dot_n[1] < temp[i][1] and cross > 0 :
                  if replace:
                     pareto_line.pop(index)
                     pareto_atoms.pop(index)
                  else:
                     pareto_line[index] = dot_n
                     pareto_atoms[index] = atom_n
               elif dot_n[1] < temp[i][1] and cross < 0:
                    pareto_line.append(dot_n)
                    pareto_atoms.append(atom_n)
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
               #if i == 0:
                  #'replace' has already been done in corner situation
               #   self.debug.write('%d\n' % (i))
               #   continue
               if replace:
                  #A previous dot has already been replaced. Discard the current one
                  #'poped' is to record the number of dots poped before current one 
                  #which will change the location of current dot
                  pareto_line.pop(index)
                  pareto_atoms.pop(index)
                  self.debug.write('%d  %s  %d\n' % (i, 'poped',  index))
               else:
                  pareto_line[index] = dot_n
                  pareto_atoms[index] = atom_n
                  self.debug.write('%d  %s  %d\n' % (i, 'replaced',  index))
                  replace = True
               continue
            elif cross_b1 < 0  and cross_f1 > 0 and cross_f2 <0:
               #If 'insert' happens, no further action to other dots is needed
               pareto_line.insert(index+1, dot_n)
               pareto_atoms.insert(index+1, atom_n)
               self.debug.write('%d  %s  %d\n' % (i, 'inserted',  index+1))
               return pareto_line
            else:
               #For other situations, no acition is needed
               self.debug.write('%d  %s  %d\n' % (i, 'nothing done',  index))
               continue
        if pareto_line != temp:
           self.logParetoLine(step, node_numb, pareto_line)
        return pareto_line
    
    def logParetoLine(self, step, node_numb, pareto_line=[]):
        if self.log_paretoLine is None:
            return
        self.log_paretoLine.write('%s = %d  %s = %d %s = %d\n' % ("step", step, 
                                   "# of node", node_numb, "dot_number", len(pareto_line)))
        for i in range(0, len(pareto_line)):
             self.log_paretoLine.write('%15.6f  %15.6f\n' % (pareto_line[i][0], pareto_line[i][1]))
        self.log_paretoLine.flush()
        

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

    def find_alpha(self, probability=[]):
        test_prob = np.random.uniform(0,1)
        if len(probability) == 1:
           return 0
        if test_prob >= 0.0 and test_prob < probability[0]:
           return 0
        if test_prob >= probability[len(probability)-2] and test_prob < probability[len(probability)-1]:
           return len(probability)-1
        for i in range(len(probability)-1):
            if test_prob >= probability[i]and probability[i+1] >test_prob:
               return i

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

    def find_scale_factor(self, pareto_line):
        if len(pareto_line) == 1:
           S_max = pareto_line[0][1]
           S_min = 0.0
           E_max = pareto_line[0][0]
           E_min = 0.0
        elif len(pareto_line) == 0:
           return
        else:
           S_max = pareto_line[0][1]
           S_min = pareto_line[len(pareto_line)-1][1]
           E_max = pareto_line[len(pareto_line)-1][0]
           E_min = pareto_line[0][0]

        S_factor = S_max - S_min
        E_factor = E_max - E_min
        
        return S_factor, E_factor

    def scale_dots(self, dots, S_factor, E_factor):
        for dot in dots: 
            E_scaled = dot[0]/E_factor
            S_scaled = dot[1]/S_factor
            dot = ([E_scaled, S_scaled])
        return dots

    def get_cross(self, config_1, config_2, config_3):
        #calculate the cross vector
        vect1 = np.subtract(config_1, config_3) #config_1 - config_3
        vect2 = np.subtract(config_2, config_3)
        cross_vect = np.cross(vect2, vect1)  # vect2 X vect1
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

