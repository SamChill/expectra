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

default_parameters=dict(
             #Switch or modify elements in structures
             move_atoms = True,
             switch = False,
             active_ratio = None, #percentage of atoms will be used to switch or modified
             cutoff=None,
             elements_lib = None, #elements used to replace the atoms
             #Structure optimization
             optimizer=FIRE,
             max_optsteps=1000,
             fmax=0.001,
             mss=0.2,
             #MD parameters
             md = False,
             md_temperature = 300 * kB,
             md_step_size = 1 * fs,
             md_steps = 10000,
             max_md_cycle = 10,
             md_trajectory = 'md.traj',
             md_interval = 10,
             in_memory_mode = True,
             specorder = None, #for 'lammps', specify the order of species which should be same to that in potential file
             #Basin Hopping parameters
             temperature = 300 * kB,
             dr=0.5,
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
             visited_configs = {}, # {'state_number': [energy, chi, repeats], ...}
             comp_eps_e = 1.e-4, #criterion to determine if two configurations are identtical in energy 
             comp_eps_r = 0.2, #criterion to determine if two configurations are identical in geometry
             #files logging data
             logfile='basin_log', 
             trajectory='lowest.xyz',
             optimizer_logfile='geo_opt.log',
             local_minima_trajectory='localminima.xyz',
             exafs_logfile = 'exafs.dat'
             )

class ParetoLineOptimize(Dynamics):

    def __init__(self, atoms,
                 opt_calculator = None,
                 exafs_calculator = None,
                 nnode = 5,
                 ncore = 2,
                 bh_steps = 10,
                 bh_steps_0 = 10,
                 scale = False,
                 sample_method = 'boltzmann',
                 boltzmann_temp = 300 *kB,
                 mini_output = True, #minima output
                 alpha = None,
                 log_paretoLine = 'paretoLine.dat',
                 log_paretoAtoms = 'paretoAtoms.traj',
                 **kwargs
                 ):
#       Dynamics.__init__(self, atoms, logfile, trajectory)
        self.atoms = atoms
        self.opt_calculator = opt_calculator
        self.exafs_calculator = exafs_calculator
        self.nnode = nnode
        self.ncore = ncore
        self.bh_steps = bh_steps
        self.bh_steps_0 = bh_steps_0
        self.scale = scale
        self.sample_method = sample_method
        self.boltzmann_temp = boltzmann_temp

        self.alpha = alpha

        for parameter in kwargs:
            if parameter not in default_parameters:
               print parameter, 'is not in the keywords included'
               break
        for (parameter, default) in default_parameters.iteritems():
            setattr(self, parameter, kwargs.get(parameter, default))

        if isinstance(self.log_paretoLine, str):
            self.log_paretoLine = open(self.log_paretoLine, 'w')

        #if isinstance(log_paretoAtoms, str):
        #    self.log_paretoAtoms = Trajectory(self.log_paretoAtoms,
        #                                          'w', atoms)

        self.initialize()

    def initialize(self):
        #self.energy = 1.e32
        self.Umin = 1.e32
        self.rmin = self.atoms.get_positions()
        self.dots = []
        self.traj = []
        self.pareto_atoms = []
        self.pareto_line = []
        self.debug = open('debug', 'w')
        self.configs_dir = os.getcwd()+'/configs'
        self.exafs_dir = os.getcwd()+'/exafs'
        self.pot_dir = os.getcwd()+'/pot'

#        self.logfile.write('   name      step accept     energy         chi_difference\n')
                
    def run(self, steps):
        """Hop the basins for defined number of steps."""

        #initialize alpha and probablility for each basin hopping jobs
        nnode = self.nnode
        interval = 1.0 / float(nnode)
        target_ratio = interval
        alpha_list = []
        prob = []
        accept_ratio = []
        pareto_base = []
        images = []
        E_factor = None
        S_factor = None
        for i in range (nnode):
            alpha_list.append(float(i) * interval)
            accept_ratio.append(interval)
            prob.append(interval)
            images.append(None)
        alpha_list.append(1.0)
        
        for step in range(steps):
            total_prob = 0.0
            alpha = []

            if step == 0:
               bh_steps = self.bh_steps_0
            else:
               bh_steps = self.bh_steps

            print "====================================================================="
            print "ParetoLine cycle ", step
            for i in range (nnode):
                print "====================================================================="
                #Fixed alpha
                if self.alpha is not None:
                   alpha.append(self.alpha)
                   if step == 0:
                      atoms = self.atoms
                      scale_ratio = 1.0
                   else:
                      if not self.scale:
                         scale_ratio = 1.0
                      else:
                         scale_ratio = E_factor/S_factor
                
                else:     
                   #dynamically select alpha
                   if step == 0:
                      index = i
                      atoms = self.atoms
                      scale_ratio = 1.0
                   else:
                      if not self.scale:
                         scale_ratio = 1.0
                         index = self.sample_index(prob)
                      else:
                         print "Scale ratio is calculated based on the current paretoLine for each paretoCycle"
                         if E_factor is None or S_factor is None:
                            print "E_factor or S_factor is not calculated correctly"
                            break
                         scale_ratio = E_factor/S_factor
                      index = self.sample_index(prob)
                   alpha.append(np.random.uniform(alpha_list[index], alpha_list[index+1]))

                if step != 0:
                   if self.sample_method == 'pl_sample':
                      atoms = self.paretoLine_sample(alpha[i], scale_ratio)
                   elif self.sample_method == 'boltzmann':
                      atoms = self.boltzmann_sample(alpha[i], scale_ratio)
                    
                print "BasinHopping cycle ", i, "alpha:", alpha[i] 
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
                                   alpha = alpha[i],
                                   scale_ratio = scale_ratio,
                                   opt_calculator = self.opt_calculator,
                                   exafs_calculator = self.exafs_calculator,
                                   ncore = self.ncore,
                                   #Switch or modify elements in structures
                                   logfile = logfile, 
                                   trajectory = trajectory,
                                   local_minima_trajectory = lm_trajectory,
                                   exafs_logfile = exafs_logfile,
                                   visited_configs = self.visited_configs
                                   **kwargs
                                   )
                configs_o = self.visited_configs.copy()
                self.visited_configs.update(opt.run(bh_steps))
                configs_n = self.differ_configs(configs_o)

                #read the dots and geometries obtained from basin hopping
                #dots = read_dots(logfile)
                #traj = read_atoms(lm_trajectory)

                #initialize pareto line
                if step == 0 and i == 0:
                   pareto_base.append([configs_n['0_0_-1'][0], configs_n['0_0_-1'][1]])
                   self.pareto_line = copy.deepcopy(pareto_base)
                   self.pareto_atoms.append(configs_n['0_0_-1'][3])
                #else:
                #   dots.pop(0)
                #   traj.pop(0)

                #pick out the dots which can push pareto line
                accepted_numb = 0
                for key in xrange (len(configs_n)):
                    if key == '0_0_-1':
                       continue
                    dot = [configs_n[key][0], configs_n[key][1]]
                    promoter = self.dots_filter(pareto_base, dot)
                    if promoter:
                       self.pareto_push(step, i, dot, configs_n[key][3])
                       accepted_numb += 1
                print "Accepted number", accepted_numb, "dots number", len(dots)
                accept_ratio[i] = accept_ratio[i] + float(accepted_numb) / float(len(dots))
                print "Accepted ratio: ", accept_ratio
                #prob[i] = prob[i] + accepted_numb / len(dots)
                total_prob = total_prob + accept_ratio[i]
                print "Total_prob: ", total_prob
                
                #store all the dots and images for sampling
                #self.dots.extend(dots)
                #self.traj.extend(traj)

            #update base pareto_line after one pareto cycle
            pareto_base = copy.deepcopy(self.pareto_line)
            if self.scale:
               S_factor, E_factor = self.find_scale_factor()

            #normalize the total probablility to 1
            temp = 0.0
            for i in range (nnode):
                if total_prob == 0.0:
                   temp = temp + 1.0/float(nnode)
                   prob[i] = temp
                else:
                   temp = temp + accept_ratio[i] / total_prob 
                   prob[i] = temp
            print "Probablility:", prob
        if self.pareto_atoms is None:
           print "Something wrong on pareto_atoms", type(self.pareto_atoms)
        else:
           print "type of pareto_atoms", type(self.pareto_atoms[0])
           print(self.pareto_atoms[0])

           write_atoms(self.log_paretoAtoms, self.pareto_atoms)
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

    #Find the new structures located
    def differ_configs(self, configs_o):
        configs_n = {}
        for key in self.visited_configs:
            if key not in configs_o:
               configs_n[key] = self.visited_configs.get(key)
               atoms = read_atoms(filename=self.configs_dir+'/'+key)
               configs_n[key][3]= atoms[0]
               continue
            if cmp(self.visited_configs[key], configs_o[key])!=0:
               configs_n[key] = self.visited_configs.get(key)
               atoms = read_atoms(filename=self.configs_dir+'/'+key)
               configs_n[key][3]= atoms[0]
        return configs_n
               
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

    def pareto_push(self, step, node_numb, dot_n, atom_n): 
        """
        Parabolic method to determine if accept rn or not
        """
        #only one dot in pareto_line
        temp = copy.deepcopy(self.pareto_line)
        if len(temp)==1:
           if dot_n[0] < temp[0][0] and dot_n[1] > temp[0][1]:
              self.pareto_line[0] = dot_n
              self.pareto_line.append(temp[0])
              self.pareto_atoms.append(self.pareto_atoms[0])
              self.pareto_atoms[0] = atom_n
           elif dot_n[0] > temp[0][0] and dot_n[1] < temp[0][1]:
              self.pareto_line.append(dot_n)
              self.pareto_atoms.append(atom_n)
           elif dot_n[0] < temp[0][0] and dot_n[1] < temp[0][1]:
              self.pareto_line[0] = dot_n
              self.pareto_atoms[0] = atom_n
           if self.pareto_line != temp:
              self.logParetoLine(step, node_numb)
           return 
        
        #more than one dot in pareto_line
        replace = False
        for i in range(0, len(temp)):
            #pareto_line may be changed after each cycle. Need to find new index for element temp[i]
            index = self.find_index(temp[i])

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
                  self.pareto_line.insert(0, dot_n)
                  self.pareto_atoms.insert(0, atom_n)
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
                     self.pareto_line.pop(index)
                     self.pareto_atoms.pop(index)
                  else:
                     self.pareto_line[index] = dot_n
                     self.pareto_atoms[index] = atom_n
               elif dot_n[1] < temp[i][1] and cross < 0:
                    self.pareto_line.append(dot_n)
                    self.pareto_atoms.append(atom_n)
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
                  self.pareto_line.pop(index)
                  self.pareto_atoms.pop(index)
                  self.debug.write('%d  %s  %d\n' % (i, 'poped',  index))
               else:
                  self.pareto_line[index] = dot_n
                  self.pareto_atoms[index] = atom_n
                  self.debug.write('%d  %s  %d\n' % (i, 'replaced',  index))
                  replace = True
               continue
            elif cross_b1 < 0  and cross_f1 > 0 and cross_f2 <0:
               #If 'insert' happens, no further action to other dots is needed
               self.pareto_line.insert(index+1, dot_n)
               self.pareto_atoms.insert(index+1, atom_n)
               self.debug.write('%d  %s  %d\n' % (i, 'inserted',  index+1))
               if self.pareto_line != temp:
                  self.logParetoLine(step, node_numb)
               return 
            else:
               #For other situations, no acition is needed
               self.debug.write('%d  %s  %d\n' % (i, 'nothing done',  index))
               continue
        if self.pareto_line != temp:
           self.logParetoLine(step, node_numb)
        return 
    
    def logParetoLine(self, step, node_numb):
        pareto_line = self.pareto_line
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

    #select index based on the probablility provided
    def sample_index(self, probability=[]):
        test_prob = np.random.uniform(0,1)
        print 'Test prob:', test_prob
        if len(probability) == 1:
           return 0
        if test_prob >= 0.0 and test_prob < probability[0]:
           return 0
        if test_prob >= probability[len(probability)-2] and test_prob < probability[len(probability)-1]:
           return len(probability)-1
        for i in range(len(probability)-1):
            if test_prob >= probability[i]and probability[i+1] >test_prob:
               return i+1

    def find_index(self, temp):
        numb = 0
        for i in range (len(self.pareto_line)):
            if self.pareto_line[i] == temp:
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

    def find_scale_factor(self):
        pareto_line = self.pareto_line
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
    
    def boltzmann_sample(self, alpha, beta):
        temp = 0.0
        dots = np.array(self.dots)
        U = (1.0 - alpha)* dots[:,0] + alpha * beta * dots[:,1] 
        min_index = np.argmin(U)

        #avoid overflow in exp
        U = U - U[min_index]
        U[U < 2**-52] = 0.0 
        p = np.exp(-U/self.boltzmann_temp)

        #avoid underflow in divide
        p[p < 2**-52] = 0.0 
        #Normalize probablility
        p = p /np.sum(p)
        for i in range(len(p)):
            temp += p[i]
            p[i] = temp
        #print "Normalized:"
        #print p
        index = self.sample_index(p)
        if self.dots[index] in self.pareto_line:
           print 'The selected dot is on the pareto line found'
        print 'The dot selected for bh: ', dots[index], dots[min_index]
        print 'atom index found: ', index, 'away from minimum:', min_index,' ', U[index]
        atoms = self.traj[index]
        atoms.set_cell([[80,0,0],[0,80,0],[0,0,80]],scale_atoms=False,fix=None)
        return atoms
    '''
    def pop_lowProb_dots(self,p):
        index = [i for i, v in enumerate(p) if p < 2**-52]
    '''
    def paretoLine_sample(self, alpha, beta):
        pareto_line = self.pareto_line 
        pareto_atoms = self.pareto_atoms
        for j in range (len(pareto_line)):
           U = (1.0-alpha) * (pareto_line[j][0]) + alpha * beta * pareto_line[j][1]
           if j == 0:
              Umin = U
              atoms = pareto_atoms[0]
           elif U < Umin:
              Umin = U
              atoms = pareto_atoms[j]
        #atoms from pareto_atoms has no pbc. This bug is to be fixed
        atoms.set_cell([[80,0,0],[0,80,0],[0,0,80]],scale_atoms=False,fix=None)
        return atoms

