import sys
import numpy as np
import copy

from ase.optimize.optimize import Dynamics
from ase.units import kB, fs
from ase.parallel import world
from ase.io import write
from ase.io.trajectory import Trajectory
from expectra.basin import BasinHopping
from expectra import default_parameters as dp
from expectra.io import read_dots, read_atoms, write_atoms

default_parameters=dp.default_parameters

class ParetoLineOptimize(Dynamics):

    def __init__(self, atoms,
                 dr = 0.5,
                 opt_calculator = None,
                 exafs_calculator = None,
                 nnode = 5,
                 ncore = 2,
                 bh_steps = 10,
                 bh_steps_0 = 10,
                 scale = False,
                 beta = 1.0,
                 sample_method = 'boltzmann',
                 boltzmann_temp = 4000 *kB,
                 visited_configs = {},
                 #mini_output = True, #minima output
                 alpha = None,
                 log_paretoLine = 'paretoLine.dat',
                 log_paretoAtoms = 'paretoAtoms.traj',
                 **kwargs
                 ):
#       Dynamics.__init__(self, atoms, logfile, trajectory)
        self.atoms = atoms
        self.dr = dr
        self.opt_calculator = opt_calculator
        self.exafs_calculator = exafs_calculator
        self.nnode = nnode
        self.ncore = ncore
        self.bh_steps = bh_steps
        self.bh_steps_0 = bh_steps_0
        self.scale = scale
        self.beta = beta
        self.sample_method = sample_method
        self.boltzmann_temp = boltzmann_temp
        self.visited_configs = visited_configs

        self.alpha = alpha
        self.log_paretoLine = log_paretoLine
        self.log_paretoAtoms = log_paretoAtoms
        for parameter in kwargs:
            if parameter not in default_parameters:
               print parameter, 'is not in the keywords included'
               sys.exit()
        for (parameter, default) in default_parameters.iteritems():
            setattr(self, parameter, kwargs.get(parameter, default))
            self.parameters = kwargs.copy()

        if isinstance(self.log_paretoLine, str):
            self.log_paretoLine = open(self.log_paretoLine, 'w')
        
        self.configs_output = open('visited_configs.dat', 'w')
        self.new_configs_output = open('new_configs.dat', 'w')
        #if isinstance(log_paretoAtoms, str):
        #    self.log_paretoAtoms = Trajectory(self.log_paretoAtoms,
        #                                          'w', atoms)

        self.initialize()

    def initialize(self):
        self.pbc = self.atoms.get_pbc()
        self.cell = self.atoms.get_cell()
        self.dots = []
        self.states = []
        self.pareto_atoms = []
        self.pareto_line = []
        self.pareto_state = []
        #self.debug = open('debug', 'w')
        self.blz_sample=open('blz_sample.dat','w')
        self.blz_sample.write("stateSelected   e_pot   chi_differ   deltaU   stateMin\n") 

    def run(self, steps):
        """Hop the basins for defined number of steps."""

        #initialize alpha and probablility for each basin hopping jobs
        nnode = self.nnode
        interval = 1.0 / float(nnode)
        target_ratio = interval
        alpha_list = []
        dr_list = []
        prob = []
        accept_ratio = []
        pareto_base = []
        images = []
        atoms_state = None
        Umin = 1.0e32
        E_factor = None
        S_factor = None
        for i in range (nnode):
            alpha_list.append(float(i) * interval)
            accept_ratio.append(interval)
            prob.append(interval)
            images.append(None)
            dr_list.append(self.dr)
        alpha_list.append(1.0)
        
        for step in range(steps):
            total_prob = 0.0
            alpha = []
            self.configs_output.write("pareto_step: %d\n" % (step))
            self.new_configs_output.write("pareto_step: %d\n" % (step))
            #self.debug.write("pareto_step: %d\n" % (step))
            #if step == 0:
            #   bh_steps = self.bh_steps_0
            #else:
            #   bh_steps = self.bh_steps
            print "====================================================================="
            print "ParetoLine cycle ", step
            for i in range (nnode):
                print "====================================================================="
                #a fixed alpha value is used
                if self.alpha is not None:
                   alpha.append(self.alpha)
                   if step == 0:
                      atoms = self.atoms
                      scale_ratio = self.beta
                   else:
                      if not self.scale:
                         scale_ratio = self.beta
                      else:
                         scale_ratio = E_factor/S_factor
                
                #dynamically select alpha
                else:     
                   if step == 0:
                      index = i
                      atoms = self.atoms
                      scale_ratio = self.beta
                   else:
                      if not self.scale:
                         scale_ratio = self.beta
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
                      atoms, atoms_state = self.paretoLine_sample(alpha[i], scale_ratio)
                   elif self.sample_method == 'boltzmann':
                      atoms, atoms_state = self.boltzmann_sample(alpha[i], scale_ratio)
                    
                print "BasinHopping cycle ", i, "alpha:", alpha[i], "step size", dr_list[i]
                #run BasinHopping
                pareto_step = str(step)
                node_numb = str(i)
                opt = BasinHopping(atoms=atoms,
                                   atoms_state = atoms_state,
                                   dr = dr_list[i],
                                   alpha = alpha[i],
                                   scale_ratio = scale_ratio,
                                   pareto_step = pareto_step,
                                   node_numb = node_numb,
                                   ncore = self.ncore,
                                   opt_calculator = self.opt_calculator,
                                   exafs_calculator = self.exafs_calculator,
                                   #Switch or modify elements in structures
                                   visited_configs = copy.deepcopy(self.visited_configs),
                                   Umin =Umin,
                                   **self.parameters
                                   )
                #old configs visited
                configs_o = copy.deepcopy(self.visited_configs)
                #updated configs after current bh runs
                new_configs, dr_list[i], Umin = opt.run(self.bh_steps)
                print 'Umin:', Umin
                #upon updating, change new_configs will not change self.visited_configs. No need to use copy
                self.visited_configs.update(new_configs)
                #new configs visited in current bh runs
                configs_n = self.differ_configs(configs_o)

                #store the dots and the corresponding states obtained from basin hopping
                dots = []
                states = []

                #initialize pareto line
                if step == 0 and i == 0:
                   pareto_base.append([configs_n['0_0_-1'][0], configs_n['0_0_-1'][1]])
                   self.pareto_line = copy.deepcopy(pareto_base)
                   self.pareto_state.append('0_0_-1')
                   self.pareto_atoms.append(configs_n['0_0_-1'][4])
                #else:
                #   dots.pop(0)
                #   traj.pop(0)

                #pick out the dots which can push pareto line
                accepted_numb = 0
                self.log_paretoLine.write("==============================================\n")
                self.log_paretoLine.write("%s: %d, %s: %d\n" % ("pl_cycle", step, "node", i))
                self.log_paretoLine.flush()
                for state in configs_n:
                    dot = [configs_n[state][0], configs_n[state][1]]
                    dots.append(dot)
                    states.append(state)
                    if state == '0_0_-1':
                       continue
                    promoter = self.dots_filter(pareto_base, dot)
                    if promoter:
                       self.pareto_push(state, dot, configs_n[state][4])
                       accepted_numb += 1
                print "Accepted number", accepted_numb, "dots number", len(dots)
                if len(dots) != 0:
                   accept_ratio[i] = accept_ratio[i] + float(accepted_numb) / float(len(dots))
                else: 
                   accept_ratio[i] = 0.0
                print "Accepted ratio: ", accept_ratio
                total_prob = total_prob + accept_ratio[i]
                print "Total_prob: ", total_prob
                
                #store all the dots and images for sampling
                self.dots.extend(dots)
                self.states.extend(states)

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
            #self.debug.flush()
            for (key, value) in configs_n.iteritems():
                self.new_configs_output.write("%s  %15.6f  %15.6f\n"%(key, value[0], value[1]))
            self.new_configs_output.flush()

            for (key, value) in self.visited_configs.iteritems():
                #For a given database, only record states that found in current job
                if len(key.split('_'))>1:
                   self.configs_output.write("%s  %15.6f  %15.6f  %d\n"%(key, value[0], value[1], value[3]))
            self.configs_output.flush()

        if self.pareto_atoms is None:
           print "Something wrong on pareto_atoms", type(self.pareto_atoms)
        else:
           print "type of pareto_atoms", type(self.pareto_atoms[0])
           print(self.pareto_atoms[0])

           write_atoms(self.log_paretoAtoms, self.pareto_atoms)
        print "ParetoOPT job is completed successfully"
        self.configs_output.close()
        if len(self.specorder)==1:
           self.log_visited_configs()
#        for atoms in pareto_atoms:
#            self.log_paretoAtoms.write(atoms)

    #Find the new structures located
    def differ_configs(self, configs_o):
        configs_n = {}
        for key in self.visited_configs:
            if key not in configs_o:
               #visited_configs[key] is a list. need to use copy
               configs_n[key] = copy.deepcopy(self.visited_configs.get(key))
               if self.in_memory_mode: #Atoms information stored in self.visited_configs
                  continue
               atoms = read_atoms(filename=self.configs_dir+'/'+key, state_number= -1)
               atoms.set_cell(self.cell)
               atoms.set_pbc(self.pbc)
               configs_n[key].append(atoms.copy())
               continue
            if cmp(self.visited_configs[key][3], configs_o[key][3])!=0:
               configs_n[key] = copy.deepcopy(self.visited_configs.get(key))
               if self.in_memory_mode: #Atoms information stored in self.visited_configs
                  continue
               atoms = read_atoms(filename=self.configs_dir+'/'+key, state_number=-1)
               atoms.set_cell(self.cell)
               atoms.set_pbc(self.pbc)
               configs_n[key].append(atoms.copy())
        return copy.deepcopy(configs_n)

    def log_visited_configs(self):
        data_base=self.visited_configs
        log_database = open('new_au55_all.xyz', 'w')
        log_exafs = open('new_au55_exafs_all.dat','w')
        for state in data_base:
            config =data_base[state]
            numb_atoms=len(config[4])
            log_database.write("%d\n"%(numb_atoms))
            log_database.write("images: %s energy: %15.6f chi_differ: %15.6f\n"%(state, config[0], config[1]))
            for atom in config[4]:
                log_database.write("%s %15.6f %15.6f %15.6f\n"%(atom.symbol, atom.x, atom.y, atom.z))
            log_database.flush()
            log_exafs.write("images: %s\n"%(state))
            for j in xrange(len(config[5])):
                log_exafs.write("%12.7f  %12.7f\n"%(config[5][j], config[6][j]))
            log_exafs.flush()
        log_database.close()
        log_exafs.close()

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

    def pareto_push(self, state_n, dot_n, atom_n): 
        """
        Parabolic method to determine if accept rn or not
        """
        #only one dot in pareto_line
        temp = copy.deepcopy(self.pareto_line)
        action = None
        if len(temp)==1:
           if dot_n[0] < temp[0][0] and dot_n[1] > temp[0][1]:
              action = 'insert'
              self.pareto_line[0] = dot_n
              self.pareto_line.append(temp[0])
              self.pareto_atoms.append(self.pareto_atoms[0])
              self.pareto_atoms[0] = atom_n
              self.pareto_state.append(self.pareto_state[0])
              self.pareto_state[0] = state_n
           elif dot_n[0] > temp[0][0] and dot_n[1] < temp[0][1]:
              action = 'append'
              self.pareto_line.append(dot_n)
              self.pareto_atoms.append(atom_n)
              self.pareto_state.append(state_n)
           elif dot_n[0] < temp[0][0] and dot_n[1] < temp[0][1]:
              action = 'replace'
              self.pareto_line[0] = dot_n
              self.pareto_atoms[0] = atom_n
              self.pareto_state[0] = state_n
           if self.pareto_line != temp:
              self.logParetoLine(state_n, action)
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
                  action = 'insert'
                  self.pareto_line.insert(0, dot_n)
                  self.pareto_atoms.insert(0, atom_n)
                  self.pareto_state.insert(0, state_n)
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
                     action = 'pop'
                     self.pareto_line.pop(index)
                     self.pareto_atoms.pop(index)
                     self.pareto_state.pop(index)
                  else:
                     action = 'replace'
                     self.pareto_line[index] = dot_n
                     self.pareto_atoms[index] = atom_n
                     self.pareto_state[index] = state_n
               elif dot_n[1] < temp[i][1] and cross < 0:
                    action = 'append'
                    self.pareto_line.append(dot_n)
                    self.pareto_atoms.append(atom_n)
                    self.pareto_state.append(state_n)
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
               #self.debug.write('%s\n' % (replace))
               #if i == 0:
                  #'replace' has already been done in corner situation
               #   self.debug.write('%d\n' % (i))
               #   continue
               if replace:
                  #A previous dot has already been replaced. Discard the current one
                  #'poped' is to record the number of dots poped before current one 
                  #which will change the location of current dot
                  action = 'pop'
                  self.pareto_line.pop(index)
                  self.pareto_atoms.pop(index)
                  self.pareto_state.pop(index)
                  #self.debug.write('%d  %s  %d\n' % (i, 'poped',  index))
               else:
                  action ='replace'
                  self.pareto_line[index] = dot_n
                  self.pareto_atoms[index] = atom_n
                  self.pareto_state[index] = state_n
                  #self.debug.write('%d  %s  %d\n' % (i, 'replaced',  index))
                  replace = True
               continue
            elif cross_b1 < 0  and cross_f1 > 0 and cross_f2 <0:
               #If 'insert' happens, no further action to other dots is needed
               action ='insert'
               self.pareto_line.insert(index+1, dot_n)
               self.pareto_atoms.insert(index+1, atom_n)
               self.pareto_state.insert(index+1, state_n)
               #self.debug.write('%d  %s  %d\n' % (i, 'inserted',  index+1))
               if self.pareto_line != temp:
                  self.logParetoLine(state_n, action)
               return 
            else:
               #For other situations, no acition is needed
               #self.debug.write('%d  %s  %d\n' % (i, 'nothing done',  index))
               continue
        if self.pareto_line != temp:
           self.logParetoLine(state_n, action)
        return 
    
    def logParetoLine(self, state_n, action=None):
        pareto_line = self.pareto_line
        if self.log_paretoLine is None:
            return
        self.log_paretoLine.write('%s : %s\n' % (state_n, action))
        for i in range(0, len(pareto_line)):
             self.log_paretoLine.write('%s %15.6f  %15.6f\n' % (self.pareto_state[i], pareto_line[i][0], pareto_line[i][1]))
        self.log_paretoLine.flush()

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
        self.blz_sample.write("%s   %15.6f   %15.6f   %15.6f   %s\n" % (self.states[index], self.visited_configs[self.states[index]][0], 
                              self.visited_configs[self.states[index]][1], U[index], self.states[min_index]))
        self.blz_sample.flush()
        if self.in_memory_mode:
           atoms=self.visited_configs[self.states[index]][4].copy()
        else:
           atoms = read_atoms(filename=self.configs_dir + '/'+ self.states[index], state_number=-1)
           atoms.set_cell(self.cell)
           atoms.set_pbc(self.pbc)
        return atoms.copy(), self.states[index]
    '''
    def pop_lowProb_dots(self,p):
        index = [i for i, v in enumerate(p) if p < 2**-52]
    '''
    def paretoLine_sample(self, alpha, beta):
        pareto_line = self.pareto_line 
        pareto_atoms = self.pareto_atoms
        pareto_states = self.pareto_state
        for j in range (len(pareto_line)):
           U = (1.0-alpha) * (pareto_line[j][0]) + alpha * beta * pareto_line[j][1]
           if j == 0:
              Umin = U
              atoms = pareto_atoms[0]
              state = pareto_states[0]
           elif U < Umin:
              Umin = U
              atoms = pareto_atoms[j]
              state = pareto_states[j]
        #atoms.set_cell([[80,0,0],[0,80,0],[0,0,80]],scale_atoms=False,fix=None)
        atoms.set_cell(self.cell)
        atoms.set_pbc(self.pbc)
        return atoms, state

