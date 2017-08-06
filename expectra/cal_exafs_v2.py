"""
This version doesn't inherit from 'Calculator' superclass and treats all
parameters as the attributes of Expectra class
"""
import numpy
import mpi4py.MPI

import argparse
import sys
import os

from expectra.exafs import exafs_first_shell, exafs_multiple_scattering
from expectra.io import read_xdatcar, read_con, read_chi
from ase.calculators.calculator import Calculator, all_changes, Parameters
from ase import Atoms
from ase.io import read
from ase.io.vasp import write_vasp
from expectra.feff import load_chi_dat
from expectra import default_parameters
from ase.io.trajectory import Trajectory
from expectra.lammps_caller import read_lammps_trj

sys.setrecursionlimit(10000)

COMM_WORLD = mpi4py.MPI.COMM_WORLD

default_parameters=default_parameters.expectra_parameters

def mpiexcepthook(type, value, traceback):
    sys.__excepthook__(type, value, traceback)
    sys.stderr.write("exception occured on rank %i\n" % COMM_WORLD.rank)
    COMM_WORLD.Abort()
sys.excepthook = mpiexcepthook

def save_result(k, chi, filename):
    if COMM_WORLD.rank != 0: return
    #print 'saving rescaled experimental chi data'
    print 'saving result to chi.dat'
    f = open(filename, 'w')
    for i in xrange(len(k)):
        f.write("%6.3f %16.8e\n" % (k[i], chi[i]))
    f.close()

#calculate the deviation of theoretical EXAFS from experimental EXAFS
def calc_deviation(chi_exp,chi_theory):
    chi_exp_array = numpy.asarray(chi_exp)
    chi_thry_array = numpy.asarray(chi_theory)
    chi_devi = numpy.sum(numpy.square(chi_exp_array - chi_thry_array))

    return chi_devi/len(chi_exp)

#Calculate difference between experiment and theory
def calc_area(y_exp, y_theory, calc_type='area', average = False):
    if len(y_exp) != len(y_theory):
        print "Warning: number of points in chi_exp and chi_theory is not equal"
        
    numb = min(len(y_exp),len(y_theory))
    area_diff = 0.00
    for i in range(0, numb):
      diff = numpy.absolute(y_exp[i] - y_theory[i])
      #if x is not None:
      #  area_diff = area_diff + diff * x[i] ** 2
      if calc_type == 'least_square':
        area_diff = area_diff + (diff/y_exp[i])**2
      elif calc_type == 'area':
        area_diff = area_diff + diff
    if average:
       area_diff = area_diff / numb
    print ('%s: %15.6f' % ("area_diff", area_diff))
    return area_diff

#linearly interpolate y values based on y_std value
def match_x(x_std, y_src, x_src, xmin, xmax):
    """
    x_std..........x values used as a standard for the rescaling
    y_src........y values required to be rescaled
    x_src..........x values corresponding to y_src
    """
    x_temp = []
    y_temp = []
    #reset chi_calc based on k_exp
    #tell if k_exp starts from a smaller value
#          try:
#          result = compareValue(k_exp[0],k_cacl[0])
#      except MyValidationError as exception:
#          print exception.message
    i = 0   
    while ( 0 <= i < len(x_std) and x_std[i] < xmax):
        if x_std[i] < xmin:
            i += 1
            continue
        for j in range(1,len(x_src)):
            if x_src[j-1] < x_std[i] and x_std[i] < x_src[j]:
                y_temp.append(numpy.interp(x_std[i],
                                           [x_src[j-1],x_src[j]],
                                       [y_src[j-1],y_src[j]]))
                x_temp.append(x_std[i])

            elif x_std[i] == x_src[j-1]:
                y_temp.append(y_src[j-1])
                x_temp.append(x_std[i])
        i += 1
    return x_temp, y_temp

class Expectra(object):

    """
    set multiple_scattering = '--multiple-scattering' to enalbe multiple
    scattering calculation. Otherwise first-shell calculation will be
    conducted.
    """
    
    def __init__(self, label='EXAFS',
                kmin = 2.50, 
                kmax = 10.00, 
                chi_deviation = 100, 
                area_diff = 100, **kwargs):
        """
        The expectra constructor:
            kmin, kmax ...........define the k window you are interested on. It is
            suggested to set them as the values appeared in experimental data
            atoms.................coordinates or trajectories
            chi_deviaiton.........used to store the calcuation deviation between
            experimental and calculated EXAFS spectra
        """
        self.lable = label
        self.kmin = kmin
        self.kmax = kmax
        self.chi_deviation = chi_deviation
        self.area_diff = area_diff
        self.results = None
        self.x = None
        self.y = None
        self.traj_filename = None

        for parameter in kwargs:
            if parameter not in default_parameters:
               print parameter, 'is not in the keywords included'
               break
        for (parameter, default) in default_parameters.iteritems():
            setattr(self, parameter, kwargs.get(parameter, default))
      
    def get_kmin(self):
        print 'getting'
        return self.kmin

    def get_chi_differ(self, k=None, chi=None):
        #self.traj_filename = filename
        x_thy, y_thy = k, chi

        #load experimental chi data
        try:
            x_exp, y_exp = read_chi(self.exp_file) 
        except:
            x_exp, y_exp = load_chi_dat(self.exp_file)

        xmin = self.kmin
        xmax = self.kmax
        #interpolate chi_exp values based on k values provided in calculated data
        x_exp, y_exp = match_x(x_thy, y_exp, x_exp, xmin, xmax)
        x_thy, y_thy = match_x(x_thy, y_thy, x_thy, xmin, xmax)

        y_thy = numpy.multiply(y_thy, numpy.power(x_thy, self.kweight))

        self.x = x_thy
        self.y = y_thy

        if self.debug:
           filename2 = 'rescaled_theory_chi.dat'
           save_result(x_thy, y_thy, filename2)
           
           filename2 = 'rescaled_exp_chi.dat'
           save_result(x_exp, y_exp, filename2)

        self.area_diff = calc_area(y_exp, y_thy)
        print "area calculation is done"
#        if properties is None:
#            self.calculate(atoms, 'chi_area')
#            return self.area_diff, numpy.array(self.x), numpy.array(self.y)
#        else:
#            self.calculate(atoms, 'chi_deviation')
#            return self.chi_deviation, numpy.array(self.x), numpy.array(self.y)

#    def get_absorber(self):
#        return self.parameters.absorber

    def calculate(self, atoms=None, properties=None):

        if self.multiple_scattering == True:
            self.first_shell = False

        trajectory = []
        #print "atom", len(atoms), type(atoms), "rank",COMM_WORLD.rank
        if COMM_WORLD.rank==0:
           if isinstance(atoms, list):
              trajectory= atoms[self.skip::self.every]
           else:
              trajectory.append(atoms)
           print '%4i configurations are received' % (len(trajectory))
           print type(trajectory)
        trajectory = COMM_WORLD.bcast(trajectory)
        self.absorber = self.get_default_absorber(trajectory[0])
      
        k, chi = self.exafs_trajectory(trajectory)
        print k, chi
        #Calculate EXAFS difference compared to the given experiment data or store data
        if COMM_WORLD.rank == 0:
           if properties == 'area':
              self.get_chi_differ(k, chi)
              return self.area_diff, k, chi
           else:
              save_result(k, chi, 'chi.dat')
              print "EXAFS Calculation is done. Data is stored in 'Chi.dat'."

    def get_default_absorber(self, atoms):
        symbols = set(atoms.get_chemical_symbols())
        if self.absorber:
            if self.absorber not in symbols:
                print 'ERROR: --absorber %s is not in the system' % self.absorber
                sys.exit(2)
            else:
                return self.absorber
        if self.ignore_elements:
            symbols -= set(self.ignore_elements)
        if len(symbols) == 1:
            return list(symbols)[0]
        else:
            print 'ERROR: must specify --absorber if more than one chemical specie'
            sys.exit(2)
   
    def exafs_trajectory(self, trajectory):
        if self.multiple_scattering:
            k, chi = exafs_multiple_scattering(self.S02, self.energy_shift, 
                    self.absorber, self.ignore_elements, self.edge, self.rmax,
                    self.sig2,
                    trajectory)
        elif self.first_shell:
            k, chi = exafs_first_shell(self.S02, self.energy_shift, 
                    self.absorber, self.ignore_elements, self.edge, 
                    self.neighbor_cutoff, self.sig2, trajectory)
   
        return k, chi


"""
version call expectra with 'mpirun'
        #write geoemtry file 'con' if trajectory file doesn't exist
        if self.traj_filename is None:
           self.traj_filename = 'CONCAR'
           write_vasp(filename = self.traj_filename, atoms = atoms, direct=True, vasp5=True)

        #prepare the command to run 'expectra'
        if parameters.ignore_elements is not None:
            ignore = '--ignore-elements ' + parameters.ignore_elements
        else:
            ignore = ''
        
        expectra_para = ['mpirun -n', str(parameters.ncore),
                         'expectra', parameters.multiple_scattering,
                         '--neighbor-cutoff', str(parameters.neighbor_cutoff),
                         '--S02', str(parameters.S02),
                         '--sig2',str(parameters.sig2),
                         '--energy-shift', str(parameters.energy_shift),
                         '--edge', parameters.edge,
                         '--absorber', parameters.absorber,
                         ignore,
                         '--specorder', parameters.specorder,
                         '--skip', str(parameters.skip),
                         '--every', str(parameters.every),
                         self.traj_filename]
        join_symbol = ' '
        expectra_cmd = join_symbol.join(expectra_para)
        print 'Expectra parameters:'
        print '   ', expectra_cmd

        #run 'expectra'
        os.system(expectra_cmd)

        if parameters.real_space:
           print "Compare exafs in real space"
           xafsft_para = ['xafsft',
                          '--kmin', str(parameters.kmin),
                          '--kmax', str(parameters.kmax),
                          '--kweight', str(parameters.kweight),
                          '--dk', str(parameters.dk),
                          '--rmin', str(parameters.rmin),
                          '--rmax', str(parameters.rmax),
                          '--ft-part', parameters.ft_part,
                          'chi.dat']
           join_symbol = ' '
           xafsft_cmd = join_symbol.join(xafsft_para)
           print 'Fourier transformation parameters used:'
           print '   ', xafsft_cmd
           os.system(xafsft_cmd)
           inputfile = 'exafs.chir'
           xmin = parameters.rmin
           xmax = parameters.rmax
        else:
           print "Compare exafs in k-space"
           inputfile = 'chi.dat'
           xmin = parameters.kmin
           xmax = parameters.kmax
        #load calculated chi data
        try:
            x_thy, y_thy = read_chi(inputfile) 
        except:
            x_thy, y_thy = load_chi_dat(inputfile)

        #load experimental chi data
        try:
            x_exp, y_exp = read_chi(parameters.exp_file) 
        except:
            x_exp, y_exp = load_chi_dat(parameters.exp_file)

        #interpolate chi_exp values based on k values provided in calculated data
        x_exp, y_exp = match_x(x_thy, y_exp, x_exp, xmin, xmax)
        x_thy, y_thy = match_x(x_thy, y_thy, x_thy, xmin, xmax)


        if not parameters.real_space:
           y_thy = numpy.multiply(y_thy, numpy.power(x_thy, parameters.kweight))

        self.x = x_thy
        self.y = y_thy

        if parameters.debug:
           filename2 = 'rescaled_theory_chi.dat'
           save_result(x_thy, y_thy, filename2)
           
           filename2 = 'rescaled_exp_chi.dat'
           save_result(x_exp, y_exp, filename2)

        if parameters.real_space:
            self.area_diff = calc_area(y_exp, y_thy, calc_type = parameters.calc_type, average = parameters.average)
        else:
            self.area_diff = calc_area(y_exp, y_thy)
"""
