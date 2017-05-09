import numpy

import argparse
import sys
import os

from expectra.exafs import exafs_first_shell, exafs_multiple_scattering
from expectra.io import read_xdatcar, read_con, read_chi
from ase.calculators.calculator import Calculator, all_changes, Parameters
from ase import Atoms
from ase.io.vasp import write_vasp
from expectra.feff import load_chi_dat

def save_result(k, chi, filename):
    print 'saving rescaled experimental chi data'
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

'''
Calculator is the superclass. Expectra is the subclass
'''
class Expectra(Calculator):

    implemented_properties = ['chi_deviation', 'chi_area']

    default_parameters = dict(
        #Following parameters used for expectra to calculate exafs
        ncore = 1,
        multiple_scattering = ' ',
        ignore_elements = None,
        neighbor_cutoff = 6.0,
        S02 = 0.89,
        energy_shift = 3.4,
        edge = 'L3',
        absorber = 'Au',
        specorder = "'Rh Au'",
        skip = 0,
        every = 1,
        exp_file = 'chi_exp.dat',
        #Following parameters used for xafsft to calculate g_r plot
        real_space = True,
        calc_type = 'area',
        average = False, #if true, the curve difference will be averaged by the number of dots
        kweight= 2,
        dk = 1.0,
        rmin = 2.0,
        rmax = 6.0,
        ft_part = 'mag',
        debug = False)
    """
    set multiple_scattering = '--multiple-scattering' to enalbe multiple
    scattering calculation. Otherwise first-shell calculation will be
    conducted.
    ncore is number of cores used for calcualtion.
    """

    def __int__(self, label='EXAFS',
                atoms = None, 
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
        self.real_space = real_space
        self.atoms = atoms
        self.kmin = kmin
        self.kmax = kmax
        self.chi_deviation = chi_deviation
        self.area_diff = area_diff
        self.parameters = None
        self.results = None
        self.x = None
        self.y = None
        self.traj_filename = None
      
        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms,
                            **kwargs)
      

    def set(self, **kwargs):
        """Set parameters like set(key1=value1, key2=value2,
        ...)."""
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
           self.reset()

    def get_chi_differ(self, atoms=None, properties=None, filename=None):
        self.traj_filename = filename
        if properties is None:
            self.calculate(atoms, 'chi_area')
            return self.area_diff, self.x, self.y
        else:
            self.calculate(atoms, 'chi_deviation')
            return self.chi_deviation, self.x, self.y

    def get_absorber(self):
        return self.parameters.absorber

    def calculate(self, atoms=None, properties=None):

        parameters = self.parameters
        
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

        self.x = x_thy
        self.y = y_thy

        if not parameters.real_space:
           y_thy = numpy.multiply(y_thy, numpy.power(x_thy, parameters.kweight))

        if parameters.debug:
           filename2 = 'rescaled_theory_chi.dat'
           save_result(x_thy, y_thy, filename2)
           
           filename2 = 'rescaled_exp_chi.dat'
           save_result(x_exp, y_exp, filename2)

        if parameters.real_space:
            self.area_diff = calc_area(y_exp, y_thy, calc_type = parameters.calc_type, average = parameters.average)
        else:
            #need debug
            self.area_diff = calc_area(y_exp, y_thy)
