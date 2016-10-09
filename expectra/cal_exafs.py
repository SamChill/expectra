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
def calc_area(y_exp, y_theory, x=None):
    if len(y_exp) != len(y_theory):
        print "Warning: number of points in chi_exp and chi_theory is not equal"
        
    area_diff = 0.00
    for i in range(0, len(chi_exp)):
        if x is not None:
           area_diff = area_diff + (numpy.absolute(y_exp[i] - y_theory[i])) * x[i] ** 2
        else:
           area_diff = area_diff + (numpy.absolute(y_exp[i] - y_theory[i]))

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
        skip = 0,
        every = 1,
        exp_file = 'chi_exp.dat',
        #Following parameters used for xafsft to calculate g_r plot
        kweight= 2.0,
        dk = 1.0,
        rmin = 0.0,
        rmax = 8.0,
        ft_part = 'mag')
    """
    set multiple_scattering = '--multiple-scattering' to enalbe multiple
    scattering calculation. Otherwise first-shell calculation will be
    conducted.
    ncore is number of cores used for calcualtion.
    """

    def __int__(self, label='EXAFS',
                g_r = False,
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
        self.atoms = atoms
        self.kmin = kmin
        self.kmax = kmax
        self.chi_deviation = chi_deviation
        self.area_diff = area_diff
        self.parameters = None
        self.results = None
        self.x = None
        self.y = None
      
        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms,
                            **kwargs)
      

    def set(self, **kwargs):
        """Set parameters like set(key1=value1, key2=value2,
        ...)."""
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
           self.reset()

    def get_potential_energy(self, atoms=None, properties=None):
        if properties is None:
            self.calculate(atoms, 'chi_area')
            return self.area_diff, self.x, self.y
        else:
            self.calculate(atoms, 'chi_deviation')
            return self.chi_deviation, self.x, self.y

    def calculate(self, atoms=None, properties=None):

        parameters = self.parameters
        
        #write 'CONTCAR' which is required for 'expectra' code
        con_filename = 'CONTCAR'
        write_vasp(filename = con_filename, atoms = atoms, direct=True, vasp5=True)

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
                         '--skip', str(parameters.skip),
                         '--every', str(parameters.every),
                         con_filename]
        join_symbol = ' '
        expectra_cmd = join_symbol.join(expectra_para)

        #run 'expectra'
        os.system(expectra_cmd)

        if parameters.g_r:
           print "g_r"
           self.gr_function()
           inputfile = 'exafs.chir'
           xmin = parameters.rmin
           xmax = parameters.rmax
        else:
           print "chi.dat"
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
            x_exp, y_exp = read_chi(parameters.exp_chi_file) 
        except:
            x_exp, y_exp = load_chi_dat(parameters.exp_chi_file)

        filename2 = 'test_exp_chi.dat'
        save_result(x_exp, y_exp, filename2)

        #interpolate chi_exp values based on k values provided in calculated data
        x_exp, y_exp = match_x(x_thy, y_exp, x_exp, xmin, xmax)
        x_thy, y_thy = match_x(x_thy, y_thy, x_thy, xmin, xmax)

        self.x = x_thy
        self.y = y_thy

        filename2 = 'rescaled_exp_chi.dat'
        save_result(x_exp, y_exp, filename2)

        if g_r:
            self.area_diff = calc_area(y_exp, y_thy)
        else:
            self.area_diff = calc_area(y_exp, y_thy, k_thy)

    def gr_function(self, parameters):
        xafsft_para = ['xafsft',
                       '--kmin', parameters.kmin,
                       '--kmax', parameters.kmax,
                       '--kweight', parameters.kweight,
                       '--dk', parameters.dk,
                       '--rmin', parameters.rmin,
                       '--rmax', parameters.rmax,
                       '--ft-part',parameter.ft_part,
                       'chi.dat']
        join_symbol = ' '
        xafsft_cmd = join_symbol.join(xafsft_para)
        os.system(xafsft_cmd)
