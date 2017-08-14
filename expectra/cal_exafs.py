import numpy

import argparse
import sys
import os
import cPickle as pickle
import subprocess
from expectra.exafs import exafs_first_shell, exafs_multiple_scattering
from expectra.io import read_xdatcar, read_con, read_chi
from ase.calculators.calculator import Calculator, all_changes, Parameters
from ase import Atoms
from ase.io.vasp import write_vasp
from expectra.feff import load_chi_dat
from expectra import default_parameters

default_parameters=default_parameters.expectra_parameters

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

class Expectra(object):


    """
    set multiple_scattering = '--multiple-scattering' to enalbe multiple
    scattering calculation. Otherwise first-shell calculation will be
    conducted.
    ncore is number of cores used for calcualtion.
    """

    def __init__(self, label='EXAFS',
                atoms = None, 
                kmin = 2.50, 
                kmax = 10.00, 
                chi_deviation = 100, 
                area_diff = 100, 
                **kwargs):
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
        self.results = None
        self.x = None
        self.y = None

        for parameter in kwargs:
            if parameter not in default_parameters:
               print parameter, 'is not in the keywords included'
               break
        for (parameter, default) in default_parameters.iteritems():
            setattr(self, parameter, kwargs.get(parameter, default))


    def get_chi_differ(self, atoms=None, properties=None, filename=''):
        self.traj_filename = filename
        if properties is None:
            self.calculate(atoms, 'chi_area')
            return self.area_diff, self.x, self.y
        else:
            self.calculate(atoms, 'chi_deviation')
            return self.chi_deviation, self.x, self.y

    def get_absorber(self):
        return self.absorber

    def calculate(self, atoms=None, properties=None,k_exp = None,chi_exp = None):


        #prepare the command to run 'expectra'
        if self.ignore_elements is not None:
            ignore = '--ignore-elements ' + self.ignore_elements
        else:
            ignore = ''
        expectra_para = ['mpirun -n', str(self.ncore),
                         #'-bind-to-socket',
                         'expectra', self.multiple_scattering,
                         '--neighbor-cutoff', str(self.neighbor_cutoff),
                         '--S02', str(self.S02),
                         '--sig2',str(self.sig2),
                         '--energy-shift', str(self.energy_shift),
                         '--edge', self.edge,
                         '--absorber', self.absorber,
                         ignore,
                         '--specorder', self.specorder,
                         '--skip', str(self.skip),
                         '--every', str(self.every),
                         '--tmpdir', self.tmpdir,
                         self.traj_filename]
        join_symbol = ' '
        expectra_cmd = join_symbol.join(expectra_para)
        #print expectra_cmd
        #run 'expectra'
        proc = subprocess.Popen(expectra_cmd, shell=True, stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        proc.stdin.write(pickle.dumps(atoms))
        proc.stdin.flush()
        output,return_value = proc.communicate(), proc.returncode

        if output[1] is not None:
           print ("Error {} raised for expectra: {}".format(return_value,output[1]))
           sys.exit()

        lines = output[0].split('\n')
        for line in lines:
            if "k is" in line:
               k = numpy.array([float(element) for element in line.split('is')[1].strip().strip('[').strip(']').split(',')])
               continue
            if "chi is" in line:
               chi = numpy.array([float(element) for element in line.split('is')[1].strip().strip('[').strip(']').split(',')])
               continue
            print line

        if self.real_space:
           print "Compare exafs in real space"
           xafsft_para = ['xafsft',
                          '--kmin', str(self.kmin),
                          '--kmax', str(self.kmax),
                          '--kweight', str(self.kweight),
                          '--dk', str(self.dk),
                          '--rmin', str(self.rmin),
                          '--rmax', str(self.rmax),
                          '--ft-part', self.ft_part,
                          'chi.dat']
           join_symbol = ' '
           xafsft_cmd = join_symbol.join(xafsft_para)
           print 'Fourier transformation parameters used:'
           print '   ', xafsft_cmd
           os.system(xafsft_cmd)
           inputfile = 'exafs.chir'
           xmin = self.rmin
           xmax = self.rmax
        else:
           #print "Compare exafs in k-space"
           xmin = self.kmin
           xmax = self.kmax

        if properties == 'area':
           self.get_difference(k, chi, k_exp, chi_exp)
           return self.area_diff, k, chi
        else:
           save_result(k, chi, 'chi.dat')
           print "EXAFS Calculation is done. Data is stored in 'Chi.dat'."

    def get_difference(self, k=None, chi=None, k_exp=None, chi_exp=None):
        #self.traj_filename = filename
        x_thy, y_thy = k, chi
        x_exp, y_exp = k_exp, chi_exp

        xmin = self.kmin
        xmax = self.kmax
        y_thy = numpy.multiply(y_thy, numpy.power(x_thy, self.kweight))
        #interpolate chi_exp values based on k values provided in calculated data
        x_exp, y_exp = match_x(x_thy, y_exp, x_exp, xmin, xmax)
        x_thy, y_thy = match_x(x_thy, y_thy, x_thy, xmin, xmax)

        self.x = x_thy
        self.y = y_thy

        if self.debug:
           filename2 = 'rescaled_theory_chi.dat'
           save_result(x_thy, y_thy, filename2)
           
           filename2 = 'rescaled_exp_chi.dat'
           save_result(x_exp, y_exp, filename2)

        self.area_diff = calc_area(y_exp, y_thy)
        #print "area calculation is done"
#        if properties is None:
#            self.calculate(atoms, 'chi_area')
#            return self.area_diff, numpy.array(self.x), numpy.array(self.y)
#        else:
#            self.calculate(atoms, 'chi_deviation')
#            return self.chi_deviation, numpy.array(self.x), numpy.array(self.y)
