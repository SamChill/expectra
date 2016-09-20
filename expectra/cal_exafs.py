import numpy

import argparse
import sys
import os

from expectra.exafs import exafs_first_shell, exafs_multiple_scattering
from expectra.io import read_xdatcar, read_con, read_chi
from ase.calculators.calculator import Calculator, all_changes, Parameters
from expectra.aselite import Atoms, write_vasp
from expectra.feff import load_chi_dat

def save_result(k, chi, filename):
    print 'saving rescaled experimental chi data'
    f = open(filename, 'w')
    for i in xrange(len(k)):
        f.write("%6.3f %16.8e\n" % (k[i], chi[i]))
    f.close()

def get_default_absorber(atoms, args):
    symbols = set(atoms.get_chemical_symbols())
    if args.absorber:
        if args.absorber not in symbols:
            print 'ERROR: --absorber %s is not in the system' % args.absorber
            sys.exit(2)
        else:
            return args.absorber
    if args.ignore_elements:
        symbols -= set(args.ignore_elements)
    if len(symbols) == 1:
        return list(symbols)[0]
    else:
        print 'ERROR: must specify --absorber if more than one chemical specie'
        sys.exit(2)

def exafs_trajectory(args, trajectory):
#    print "trajectory: "
#    print(type(trajectory))
    if args.multiple_scattering:
        k, chi = exafs_multiple_scattering(args.S02, args.energy_shift, 
                args.absorber, args.ignore_elements, args.edge, args.rmax, 
                trajectory)

    elif args.first_shell:
        k, chi = exafs_first_shell(args.S02, args.energy_shift, 
                args.absorber, args.ignore_elements, args.edge, 
                args.neighbor_cutoff, trajectory)
    
    return k, chi

#calculate the deviation of theoretical EXAFS from experimental EXAFS
def calc_deviation(chi_exp,chi_theory):
    chi_exp_array = numpy.asarray(chi_exp)
    chi_thry_array = numpy.asarray(chi_theory)
    chi_devi = numpy.sum(numpy.square(chi_exp_array - chi_thry_array))

    return chi_devi/len(chi_exp)

def calc_area(chi_exp,chi_theory):
    if len(chi_exp) != len(chi_theory):
        print "Warning: number of points in chi_exp and chi_theory is not equal"
        
    chi_area = 0.00
    for i in range(0, len(chi_exp)):
        chi_area = chi_area + numpy.absolute(chi_exp[i] - chi_theory[i])

    return chi_area

#linearly interpolate chi values based on k_std value
def rescale_chi_calc(k_std, chi_src, k_src, kmin, kmax):
    """
    k_std..........k values used as a standard for the rescaling
    chi_src........chi values required to be rescaled
    k_src..........k values corresponding to chi_src
    """
    k_temp = []
    chi_temp = []
    #reset chi_calc based on k_exp
    #tell if k_exp starts from a smaller value
#          try:
#          result = compareValue(k_exp[0],k_cacl[0])
#      except MyValidationError as exception:
#          print exception.message
    i = 0   

    while ( 0 <= i < len(k_std) and k_std[i] < kmax):
        if k_std[i] < kmin:
            i += 1
            continue
        for j in range(1,len(k_src)):
            if k_src[j-1] < k_std[i] and k_std[i] < k_src[j]:
                chi_temp.append(numpy.interp(k_std[i],
                                           [k_src[j-1],k_src[j]],
                                       [chi_src[j-1],chi_src[j]]))
                k_temp.append(k_std[i])

            elif k_std[i] == k_src[j-1]:
                chi_temp.append(chi_src[j-1])
                k_temp.append(k_std[i])
        i += 1
    return k_temp, chi_temp

'''
Calculator is the superclass. expectra is the subclass
'''
class Expectra(Calculator):

    implemented_properties = ['chi_deviation', 'chi_area']

    default_parameters = dict(
        ncore = 1,
        multiple_scattering = ' ',
        ignore_elements = None,
        neighbor_cutoff = 6.0,
        rmax = 6.0,
        S02 = 0.89,
        energy_shift = 3.4,
        edge = 'L3',
        absorber = 'Au',
        skip = 0,
        every = 1,
        exp_chi_file = 'chi_exp.dat',
        output_file = 'chi.dat')
    """
    set multiple_scattering = '--multiple-scattering' to enalbe multiple
    scattering calculation. Otherwise first-shell calculation will be
    conducted.
    ncore is number of cores used for calcualtion.
    """

    def __int__(self, label='EXAFS', 
                atoms=None, kmin=0.00, kmax=10.00, chi_deviation=100, chi_area
                = 100, **kwargs):
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
        self.chi_area = chi_area
        self.parameters = None
        self.results = None
        self.k = None
        self.chi =None
      
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
            return self.chi_area, self.k, self.chi
        else:
            self.calculate(atoms, 'chi_deviation')
            return self.chi_deviation, self.k, self.chi

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
        cmd = join_symbol.join(expectra_para)

        #run 'expectra'
        os.system(cmd)

        #load calculated chi data
        try:
            k, chi = read_chi('chi.dat') 
        except:
            k, chi = load_chi_dat('chi.dat')

        #load experimental chi data
        try:
            k_exp, chi_exp = read_chi(parameters.exp_chi_file) 
        except:
            k_exp, chi_exp = load_chi_dat(parameters.exp_chi_file)

        filename2 = 'test_exp_chi.dat'
        save_result(k, chi, filename2)
        #interpolate chi_exp values based on k values provided in calculated data
        k_exp, chi_exp = rescale_chi_calc(k, chi_exp, k_exp, parameters.kmin,
                                          parameters.kmax)
        k, chi = rescale_chi_calc(k, chi, k, parameters.kmin, parameters.kmax)

        self.k = k
        self.chi = chi

        filename2 = 'rescaled_exp_chi.dat'
        save_result(k_exp, chi_exp, filename2)


        if properties == 'chi_area':
            self.chi_area = calc_area(chi_exp, chi)
        else:
            self.chi_deviation = calc_deviation(chi_exp, chi)

