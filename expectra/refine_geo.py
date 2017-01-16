import sys
import os
from ase import Atoms
from ase.io import read, write, iread
from ase.optimize.lbfgs import LBFGS
from ase.io.trajectory import Trajectory
from expectra.cal_exafs import Expectra

class refine_geos:
    """This class can help select and refine  the geometries based on the maximum energy
    (Emax) and area (Smax) between exp. and cal. chi curve.
    """
    def __init__(self, geo_inputfile=None,
                pot_log = None,
                outputfile1 = None,
                outputfile2 = None,
                chi_logfile = None,
                S_E_logfile = None,
                fmt1 = None,
                fmt2 = None,
                optimizer = LBFGS,
                calc1 = None,
                calc2 = None):
        """
        geo_inputfile: traj format file from exafs_basin calculation
        pot_log: the file logged with E and S from exafs_basin calculation
        outputfile1: used to log initial geometries from traj file
        outputfile2: used to log optimized geometries
        logfile: used to log re-calculated energy E and area S
        calc1: calculator for geometry optimization
        calc2: exafs calculator
        """
        self.geo_inputfile = geo_inputfile
        self.pot_log = pot_log
        self.fmt1 = fmt1
        self.outputfile1 = outputfile1
        self.outputfile2 = outputfile2
        self.fmt2 = fmt2
        self.calc1 = calc1
        self.calc2 = calc2
        self.optimizer = optimizer
        self.chi_logfile = chi_logfile
        self.S_E_logfile = S_E_logfile
        
        self.log_S_E = open(self.S_E_logfile, 'w')
        self.chi_log = open(self.chi_logfile, 'w')

    def run(self, fmax,
            Emax = None,
            Smax = None):

        p1 = []
        p2 = []

        if Emax is None:
           print "please assign Emax"
        if Smax is None:
           print "please assign Smax"

        if self.geo_inputfile is None:
           print "please provide traj input file"
        else:
           traj = Trajectory(self.geo_inputfile)

        if self.pot_log is None:
           print "no pot_log file name is provided"
        else:
           f = open(self.pot_log,'r')

        f.readline()

        for line in f:
            column = line.split( )
            if float(column[4]) < Smax and float(column[3]) < Emax:
               image_numb = int(column[1]) + 1
               print(str(image_numb)+'     '+column[3]+'     '+column[4])
               
               atoms = traj[image_numb]

               #store the initial geometry
               p1.append(atoms)

               #optimize the atoms and store the optimized structure
               atoms.set_calculator(self.calc1)
               opt = self.optimizer(atoms,logfile = '-')
               opt.run(fmax)
               energy = atoms.get_potential_energy()
               p2.append(atoms)


               #calculate and log the exafs and area S
               atoms.set_calculator(self.calc2)
               S, k, chi = atoms.get_potential_energy()
               log_chi(image_numb, k, chi)
               log_S_E(image_numb, energy, S)

        f.close()
        write(filename=self.outputfile1, images=p1, format = self.fmt2)
        write(filename=self.outputfile2, images=p2, format = self.fmt2)

    def log_chi(self, step, k, chi):
        self.chi_log.write("step: %d\n" % (step))
        for i in xrange(len(k)):
            self.chi_log.write("%6.3f %16.8e\n" % (k[i], chi[i]))
        self.chi_log.flush()

    def log(self, step, E, S):
        if self.log_S_E is None:
           return
        self.log_S_E.write('%d  %15.6f  %15.6f\n'
                            % (step, E, S))
        self.log_S_E.flush()

