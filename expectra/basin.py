import sys
import numpy as np

from ase.optimize.optimize import Dynamics
from ase.optimize.fire import FIRE
#from ase.optimize.LBFGS import LBFGS
from ase.units import kB
from ase.parallel import world
from ase.io.trajectory import Trajectory

class BasinHopping(Dynamics):
    """Basin hopping algorithm.

    After Wales and Doye, J. Phys. Chem. A, vol 101 (1997) 5111-5116

    and 

    David J. Wales and Harold A. Scheraga, Science, Vol. 285, 1368 (1999)
    """

    def __init__(self, atoms,
                 opt_calculator = None,
                 exafs_calculator = None,
                 ratio = 0.01,
                 temperature=100 * kB,
                 optimizer=FIRE,
                 fmax=0.1,
                 dr=0.1,
                 logfile='-',
                 chi_logfile='chi_log.dat',
                 trajectory='lowest.traj',
                 optimizer_logfile='-',
                 local_minima_trajectory='local_minima.traj',
                 adjust_cm=True):
        """Parameters:

        atoms: Atoms object
            The Atoms object to operate on.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.
        pseudo_pot: pseudo potential defined
        """
        Dynamics.__init__(self, atoms, logfile, trajectory)
        self.kT = temperature
        self.optimizer = optimizer
        self.fmax = fmax
        self.dr = dr
        self.opt_calculator = opt_calculator
        self.exafs_calculator = exafs_calculator
        self.ratio = ratio
        self.chi_logfile = chi_logfile
        self.k = None
        self.chi = None

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
        self.positions = 0.0 * self.atoms.get_positions()
        self.energy = 1.e32
        self.Umin = self.energy
        self.chi_deviation = 100.00
        self.rmin = self.atoms.get_positions()
        self.positions = self.atoms.get_positions()
        self.call_observers()

        #'logfile' is defined in the superclass Dynamics in 'optimize.py'
        self.logfile.write('   name      step accept       alpha      energy         chi_deviation         pseudoPot        Umin \n')
#        self.log(-1, self.Umin, self.chi_deviation, self.Umin,self.Umin)
                
    def run(self, steps):
        """Hop the basins for defined number of steps."""

        ratio = self.ratio
        ro = self.positions
        Eo = self.get_energy(ro)
        chi_devi_o = self.get_chi_deviation(self.atoms.get_positions())

        alpha = self.get_alpha()

        Uo = Eo + alpha * chi_devi_o
        
        self.chi_log = open(self.chi_logfile, 'w')
        self.log_chi(-1)

        self.log(-1,'Yes', alpha, Eo, chi_devi_o, Uo, self.Umin)

        for step in range(steps):
            Un = None
            while Un is None:
                rn = self.move(ro)
                if np.sometrue(rn != ro):

                    En = self.get_energy(rn)

                    chi_devi_n = self.get_chi_deviation(self.atoms.get_positions())

                    alpha_dynam = self.get_alpha()

                    Un = En + alpha * chi_devi_n
                    self.log_chi(step)
                else:
                    Un = self.energy + alpha * self.chi_deviation

            if Un < self.Umin:
                # new minimum found
                self.Umin = Un
                self.rmin = self.atoms.get_positions()
                self.call_observers()


            #accept or reject?
            accept = np.exp((Uo - Un) / self.kT) > np.random.uniform()
            if accept:
                ro = rn.copy()
                Uo = Un
            self.log(step, accept, alpha_dynam, En, chi_devi_n, Un, self.Umin)

    def log(self, step, accept, alpha, En, chi_devi_n, Un, Umin):
        if self.logfile is None:
            return
        name = self.__class__.__name__
        self.logfile.write('%s: %d  %s  %15.6f  %15.6f  %15.8f  %15.6f  %15.6f\n'
                           % (name, step, accept, alpha, En, chi_devi_n, Un, Umin))
        self.logfile.flush()

    def log_chi(self, step):
        self.chi_log.write("step: %d\n" % (step))
        k = self.k
        chi = self.chi
        for i in xrange(len(k)):
            self.chi_log.write("%6.3f %16.8e\n" % (k[i], chi[i]))
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

    def get_alpha(self):
        alpha = self.energy * self.ratio / self.chi_deviation
        return alpha

    def get_minimum(self):
        """Return minimal energy and configuration."""
        atoms = self.atoms.copy()
        atoms.set_positions(self.rmin)
        return self.Umin, atoms

    def get_energy(self, positions):
        """Return the energy of the nearest local minimum."""
        self.positions = positions
        self.atoms.set_positions(positions)

        #opt_calculator can be any calculator compatible with ASE
        """if self.opt_calculator is not None:
            try:
                self.atoms.set_calculator(self.opt_calculator)
                self.energy = self.atoms.get_potential_energy()
            except:
                # Something went wrong.
                # In GPAW the atoms are probably to near to each other.
                return None

            return self.energy
        """
        try:
            self.atoms.set_calculator(self.opt_calculator)
            opt = self.optimizer(self.atoms, 
                                 logfile=self.optimizer_logfile)
            opt.run(fmax=self.fmax)
            if self.lm_trajectory is not None:
                self.lm_trajectory.write(self.atoms)

            self.energy = self.atoms.get_potential_energy()
        except:
            # Something went wrong.
            # In GPAW the atoms are probably to near to each other.
            return None

        return self.energy


    def get_chi_deviation(self, positions):
        """Return the standard deviation of chi between calculated and
        experimental."""
        self.positions = positions
        self.atoms.set_positions(positions)

        try:
            self.atoms.set_calculator(self.exafs_calculator)
            self.chi_deviation, self.k, self.chi = self.atoms.get_potential_energy()
        except:
            # Something went wrong.
            # In GPAW the atoms are probably to near to each other.
            return None
   
        return self.chi_deviation

