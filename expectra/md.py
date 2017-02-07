from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.units import kB, fs

def printenergy(a):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * kB), epot + ekin))

def run_md(atoms=None, md_step=100, temperature = 300 * kB, step_size = 1 * fs, trajectory=None):
    print "Running MD simulation:"
    # Set the momenta corresponding to md_temperature
    MaxwellBoltzmannDistribution(atoms, temperature)
    # We want to run MD with constant temperature using the Langevin algorithm
    #dyn = VelocityVerlet(atoms, step_size, 
    #                     trajectory=trajectory)
    dyn = Langevin(atoms, step_size, 
                   temperature, 0.002)
    # Now run the dynamics
    printenergy(atoms)
    for i in range(md_step):
        dyn.run(1)
        printenergy(atoms)


