import numpy
from time import strftime
from expectra.aselite import bulk, units, NeighborList
from expectra.feff import feff_edge_number, run_feff

from mpi4py import MPI
COMM_WORLD = MPI.COMM_WORLD

def getxk(e):
    return numpy.sign(e)*numpy.sqrt(numpy.abs(e))

def xk2xkp(xk, vrcorr):
    xk0 = xk*units.Bohr
    vr = vrcorr / units.Rydberg
    xksign = numpy.sign(xk0)
    e = xksign*xk0**2 + vr
    
    return getxk(e) / units.Bohr

def xkp2xk(xk, vrcorr):
    xk0 = xk*units.Bohr
    vr = vrcorr / units.Rydberg
    xksign = numpy.sign(xk0)
    e = xksign*xk0**2 - vr
    
    return getxk(e) / units.Bohr

def chi_path(path, r, sig2, energy_shift, s02, N):
    #this gives errors at small k (<0.1), but it doesn't matter
    #as the k-window should always be zero in this region
    #also uses real momentum for dw factor
    delta_k = 0.05
    vrcorr = energy_shift

    xkpmin = xk2xkp(path['xk'][0], vrcorr)
    n = int(xkpmin / delta_k)
    if xkpmin > 0.0: n += 1
    xkmin = n*delta_k

    xkout = numpy.arange(xkmin, 20.0+delta_k, delta_k)
    xk0 = xkp2xk(xkout, vrcorr)

    f0      =  numpy.interp(xk0, path['xk'], path['afeff'])
    lambda0 =  numpy.interp(xk0, path['xk'], path['xlam'])
    delta0  =  numpy.interp(xk0, path['xk'], path['cdelta'] + path['phfeff'])
    redfac0 =  numpy.interp(xk0, path['xk'], path['redfac'])
    rep0    =  numpy.interp(xk0, path['xk'], path['rep'])
    p0      =  rep0 + 1j/lambda0
    dr = r - path['reff']

    chi = numpy.zeros(len(xk0), dtype=complex)
    
    chi[1:]  = redfac0[1:]*s02*N*f0[1:]/(xk0[1:]*(path['reff']+dr)**2.0)
    chi[1:] *= numpy.exp(-2*path['reff']/lambda0[1:])
    chi[1:] *= numpy.exp(-2*(p0[1:]**2.0)*sig2)
    chi[1:] *= numpy.exp(1j*(2*p0[1:]*dr - 4*p0[1:]*sig2/path['reff']))
    chi[1:] *= numpy.exp(1j*(2*xk0[1:]*path['reff'] + delta0[1:]))

    return xkout, numpy.imag(chi)

def exafs_reference_path(z, feff_options):
    atoms = bulk(z, orthorhombic=True, cubic=True)
    atoms = atoms.repeat((4,4,4))
    center = numpy.argmin(numpy.sum((atoms.get_scaled_positions() -
        numpy.array( (.5,.5,.5) ))**2.0, axis=1))
    #do the bulk reference scattering calculation and get the path
    #data from feff
    path = run_feff(atoms, center, feff_options, get_path=True)[2]
    return path

def exafs_first_shell(S02, energy_shift, absorber, 
    ignore_elements, edge, neighbor_cutoff, trajectory):
    feff_options = {
            'RMAX':str(neighbor_cutoff),
            'HOLE':'%i %.4f' % (feff_edge_number(edge), S02),
            'CORRECTIONS':'%.4f %.4f' % (energy_shift, 0.0),
    }

    #get the bulk reference state
    path = exafs_reference_path(absorber, feff_options)

    k = None
    chi_total = None

    counter = -1
    interactions = 0
    nl = None

    for step, atoms in enumerate(trajectory):
        if COMM_WORLD.rank == 0:
            time_stamp = strftime("%F %T")
            print '[%s] step %i/%i' % (time_stamp, step+1, len(trajectory))
        atoms = atoms.copy()
        if ignore_elements:
            ignore_indicies = [atom.index for atom in atoms 
                               if atom.symbol in ignore_elements]
            del atoms[ignore_indicies]
        if nl == None:
            nl = NeighborList(len(atoms)*[neighbor_cutoff], skin=0.3, 
                    self_interaction=False)
        nl.update(atoms)

        for i in xrange(len(atoms)):
            if atoms[i].symbol != absorber:
                continue
            indicies, offsets = nl.get_neighbors(i)
            for j, offset in zip(indicies, offsets):
                counter += 1
                if counter % COMM_WORLD.size != COMM_WORLD.rank: 
                    continue

                r = atoms.get_distance(i,j,True)
                if r >= neighbor_cutoff: continue
                interactions += 1
                k, chi = chi_path(path, r, 0.0, energy_shift, S02, 1)

                if chi_total != None:
                    chi_total += chi
                else:
                    chi_total = chi
    chi_total = COMM_WORLD.allreduce(chi_total)
    chi_total /= atoms.get_chemical_symbols().count(absorber)
    chi_total /= len(trajectory)
    chi_total *= 2
    return k, chi_total

def get_neighbors(atoms):
    neighbors = [ 0 for i in range(len(atoms)) ]
    for i in range(len(atoms)):
        for j in range(i+1,len(atoms)):
            r = atoms.get_distance(i,j,True)
            if r < 3.4:
                neighbors[i] += 1
                neighbors[j] += 1
    return neighbors

def exafs_multiple_scattering(S02, energy_shift, absorber, 
    ignore_elements, edge, rmax, trajectory):
    feff_options = {
            'RMAX':str(rmax),
            'HOLE':'%i %.4f' % (feff_edge_number(edge), S02),
            'CORRECTIONS':'%.4f %.4f' % (energy_shift, 0.0),
            'NLEG':'4',
    }

    k = None
    chi_total = None
    counter = -1
    for step, atoms in enumerate(trajectory):
        if COMM_WORLD.rank == 0:
            time_stamp = strftime("%F %T")
            print '[%s] step %i/%i' % (time_stamp, step+1, len(trajectory))

        atoms = atoms.copy()

        first_shell_idx = [ i for i,n in enumerate(get_neighbors(atoms)) if n != 12 ]
        first_shell  = atoms[first_shell_idx]
        tmp = atoms.copy()
        del tmp[first_shell_idx]
        second_shell = tmp[[ i for i,n in enumerate(get_neighbors(tmp)) if n != 12 ]]

        nfirst = len(first_shell)
        nsecond = len(second_shell)

        #for i in xrange(0,nsecond): #second shell
        for i in xrange(0,nfirst): #first shell
            counter += 1
            if counter % COMM_WORLD.size != COMM_WORLD.rank: 
                continue

            atoms = second_shell + first_shell[i]
            #atoms = first_shell + second_shell[i]
            if i == 0:
                import ase.io
                ase.io.write('shell_%i_%i.xyz'%(step,i),atoms)

            k, chi = run_feff(atoms, len(atoms)-1, feff_options)
            if k == None and chi == None:
                continue

            if chi_total != None:
                chi_total += chi
            else:
                chi_total = chi

    #in case too many ranks
    k = COMM_WORLD.bcast(k)
    if chi_total == None:
        chi_total = numpy.zeros(len(k))

    chi_total = COMM_WORLD.allreduce(chi_total)
    chi_total /= nfirst
    #chi_total /= nsecond
    chi_total /= len(trajectory)

    return k, chi_total
