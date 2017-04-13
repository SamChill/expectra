"""Following codes are copied and modfied from atom.py in EON"""

import time 
from math import cos, sin, acos
import numpy

def pbc(r, box, ibox = None):
    """
    Applies periodic boundary conditions.
    Parameters:
        r:      the vector the boundary conditions are applied to
        box:    the box that defines the boundary conditions
        ibox:   the inverse of the box. This will be calcluated if not provided.
    """
    if ibox is None:    
        ibox = numpy.linalg.inv(box)
    vdir = numpy.dot(r, ibox)
    vdir = (vdir % 1.0 + 1.5) % 1.0 - 0.5
    return numpy.dot(vdir, box)

def per_atom_norm(v, box, ibox = None):
    '''
    Returns a length N numpy array containing per atom distance
        v:      an Nx3 numpy array
        box:    box matrix that defines the boundary conditions
        ibox:   the inverse of the box. will be calculated if not provided
    '''
    diff = pbc(v, box, ibox)
    return numpy.sqrt(numpy.sum(diff**2.0, axis=1)) 

def rot_match(a, b, comp_eps_r, readtime):
    """
       Following codes are modfied from atom.py in EON
       Determine if Configuration 'a' and 'b' is matching.
       Using the method described in the paper 'Horn, J. Opt. Soc. Am. A, 1987' to traslate and rotate the configuration.
       All the atoms are considered distinguishable.
       Input configurations should be 'atoms' defined in ASE.
       comp_eps_r is distance tolerance to considered two configurations are identical
    """
    #TODO: method to map atoms in 'a' to those 'b' so that to deal with 'indistinguishable' situation
#    if not (a.free.all() and b.free.all()):
#        print "Comparing structures with frozen atoms with rotational matching; check_rotation may be set incorrectly"
    acm = a.get_center_of_mass()
    bcm = b.get_center_of_mass()

    ta = a.copy()
    tb = b.copy()
    
    ta.translate(-acm)
    tb.translate(-bcm)

    starttime = time.time()

    #Horn, J. Opt. Soc. Am. A, 1987
    m = numpy.dot(tb.get_positions().transpose(), ta.get_positions())
    sxx = m[0][0]
    sxy = m[0][1]
    sxz = m[0][2]
    syx = m[1][0]
    syy = m[1][1]
    syz = m[1][2]
    szx = m[2][0]
    szy = m[2][1]
    szz = m[2][2]

    n = numpy.zeros((4,4))
    n[0][1] = syz-szy
    n[0][2] = szx-sxz
    n[0][3] = sxy-syx

    n[1][2] = sxy+syx
    n[1][3] = szx+sxz

    n[2][3] = syz + szy

    n += n.transpose()

    n[0][0] = sxx + syy + szz
    n[1][1] = sxx-syy-szz
    n[2][2] = -sxx + syy -szz
    n[3][3] = -sxx -syy + szz

    w,v = numpy.linalg.eig(n)
    maxw = 0
    maxv = 0
    for i in range(len(w)):
        if w[i] > maxw:
            maxw = w[i]
            maxv = v[:,i]

    R = numpy.zeros((3,3))

    aa = maxv[0]**2
    bb = maxv[1]**2
    cc = maxv[2]**2
    dd = maxv[3]**2
    ab = maxv[0]*maxv[1]
    ac = maxv[0]*maxv[2]
    ad = maxv[0]*maxv[3]
    bc = maxv[1]*maxv[2]
    bd = maxv[1]*maxv[3]
    cd = maxv[2]*maxv[3]

    R[0][0] = aa + bb - cc - dd
    R[0][1] = 2*(bc-ad) 
    R[0][2] = 2*(bd+ac) 
    R[1][0] = 2*(bc+ad) 
    R[1][1] = aa - bb + cc - dd
    R[1][2] = 2*(cd-ab) 
    R[2][0] = 2*(bd-ac) 
    R[2][1] = 2*(cd+ab) 
    R[2][2] = aa - bb - cc + dd
    tb.set_positions(numpy.dot(tb.get_positions(), R.transpose()))

    dist = max(per_atom_norm(ta.get_positions() - tb.get_positions(), ta.get_cell()))

    donetime = time.time()
    print "max differece between two configs:", dist, readtime, donetime - starttime
    return dist < comp_eps_r

    ### This gives the RMSD faster, but does not give the optimial rotation
    ### this could be amended by solving for the eigenvector corresponding to the largest eigenvalue
    ### Theobald, Acta Crystallographica A, 2005
    #
    ##could be faster if done explicitly
    #c0 = numpy.linalg.det(k)
    #c1 = -8*numpy.linalg.det(m)
    #c2 = -2*numpy.trace(numpy.dot(m.transpose(), m))

    #ga = numpy.trace(numpy.dot(ta.r.transpose(), ta.r))
    #gb = numpy.trace(numpy.dot(tb.r.transpose(), tb.r))
    #
    #lold = 0.0
    #l = (ga + gb)/2.0
    #while abs(lold - l) > 0.00001:
    #    lold = l
    #    l -= (l**4 + c2*l**2 + c1*l + c0)/(4*l**3 + 2*c2*l + c1)
    #rmsd = sqrt((ga + gb - 2*l)/len(a))
    #return rmsd < config.comp_rot_rmsd

def brute_neighbor_list(p, cutoff):
    nl = []
    ibox = numpy.linalg.inv(p.box)
    for a in range(len(p)):
        nl.append([])
        for b in range(len(p)):
            if b != a:
                dist = numpy.linalg.norm(pbc(p.r[a] - p.r[b], p.box, ibox))
                if dist < cutoff:
                    nl[a].append(b)
    return nl

def sweep_and_prune(p_in, cutoff, strict = True, bc = True):
    """ Returns a list of nearest neighbors within cutoff for each atom.
        Parameters:
            p_in:   Atoms object
            cutoff: the radius within which two atoms are considered to intersect.
            strict: perform an actual distance check if True
            bc:     include neighbors across pbc's """
    #TODO: Get rid of 'cutoff' and use the covalent bond radii. (Rye can do)
        # Do we want to use covalent radii? I think the displace class wants to allow for user-defined cutoffs.
            # We should have both options available. -Rye
    #TODO: Make work for nonorthogonal boxes.
    p = p_in.copy()
    p.r = pbc(p.r, p.box)
    p.r -= numpy.array([min(p.r[:,0]), min(p.r[:,1]), min(p.r[:,2])])
    numatoms = len(p)
    coord_list = []
    for i in range(numatoms):
        coord_list.append([i, p.r[i]])
    for axis in range(3):
        sorted_axis = sorted(coord_list, key = lambda foo: foo[1][axis])
        intersect_axis = []
        for i in range(numatoms):
            intersect_axis.append([])
        for i in range(numatoms):
            done = False
            j = i + 1
            if not bc and j >= numatoms:
                done = True
            while not done:
                j = j % numatoms
                if j == i:
                    done = True
                dist = abs(sorted_axis[j][1][axis] - sorted_axis[i][1][axis])
                if p.box[axis][axis] - sorted_axis[i][1][axis] < cutoff:
                    dist = min(dist, (p.box[axis][axis] - sorted_axis[i][1][axis]) + sorted_axis[j][1][axis])
                if dist < cutoff:
                    intersect_axis[sorted_axis[i][0]].append(sorted_axis[j][0]) 
                    intersect_axis[sorted_axis[j][0]].append(sorted_axis[i][0]) 
                    j += 1
                    if not bc and j >= numatoms:
                        done = True
                else:
                    done = True
        if axis == 0:
            intersect = []
            for i in range(numatoms):
                intersect.append([])
                intersect[i] = intersect_axis[i]
        else:
            for i in range(numatoms):
                intersect[i] = list(set(intersect[i]).intersection(intersect_axis[i]))
    if strict:
        ibox = numpy.linalg.inv(p.box)
        for i in range(numatoms):
            l = intersect[i][:]
            for j in l:
                dist = numpy.linalg.norm(pbc(p.r[i] - p.r[j], p.box, ibox))
                if dist > cutoff:
                    intersect[i].remove(j)
                    intersect[j].remove(i)
    return intersect

def neighbor_list(p, cutoff, brute=False):
    if brute:
        nl = brute_neighbor_list(p, cutoff)
    else:
        nl = sweep_and_prune(p, cutoff)
    return nl

def coordination_numbers(p, cutoff, brute=False):
    """ Returns a list of coordination numbers for each atom in p """
    nl = neighbor_list(p, cutoff, brute)
    return [len(l) for l in nl]

import sys
sys.setrecursionlimit(10000)
def get_mappings(a, b, eps_r, neighbor_cutoff, mappings = None):
    """ A recursive depth-first search for a complete set of mappings from atoms
        in configuration a to atoms in configuration b. Do not use the mappings
        argument, this is only used internally for recursion.

        Returns None if no mapping was found, or a dictionary mapping atom
        indices a to atom indices b.

        Note: If a and b are mirror images, this function will still return a
        mapping from a to b, even though it may not be possible to align them
        through translation and rotation. """
    # If this is the top-level user call, create and loop through top-level
    # mappings.
    if mappings is None:
        # Find the least common coordination number in b.
        bCoordinations = coordination_numbers(b, neighbor_cutoff)
        bCoordinationsCounts = {}
        for coordination in bCoordinations:
            if coordination in bCoordinationsCounts:
                bCoordinationsCounts[coordination] += 1
            else:
                bCoordinationsCounts[coordination] = 1
        bLeastCommonCoordination = bCoordinationsCounts.keys()[0]
        for coordination in bCoordinationsCounts.keys():
            if bCoordinationsCounts[coordination] < bCoordinationsCounts[bLeastCommonCoordination]:
                bLeastCommonCoordination = coordination
        # Find one atom in a with the least common coordination number in b.
        # If it does not exist, return None.
        aCoordinations = coordination_numbers(a, neighbor_cutoff)
        try:
            aAtom = aCoordinations.index(bLeastCommonCoordination)
        except ValueError:
            return None
        # Create a mapping from the atom chosen from a to each of the atoms with
        # the least common coordination number in b, and recurse.
        for i in range(len(bCoordinations)):
            if bCoordinations[i] == bLeastCommonCoordination:
                # Make sure the element types are the same.
                if a.names[aAtom] != b.names[i]:
                    continue
                mappings = get_mappings(a, b, eps_r, neighbor_cutoff, {aAtom:i})
                # If the result is not none, then we found a successful mapping.
                if mappings is not None:
                    return mappings
        # There were no mappings.
        return None

    # This is a recursed invocation of this function.
    else:
        # Find an atom from a that has not yet been mapped.
        unmappedA = 0
        while unmappedA < len(a):
            if unmappedA not in mappings.keys():
                break
            unmappedA += 1
        # Calculate the distances from unmappedA to all mapped a atoms.
        distances = {}
        for i in mappings.keys():
            distances[i] = numpy.linalg.norm(pbc(a.r[unmappedA] - a.r[i], a.box))
        # Loop over each unmapped b atom. Compare the distances between it and
        # the mapped b atoms to the corresponding distances between unmappedA
        # and the mapped atoms. If everything is similar, create a new mapping
        # and recurse.
        for bAtom in range(len(b)):
            if bAtom not in mappings.values():
                for aAtom in distances:
                    # Break if type check fails.
                    if b.names[bAtom] != a.names[unmappedA]:
                        break
                    # Break if distance check fails
                    bDist = numpy.linalg.norm(pbc(b.r[bAtom] - b.r[mappings[aAtom]], b.box))
                    if abs(distances[aAtom] - bDist) > eps_r:
                        break
                else:
                    # All distances were good, so create a new mapping.
                    newMappings = mappings.copy()
                    newMappings[unmappedA] = bAtom
                    # If this is now a complete mapping from a to b, return it.
                    if len(newMappings) == len(a):
                        return newMappings
                    # Otherwise, recurse.
                    newMappings = get_mappings(a, b, eps_r, neighbor_cutoff, newMappings)
                    # Pass any successful mapping up the recursion chain.
                    if newMappings is not None:
                        return newMappings
        # There were no mappings.
        return None 

