"""Following codes are copied and modfied from atom.py in EON"""

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

def rot_match(a, b, comp_eps_r):
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
    print "max differece between two configs:", dist
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


