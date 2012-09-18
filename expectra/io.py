from expectra.aselite import Atoms, Atom
import numpy

def read_xdatcar(filename, skip=0, every=1):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    lattice_constant = float(lines[1].strip())
    cell = numpy.array([[float(x) * lattice_constant for x in lines[2].split()], 
                        [float(x) * lattice_constant for x in lines[3].split()], 
                        [float(x) * lattice_constant for x in lines[4].split()]])
    elements = lines[5].split()
    natoms = [int(x) for x in lines[6].split()]
    nframes = (len(lines)-7)/(sum(natoms) + 1)
    trajectory = []
    for i in range(skip, nframes, every):
        a = Atoms('H'*sum(natoms))
        a.masses = [1.0] * len(a)
        a.set_chemical_symbols(''.join([n*e for (n, e) in zip(natoms, elements)]))
        a.cell = cell.copy()
        j = 0
        for N, e in zip(natoms, elements):
            for k in range(N):
                split = lines[8 + i * (sum(natoms) + 1) + j].split()
                a[j].position = [float(l) for l in split[0:3]]
                j += 1
        a.positions = numpy.dot(a.positions, cell)
        trajectory.append(a)
    return trajectory

def write_chir(filename, r, chir):                                                                       
    f = open(filename, 'w') 
    for i in xrange(len(r)):
        f.write('%e %e\n' % (r[i], numpy.abs(chir[i])))
    f.close()

def read_chi(filename):
    f = open(filename)
    ks = []
    chis = []
    for line in f:
        if line.startswith('#'):
            continue
        fields = [ float(field) for field in line.split() ]
        k = fields[0]                                                           
        chi = fields[1]
        ks.append(k)
        chis.append(chi)
    f.close()
    ks = numpy.array(ks)
    chis = numpy.array(chis)
    return ks, chis

