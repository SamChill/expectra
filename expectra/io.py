from expectra.aselite import Atoms, FixAtoms
import numpy

def length_angle_to_box(boxlengths, angles):
    box = numpy.zeros( (3,3) )
    angles *= numpy.pi/180.0
    box[0][0] = 1.0
    box[1][0] = numpy.cos(angles[0])
    box[1][1] = numpy.sin(angles[0])
    box[2][0] = numpy.cos(angles[1])
    box[2][1] = (numpy.cos(angles[2]) - box[1][0] * box[2][0])/box[1][1]
    box[2][2] = numpy.sqrt(1.0 - box[2][0]**2 - box[2][1]**2)
    box[0,:]*=boxlengths[0]
    box[1,:]*=boxlengths[1]
    box[2,:]*=boxlengths[2]
    return box

def box_to_length_angle(box):
    lengths = numpy.zeros(3)
    lengths[0] = numpy.linalg.norm(box[0,:])
    lengths[1] = numpy.linalg.norm(box[1,:])
    lengths[2] = numpy.linalg.norm(box[2,:])
    angles = numpy.zeros(3)
    angles[0] = numpy.arccos(numpy.dot(box[0,:]/lengths[0],box[1,:]/lengths[1]))
    angles[1] = numpy.arccos(numpy.dot(box[0,:]/lengths[0],box[2,:]/lengths[2]))
    angles[2] = numpy.arccos(numpy.dot(box[1,:]/lengths[1],box[2,:]/lengths[2]))
    angles *= 180.0/numpy.pi
    return lengths, angles
    
    
def read_con(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    trajectory = []
    line_index = 0
    while True:
        try:
            boxlengths = numpy.array([float(length) for length in lines[line_index+2].split()[0:3]])
            boxangles = numpy.array([float(angle) for angle in lines[line_index+3].split()[0:3]])
            cell = length_angle_to_box(boxlengths, boxangles)
            num_types = int(lines[line_index+6].split()[0])
            num_each_type = [int(n) for n in lines[line_index+7].split()[0:num_types]]
            mass_each_type = [float(n) for n in lines[line_index+8].split()[0:num_types]]
            a = Atoms('H'*sum(num_each_type))
            a.cell = cell
            a.set_pbc((True, True, True))
            frozen = []
            positions = []
            symbols = []
            masses = []
            line_index += 9
            atom_index = 0
            for i in range(num_types):
                symbol = lines[line_index].strip()
                mass = mass_each_type[i]
                line_index += 2
                for j in range(num_each_type[i]):
                    split = lines[line_index].split()
                    positions.append([float(s) for s in split[0:3]])
                    symbols.append(symbol)
                    masses.append(mass)
                    if split[3] != '0':
                        frozen.append(atom_index)
                    atom_index += 1
                    line_index += 1
            a.set_chemical_symbols(symbols)
            a.set_positions(positions)
            a.set_masses(masses)
            a.set_constraint(FixAtoms(frozen))
        except:
            if len(trajectory) == 1:
                return trajectory[0]
            if len(trajectory) == 0:
                raise IOError, "Could not read con file."
            return trajectory
        trajectory.append(a)        
            
            
def write_con(filename, p, w = 'w'):
    con = open(filename, w)
    print >> con, "Generated by tsase"
    print >> con
    lengths, angles = box_to_length_angle(p.cell)
    print >> con, " ".join(['%12.6f' % s for s in lengths])
    print >> con, " ".join(['%12.6f' % s for s in angles])
    print >> con
    print >> con
    atom_count = {}
    name_order = []
    for i in range(len(p)):
        name = p[i].symbol
        if name not in name_order:
            name_order.append(name)
        if name in atom_count:
            atom_count[name] += 1
        else:
            atom_count[name] = 1
    print >> con, len(name_order)
    print >> con, " ".join([str(atom_count[i]) for i in name_order])
    printmasses = []
    index = 0
    for i in range(len(name_order)):
        printmasses.append(p[index].mass)
        index += atom_count[name_order[i]]
    print >> con, " ".join(["%12.6f"% i for i in printmasses])
    index = 0
    for i in range(len(name_order)):
        print >> con, name_order[i]
        print >> con, "Coordinates of Component", i+1
        for j in range(atom_count[name_order[i]]):
            free = 0
            if len(p.constraints) > 0:
                if index in p.constraints[0].index:
                    free = 1
            con.write("%12.6f %12.6f %12.6f %d %d\n" % (p[index].position[0],
                      p[index].position[1], p[index].position[2], free, index))
            index += 1

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
        a.set_pbc((True, True, True))
        j = 0
        for N, e in zip(natoms, elements):
            for k in range(N):
                split = lines[8 + i * (sum(natoms) + 1) + j].split()
                a[j].position = [float(l) for l in split[0:3]]
                j += 1
        a.positions = numpy.dot(a.positions, cell)
        trajectory.append(a)
    return trajectory

def read_vasp_multiframe(filename):
    try:
        xdat = read_xdatcar(filename)
        if type(xdat) == list and len(xdat) > 1:
            return xdat
    except:
        pass
    f = open(filename, 'r')
    data = []
    while True:
        try:
            data.append(ase.io.read(f, format='vasp'))
        except:
            f.close()
            break
    if len(data) < 1:
        raise IOError, "Could not read file %s as vasp file." % filename
    if len(data) < 2:
        return data[0]
    return data

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
        a.set_pbc((True, True, True))
        j = 0
        for N, e in zip(natoms, elements):
            for k in range(N):
                split = lines[8 + i * (sum(natoms) + 1) + j].split()
                a[j].position = [float(l) for l in split[0:3]]
                j += 1
        a.positions = numpy.dot(a.positions, cell)
        trajectory.append(a)
    return trajectory

def write_chir(filename, r, chir, comments=None):
    f = open(filename, 'w') 

    if type(comments) == str:
        f.write('#%s\n' % comments)
    elif type(comments) == list or type(comments) == tuple:
        for comment in comments:
            f.write('#%s\n' % comment)

    for i in xrange(len(r)):
        f.write('%e %e\n' % (r[i], chir[i]))

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

