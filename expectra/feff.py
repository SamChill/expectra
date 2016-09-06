import tempfile
import os
import subprocess
import shutil
import sys

import numpy

from ase import Atoms, Atom


def feff_edge_number(edge):
    edge_map = {}
    edge_map['k'] = 1
    edge_map['l1'] = edge_map['li'] = 1
    edge_map['l2'] = edge_map['lii'] = 3
    edge_map['l3'] = edge_map['liii'] = 4
    return edge_map[edge.lower()]

def load_chi_dat(filename):
    f = open(filename)
    chi_section = False
    k = []
    chi = []
    for line in f:
        line = line.strip()
        if len(line) == 0:
            continue

        fields = line.split()
        if fields[0] == "k" and fields[1] == "chi" and fields[2] == "mag":
            chi_section = True
            continue

        if chi_section:
            k.append(float(fields[0]))
            chi.append(float(fields[1]))
    return numpy.array(k), numpy.array(chi)

def load_feff_dat(filename):
    xk = []
    cdelta = []
    afeff = []
    phfeff = []
    redfac = []
    xlam = []
    rep = []

    atoms = Atoms()
    atoms.set_pbc((False,False,False))

    path_section = False
    atoms_section = False
    data_section = False
    f = open(filename)
    for line in f:
        line = line.strip()
        fields = line.split()
        if "genfmt" in line:
            path_section = True
            continue
        if fields[0] == "x" and fields[1] == "y" and fields[2] == "z":
            atoms_section = True
            path_section = False
            continue
        if fields[0] == "k" and fields[1] == "real[2*phc]":
            data_section = True
            atoms_section = False
            continue

        if path_section:
            if "---------------" in line:
                continue
            reff = float(fields[2])
            path_section = False

        if atoms_section:
            x = float(fields[0])
            y = float(fields[1])
            z = float(fields[2])
            pot = int(fields[3])
            atomic_number = int(fields[4])
            atoms.append(Atom(symbol=atomic_number, position=(x,y,z),
                         tag=pot))

        if data_section:
            fields = [ float(f) for f in fields ]
            xk.append(fields[0])
            cdelta.append(fields[1])
            afeff.append(fields[2])
            phfeff.append(fields[3])
            redfac.append(fields[4])
            xlam.append(fields[5])
            rep.append(fields[6])

    xk = numpy.array(xk)
    cdelta = numpy.array(cdelta)
    afeff = numpy.array(afeff)
    phfeff = numpy.array(phfeff)
    redfac = numpy.array(redfac)
    xlam = numpy.array(xlam)
    rep = numpy.array(rep)

    return {
             "atoms":atoms,
             "reff":reff,
             "xk":xk,
             "cdelta":cdelta,
             "afeff":afeff,
             "phfeff":phfeff,
             "redfac":redfac,
             "xlam":xlam,
             "rep":rep,
           }

def write_feff(filename, atoms, absorber, feff_options={}):
    f = open(filename, "w")
    f.write("TITLE %s\n" % str(atoms))
    for key, value in feff_options.iteritems():
        f.write("%s %s\n" % (key, value))
    f.write("\nPOTENTIALS\n")
    absorber_z = atoms[absorber].number
    f.write("%i %i\n" % (0, absorber_z))

    unique_z = list(set(atoms.get_atomic_numbers()))
    pot_map = {}
    i = 1
    for z in unique_z:
        nz = len( [ a for a in atoms if a.number == z ] )
        if z == absorber_z and nz-1==0:
            continue
        f.write("%i %i\n" % (i, z))
        pot_map[z] = i
        i+=1

    f.write("\nATOMS\n")
    for i,atom in enumerate(atoms):
        if i == absorber:
            pot = 0
        else:
            pot = pot_map[atom.number]
        f.write("%f %f %f %i\n" % (atom.x, atom.y, atom.z, pot))

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

def absorber_sphere(atoms, absorber, radius):
    box = atoms.get_cell()
    ibox = numpy.linalg.inv(box)
    pos = atoms.get_positions()
    elements = atoms.get_chemical_symbols()
    atoms_sphere = [Atom(elements[absorber], (0.,0.,0.))]
    for i in xrange(len(atoms)):
        if i == absorber: continue
        r = pbc(pos[i] - pos[absorber], box, ibox)
        d = numpy.linalg.norm(r)
        if d <= radius:
            atoms_sphere.append(Atom(elements[i], r))
    return Atoms(atoms_sphere)

def run_feff(atoms, absorber, feff_options={}, tmp_dir=None, get_path=False):
    tmp_dir_path = tempfile.mkdtemp(prefix="tmp_feff_", dir=tmp_dir)
    feff_inp_path = os.path.join(tmp_dir_path, "feff.inp")

    if any(atoms.get_pbc()):
        #pick out a sphere around the absorber atom, important for PBC to work with feff
        #atom index 0 is now the absorber
        atoms = absorber_sphere(atoms, absorber, radius=float(feff_options['RMAX'])+0.01)
        absorber = 0

    write_feff(feff_inp_path, atoms, absorber, feff_options)

    try:
        p = subprocess.Popen(["feff"], cwd=tmp_dir_path,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        from sys import exit, stderr
        stderr.write('unable to locate feff executable in PATH\n')
        exit(1)
    retval = p.wait()

    #function to deal with errors
    def feff_error():
        print 'Problem with feff calculation in %s' % tmp_dir_path
        tmp_f = tempfile.NamedTemporaryFile(dir='.', prefix='feff.inp.')
        print 'feff.inp saved to:', tmp_f.name
        tmp_f.close()
        write_feff(tmp_f.name, atoms, absorber, feff_options)
        shutil.rmtree(tmp_dir_path)

    if retval != 0:
        feff_error()
        return

    stdout, stderr = p.communicate()
    stderr = stderr.strip()
    if stderr == "hash error":
        atoms[absorber].set_position(atoms[absorber].get_position()+0.001)
        sys.stderr.write("%s\n"%stderr)
        return run_feff(atoms, absorber, feff_options, tmp_dir)

    #check to see if we found any paths
    for line in stdout.split('\n'):
        line = line.strip()
        if line.startswith('Paths found'):
            npaths = int(line.split()[2])
            if npaths == 0:
                shutil.rmtree(tmp_dir_path)
                if get_path:
                    return None, None, None
                else:
                    return None, None

    try:
        k, chi = load_chi_dat(os.path.join(tmp_dir_path, "chi.dat"))
    except IOError:
        feff_error()
        raise
    if get_path:
        path = load_feff_dat(os.path.join(tmp_dir_path, "feff0001.dat"))
    shutil.rmtree(tmp_dir_path)

    if get_path:
        return k, chi, path
    else:
        return k, chi
