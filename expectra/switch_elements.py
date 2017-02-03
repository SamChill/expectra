import random
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import NeighborList

def switch_elements(atoms, symbols, cutoff):
    atoms.set_chemical_symbols(symbols)
    elements_lib = ['Rh', 'Au']
    #switch_space = int(self.switch_ratio * len(atoms))
    chemical_symbols = atoms.get_chemical_symbols()

    cutoffs=[]
    for i in range (len(atoms)):
        cutoffs.append(cutoff)
    nl_zero = NeighborList(cutoffs, self_interaction = True, bothways=True)
    nl_zero.update(atoms)

    cutoffs=[]
    for i in range (len(atoms)):
        cutoffs.append(1.5)
    nl_one = NeighborList(cutoffs, self_interaction = False, bothways=True)
    nl_one.update(atoms)


    elements_numb=[]
    for i in range(len(elements_lib)):
        elements_numb.append(0)
    for i in range(len(atoms)):
        for j in range (len(elements_lib)):
            if chemical_symbols[i] == elements_lib[j]:
               elements_numb[j] += 1
    print "Elements_numb before switch:", elements_numb

    spec_index=[]
    elements_numb=[]
    for i in range(len(elements_lib)):
        spec_index.append([])
        elements_numb.append(0)
    for i in xrange(len(atoms)):
        for j in range (len(elements_lib)):
            if chemical_symbols[i] == elements_lib[j]:
               spec_index[j].append(i)
               elements_numb[j] += 1

    #find gold atoms at Au_Rh boundary
    switchable_lib_o=[]
    for index in spec_index[1]:
        #alien_numb = 0
        indices, offsets = nl_one.get_neighbors(index)
        for j in range(len(indices)):
            if chemical_symbols[indices[j]] == elements_lib[0]:
               switchable_lib_o.append(index)
               break
               

    #pick and replace Rh with Au
    numb_switched = 0
    while (numb_switched == 0):
        index_zero=random.sample(spec_index[0], 1)
        indices, offsets=nl_zero.get_neighbors(index_zero[0])
        for j in range(len(indices)):
            if chemical_symbols[indices[j]] == elements_lib[0]:
               chemical_symbols[indices[j]] = elements_lib[1]
               numb_switched += 1 
            if numb_switched == len(switchable_lib_o):
               break 
    #pick and replace Au with Rh
    index_one=random.sample(switchable_lib_o, numb_switched)
    for i in range(numb_switched):
        chemical_symbols[index_one[i]]=elements_lib[0]

    elements_numb=[]
    for i in range(len(elements_lib)):
        elements_numb.append(0)
    for i in range(len(atoms)):
        for j in range (len(elements_lib)):
            if chemical_symbols[i] == elements_lib[j]:
               elements_numb[j] += 1
    print "Elements_numb after switch:", elements_numb
    return chemical_symbols

