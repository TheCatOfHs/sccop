import os, sys
import time
import random
import copy
import numpy as np
from collections import Counter


from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure, IStructure
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter


sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.utils import *
from core.grid_divide import PlanarSpaceGroup


if __name__ == '__main__':
    strus = [1,1,2,3,4,5,6,7,1,2,3,4,9,0]
    strus_num = len(strus)
    strus_idx = np.arange(strus_num)
    idx, store, delet = [], [], []
    while True:
        i = strus_idx[0]
        stru_1 = strus[i]
        for k in range(1, len(strus_idx)):
            j = strus_idx[k]
            stru_2 = strus[j]
            same = stru_1==stru_2
            if same:
                store.append(j)
                delet.append(k)
        #update
        idx += store
        strus_idx = np.delete(strus_idx, [0]+delet)
        store, delet = [], []
        if len(strus_idx) == 0:
            break
    
    all_idx = np.arange(strus_num)
    idx = np.setdiff1d(all_idx, idx)
    print(np.array(strus)[idx])
    
    
    
    
    
    
    '''
    crystal_system = 5
    
    #space group
    if crystal_system == 0:
        groups = [1, 2]
    if crystal_system == 2:
        groups = [3, 4, 5, 25, 28, 32, 35]
    if crystal_system == 3:
        groups = [75, 99, 100]
    if crystal_system == 5:
        groups = [143, 156, 157, 168, 183]
    
    a, b, c = 5, 5, 10
    alpha, beta, gamma = 90, 90, 120
    latt = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    atoms = [1]
    
    plane = PlanarSpaceGroup()
    sg = groups[1]
    all_grid, mapping = plane.get_grid_points(sg, grain, latt)
    
    pos = [i for i in range(4)]
    type = [i for i in range(4)]
    symm = [1, 2, 3, 6]
    '''
    
    
    '''
    min_grid, area = [], []
    sg = groups[3]
    for i in np.arange(0, 1, 1/120):
        #inner
        for j in np.arange(0, 1, 1/120):
            cond_1 = j <= i
            cond_2 = 2*i-1 <= j
            cond_3 = j <= -i+1
            cond_4 = True
            if cond_1 and cond_2 and cond_3 and cond_4:
            #if True:
                stru = Structure.from_spacegroup(sg, latt, atoms, [[i, j, 0]])
                sites = stru.sites
                min_grid.append([i, j])
                for site in sites:
                    coor = site.frac_coords.tolist()
                    area.append(coor[:2]+[len(sites)])
        
    
    rwtools = ListRWTools()
    rwtools.write_list2d(f'test/hexagonal-{sg}.dat', area)
    rwtools.write_list2d(f'test/hexagonal-point-{sg}.dat', min_grid)
    '''
    
    '''
    s = []
    for i in ['P1', 'P211', 'P1m1', 'P12_11', 'C1m1', 'P2mm', 'Pma2', 'Pba2', 'C2mm', 'P4', 'P4mm', 'P4bm', 'P3', 'P3m1', 'P31m', 'P6', 'P6mm']:
        sg = SpaceGroup(i)
        s.append(f'{i}:{sg.int_number}')
        print(f'{i}:{sg.int_number}')
    print(s)
    print(len(s))
    
    CifWriter(stru, symprec=.1).write_file('test/test.cif')
    print(stru.formula)
    print(np.arange(1,3))
    '''
    
    