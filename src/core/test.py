from ctypes import Structure
from fileinput import filename
import os, sys
import time
import random
import numpy as np


from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure, IStructure
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.utils import *


if __name__ == '__main__':
    
    crystal_system = 1
    #monoclinic
    if crystal_system == 1:
        a, b, c = np.random.normal(len_mu, len_sigma, 3)
        alpha, gamma = 90, 90
        beta = random.normalvariate(ang_mu, ang_sigma)
    #orthorhombic
    if crystal_system == 2:
        a, b, c = np.random.normal(len_mu, len_sigma, 3)
        alpha, beta, gamma = 90, 90, 90
    #tetragonal
    if crystal_system == 3:
        a, c = np.random.normal(len_mu, len_sigma, 2)
        b = a
        alpha, beta, gamma = 90, 90, 90
    #hexagonal
    if crystal_system == 5:
        a, c = np.random.normal(len_mu, len_sigma, 2)
        b = a
        alpha, beta, gamma = 90, 90, 120
    
    #space group
    if crystal_system == 1:
        groups = [1, 3]
    if crystal_system == 2:
        groups = [4, 6, 8, 25, 28, 32, 38]
    if crystal_system == 3:
        groups = [75, 99, 100]
    if crystal_system == 5:
        groups = [143, 156, 157, 168, 183]
    
    latt = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    atoms = [1]
    
    min_area, area = [], []
    sg = groups[1]
    for i in np.arange(0, 1, 1/120):
        #inner
        for j in np.arange(0, 1, 1/120):
            cond_1 = i <= .5
            cond_2 = True
            cond_3 = True
            cond_4 = True
            if cond_1 and cond_2 and cond_3 and cond_4:
            #if True:
                stru = IStructure.from_spacegroup(sg, latt, atoms, [[i, j, 0]])
                sites = stru.sites
                min_area.append([i, j])
                for site in sites:
                    coor = site.frac_coords.tolist()
                    area.append(coor[:2]+[len(sites)])
        
    
    rwtools = ListRWTools()
    rwtools.write_list2d(f'test/monoclinic-{sg}.dat', area)
    rwtools.write_list2d(f'test/monoclinic-point-{sg}.dat', min_area)
    
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