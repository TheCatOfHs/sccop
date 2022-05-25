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
    
    def get_all_pos(min_pos, mapping):
        """
        get all equivalnent pos and type
        
        Parameters
        ----------
        min_pos [int, 1d]: pos in minimum grid
        mapping [int, 2d]: mapping between min and all grid

        Returns
        ----------
        all_pos [int, 1d]: pos in all grid
        """
        all_pos = np.array(mapping, dtype=object)[min_pos]
        all_pos = np.concatenate(all_pos).tolist()
        return all_pos
    print(get_all_pos(pos, mapping))
    
    #all_pos, all_type = plane.get_all_sites(pos, type, mapping)
    symm_site = plane.group_symm_sites(mapping)
    assign = plane.assign_by_spacegroup({5:2, 6:6}, symm_site)
    
    import json
    with open('test/test.json', 'w') as f:
        json.dump(assign, f)
    
    with open('test/test.json', 'r') as f:
        json_dict = json.load(f)
    
    print(assign)
    print(symm_site)
    
    def get_type_and_symm(assign):
        """
        get list of atom type and symmetry
        
        Parameters
        ----------
        assign [dict, int:list]: site assignment of atom_num

        Returns
        ----------
        type [int, 1d]: type of atoms
        symm [int, 1d]: symmetry of atoms
        """
        type, symm = [], []
        for atom in assign.keys():
            value = assign[atom]
            symm += sorted(value)
            type += [atom for _ in range(len(value))]
        return type, symm

    def get_pos(symm, symm_site):
        """
        sampling position of atoms by symmetry
        
        Parameters
        ----------
        symm [int, 1d]: symmetry of atoms
        symm_site [dict, int:list]: site position grouped by symmetry

        Returns
        ----------
        pos [int, 1d]: position of atoms
        """
        site_copy = copy.deepcopy(symm_site)
        pos = []
        for i in symm:
            pool = site_copy[i]
            sample = np.random.choice(pool)
            pos.append(sample)
            pool.remove(sample)
        return pos
    
    type, symm = get_type_and_symm(assign[0])
    print(type)
    print(get_type_and_symm(assign[0]))
    print(get_pos(symm, symm_site))
    #rwtools = ListRWTools()
    #rwtools.write_list2d(f'test/test.dat', coors)
    
    def sort_by_grid_sg(grid, sg):
        """
        sort pos, type, symm in order of grid and space group
        
        Parameters
        ----------
        grid [int, 1d]: name of grid
        sg [int, 1d]: space group number 

        Returns
        ----------
        idx [int, 1d, np]: index of grid-sg order
        """
        idx = np.arange(len(sg))
        grid_sg = np.stack((idx, grid, sg), axis=1).tolist()
        order = sorted(grid_sg, key=lambda x:(x[1], x[2]))
        idx = np.array(order)[:,0]
        return idx

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
    
    