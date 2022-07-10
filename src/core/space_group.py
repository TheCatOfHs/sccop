import os, sys
import copy
import numpy as np
from decimal import Decimal
from collections import Counter

from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *


class PlanarSpaceGroup:
    #17 planar space groups
    def __init__(self):
        pass    
    
    def get_grid_points(self, group, grain, latt):
        """
        generate grid point according to space group
        
        Parameters
        ----------
        group [int, 0d]: international number of group number
        grain [float, 1d]: grain of grid points
        latt [obj]: Lattice object of pymatgen

        Returns
        ----------
        all_grid [float, 2d, np]: fraction coordinates of grid points
        mapping [int, 2d]: mapping between min and all grid
        """
        norms = latt.lengths
        n = [norms[i]//grain[i] for i in range(3)]
        frac_grain = [1/i for i in n]
        #monoclinic
        if group == 1:
            min_grid = self.triclinic_1(frac_grain)
        if group == 2:
            min_grid = self.triclinic_2(frac_grain)
        #orthorhombic
        if group == 3:
            min_grid = self.orthorhombic_3(frac_grain)
        if group == 4:
            min_grid = self.orthorhombic_4(frac_grain)
        if group == 5:
            min_grid = self.orthorhombic_5(frac_grain)
        if group == 25:
            min_grid = self.orthorhombic_25(frac_grain)
        if group == 28:
            min_grid = self.orthorhombic_28(frac_grain)
        if group == 32:
            min_grid = self.orthorhombic_32(frac_grain)
        if group == 35:
            min_grid = self.orthorhombic_35(frac_grain)
        #tetragonal
        if group == 75:
            min_grid = self.tetragonal_75(frac_grain)
        if group == 99:
            min_grid = self.tetragonal_99(frac_grain)
        if group == 100:
            min_grid = self.tetragonal_100(frac_grain)
        #hexagonal
        if group == 143:
            min_grid = self.hexagonal_143(frac_grain)
        if group == 156:
            min_grid = self.hexagonal_156(frac_grain)
        if group == 157:
            min_grid = self.hexagonal_157(frac_grain)
        if group == 168:
            min_grid = self.hexagonal_168(frac_grain)
        if group == 183:
            min_grid = self.hexagonal_183(frac_grain)
        #get all equivalent sites and mapping relationship
        mapping = []
        all_grid = min_grid.copy()
        for i, point in enumerate(min_grid):
            stru = Structure.from_spacegroup(group, latt, [1], [point], tol=1e-8)
            equal_coords = self.get_equal_coords(point, stru.frac_coords.tolist())
            start = len(all_grid)
            end = start + len(equal_coords)
            mapping.append([i] + [j for j in range(start, end)])
            all_grid += equal_coords
        return np.array(all_grid), mapping
    
    def get_equal_coords(self, point, coords):
        """
        get equivalent coordinates
        
        Parameters
        ----------
        point [float, 1d]: point in minimum area
        coords [float, 2d]: all symmetry coordinates

        Returns
        ----------
        equal_coords [float, 2d]: equivalent coordinates
        """
        equal_coords = coords.copy()
        for i, coord in enumerate(equal_coords):
            delta = np.array(point) - np.array(coord)
            if np.linalg.norm(delta) < 1e-6:
                del equal_coords[i]
                break
        return equal_coords
    
    def assign_by_spacegroup(self, atom_num, symm_site):
        """
        generate site assignments by space group
        
        Parameters
        ----------
        atom_num [dict, int:int]: number of different atoms\\
        symm_site [dict, int:list]: site position grouped by symmetry

        Returns
        ----------
        assign_plan [list, dict, 1d]: site assignment of atom_num
        e.g. [{5:[1, 1, 2, 2], 6:[6, 12]}]
        """
        symm = list(symm_site.keys())
        site_num = [len(i) for i in symm_site.values()]
        symm_num = dict(zip(symm, site_num))
        #initialize assignment
        store = []
        init_assign = {}
        for atom in atom_num.keys():
            init_assign[atom] = []
        for site in symm_site.keys():
            for atom in atom_num.keys():
                num = atom_num[atom]
                if site <= num:
                    assign = copy.deepcopy(init_assign)
                    assign[atom] = [site]
                    store.append(assign)
        #find site assignment of different atom_num
        new_store, assign_plan = [], []
        while True:
            for assign in store:
                for site in symm_site.keys():
                    for atom in atom_num.keys():
                        new_assign = copy.deepcopy(assign)
                        new_assign[atom] += [site]
                        save = self.check_assign(atom_num, symm_num, new_assign)
                        if save == 0:
                            assign_plan.append(new_assign)
                        if save == 1:
                            new_store.append(new_assign)
            if len(new_store) == 0:
                break
            store = new_store
            store = self.delete_same_assign(store)
            new_store = []
        assign_plan = self.delete_same_assign(assign_plan)
        return assign_plan
    
    def group_symm_sites(self, mapping):
        """
        group sites by symmetry
        
        Parameters
        ----------
        mapping [int, 2d]: mapping between min and all grid

        Returns
        ----------
        symm_site [dict, int:list]: site position in min grouped by symmetry
        """
        symm = [len(i) for i in mapping]
        last = symm[0]
        store, symm_site = [], {}
        for i, s in enumerate(symm):
            if s == last:
                store.append(i)
            else:
                symm_site[last] = store
                store = []
                last = s
                store.append(i)
        symm_site[s] = store
        return symm_site
    
    def check_assign(self, atom_num, symm_num_dict, assign):
        """
        check site assignment of different atom_num
        
        Parameters
        ----------
        atom_num [dict, int:int]: number of different atoms\\
        symm_num [dict, int:int]: number of each symmetry site\\
        assign [dict, int:list]: site assignment of atom_num

        Returns
        ----------
        save [int, 0d]: 0, right. 1, keep. 2, delete.
        """
        assign_num, site_used = {}, []
        #check number of atom_num
        save = 1
        for atom in atom_num.keys():
            site = assign[atom]
            num = sum(site)
            if num <= atom_num[atom]:
                assign_num[atom] = num
                site_used += site
            else:
                save = 2
                break
        if save == 1:
            #check number of used sites
            site_used = Counter(site_used)
            for site in site_used.keys():
                if site_used[site] > symm_num_dict[site]:
                    save = 2
                    break
            #whether find a right assignment
            if assign_num == atom_num and save == 1:
                save = 0
        return save
    
    def delete_same_assign(self, store):
        """
        delete same site assignment 
        
        Parameters
        ----------
        store [list, dict, 1d]: assignment of site

        Returns
        ----------
        new_store [list, dict, 1d]: unique assignment of site
        """
        idx = []
        num = len(store)
        for i in range(num):
            assign_1 = store[i]
            for j in range(i+1, num):
                same = True
                assign_2 = store[j]
                for k in assign_1.keys():
                    if sorted(assign_1[k]) != sorted(assign_2[k]):
                        same = False
                        break
                if same:
                    idx.append(i)
                    break
        new_store = np.delete(store, idx, axis=0).tolist()
        return new_store
    
    def judge_crystal_system(self, latt):
        """
        judge crystal system by lattice constants

        Parameters
        ----------
        latt [obj]: lattice object in pymatgen 

        Returns
        ----------
        system [int, 0d]: crystal system number
        params [float, 1d]: approximal lattice constants
        """
        approx = [self.convert_digit(i) for i in latt.parameters]
        a, b, c, alpha, beta, gamma = approx
        system = 0
        if a != b and gamma != 90:
            system = 0
        if a != b and gamma == 90:
            system = 2
        if a == b and gamma == 90:
            system = 3
        if a == b and gamma == 120:
            system = 5
        if add_vacuum:
            c = vacuum_space
            alpha, beta = 90, 90
        params = [a, b, c, alpha, beta, gamma]
        return system, params
    
    def convert_digit(self, num):
        """
        keep 2 digits after decimal point
        
        Parameters
        ----------
        num [flat, 0d]: float number
        """
        return float(Decimal(num).quantize(Decimal('0.01'),
                                           rounding = 'ROUND_HALF_UP'))
    
    def triclinic_1(self, frac_grain):
        """
        space group P1
        """
        equal_1 = []
        #inner
        for i in np.arange(0, 1, frac_grain[0]):
            for j in np.arange(0, 1, frac_grain[1]):
                equal_1.append([i, j, 0])
        min_grid = equal_1
        return min_grid
    
    def triclinic_2(self, frac_grain):
        """
        space group P211
        """
        equal_2 = []
        #point
        equal_1 = [[0, 0, 0], [0, .5, 0], [.5, 0, 0], [.5, .5, 0]]
        #boundary
        for i in np.arange(0, .5, frac_grain[0]):
            if 0 < i:
                equal_2.append([i, 0, 0])
                equal_2.append([i, .5, 0])
        for j in np.arange(0, .5, frac_grain[1]):
            if 0 < j:
                equal_2.append([0, j, 0])
        #inner
        for i in np.arange(0, 1, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 0 < i and 0 < j:
                    equal_2.append([i, j, 0])
        min_grid = equal_1 + equal_2
        return min_grid
    
    def orthorhombic_3(self, frac_grain):
        """
        space group P1m1
        """
        equal_1, equal_2 = [], []
        #boundary
        for j in np.arange(0, 1, frac_grain[1]):
            equal_1.append([0, j, 0])
            equal_1.append([.5, j, 0])
        #inner
        for i in np.arange(0, .5, frac_grain[0]):
            for j in np.arange(0, 1, frac_grain[1]):
                if 0 < i:
                    equal_2.append([i, j, 0])
        min_grid = equal_1 + equal_2
        return min_grid
    
    def orthorhombic_4(self, frac_grain):
        """
        space group P1g1
        """
        equal_2 = []
        #inner
        for i in np.arange(0, 1, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                equal_2.append([i, j, 0])
        min_grid = equal_2
        return min_grid
    
    def orthorhombic_5(self, frac_grain):
        """
        space group C1m1
        """
        equal_2, equal_4 = [], []
        #boundary
        for j in np.arange(0, .5, frac_grain[1]):
            equal_2.append([0, j, 0])
            equal_2.append([.5, j, 0])
        #inner
        for i in np.arange(0, .5, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 0 < i:
                    equal_4.append([i, j, 0])
        min_grid = equal_2 + equal_4
        return min_grid
    
    def orthorhombic_25(self, frac_grain):
        """
        space group P2mm
        """
        equal_2, equal_4 = [], []
        #point
        equal_1 = [[0, 0, 0], [0, .5, 0], [.5, 0, 0], [.5, .5, 0]]
        #boundary
        for i in np.arange(0, .5, frac_grain[0]):
            if 0 < i:
                equal_2.append([i, 0, 0])
                equal_2.append([i, .5, 0])
        for j in np.arange(0, .5, frac_grain[1]):
            if 0 < j:
                equal_2.append([0, j, 0])
                equal_2.append([.5, j, 0])
        #inner
        for i in np.arange(0, .5, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 0 < i and 0 < j:
                    equal_4.append([i, j, 0])
        min_grid = equal_1 + equal_2 + equal_4
        return min_grid
    
    def orthorhombic_28(self, frac_grain):
        """
        space group P2mg
        """
        equal_4 = []
        #point
        equal_2 = [[0, 0, 0], [0, .5, 0]]
        #boundary
        for j in np.arange(0, 1, frac_grain[1]):
            equal_2.append([.25, j, 0])
        for i in np.arange(0, .25, frac_grain[0]):
            if 0 < i:
                equal_4.append([i, 0, 0])
        for j in np.arange(0, .5, frac_grain[1]):
            if 0 < j:
                equal_4.append([0, j, 0])
        #inner
        for i in np.arange(0, .25, frac_grain[0]):
            for j in np.arange(0, 1, frac_grain[1]):
                if 0 < i and 0 < j:
                    equal_4.append([i, j, 0])
        min_grid = equal_2 + equal_4
        return min_grid
    
    def orthorhombic_32(self, frac_grain):
        """
        space group P2gg
        """
        equal_4 = []
        #point
        equal_2 = [[0, 0, 0], [0, .5, 0]]
        #boundary
        for i in np.arange(0, .5, frac_grain[0]):
            if 0 < i:
                equal_4.append([i, 0, 0])
        for j in np.arange(0, .5, frac_grain[1]):
            if 0 < j:
                equal_4.append([0, j, 0])
        #inner
        for i in np.arange(0, .5, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 0 < i and 0 < j:
                    equal_4.append([i, j, 0])
        min_grid = equal_2 + equal_4
        return min_grid
    
    def orthorhombic_35(self, frac_grain):
        """
        space group C2mm
        """
        equal_8 = []
        #point
        equal_2 = [[0, 0, 0], [0, .5, 0]]
        equal_4 = [[.25, .25, 0], [.25, 0., 0]]
        #boundary
        for i in np.arange(0, .25, frac_grain[0]):
            if 0 < i:
                equal_4.append([i, 0, 0])
                equal_4.append([i, .5, 0])
        for j in np.arange(0, .5, frac_grain[1]):
            if 0 < j:
                equal_4.append([0, j, 0])
            if 0 < j < .25:
                equal_8.append([.25, j, 0])
        #inner
        for i in np.arange(0, .25, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 0 < i and 0 < j:
                    equal_8.append([i, j, 0])
        min_grid = equal_2 + equal_4 + equal_8
        return min_grid
    
    def tetragonal_75(self, frac_grain):
        """
        space group P4
        """
        equal_4 = []
        #point
        equal_1 = [[0, 0, 0], [.5, .5, 0]]
        equal_2 = [[0, .5, 0]]
        #boundary
        for i in np.arange(0, .5, frac_grain[0]):
            if 0 < i:
                equal_4.append([i, .5, 0])
        for j in np.arange(0, .5, frac_grain[1]):
            if 0 < j:
                equal_4.append([0, j, 0])
        #inner
        for i in np.arange(0, .5, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 0 < i and 0 < j:
                    equal_4.append([i, j, 0])
        min_grid = equal_1 + equal_2 + equal_4
        return min_grid
    
    def tetragonal_99(self, frac_grain):
        """
        space group P4mm
        """
        equal_4, equal_8 = [], []
        #point
        equal_1 = [[0, 0, 0], [.5, .5, 0]]
        equal_2 = [[0, .5, 0]]
        #boundary
        for i in np.arange(0, .5, frac_grain[0]):
            if 0 < i:
                equal_4.append([i, 0, 0])
                equal_4.append([i, i, 0])
        for j in np.arange(0, .5, frac_grain[1]):
            if 0 < j:
                equal_4.append([.5, j, 0])
        #inner
        for i in np.arange(0, .5, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 0 < i and 0 < j < i:
                    equal_4.append([i, j, 0])
        min_grid = equal_1 + equal_2 + equal_4 + equal_8
        return min_grid
    
    def tetragonal_100(self, frac_grain):
        """
        space group P4gm
        """
        equal_4, equal_8 = [], []
        #point
        equal_2 = [[0, 0, 0], [0, .5, 0]]
        #boundary
        for i in np.arange(0, .5, frac_grain[0]):
            if 0 < i:
                equal_4.append([i, -i+.5, 0])
                equal_8.append([i, 0, 0])
        #inner
        for i in np.arange(0, .5, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 0 < i and 0 < j < -i+.5:
                    equal_8.append([i, j, 0])
        min_grid = equal_2 + equal_4 + equal_8
        return min_grid
    
    def hexagonal_143(self, frac_grain):
        """
        space group P3
        """
        equal_3 = []
        #point
        equal_1 = [[0, 0, 0], [1/3, 2/3, 0], [2/3, 1/3, 0]]
        #boundary
        for i in np.arange(0, 2/3, frac_grain[0]):
            if 0 < i:
                equal_3.append([i, .5*i, 0])
        for j in np.arange(0, 2/3, frac_grain[1]):
            if 0 < j:
                equal_3.append([.5*j, j, 0])
        #inner
        for i in np.arange(0, 1, frac_grain[0]):
            for j in np.arange(0, 1, frac_grain[1]):
                if .5*i < j and 2*i-1 < j and j < 2*i and j < .5*i+.5:
                    equal_3.append([i, j, 0])
        min_grid = equal_1 + equal_3
        return min_grid

    def hexagonal_156(self, frac_grain):
        """
        space group P3m1
        """
        equal_3, equal_6 = [], []
        #point
        equal_1 = [[0, 0, 0], [1/3, 2/3, 0], [2/3, 1/3, 0]]
        #boundary
        for i in np.arange(0, 2/3, frac_grain[0]):
            if 0 < i:
                equal_3.append([i, .5*i, 0])
            if 1/3 < i:
                equal_3.append([i, -i+1, 0])
        for j in np.arange(0, 2/3, frac_grain[1]):
            if 0 < j:
                equal_3.append([.5*j, j, 0])
        #inner
        for i in np.arange(0, 2/3, frac_grain[0]):
            for j in np.arange(0, 2/3, frac_grain[1]):
                if .5*i < j and 0 < j < 2*i and j < -i+1:
                    equal_6.append([i, j, 0])
        min_grid = equal_1 + equal_3 + equal_6 
        return min_grid
    
    def hexagonal_157(self, frac_grain):
        """
        space group P31m
        """
        equal_6 = []
        #point
        equal_1 = [[0, 0, 0]]
        equal_2 = [[1/3, 2/3, 0]]
        equal_3 = [[.5, .5, 0]]
        #boundary
        for i in np.arange(0, .5, frac_grain[0]):
            if 0 < i:
                equal_3.append([i, i, 0])
                equal_3.append([i, 0, 0])
        for i in np.arange(.5, 2/3, frac_grain[0]):
            if .5 < i:
                equal_6.append([i, 2*i-1, 0])
        #inner
        for i in np.arange(0, 2/3, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 2*i-1 < j and 0 < j < i and j < -i+1:
                    equal_6.append([i, j, 0])
        min_grid = equal_1 + equal_2 + equal_3 + equal_6
        return min_grid
    
    def hexagonal_168(self, frac_grain):
        """
        space group P6
        """
        equal_6 = []
        #point
        equal_1 = [[0, 0, 0]]
        equal_2 = [[1/3, 2/3, 0]]
        equal_3 = [[.5, .5, 0]]
        #boundary
        for i in np.arange(0, .5, frac_grain[0]):
            if 0 < i:
                equal_6.append([i, 0, 0])
        for i in np.arange(.5, 2/3, frac_grain[0]):
            if .5 < i:
                equal_6.append([i, 2*i-1, 0])
        #inner
        for i in np.arange(0, 2/3, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 2*i-1 < j and 0 < j < i and j < -i+1:
                    equal_6.append([i, j, 0])
        min_grid = equal_1 + equal_2 + equal_3 + equal_6
        return min_grid
    
    def hexagonal_183(self, frac_grain):
        """
        space group P6mm
        """
        equal_6, equal_12 = [], []
        #point
        equal_1 = [[0, 0, 0]]
        equal_2 = [[1/3, 2/3, 0]]
        equal_3 = [[.5, .5, 0]]
        #boundary
        for i in np.arange(0, 2/3, frac_grain[0]):
            if 0 < i:
                equal_6.append([i, .5*i, 0])
            if 0 < i < .5:
                equal_6.append([i, 0, 0])
        for j in np.arange(0, 1/3, frac_grain[1]):
            if 0 < j:
                equal_6.append([.5*(j+1), j, 0])
        #inner
        for i in np.arange(0, 2/3, frac_grain[0]):
            for j in np.arange(0, 1/3, frac_grain[1]):
                if 2*i-1 < j and 0 < j < .5*i:
                    equal_12.append([i, j, 0])
        min_grid = equal_1 + equal_2 + equal_3 + equal_6 + equal_12
        return min_grid


def get_space_group(num, system):
    """
    get space group number according to crystal system
        
    Parameters
    ----------
    num [int, 0d]: number of space groups
    system [int, 0d]: crystal system

    Returns
    ----------
    space_group [int, 1d, np]: international number of space group
    """
    #space group of 2-dimensional structure
    if num_dim == 2:
        if system == 0:
            groups = [1, 2]
        if system == 2:
            groups = [3, 4, 5, 25, 28, 32, 35]
        if system == 3:
            groups = [75, 99, 100]
        if system == 5:
            groups = [143, 156, 157, 168, 183]
    #space group of 3-dimensional structure
    if num_dim == 3:
        if system == 0:
            groups = np.arange(1, 3)
        if system == 1:
            groups = np.arange(3, 16)
        if system == 2:
            groups = np.arange(16, 75)
        if system == 3:
            groups = np.arange(75, 143)
        if system == 4:
            groups = np.arange(143, 168)
        if system == 5:
            groups = np.arange(168, 195)
        if system == 6:
            groups = np.arange(195, 231)
    space_group = np.sort(np.random.choice(groups, num*len(groups)))
    return space_group


if __name__ == '__main__':
    psg = PlanarSpaceGroup()
    stru = Structure.from_file('data/poscar/initial_strus_0/POSCAR-RCSD-045', sort=True)
    latt = stru.lattice
    all_grid, _ = psg.get_grid_points(3, grain, latt)
    print(all_grid)
    print(len(all_grid))
    
    from core.utils import ListRWTools
    rw = ListRWTools()
    coords = rw.import_list2d(f'data/grid/045_frac_coords_3.bin', float, binary=True)
    #print(coords)
    #print(len(coords))