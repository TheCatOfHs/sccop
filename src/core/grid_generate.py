import os, sys
import copy
import numpy as np

from collections import Counter
from pymatgen.core.periodic_table import Element
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.util.coord import pbc_diff

sys.path.append(f'{os.getcwd()}/src')
from core.path import *
from core.input import *
from core.utils import ListRWTools
from core.space_group import PlaneSpaceGroup, BulkSpaceGroup

    
class GridGenerate(ListRWTools):
    #Build the grid
    def __init__(self):
        pass
    
    def crystal_system_sampling(self, number):
        """
        get crystal system according to space group
        
        Parameters
        ----------
        number [int, 0d]: number of grid
        
        Returns
        ----------
        crystal_system [int, 1d, np]: crystal system list
        """
        #list of target space groups
        sg_buffer = []
        for item in space_group:
            item_len = len(item)
            if item_len == 1:
                sg_buffer += item
            elif item_len == 2:
                sg = [i for i in range(item[0], item[1]+1)]
                sg_buffer += sg
        #find corresponding space group
        cs_buffer = self.crystal_system_classify(sg_buffer)
        cs_sg_dict = self.space_group_divide(cs_buffer, sg_buffer)
        cs_list = list(cs_sg_dict.keys())
        crystal_system = np.random.choice(cs_list, number)
        return sorted(crystal_system)
    
    def crystal_system_classify(self, space_group):
        """
        classify crystal system by space group

        Parameters
        ----------
        space_group [int, 1d]: space group number
        
        Returns
        ----------
        crystal_system [int, 1d]: crystal system list
        """
        space_group = np.unique(space_group)
        crystal_system = []
        if dimension == 2:
            for sg in space_group:
                if sg in range(1, 3):
                    crystal_system.append(1)
                elif sg in range(3, 10):
                    crystal_system.append(3)
                elif sg in range(10, 13):
                    crystal_system.append(4)
                elif sg in range(13, 18):
                    crystal_system.append(6)
        elif dimension == 3:
            for sg in space_group:
                if sg in range(1, 3):
                    crystal_system.append(1)
                elif sg in range(3, 16):
                    crystal_system.append(2)
                elif sg in range(16, 75):
                    crystal_system.append(3)
                elif sg in range(75, 143):
                    crystal_system.append(4)
                elif sg in range(143, 168):
                    crystal_system.append(5)
                elif sg in range(168, 195):
                    crystal_system.append(6)
                elif sg in range(195, 231):
                    crystal_system.append(7)
        return crystal_system
    
    def plane_space_group_convert(self, space_group):
        """
        convert plane space group to corresponding 3d space group
        
        Parameters
        ----------
        space_group [int, 1d]: space group 2d

        Returns
        ----------
        new_space_group [int, 1d]: corresponding 3d space group
        """
        mapping = [1, 2, 3, 4, 5, 25, 28, 32, 35, 75, 99, 100, 143, 156, 157, 168, 183]
        new_space_group = []
        for sg in space_group:
            new_space_group.append(mapping[sg-1])
        return new_space_group
    
    def space_group_divide(self, crystal_system, space_group):
        """
        divide space group into different crystal system
        
        Parameters
        ----------
        crystal_system [int, 1d]: crystal system list
        space_group [int, 1d]: space group list

        Returns
        ----------
        cs_sg_dict [dict, int:list]: crystal system and its space group
        """
        #convert 2d space group
        if dimension == 2:
            space_group = self.plane_space_group_convert(space_group)
        #space group sort by crystal system
        order = np.argsort(crystal_system)
        crystal_system = np.array(crystal_system)[order].tolist()
        space_group = np.array(space_group)[order].tolist()
        #divide space group by crystal system
        last = crystal_system[0]
        store, cs_sg_dict = [], {}
        for i, cs in enumerate(crystal_system):
            if cs == last:
                store.append(space_group[i])
            else:
                cs_sg_dict[last] = sorted(store)
                store = []
                last = cs
                store.append(space_group[i])
        cs_sg_dict[cs] = sorted(store)
        #export dict of crystal system and space group
        self.write_dict(f'{grid_path}/space_group_sampling.json', cs_sg_dict)
        return cs_sg_dict
    
    def get_space_group(self, num, system):
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
        sg_buffer = self.import_dict(f'{grid_path}/space_group_sampling.json')[system]
        space_group = np.random.choice(sg_buffer, num)
        return space_group
    
    def get_grid_points_2d(self, sg, grain, latt, atom_num):
        """
        generate grid point according to 2d space group
        
        Parameters
        ----------
        sg [int, 0d]: international number of space group
        grain [float, 1d]: grain of grid points
        latt [obj]: lattice object of pymatgen
        atom_num [list, 1d]: number of different atoms

        Returns
        ----------
        all_grid [float, 2d, np]: fraction coordinates of grid points
        mapping [int, 2d]: mapping between min and all grid
        """
        plane_funcs = vars(PlaneSpaceGroup)
        if sg in [1, 2]:
            grid_func = plane_funcs[f'triclinic_{sg:03g}']
        elif sg in [3, 4, 5, 25, 28, 32, 35]:
            grid_func = plane_funcs[f'orthorhombic_{sg:03g}']
        elif sg in [75, 99, 100]:
            grid_func = plane_funcs[f'tetragonal_{sg:03g}']
        elif sg in [143, 156, 157, 168, 183]:
            grid_func = plane_funcs[f'hexagonal_{sg:03g}']
        #get dau grid
        norms = latt.lengths
        n = [norms[i]//grain[i] for i in range(3)]
        frac_grain = [1/i for i in n]
        dau_grid = grid_func(frac_grain)
        dau_grid = self.dau_grid_sampling_2d(atom_num, sg, dau_grid)
        #get all equivalent sites and mapping relationship
        mapping = []
        all_grid = dau_grid.copy()
        if len(dau_grid) > 0:
            spg = SpaceGroup.from_int_number(sg)
            for i, point in enumerate(dau_grid):
                coords = spg.get_orbit(point)
                equal_coords = self.get_equal_coords(point, coords)
                start = len(all_grid)
                end = start + len(equal_coords)
                mapping.append([i] + [j for j in range(start, end)])
                all_grid += equal_coords
        return np.array(all_grid), mapping
    
    def get_grid_points_3d(self, sg, grain, latt, atom_num):
        """
        generate grid point according to 3d space group
        
        Parameters
        ----------
        sg [int, 0d]: international number of space group
        grain [float, 1d]: grain of grid points
        latt [obj]: lattice object of pymatgen
        atom_num [list, 1d]: number of different atoms

        Returns
        ----------
        all_grid [float, 2d, np]: fraction coordinates of grid points
        mapping [int, 2d]: mapping between min and all grid
        """
        bulk_funcs = vars(BulkSpaceGroup)
        if sg in range(1, 3):
            grid_func = bulk_funcs[f'triclinic_{sg:03g}']
        elif sg in range(3, 16):
            grid_func = bulk_funcs[f'monoclinic_{sg:03g}']
        elif sg in range(16, 75):
            grid_func = bulk_funcs[f'orthorhombic_{sg:03g}']
        elif sg in range(75, 143):
            grid_func = bulk_funcs[f'tetragonal_{sg:03g}']
        elif sg in range(143, 168):
            grid_func = bulk_funcs[f'trigonal_{sg:03g}']
        elif sg in range(168, 195):
            grid_func = bulk_funcs[f'hexagonal_{sg:03g}']
        elif sg in range(195, 231):
            grid_func = bulk_funcs[f'cubic_{sg:03g}']
        #get dau grid
        dau_grid = grid_func()
        dau_grid = self.dau_grid_sampling_3d(grain, atom_num, sg, dau_grid, latt)
        #get all equivalent sites and mapping relationship
        mapping = []
        all_grid = dau_grid.copy()
        if len(dau_grid) > 0:
            spg = SpaceGroup.from_int_number(sg)
            for i, point in enumerate(dau_grid):
                coords = spg.get_orbit(point)
                equal_coords = self.get_equal_coords(point, coords)
                start = len(all_grid)
                end = start + len(equal_coords)
                mapping.append([i] + [j for j in range(start, end)])
                all_grid += equal_coords
        return np.array(all_grid), mapping
    
    def dau_grid_sampling_2d(self, atom_num, sg, grid):
        """
        sampling on min grid to sparse dau grid

        Parameters
        ----------
        atom_num [int, 1d]: number of different atoms
        sg [int, 0d]: international number of space group
        grid [float, 2d, np]: dense min grid includes all symmetry sites

        Returns
        ----------
        sparse_grid [float, 2d]: coordinates of symmetry sites
        """
        #group by symmetry
        spg = SpaceGroup.from_int_number(sg)
        symm = [len(spg.get_orbit(i)) for i in grid]
        index = np.arange(0, len(grid))
        order = np.argsort(symm)
        symm = np.array(symm)[order]
        index = index[order]
        symm_site = self.group_by_symm(index, symm)
        #multiplicity upper boundary
        mul_upper = max(atom_num)
        del_mul = [i for i in symm_site.keys() if i > mul_upper]
        for mul in del_mul:
            symm_site.pop(mul)
        if len(symm_site.keys()) > 0:
            sample_coords = []
            for mul, idx in symm_site.items():
                sample_coords += grid[idx].tolist()
            dau_grid = sample_coords
        else:
            dau_grid = []
        return dau_grid
    
    def dau_grid_sampling_3d(self, grain, atom_num, sg, grid, latt):
        """
        sampling on min grid to sparse dau grid

        Parameters
        ----------
        grain [float, 1d]: grain of grid points
        atom_num [int, 1d]: number of different atoms
        sg [int, 0d]: international number of space group
        grid [float, 2d, np]: dense min grid includes all symmetry sites
        latt [obj]: lattice object in pymatgen
        
        Returns
        ----------
        sparse_grid [float, 2d]: coordinates of symmetry sites
        """
        #group by symmetry
        spg = SpaceGroup.from_int_number(sg)
        symm = [len(spg.get_orbit(i)) for i in grid]
        index = np.arange(0, len(grid))
        order = np.argsort(symm)
        symm = np.array(symm)[order]
        index = index[order]
        symm_site = self.group_by_symm(index, symm)
        #multiplicity upper boundary
        mul_upper = max(atom_num)
        del_mul = [i for i in symm_site.keys() if i > mul_upper]
        for mul in del_mul:
            symm_site.pop(mul)
        #down sampling
        norms = latt.lengths
        total_n = int(np.prod([norms[i]//grain[i] for i in range(3)]))
        total_n = min(total_n, 20*sum(atom_num))
        if len(symm_site.keys()) > 0:
            un_best, sample_coords = 0, []
            for _ in range(100):
                #sampling on different symmetry
                sampling_num = self.get_symm_sampling_num(total_n, symm_site)
                for mul, idx in symm_site.items():
                    sample_idx = np.random.choice(idx, sampling_num[mul], replace=False)
                    sample_coords += grid[sample_idx].tolist()
                #calculate uniformity
                sparse_grid = np.vstack([spg.get_orbit(i) for i in sample_coords])
                hist, _ = np.histogramdd(sparse_grid, bins=[20, 20, 20])
                un = np.count_nonzero(hist)
                #choose highest uniformity
                if un > un_best:
                    dau_grid = sample_coords
                    un_best = un
                sample_coords = []
        else:
            dau_grid = []
        return dau_grid
    
    def group_by_symm(self, index, symm):
        """
        group sites by symmetry

        Parameters
        ----------
        index [int, 1d]: index of sites
        symm [int, 1d]: symmetry of sites

        Returns
        ----------
        symm_site [dict, int]: index of symmetry sites
        """
        last = symm[0]
        store, symm_site = [], {}
        for i, mul in enumerate(symm):
            if mul == last:
                store.append(index[i])
            else:
                symm_site[last] = store
                store = []
                last = mul
                store.append(index[i])
        symm_site[mul] = store
        return symm_site
    
    def get_symm_sampling_num(self, total_n, symm_site):
        """
        get sampling number for different symmetry
        
        Parameters
        ----------
        total_n [int, 0d]: total number of sparse grid
        symm_site [dict, int:list]: multiplicity and its index

        Returns
        ----------
        symm_num [dict, int:int]: multiplicity and its sampling number
        """
        total = total_n
        sampling_num = {}
        mul_max = list(symm_site.keys())[-1]
        for mul, idx in symm_site.items():
            if mul < mul_max:
                upper = total//mul
                if upper > 1:
                    num = np.random.randint(1, upper)
                else:
                    num = 1
                num = min(num, len(idx))
                sampling_num[mul] = num
                total -= mul*num
            else:
                left = min(total//mul, len(idx))
                if left > 0:
                    sampling_num[mul] = left
                else:
                    sampling_num[mul] = 1
        return sampling_num
    
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
            vec = pbc_diff(point, coord)
            if np.linalg.norm(vec) < 1e-5:
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
    
    def build(self, grid, cutoff):
        """
        save nbr_idx, nbr_dis of grid
        
        Parameters
        ----------
        grid [int, 0d]: name of grid
        cutoff [float, 0d]: cutoff distance
        """
        head = f'{grid_path}/{grid:03.0f}'
        latt_vec = self.import_list2d(f'{head}_latt_vec.bin',
                                      float, binary=True)
        #get near neighbors and distance
        space_group = self.grid_space_group(grid)
        for sg in space_group:
            coords = self.import_list2d(f'{head}_frac_coords_{sg}.bin',
                                        float, binary=True)
            mapping = self.import_list2d(f'{head}_mapping_{sg}.bin',
                                        int, binary=True)
            atoms = [1 for _ in range(len(coords))]
            stru = Structure.from_spacegroup(1, latt_vec, atoms, coords)
            #export neighbor index and distance in min area
            nbr_idx, nbr_dis = self.near_property(stru, cutoff, mapping)
            self.write_list2d(f'{head}_nbr_idx_{sg}.bin', 
                              nbr_idx, binary=True)
            self.write_list2d(f'{head}_nbr_dis_{sg}.bin', 
                              nbr_dis, binary=True)
    
    def grid_space_group(self, grid):
        """
        get space groups of grid

        Parameters
        ----------
        grid [int, 0d]: name of grid
        
        Returns
        ---------
        space_group [str, 1d]: space group number
        """
        grid_file = os.listdir(grid_path)
        coords_file = [i for i in grid_file 
                       if i.startswith(f'{grid:03.0f}_frac')]
        name = [i.split('.')[0] for i in coords_file]
        space_group = [i.split('_')[-1] for i in name]
        return space_group
    
    def near_property(self, stru, cutoff, mapping):
        """
        index and distance of near grid points
        
        Parameters
        ----------
        stru [obj]: structure object in pymatgen
        cutoff [float, 0d]: cutoff distance
        mapping [int, 2d]: mapping between min and all grid
        
        Returns
        ----------
        nbr_idx [int, 2d, np]: index of near neighbor in min
        nbr_dis [float, 2d, np]: distance of near neighbor in min
        """
        #neighbor index and distance of DAU
        min_atom_num = len(mapping)
        all_nbrs = stru.get_all_neighbors(cutoff, sites=stru.sites[:min_atom_num])
        all_nbrs = [sorted(nbrs, key = lambda x: x[1]) for nbrs in all_nbrs]
        #constrain number of neighbors
        num_near = min(map(lambda x: len(x), all_nbrs))
        nbr_idx, nbr_dis = [], []
        for nbr in all_nbrs:
            nbr_dis.append(list(map(lambda x: x[1], nbr[:num_near])))
            nbr_idx.append(list(map(lambda x: x[2], nbr[:num_near])))
        nbr_idx, nbr_dis = self.reduce_to_dau(nbr_idx, nbr_dis, mapping)
        return nbr_idx, nbr_dis
    
    def reduce_to_dau(self, nbr_idx, nbr_dis, mapping):
        """
        reduce neighbors to DAU
        
        Parameters
        ----------
        nbr_idx [int, 2d]: index of near neighbor 
        nbr_dis [float, 2d]: distance of near neighbor 
        mapping [int, 2d]: mapping between DAU and all grid

        Returns
        ----------
        nbr_idx [int, 2d, np]: index of near neighbor in DAU
        nbr_dis [float, 2d, np]: distance of near neighbor in DAU
        """
        nbr_idx = np.array(nbr_idx)
        nbr_dis = np.array(nbr_dis)
        #reduce to min area
        for line in mapping:
            if len(line) > 1:
                dau_atom = line[0]
                for atom in line[1:]:
                    nbr_idx[nbr_idx==atom] = dau_atom
        return nbr_idx, nbr_dis
    
    def lattice_generate_2d(self, crystal_system, params):
        """
        generate lattice by 2d crystal system
        
        Parameters
        ----------
        crystal_system [int, 0d]: crystal system number
        params [float, tuple]: parameters used to predict lattice volume
        
        Returns
        ----------
        latt [obj]: Lattice object of pymatgen
        """
        area = 0
        lower, upper = self.area_boundary_predict(params)
        while area < lower or area > upper:
            #triclinic
            if crystal_system == 1:
                a = np.random.gamma(9.74078, 0.513692)
                b = np.random.gamma(8.09637, 1.00711)
                gamma = np.random.choice([np.random.normal(105.857, 8.12514),
                                          np.random.normal(75.378, 6.33747)], p=[.5, .5])
            #orthorhombic
            elif crystal_system == 3:
                a = np.random.gamma(10.2258, 0.458184)
                b = np.random.gamma(7.26625, 1.17494)
                gamma = 90
            #tetragonal
            elif crystal_system == 4:
                a = np.random.gamma(10.6728, 0.441636)
                b = a
                gamma = 90
            #hexagonal
            elif crystal_system == 6:
                a = np.random.choice([np.random.gamma(13.4302, 0.460956),
                                      np.random.gamma(65.1615, 0.0554826)], p=[.5, .5])
                b = a
                gamma = 120
            #add vaccum layer
            c = vacuum_space
            alpha, beta = 90, 90
            #check validity of lattice vector
            if 0 < a and 0 < b and 0 < gamma and gamma < 180:
                latt = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
                area = latt.volume/c
            else:
                area = 0
        return latt
    
    def lattice_generate_3d(self, crystal_system, params):
        """
        generate lattice by 3d crystal system
        
        Parameters
        ----------
        crystal_system [int, 0d]: crystal system number
        params [float, tuple]: parameters used to predict lattice volume
        
        Returns
        ----------
        latt [obj]: lattice object of pymatgen
        """
        volume = 0
        lower, upper = self.volume_boundary_predict(params)
        while volume < lower or volume > upper:
            #triclinic
            if crystal_system == 1:
                a = np.random.gamma(11.1407, 0.54363)
                b = np.random.gamma(14.9172, 0.423919)
                c = np.random.gamma(9.3193, 0.751272)
                alpha = np.random.gamma(19.6702, 5.48192)
                beta = np.random.gamma(21.638, 4.76449)
                gamma = np.random.gamma(6.80769, 12.8246)
            #monoclinic
            elif crystal_system == 2:
                a = np.random.gamma(8.93408, 0.731207)
                b = np.random.gamma(7.51874, 0.906048)
                c = np.random.gamma(9.99707, 0.837898)
                alpha, gamma = 90, 90
                beta = np.random.choice([np.random.normal(108.242, 8.77108),
                                         np.random.normal(74.8068, 4.97432)], p=[.85, .15])
            #orthorhombic
            elif crystal_system == 3:
                a = np.random.gamma(5.84602, 1.05456)
                b = np.random.gamma(6.95415, 0.93105)
                c = np.random.gamma(6.40244, 1.09067)
                alpha, beta, gamma = 90, 90, 90
            #tetragonal
            elif crystal_system == 4:
                a = np.random.gamma(17.2155, 0.270062)
                c = np.random.gamma(11.6461, 0.691331)
                b = a
                alpha, beta, gamma = 90, 90, 90
            #trigonal or hexagonal
            elif crystal_system == 5 or crystal_system == 6:
                a = np.random.gamma(35.5889, 0.137857)
                c = np.random.gamma(20.7842, 0.242435)
                b = a
                alpha, beta, gamma = 90, 90, 120
            #cubic
            elif crystal_system == 7:
                a = np.random.gamma(17.2494, 0.271047)
                b, c = a, a
                alpha, beta, gamma = 90, 90, 90
            #check validity of lattice vector
            if 0 < a and 0 < b and 0 < c and 0 < alpha and 0 < beta and 0 < gamma and alpha + beta + gamma < 360:
                latt = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
                volume = latt.volume
            else:
                volume = 0
        return latt
    
    def get_parameters(self, atom_type):
        """
        get parameters to predict area or volume 
        
        Parameters
        ----------
        atom_type [int or str, 1d]: type of atoms

        Returns:
        ----------
        radiu [float, 1d]: atom radius
        negativity [float, 1d]: electronic negativity
        affinity [float, 1d]: electronic affinity
        """
        radius, negativity, affinity = [], [], []
        for i in atom_type:
            if isinstance(i, str):
                atom = Element(i)
            elif isinstance(i, int):
                atom = Element.from_Z(i)
            radius += [atom.atomic_radius.real]
            negativity += [atom.X]
            affinity += [atom.electron_affinity]
        return radius, negativity, affinity
    
    def get_atom_type_all(self, atom_type, atom_symm):
        """
        get type of all symmetry atoms
        
        Parameters
        ----------
        atom_type [int, 1d]: type of atoms in DAU
        atom_symm [int, 1d]: symmetry of atoms

        Returns
        ----------
        atom_type_all [int, 1d]: type of all atoms
        """
        atom_type_all = []
        for i, symm in enumerate(atom_symm):
            for _ in range(symm):
                atom_type_all.append(atom_type[i])
        return atom_type_all
        
    def area_boundary_predict(self, params):
        """
        predict lattice area boundary of specific composition
        
        Parameters
        ----------
        params [tuple, 2d]: parameters used to predict lattice area

        Returns
        ----------
        lower [float, 0d]: lower boundary of area
        upper [float, 0d]: upper boundary of area
        """
        radius, negativity, affinity = params
        #summation over atoms
        radiu = sum(radius)
        negativity = sum(negativity)
        affinity = sum(affinity)
        #predict area
        area_pred = 2.77505*radiu - 0.213984*negativity + 0.500184*affinity + 2.85521
        dense_stack = np.sum(.5*np.pi*np.array(radius)**2)
        #get boundary by occupy rate distribution
        lower_stack, upper_stack = 0.96*dense_stack, 2.66*dense_stack
        lower = max(lower_stack, area_pred*0.65)
        upper = min(upper_stack, area_pred*3.85)
        if lower >= upper:
            lower, upper = lower_stack, upper_stack
        return lower, upper
    
    def volume_boundary_predict(self, params):
        """
        predict lattice volume boundary of specific composition
        
        Parameters
        ----------
        params [tuple, 2d]: parameters used to predict lattice volume

        Returns
        ----------
        lower [float, 0d]: lower boundary of volume
        upper [float, 0d]: upper boundary of volume
        """
        radius, negativity, affinity = params
        #summation over atoms
        radiu = sum(radius)
        negativity = sum(negativity)
        affinity = sum(affinity)
        #predict volume
        volume_pred = 17.9984*radiu - 3.10083*negativity + 4.50924*affinity - 8.49767
        dense_stack = np.sum(4/3*np.pi*np.array(radius)**3)
        #get boundary by occupy rate distribution
        lower_stack, upper_stack = 0.96*dense_stack, 2.66*dense_stack
        lower = max(lower_stack, volume_pred*0.70)
        upper = min(upper_stack, volume_pred*2.21)
        if lower >= upper:
            lower, upper = lower_stack, upper_stack
        return lower, upper
    
    def add_symmetry_manually(self, path):
        """
        add symmetry to poscars from last iteration
        
        Parameters
        ----------
        path [str, 0d]: poscar saved path
        """
        poscars = os.listdir(path)
        for i, poscar in enumerate(poscars):
            stru = Structure.from_file(f'{path}/{poscar}')
            params = stru.lattice.parameters
            if dimension == 2:
                a, b, c, alpha, beta, gamma, crystal_system = self.add_symmetry_2d(params)
            elif dimension == 3:
                a, b, c, alpha, beta, gamma, crystal_system = self.add_symmetry_3d(params)
            new_latt = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
            new_stru = Structure(new_latt, stru.species, stru.frac_coords)
            new_stru.to(filename=f'{path}/POSCAR-Last-{crystal_system}-{i:03.0f}', fmt='poscar')
    
    def add_symmetry_2d(self, params, ang_tol=5, len_tol=0.01):
        """
        add symmetry for 2d structure
        
        Parameters
        ----------
        params [tuple, float]: lattice parameters
        ang_tol [float, 0d]: tolerance of angle
        len_tol [float, 0d]: tolerance of length

        Returns
        ----------
        params [tuple, float]: lattice parameters and crystal system number
        """
        a, b, c, alpha, beta, gamma = params
        #hexagonal
        if np.abs(a-b) < len_tol and np.abs(gamma-120) < ang_tol:
            b = a
            gamma = 120
            crystal_system = 6
        #tetragonal
        elif np.abs(a-b) < len_tol and np.abs(gamma-90) < ang_tol:
            b = a
            gamma = 90
            crystal_system = 4
        #orthorhombic
        elif np.abs(gamma-90) < ang_tol:
            gamma = 90
            crystal_system = 3
        #monoclinic
        else:
            crystal_system = 1
        #add vaccum layer
        c = vacuum_space
        alpha, beta = 90, 90
        return a, b, c, alpha, beta, gamma, crystal_system
    
    def add_symmetry_3d(self, params, ang_tol=5, len_tol=0.01):
        """
        add symmetry for 3d structure
        
        Parameters
        ----------
        params [tuple, float]: lattice parameters
        ang_tol [float, 0d]: tolerance of angle
        len_tol [float, 0d]: tolerance of length

        Returns
        ----------
        params [tuple, float]: lattice parameters and crystal system number
        """
        a, b, c, alpha, beta, gamma = params
        #cubic
        if np.abs(a-b) < len_tol and np.abs(a-c) < len_tol and np.abs(alpha-90) < ang_tol and np.abs(beta-90) < ang_tol and np.abs(gamma-90) < ang_tol:
            b, c = a, a
            alpha, beta, gamma = 90, 90, 90    
            crystal_system = 7
        #hexagonal
        elif np.abs(a-b) < len_tol and np.abs(alpha-90) < ang_tol and np.abs(beta-90) < ang_tol and np.abs(gamma-120) < ang_tol:
            b = a
            alpha, beta, gamma = 90, 90, 120
            crystal_system = 6
        #trigonal
        elif np.abs(a-b) < len_tol and np.abs(a-c) < len_tol and np.abs(alpha-beta) < ang_tol and np.abs(alpha-gamma) < ang_tol and np.abs(beta-gamma) < ang_tol:
            b = a
            alpha, beta, gamma = 90, 90, 120
            crystal_system = 5
        #tetragonal
        elif np.abs(a-b) < len_tol and np.abs(alpha-90) < ang_tol and np.abs(beta-90) < ang_tol and np.abs(gamma-90) < ang_tol:
            b = a
            alpha, beta, gamma = 90, 90, 90
            crystal_system = 4
        #orthorhombic
        elif np.abs(alpha-90) < ang_tol and np.abs(beta-90) < ang_tol and np.abs(gamma-90) < ang_tol:
            alpha, beta, gamma = 90, 90, 90
            crystal_system = 3
        #monoclinic
        elif np.abs(alpha-90) < ang_tol and np.abs(gamma-90) < ang_tol:
            alpha, gamma = 90, 90
            crystal_system = 2
        #triclinic
        else:
            crystal_system = 1
        return a, b, c, alpha, beta, gamma, crystal_system
    

if __name__ == '__main__':
    from core.utils import get_min_bond
    min_bond = get_min_bond(composition)
    grain = [min_bond/2, min_bond/2, min_bond/2]
    latt = Lattice.from_parameters(5, 5, 5, 90, 90, 120)
    gg = GridGenerate()
    
    all_grid, mapping = gg.get_grid_points_2d(183, grain, latt, {5:2, 6:6})
    if len(all_grid) > 0:
        gg.write_list2d('grid.dat', all_grid)
    else:
        print(all_grid)

    '''
    for i in [1, 2]:
        all_grid, mapping = gg.get_grid_points_2d(i, grain, latt, {5:2*4, 6:6*4})
        symm = np.unique([len(i) for i in mapping])
        gg.write_list2d(f'check_area_2d/{i:03g}.dat', all_grid)
        gg.write_list2d(f'check_symm_2d/{i:03g}.dat', [symm])
        print(i)
    '''