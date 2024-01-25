import os, sys
import copy
import numpy as np
import multiprocessing as pythonmp

from collections import Counter
from pymatgen.core.periodic_table import Element
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.groups import SpaceGroup

sys.path.append(f'{os.getcwd()}/src')
from core.log_print import *
from core.neighbors import Neighbors
from core.utils import ListRWTools
from core.space_group import PlaneSpaceGroup, BulkSpaceGroup


class LatticePrediction(ListRWTools):
    #predict lattice vectors
    def __init__(self):
        pass
    
    def get_parameters_stack(self, atom_type):
        """
        get parameters to predict area or volume
        
        Parameters
        ----------
        atom_type [int or str, 1d]: type of atoms

        Returns:
        ----------
        radius [float, 1d]: atom radius
        """
        if Cluster_Search:
            property_dict = self.import_data('property')
        radius = []
        for i in atom_type:
            if i > 0:
                atom = Element.from_Z(i)
                radius += [atom.atomic_radius.real]
            else:
                properties = property_dict[str(i)]
                radius += [properties['ave_radiu']]
        return radius
    
    def get_parameters_pred(self, atom_type):
        """
        get parameters to predict area or volume
        
        Parameters
        ----------
        atom_type [int or str, 1d]: type of atoms

        Returns:
        ----------
        radius [float, 1d]: atom radius
        negativity [float, 1d]: electronic negativity
        affinity [float, 1d]: electronic affinity
        """
        if Cluster_Search:
            property_dict = self.import_data('property')
            tmp_type = []
            for i in atom_type:
                if i > 0:
                    tmp_type.append(i)
                else:
                    tmp_type += property_dict[str(i)]['types']
            atom_type = tmp_type
        #get parameters to predict lattice
        radius, negativity, affinity = [], [], []
        for i in atom_type:
            atom = Element.from_Z(i)
            radius += [atom.atomic_radius.real]
            negativity += [atom.X]
            affinity += [atom.electron_affinity]
        return radius, negativity, affinity
    
    def area_boundary_stack(self, radius):
        """
        predict lattice area boundary of specific composition

        Parameters
        ----------
        radius [float, 1d]: atom radius

        Returns
        ----------
        lower_stack [float, 0d]: lower boundary of area
        upper_stack [float, 0d]: upper boundary of area
        """
        dense_stack = np.sum(.5*np.pi*np.array(radius)**2)
        #get boundary by occupy rate distribution
        lower_stack, upper_stack = 1.06*dense_stack, 5.07*dense_stack
        return lower_stack, upper_stack
    
    def area_boundary_predict(self, params):
        """
        predict lattice area boundary of specific composition
        
        Parameters
        ----------
        params [tuple, 2d]: parameters used to predict lattice area

        Returns
        ----------
        lower_pred [float, 0d]: lower boundary of area
        upper_pred [float, 0d]: upper boundary of area
        """
        radius, negativity, affinity = params
        #summation over atoms
        radiu = sum(radius)
        negativity = sum(negativity)
        affinity = sum(affinity)
        #predict area
        area_pred = 3.19353*radiu - 0.0126031*negativity + 0.253931*affinity + 1.18601
        lower_pred, upper_pred = area_pred*0.68, area_pred*2.99
        return lower_pred, upper_pred
    
    def volume_boundary_stack(self, radius):
        """
        predict lattice volume boundary of specific composition

        Parameters
        ----------
        radius [float, 1d]: atom radius

        Returns
        ----------
        lower_stack [float, 0d]: lower boundary of volume
        upper_stack [float, 0d]: upper boundary of volume
        """
        dense_stack = np.sum(4/3*np.pi*np.array(radius)**3)
        #get boundary by occupy rate distribution
        lower_stack, upper_stack = 0.96*dense_stack, 2.66*dense_stack
        return lower_stack, upper_stack
    
    def volume_boundary_predict(self, params):
        """
        predict lattice volume boundary of specific composition
        
        Parameters
        ----------
        params [tuple, 2d]: parameters used to predict lattice volume

        Returns
        ----------
        lower_pred [float, 0d]: lower boundary of volume
        upper_pred [float, 0d]: upper boundary of volume
        """
        radius, negativity, affinity = params
        #summation over atoms
        radiu = sum(radius)
        negativity = sum(negativity)
        affinity = sum(affinity)
        #predict volume
        volume_pred = 17.9984*radiu - 3.10083*negativity + 4.50924*affinity - 8.49767
        lower_pred, upper_pred = volume_pred*0.70, volume_pred*2.21
        return lower_pred, upper_pred
    
    def lattice_generate_2d_database(self, crystal_system, radius, params, max_try=100):
        """
        generate lattice by 2d crystal system from database
        
        Parameters
        ----------
        crystal_system [int, 0d]: crystal system number
        radius [float, 1d]: atom radius
        params [float, tuple]: parameters used to predict lattice volume
        max_try [int, 0d]: maximum number of trying
        
        Returns
        ----------
        latt [obj]: Lattice object of pymatgen
        """
        area = 0
        counter, store, score = 0, [], []
        latt_limit = 2*np.sum(params[0])
        lower_stack, upper_stack = self.area_boundary_stack(radius)
        lower_pred, upper_pred = self.area_boundary_predict(params)
        lower = max(lower_stack, lower_pred)
        upper = min(upper_stack, upper_pred)
        if lower >= upper:
            lower, upper = lower_stack, upper_stack
        while area < lower or area > upper:
            #triclinic
            if crystal_system == 1:
                a = np.random.gamma(9.21681, 0.534819)
                b = np.random.gamma(7.04037, 1.06967)
                gamma = np.random.choice([np.random.normal(107.127, 8.59761),
                                          np.random.normal(73.3422, 6.21229)], p=[.58, .42])
            #orthorhombic
            elif crystal_system == 3:
                a = np.random.gamma(10.2367, 0.431989)
                b = np.random.gamma(8.51868, 0.881737)
                gamma = 90
            #tetragonal
            elif crystal_system == 4:
                a = np.random.gamma(12.7283, 0.356367)
                b = a
                gamma = 90
            #hexagonal
            elif crystal_system == 6:
                a = np.random.choice([np.random.gamma(24.8229, 0.263644),
                                      np.random.gamma(57.0826, 0.0639401)], p=[.4, .6])
                b = a
                gamma = 120
            #add vaccum layer
            c = Vacuum_Space
            alpha, beta = 90, 90
            #check validity of lattice vector
            if 0 < a and 0 < b and 0 < gamma and gamma < 180:
                if a < latt_limit and b < latt_limit:
                    latt = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
                    area = latt.volume/c
                    store.append(latt)
                    score.append(np.abs(area-lower)+np.abs(area-upper))
                    counter += 1
                    if counter > max_try:
                        idx = np.argmin(score)
                        latt = store[idx]
                        break
                    if  lower < area < upper:
                        break
            else:
                area = 0
        return latt
    
    def lattice_generate_2d_rand(self, crystal_system, radius, params, max_try=100):
        """
        generate lattice by 2d crystal system randomly
        
        Parameters
        ----------
        crystal_system [int, 0d]: crystal system number
        radius [float, 1d]: atom radius
        params [float, tuple]: parameters used to predict lattice volume
        max_try [int, 0d]: maximum number of trying
        
        Returns
        ----------
        latt [obj]: lattice object of pymatgen
        """
        counter, store, score = 0, [], []
        latt_limit = 2*np.sum(params[0])
        lower_stack, upper_stack = self.area_boundary_stack(radius)
        lower_pred, upper_pred = self.area_boundary_predict(params)
        lower = max(lower_stack, lower_pred)
        upper = min(upper_stack, upper_pred)
        if lower >= upper:
            lower, upper = lower_stack, upper_stack
        for _ in range(max_try):
            area = np.random.uniform(lower, upper)
            #triclinic
            if crystal_system == 1:
                a = np.random.uniform(0.7*area**0.5, area**0.5)
                b = area/a
                gamma = np.random.uniform(30, 150)
            #orthorhombic
            elif crystal_system == 3:
                a = np.random.uniform(0.7*area**0.5, area**0.5)
                b = area/a
                gamma = 90
            #tetragonal
            elif crystal_system == 4:
                a = area**0.5
                b = a
                gamma = 90
            #hexagonal
            elif crystal_system == 6:
                a = (2*area/(3**0.5))**0.5
                b = a
                gamma = 120
            #add vaccum layer
            c = Vacuum_Space
            alpha, beta = 90, 90
            #check validity of lattice vector
            if 0 < a and 0 < b and 0 < gamma and gamma < 180:
                if a < latt_limit and b < latt_limit:
                    latt = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
                    area = latt.volume/c
                    store.append(latt)
                    score.append(np.abs(area-lower)+np.abs(area-upper))
                    counter += 1
                    if counter > max_try:
                        idx = np.argmin(score)
                        latt = store[idx]
                        break
                    if  lower < area < upper:
                        break
        return latt
    
    def lattice_generate_3d_database(self, crystal_system, radius, params, max_try=100):
        """
        generate lattice by 3d crystal system from database
        
        Parameters
        ----------
        crystal_system [int, 0d]: crystal system number
        radius [float, 1d]: atom radius
        params [float, tuple]: parameters used to predict lattice volume
        max_try [int, 0d]: maximum number of trying
        
        Returns
        ----------
        latt [obj]: lattice object of pymatgen
        """
        volume = 0
        counter, store, score = 0, [], []
        latt_limit = 2*np.sum(params[0])
        lower_stack, upper_stack = self.volume_boundary_stack(radius)
        lower_pred, upper_pred = self.volume_boundary_predict(params)
        lower = max(lower_stack, lower_pred)
        upper = min(upper_stack, upper_pred)
        if lower >= upper:
            lower, upper = lower_stack, upper_stack
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
                if a < latt_limit and b < latt_limit and c < latt_limit:
                    latt = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
                    volume = latt.volume
                    store.append(latt)
                    score.append(np.abs(volume-lower)+np.abs(volume-upper))
                    counter += 1
                    if counter > max_try:
                        idx = np.argmin(score)
                        latt = store[idx]
                        break
                    if  lower < volume < upper:
                        break
            else:
                volume = 0
        return latt
    
    def lattice_generate_3d_rand(self, crystal_system, radius, params, max_try=100):
        """
        generate lattice by 3d crystal system randomly
        
        Parameters
        ----------
        crystal_system [int, 0d]: crystal system number
        radius [float, 1d]: atom radius
        params [float, tuple]: parameters used to predict lattice volume
        max_try [int, 0d]: maximum number of trying
        
        Returns
        ----------
        latt [obj]: lattice object of pymatgen
        """
        counter, store, score = 0, [], []
        latt_limit = 2*np.sum(params[0])
        lower_stack, upper_stack = self.volume_boundary_stack(radius)
        lower_pred, upper_pred = self.volume_boundary_predict(params)
        lower = max(lower_stack, lower_pred)
        upper = min(upper_stack, upper_pred)
        if lower >= upper:
            lower, upper = lower_stack, upper_stack
        for _ in range(max_try):
            volume = np.random.uniform(lower, upper)
            #triclinic
            if crystal_system == 1:
                factors = [np.random.random(), np.random.random(), np.random.random()]
                norm = sum(factors)
                a, b, c = [factor*volume**(1/3) for factor in factors]
                a, b, c = a/norm, b/norm, c/norm
                alpha = np.random.uniform(60, 120)
                beta = np.random.uniform(60, 120) 
                gamma = np.random.uniform(60, 120)
            #monoclinic
            elif crystal_system == 2:
                a = np.random.uniform(0.7*volume**(1/3), volume**(1/3))
                b = np.random.uniform(0.7*volume**(1/3), volume**(1/3))
                c = volume/(a*b)
                beta = np.random.uniform(90, 120)
                alpha, gamma = 90, 90
            #orthorhombic
            elif crystal_system == 3:
                factors = [np.random.random(), np.random.random(), np.random.random()]
                norm = sum(factors)
                a, b, c = [factor*volume**(1/3) for factor in factors]
                a, b, c = a/norm, b/norm, c/norm
                alpha, beta, gamma = 90, 90, 90
            #tetragonal
            elif crystal_system == 4:
                c = np.random.uniform(0.5*volume**(1/3), 1.5*volume**(1/3))
                a = (volume/c)**(1/2)
                b = a
                alpha, beta, gamma = 90, 90, 90
            #trigonal or hexagonal
            elif crystal_system == 5 or crystal_system == 6:
                c = np.random.uniform(0.5*volume**(1/3), 1.5*volume**(1/3))
                a = (volume/c)**(1/2)
                b = a
                alpha, beta, gamma = 90, 90, 120
            #cubic
            elif crystal_system == 7:
                a = volume**(1/3)
                b, c = a, a
                alpha, beta, gamma = 90, 90, 90
            #check validity of lattice vector
            if 0 < a and 0 < b and 0 < c and 0 < alpha and 0 < beta and 0 < gamma and alpha + beta + gamma < 360:
                if a < latt_limit and b < latt_limit and c < latt_limit:
                    latt = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
                    volume = latt.volume
                    store.append(latt)
                    score.append(np.abs(volume-lower)+np.abs(volume-upper))
                    counter += 1
                    if counter > max_try:
                        idx = np.argmin(score)
                        latt = store[idx]
                        break
                    if  lower < volume < upper:
                        break
        return latt


class GridGenerate(Neighbors, LatticePrediction):
    #Build the grid
    def __init__(self):
        Neighbors.__init__(self)
    
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
        for item in Space_Group:
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
        #sampling crystal systems
        cs_num = len(cs_list)
        repeat = number//cs_num
        left = np.mod(number, cs_num)
        uniform = np.ravel([cs_list for _ in range(repeat)])
        random = np.random.choice(cs_list, left, replace=False)
        crystal_system = np.concatenate((uniform, random))
        return np.array(crystal_system, dtype=int)
    
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
        if Dimension == 2:
            for sg in space_group:
                if sg in range(1, 3):
                    crystal_system.append(1)
                elif sg in range(3, 10):
                    crystal_system.append(3)
                elif sg in range(10, 13):
                    crystal_system.append(4)
                elif sg in range(13, 18):
                    crystal_system.append(6)
        elif Dimension == 3:
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
        if Dimension == 2:
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
        self.write_dict(f'{Grid_Path}/space_group_sampling.json', cs_sg_dict)
        return cs_sg_dict
    
    def get_space_group(self, num, crys_system):
        """
        get space group number according to crystal system
        
        Parameters
        ----------
        num [int, 0d]: number of space groups
        crys_system [int, 0d]: crystal system

        Returns
        ----------
        space_group [int, 1d, np]: international number of space group
        """
        sg_buffer = self.import_dict(f'{Grid_Path}/space_group_sampling.json', trans=True)[crys_system]
        #sampling space groups uniformly
        sample_num = min(num, len(sg_buffer))
        space_group = np.random.choice(sg_buffer, sample_num, replace=False)
        return space_group
    
    def generate_latt(self, poscar, crys_system):
        """
        generate lattice poscar
        
        Parameters
        ----------
        poscar [str, 0d]: poscar name 
        crys_system [int, 0d]: crystal system number
        """
        max_atom_type = self.import_list2d(f'{Grid_Path}/atom_types.dat', int)[-1]
        radius = self.get_parameters_stack(max_atom_type)
        params = self.get_parameters_pred(max_atom_type)
        #estimate size of lattice
        if Dimension == 2:
            if np.random.rand() > Rand_Latt_Ratio:
                latt = self.lattice_generate_2d_database(crys_system, radius, params)
            else:
                latt = self.lattice_generate_2d_rand(crys_system, radius, params)
        elif Dimension == 3:
            if np.random.rand() > Rand_Latt_Ratio:
                latt = self.lattice_generate_3d_database(crys_system, radius, params)
            else:
                latt = self.lattice_generate_3d_rand(crys_system, radius, params)
        stru = Structure(latt, [1], [[0, 0, 0]])
        stru.to(filename=poscar, fmt='poscar')
        
    def get_grid_points_seeds(self, stru):
        """
        generate grid point according to initial seeds
        
        Parameters
        ----------
        stru [obj, 0d]: pymatgen structure object

        Returns
        ----------
        all_grid [float, 2d, np]: fraction coordinates of grid points
        mapping [int, 2d]: mapping between min and all grid
        """
        all_grid = stru.frac_coords
        mapping = [[i] for i in range(len(all_grid))]
        return np.array(all_grid), mapping
    
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
        grid [float, 2d, np]: DAU grid includes all symmetry sites

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
    
    def dau_grid_sampling_3d(self, grain, atom_num, sg, grid, latt, limit=8, repeat=100):
        """
        sampling on min grid to sparse dau grid
        
        Parameters
        ----------
        grain [float, 1d]: grain of grid points
        atom_num [int, 1d]: number of different atoms
        sg [int, 0d]: international number of space group
        grid [float, 2d, np]: DAU grid includes all symmetry sites
        latt [obj]: lattice object in pymatgen
        limit [int, 0d]: limit of parallel cores
        repeat [int, 0d]: repeat times of generating grids
        
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
        #downsampling
        volume = latt.volume
        equal_a = np.cbrt(volume)
        total_n = int(np.prod([equal_a//grain[i]+1 for i in range(3)]))
        if len(symm_site.keys()) > 0:
            latt_vec = latt.matrix
            un_best, cutoff = -1e6, np.mean(grain)
            #generate jobs
            args_list = []
            for _ in range(repeat):
                args_list.append((sg, total_n, grid, latt_vec, symm_site, cutoff))
            #multi-cores
            cores = limit
            with pythonmp.get_context('fork').Pool(processes=cores) as pool:
                #get jobs
                jobs = [pool.apply_async(self.discretize_space, args) for args in args_list]
                pool.close()
                pool.join()
                #get results
                jobs_pool = [p.get() for p in jobs]
                for un, sample_coords in jobs_pool:
                    if un > un_best:
                        dau_grid = sample_coords
                        un_best = un
            del pool
        else:
            dau_grid = []
        return dau_grid
    
    def discretize_space(self, sg, total_n, grid, latt_vec, symm_site, cutoff):
        """
        discretize space into grid
        
        Parameters
        ----------
        sg [int, 0d]: space group number
        total_n [int, 0d]: total number of grid points
        grid [float, 2d, np]: DAU grid includes all symmetry sites
        latt_vec [float, 2d, np]: lattice vector
        symm_site [dict, int]: index of symmetry sites
        cutoff [float, 0d]: cutoff distance 

        Returns
        ----------
        un [float, 0d]: uniformity of grid points 
        """
        #sampling on different symmetry
        sample_coords = []
        sampling_num = self.get_symm_sampling_num(total_n, symm_site)
        for mul, idx in symm_site.items():
            sample_idx = self.sample_uniform_index(idx, sampling_num[mul], grid, latt_vec)
            sample_coords += grid[sample_idx].tolist()
        #calculate uniformity
        un = self.calculate_uniformity(sg, latt_vec, sample_coords, cutoff)
        return un, sample_coords
        
    def sample_uniform_index(self, idx, num, grid, latt_vec):
        """
        sampling uniformally in space

        Parameters
        ----------
        idx [int, 1d]: points index
        num [int, 0d]: number of sampling points
        grid [float, 2d, np]: dense min grid includes all symmetry sites
        latt_vec [float, 2d, np]: lattice vector
        
        Returns
        ----------
        sample_idx [int, 1d]: index of samples
        """
        np.random.seed()
        slt_idx = np.random.choice(idx, 1)
        coords = grid[idx]
        center_coord = np.dot(grid[slt_idx], latt_vec)
        all_coords = np.dot(coords, latt_vec)
        dis = np.linalg.norm(all_coords-center_coord, axis=1)
        order = np.argsort(dis)
        idx_order = np.array(idx)[order].tolist()
        #get index
        num_idx = len(idx_order)
        interval = num_idx//num
        sample_idx, left_idx = self.slice_by_interval(idx_order, interval)
        left = num - len(sample_idx)
        if left > 0:
            np.random.seed()
            sample_idx += np.random.choice(left_idx, left, replace=False).tolist()
        return sample_idx
    
    def slice_by_interval(self, index, interval):
        """
        slice the index list by interval

        Parameters
        ----------
        index [int, 1d]: index list 
        interval [int ,0d]: interval of sampling 
        
        Returns
        ----------
        store [int, 1d]: select index 
        left [int, 1d]: left index
        """
        select, left = [], []
        for i, idx in enumerate(index):
            if np.mod(i, interval) == 0:
                select.append(idx)
            else:
                left.append(idx)
        return select, left
    
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
        total_n [int, 0d]: total number of sparse grid\\
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
                    np.random.seed()
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
    
    def calculate_uniformity(self, sg, latt_vec, dau_coords, cutoff=1):
        """
        get uniformity of DAU grid
        
        Parameters
        ----------
        sg [int, 0d]: space group number 
        latt_vec [float, 2d, np]: lattice vector
        dau_coords [float, 2d]: coordinates in DAU 
        cutoff [float, 0d]: cutoff distance
        
        Returns
        ----------
        un [float, 0d]: uniformity of grid
        """
        #get all grid points in unit cell
        dau_atom_num = len(dau_coords)
        all_grid = dau_coords.copy()
        if len(dau_coords) > 0:
            spg = SpaceGroup.from_int_number(sg)
            for point in dau_coords:
                coords = spg.get_orbit(point)
                equal_coords = self.get_equal_coords(point, coords)
                all_grid += equal_coords
        #neighbor index and distance of DAU
        atoms = [1 for _ in range(len(all_grid))]
        stru = Structure(latt_vec, atoms, all_grid)
        distances = stru.get_neighbor_list(cutoff, sites=stru.sites[:dau_atom_num])[-1]
        #calculate uniformity
        near_num = len(distances)
        if near_num > 0:
            un = np.mean(distances)/near_num
        else:
            un= cutoff
        return un
    
    def get_assign_parallel(self, atom_num, symm_site, times=10, limit=8):
        """
        generate site assignments by space group in parallel
        
        Parameters
        ----------
        atom_num [dict, int:int]: number of different atoms
        symm_site [dict, int:list]: list of different symmetry sites
        times [int, 0d]: times for sampling sites assignments
        limit [int, 0d]: limit of core number

        Returns
        ----------
        assign_plan [list, dict, 1d]: site assignment of atom_num
        e.g. [{5:[1, 1, 2, 2], 6:[6, 12]}]
        """
        symms = list(symm_site.keys())
        site_num = [len(i) for i in symm_site.values()]
        symm_num = dict(zip(symms, site_num))
        atoms = list(atom_num.keys())
        #initialize assignment
        init_assign = {}
        for atom in atoms:
            init_assign[atom] = []
        #find site assignment of different atom_num
        args_list = []
        for _ in range(times):
            args = (atom_num, atoms, symm_num, symms, init_assign)
            args_list.append(args)
        #multi-cores
        assign_plan = []
        cores = limit
        with pythonmp.get_context('fork').Pool(processes=cores) as pool:
            #put atoms into grid with symmetry constrain
            jobs = [pool.apply_async(self.assign_by_spacegroup, args) for args in args_list]
            #get results
            pool.close()
            pool.join()
            jobs_pool = [p.get() for p in jobs]
            for flag, assign in jobs_pool:
                if flag:
                    assign_plan += assign
        del pool
        assign_plan = self.delete_same_assign(assign_plan)
        return assign_plan
        
    def assign_by_spacegroup(self, atom_num, atoms, symm_num, symms, init_assign, max_store=100, max_assign=50):
        """
        generate site assignments by space group
        
        Parameters
        ----------
        atom_num [dict, int:int]: number of different atoms
        atoms [int, 1d]: type of atoms
        symm_num [dict, int:int]: symmetry of different atoms
        symms [int, 1d]: symmetry of atoms
        init_assign [dict, 1d]: assignment of sites
        max_store [int, 0d]: max number of store during assignment
        max_assign [int, 0d]: max number of assignments
        
        Returns
        ----------
        assign_plan [list, dict, 1d]: site assignment of atom_num
        e.g. [{5:[1, 1, 2, 2], 6:[6, 12]}]
        """
        counter = 0
        new_store, assign_plan = [], []
        store = [init_assign]
        while True:
            for assign in store:
                np.random.shuffle(symms)
                for site in symms:
                    np.random.shuffle(atoms)
                    for atom in atoms:
                        new_assign = copy.deepcopy(assign)
                        new_assign[atom] += [site]
                        save = self.check_assign(atom_num, symm_num, new_assign)
                        if save == 0:
                            assign_plan.append(new_assign)
                            counter += 1
                        if save == 1:
                            new_store.append(new_assign)
                        #constrain leafs
                        if len(new_store) > max_store:
                            break
                    if len(new_store) > max_store:
                        break
                if len(new_store) > max_store:
                    break
            if len(new_store) == 0:
                break
            if counter > max_assign:
                break
            store = new_store
            store = self.delete_same_assign(store)
            np.random.shuffle(store)
            new_store = []
        flag = False
        if len(assign_plan) > 0:
            flag = True
        return flag, assign_plan
    
    def group_symm_sites(self, mapping):
        """
        group sites by symmetry
        
        Parameters
        ----------
        mapping [int, 2d]: mapping between DAU and all grid

        Returns
        ----------
        symm_site [dict, int:list]: site position in DAU grouped by symmetry
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
    
    def build(self, grid, cutoff, limit=8):
        """
        save nbr_idx, nbr_dis of grid
        
        Parameters
        ----------
        grid [int, 0d]: name of grid
        cutoff [float, 0d]: cutoff distance
        """
        head = f'{Grid_Path}/{grid:03.0f}'
        latt_vec = self.import_list2d(f'{head}_latt_vec.bin',
                                      float, binary=True)
        space_group = self.get_grid_sg(grid)
        #generate jobs
        args_list = []
        for sg in space_group:
            coords = self.import_list2d(f'{head}_frac_coords_{sg}.bin',
                                        float, binary=True)
            mapping = self.import_list2d(f'{head}_mapping_{sg}.bin',
                                        int, binary=True)
            args_list.append((head, latt_vec, sg, coords, cutoff, mapping))
        #multi-cores
        cores = limit
        with pythonmp.get_context('fork').Pool(processes=cores) as pool:
            #get jobs
            [pool.apply_async(self.build_grid, args) for args in args_list]
            pool.close()
            #start jobs
            pool.join()
        del pool
    
    def build_grid(self, head, latt_vec, sg, coords, cutoff, mapping):
        """
        calculate neighbors for a sparse grid

        Parameters
        ----------
        head [str, 0d]: store path
        latt_vec [float, 2d, np]: lattice vector
        sg [int, 0d]: space group number
        coords [float, 2d, np]: DAU coordinates
        cutoff [float, 0d]: neighbor cutoff distance
        mapping [int, 2d]: DAU mapping relationship 
        """
        atoms = [1 for _ in range(len(coords))]
        stru = Structure(latt_vec, atoms, coords)
        #export neighbor index and distance in DAU
        nbr_idx, nbr_dis = self.get_neighbors_DAU(stru, cutoff, mapping)
        self.write_list2d(f'{head}_nbr_idx_{sg}.bin', 
                        nbr_idx, binary=True)
        self.write_list2d(f'{head}_nbr_dis_{sg}.bin', 
                        nbr_dis, binary=True)
    
    def get_grid_sg(self, grid):
        """
        get space groups of grid

        Parameters
        ----------
        grid [int, 0d]: name of grid
        
        Returns
        ---------
        space_group [str, 1d]: space group number
        atom_num [str, 1d]:
        """
        grid_file = os.listdir(Grid_Path)
        coords_file = [i for i in grid_file 
                       if i.startswith(f'{grid:03.0f}_frac')]
        name = [i.split('.')[0] for i in coords_file]
        space_group = [i.split('_')[-1] for i in name]
        return space_group
    
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
        
    def succeed_grid(self, init_path, opt_path):
        """
        keep grid to next iteration
        
        Parameters
        ----------
        init_path [str, 0d]: initial poscar path
        opt_path [str, 0d]: optimization poscar path
        """
        poscars = [i for i in os.listdir(opt_path) if len(i.split('-'))==3]
        grids = []
        for poscar in poscars:
            stru = Structure.from_file(f'{opt_path}/{poscar}')
            grids.append(stru)
        for i, grid in enumerate(grids):
            grid.to(filename=f'{init_path}/POSCAR-Template-0-{i:03.0f}', fmt='poscar')
    
    def succeed_latt(self, init_path, opt_path):
        """
        keep lattice to next iteration
        
        Parameters
        ----------
        path [str, 0d]: poscar saved path
        """
        poscars = [i for i in os.listdir(opt_path) if len(i.split('-'))==3]
        for i, poscar in enumerate(poscars):
            stru = Structure.from_file(f'{opt_path}/{poscar}')
            params = stru.lattice.parameters
            if Dimension == 2:
                a, b, c, alpha, beta, gamma, crystal_system = self.add_symmetry_2d(params)
            elif Dimension == 3:
                a, b, c, alpha, beta, gamma, crystal_system = self.add_symmetry_3d(params)
            new_latt = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
            new_stru = Structure(new_latt, stru.species, stru.frac_coords)
            new_stru.to(filename=f'{init_path}/POSCAR-Latt-{crystal_system}-{i:03.0f}', fmt='poscar')
    
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
        c = Vacuum_Space
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
    pass