import os, sys
import numpy as np
import multiprocessing as pythonmp

from pymatgen.core.structure import Structure
from pymatgen.symmetry.groups import SpaceGroup

sys.path.append(f'{os.getcwd()}/src')
from core.log_print import *
from core.neighbors import Neighbors
from core.cluster import AtomManipulate


class Transfer(Neighbors, AtomManipulate):
    #transfer data to structure object or input of GNN
    def __init__(self):
        Neighbors.__init__(self)
    
    def get_gnn_input_general(self, atom_pos, atom_type, elem_embed,
                              ratio, sg, latt_vec, grid_coords):
        """
        get input of GNN model
        
        Parameters
        ----------
        atom_pos [int, 1d]: position of atoms
        atom_type [int, 1d]: type of atoms
        elem_embed [int, 2d, np]: embedding of elements
        ratio [float, 0d]: grid ratio
        sg [int, 0d]: space group number
        latt_vec [float, 2d, np]: lattice vector
        grid_coords [float, 2d, np]: fraction coordinates of grid
        
        Returns
        ----------
        atom_fea [int, 2d, np]: atom feature
        nbr_fea [float, 3d, np]: neighbor feature
        nbr_idx [float, 2d, np]: neighbor index
        """
        #get atom features
        atom_fea = self.get_atom_fea(atom_type, elem_embed)
        #get bond features and neighbor index
        nbr_fea, nbr_idx = \
            self.get_nbr_fea_general(atom_pos, ratio, sg, latt_vec, grid_coords)
        return atom_fea, nbr_fea, nbr_idx
    
    def get_gnn_input_template(self, atom_pos, atom_type, elem_embed,
                               ratio, grid_idx, grid_dis):
        """
        get input of GNN model
        
        Parameters
        ----------
        atom_pos [int, 1d]: position of atoms
        atom_type [int, 1d]: type of atoms
        elem_embed [int, 2d, np]: embedding of elements
        ratio [float, 0d]: grid ratio
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        
        Returns
        ----------
        atom_fea [int, 2d, np]: atom feature
        nbr_fea [float, 3d, np]: neighbor feature
        nbr_idx [float, 2d, np]: neighbor index
        """
        #get atom features
        atom_fea = self.get_atom_fea(atom_type, elem_embed)
        #get bond features and neighbor index
        nbr_fea, nbr_idx = \
            self.get_nbr_fea_template(atom_pos, ratio, grid_idx, grid_dis)
        return atom_fea, nbr_fea, nbr_idx
    
    def get_gnn_input_from_stru(self, stru, elem_embed, type='structure'):
        """
        get input of GNN from structure
        
        Parameters
        ----------
        stru [obj/str, 0d]: structure object or path of POSCAR
        elem_embed [int, 2d, np]: embedding of atoms
        type [str, 0d]: type of structure
        
        Returns
        ----------
        atom_fea [int, 2d, np]: feature of atoms
        nbr_fea [float, 3d, np]: distance of neighbors 
        nbr_idx [int, 2d, np]: index of neighbors
        """
        if type == 'structure':
            pass
        elif type == 'POSCAR':
            stru = Structure.from_file(stru)
        atom_type = np.array(stru.atomic_numbers)
        nbr_idx, nbr_dis = self.get_nbr_stru(stru)
        #get atom and bond features
        nbr_fea = self.expand(nbr_dis)
        atom_fea = self.get_atom_fea(atom_type, elem_embed)
        return atom_fea, nbr_fea, nbr_idx
    
    def get_gnn_input_from_stru_batch(self, strus, type='structure'):
        """
        get input of GNN in batch
        
        Parameters
        ----------
        strus [obj, 1d]: structure object
        type [str, 0d]: type of structure

        Returns
        ----------
        atom_fea_bh [float, 3d]: batch atom feature
        nbr_fea_bh [float, 4d]: batch bond feature
        nbr_idx_bh [int, 3d]: batch neighbor index
        """
        elem_embed = self.import_data('elem')
        atom_fea_bh, nbr_fea_bh, nbr_idx_bh = [], [], []
        for stru in strus:
            atom_fea, nbr_fea, nbr_idx = self.get_gnn_input_from_stru(stru, elem_embed)
            atom_fea_bh.append(atom_fea)
            nbr_fea_bh.append(nbr_fea)
            nbr_idx_bh.append(nbr_idx)
        return atom_fea_bh, nbr_fea_bh, nbr_idx_bh
    
    def get_gnn_input_from_stru_batch_parallel(self, strus, type='structure', limit=0):
        """
        get input of GNN in batch
        
        Parameters
        ----------
        strus [obj, 1d]: structure object
        type [str, 0d]: type of structure
        limit [int, 0d]: limit of core number
        
        Returns
        ----------
        atom_fea_bh [float, 3d]: batch atom feature
        nbr_fea_bh [float, 4d]: batch bond feature
        nbr_idx_bh [int, 3d]: batch neighbor index
        """
        elem_embed = self.import_data('elem')
        atom_fea_bh, nbr_fea_bh, nbr_idx_bh = [], [], []
        #multi-cores
        args_list = []
        if limit == 0:
            cores = pythonmp.cpu_count()
        else:
            cores = min(limit, pythonmp.cpu_count())
        with pythonmp.get_context('fork').Pool(processes=cores) as pool:
            for stru in strus:
                args_list.append((stru, elem_embed, type))
            #put atoms into grid with symmetry constrain
            jobs = [pool.apply_async(self.get_gnn_input_from_stru, args) for args in args_list]
            pool.close()
            pool.join()
            #get results
            jobs_pool = [p.get() for p in jobs]
            for atom_fea, nbr_fea, nbr_idx in jobs_pool:
                atom_fea_bh.append(atom_fea)
                nbr_fea_bh.append(nbr_fea)
                nbr_idx_bh.append(nbr_idx)
        pool.close()
        del pool
        return atom_fea_bh, nbr_fea_bh, nbr_idx_bh
    
    def get_gnn_input_seq_general(self, atom_pos, atom_type, grid, grid_ratio, space_group,
                                  latt_vec, elem_embed):
        """
        get input of GNN under same grid
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid [int, 0d]: name of grid
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        latt_vec [float, 2d, np]: lattice vector
        elem_embed [int, 2d, np]: embedding of elements
        
        Returns
        ----------
        atom_fea_seq [float, 3d]: atom feature
        nbr_fea_seq [float, 4d]: bond feature
        nbr_idx_seq [int, 3d]: near index
        """
        last_sg = space_group[0]
        atom_fea_seq, nbr_fea_seq, nbr_idx_seq = [], [], []
        #transform pos, type into input of gnn for different space groups
        grid_coords = self.import_data('frac', grid, last_sg)
        atom_fea, nbr_fea, nbr_idx = \
            self.get_gnn_input_general(atom_pos[0], atom_type[0], elem_embed,
                                       grid_ratio[0], last_sg, latt_vec, grid_coords)
        for i, sg in enumerate(space_group):
            #update fraction coordinates
            if sg != last_sg:
                grid_coords = self.import_data('frac', grid, sg)
                last_sg = sg
            atom_fea, nbr_fea, nbr_idx = \
                self.get_gnn_input_general(atom_pos[i], atom_type[i], elem_embed, 
                                           grid_ratio[i], sg, latt_vec, grid_coords)
            atom_fea_seq.append(atom_fea)
            nbr_fea_seq.append(nbr_fea)
            nbr_idx_seq.append(nbr_idx)
        return atom_fea_seq, nbr_fea_seq, nbr_idx_seq
    
    def get_gnn_input_seq_template(self, atom_pos, atom_type, grid, grid_ratio, space_group):
        """
        get input of GNN under same grid
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid [int, 0d]: name of grid
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        
        Returns
        ----------
        atom_fea_seq [float, 3d]: atom feature
        nbr_fea_seq [float, 4d]: bond feature
        nbr_fea_idx_seq [int, 3d]: near index
        """
        last_sg = space_group[0]
        atom_fea_seq, nbr_fea_seq, nbr_fea_idx_seq = [], [], []
        #transform pos, type into input of gnn under different space groups
        elem_embed = self.import_data('elem')
        grid_idx, grid_dis = self.import_data('grid', grid, last_sg)
        for i, sg in enumerate(space_group):
            if sg == last_sg:
                atom_fea, nbr_fea, nbr_fea_idx = \
                    self.get_gnn_input_template(atom_pos[i], atom_type[i], elem_embed,
                                                grid_ratio[i], grid_idx, grid_dis)
                atom_fea_seq.append(atom_fea)
                nbr_fea_seq.append(nbr_fea)
                nbr_fea_idx_seq.append(nbr_fea_idx)
            else:
                grid_idx, grid_dis = self.import_data('grid', grid, sg)
                atom_fea, nbr_fea, nbr_fea_idx = \
                    self.get_gnn_input_template(atom_pos[i], atom_type[i], elem_embed,
                                                grid_ratio[i], grid_idx, grid_dis)
                atom_fea_seq.append(atom_fea)
                nbr_fea_seq.append(nbr_fea)
                nbr_fea_idx_seq.append(nbr_fea_idx)
                last_sg = sg
        return atom_fea_seq, nbr_fea_seq, nbr_fea_idx_seq
    
    def get_stru_seq(self, atom_pos, atom_type, grid, grid_ratio, space_group, angles, thicks, latt_vec):
        """
        get strucutre object in same lattice
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid [int, 0d]: grid number 
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        latt_vec [float, 2d, np]: lattice vector
        
        Returns
        ----------
        stru_seq [obj, 1d]: structure object in pymatgen
        """
        #import lattice and grid
        last_sg = space_group[0]
        grid_coords = self.import_data('frac', grid, last_sg)
        cluster_angles, property_dict = [], []
        if Cluster_Search:
            cluster_angles = self.import_data('angles')
            property_dict = self.import_data('property') 
        #get structure objects for different space groups
        stru_seq = []
        for i, sg in enumerate(space_group):
            if sg == last_sg:
                stru = self.get_stru(atom_pos[i], atom_type[i], latt_vec, grid_ratio[i],
                                     grid_coords, sg, cluster_angles, property_dict, angles[i], thicks[i])
                stru_seq.append(stru)
            else:
                grid_coords = self.import_data('frac', grid, sg)
                stru = self.get_stru(atom_pos[i], atom_type[i], latt_vec, grid_ratio[i],
                                     grid_coords, sg, cluster_angles, property_dict, angles[i], thicks[i])
                stru_seq.append(stru)
                last_sg = sg
        return stru_seq

    def get_stru(self, atom_pos, atom_type, latt_vec, ratio, grid_coords, sg, 
                 cluster_angles, property_dict, angle, thick):
        """
        get structure object in pymatgen
        
        Parameters
        ----------
        atom_pos [int, 1d]: position of atoms
        atom_type [int, 1d]: type of atoms
        latt_vec [float, 2d, np]: lattice vector
        ratio [float, 0d]: grid ratio
        grid_coords [float, 2d, np]: fraction coordinates of grid
        sg [int, 0d]: space group number
        cluster_angles [float, 2d, np]: cluster rotations
        property_dict [dict, 2d, str:list]: property dictionary for new atoms
        angle [int, 1d]: cluster rotation angles
        thick [int, 1d]: atom displacement in z-direction
        
        Returns
        ----------
        stru [obj, 0d]: structure object in pymatgen
        """
        if Dimension == 2:
            latt = latt_vec*[[ratio], [ratio], [1]]
        else:
            latt = latt_vec*ratio
        coords = grid_coords[atom_pos]
        #whether get puckered structure
        if Cluster_Search:
            stru = self.generate_stru_cluster(sg, latt, atom_type, coords,
                                              cluster_angles, property_dict, angle, thick)
        elif General_Search and Dimension == 2 and Thickness > 0:
            stru = self.generate_stru_puckered(sg, latt, atom_type, coords, thick)
        else:
            stru = Structure.from_spacegroup(sg, latt, atom_type, coords)
        return stru
    
    def generate_stru_cluster(self, sg, latt, atom_type, coords, cluster_angles, property_dict, angle, thick):
        """
        generate cluster structure object
        
        Parameters
        ----------
        sg [int, 0d]: space group number
        latt [float, 2d, np]: lattice vector
        atom_type [int, 1d]: type of atoms
        coords [float, 2d, np]: fraction coordinates of atoms
        cluster_angles [float, 2d, np]: cluster rotations
        property_dict [dict, 2d, str:list]: property dictionary for new atoms
        angle [int, 1d]: cluster rotation angles
        thick [int, 1d]: atom displacement in z-direction
        
        Returns
        ----------
        stru [obj, 0d]: structure object in pymatgen
        """
        frac_thick = Thickness/Vacuum_Space
        all_coords, all_types = [], []
        spg = SpaceGroup.from_int_number(sg)
        for i, atom in enumerate(atom_type):
            equal_coords = spg.get_orbit(coords[i])
            carte_coords = np.dot(equal_coords, latt)
            equal_num = len(equal_coords)
            if atom > 0:
                equal_coords = carte_coords.tolist()
                all_coords += equal_coords
                all_types += [atom for _ in range(equal_num)]
            #replace atom with cluster
            else:
                properties = property_dict[str(atom)]
                cluster_coords = properties['coords']
                move_z = [0, 0, frac_thick*thick[i]/Z_Layers]
                alpha, beta, gamma = cluster_angles[angle[i]]
                cluster_coords = self.rotate_atom(alpha, beta, gamma, cluster_coords)
                for center in carte_coords:
                    tmp_coords = self.move_atom(center+move_z, cluster_coords)
                    all_coords += tmp_coords
                    all_types += properties['types']
        stru = Structure(latt, all_types, all_coords, coords_are_cartesian=True, to_unit_cell=True)
        return stru
    
    def generate_stru_puckered(self, sg, latt, atom_type, coords, thick):
        """
        generate puckered structure object
        
        Parameters
        ----------
        sg [int, 0d]: space group number
        latt [float, 2d, np]: lattice vector
        atom_type [int, 1d]: type of atoms
        coords [float, 2d, np]: fraction coordinates of atoms
        thick [int, 1d]: atom displacement in z-direction
        
        Returns
        ----------
        stru [obj, 0d]: structure object in pymatgen
        """
        atom_num = len(atom_type)
        frac_thick = Thickness/Vacuum_Space
        disturb = [[0, 0, .5+frac_thick*i/Z_Layers] for i in thick]
        #disturb atoms in z direction
        disturb_types, disturb_coords = [], []
        spg = SpaceGroup.from_int_number(sg)
        for i in range(atom_num):
            equal_coords = spg.get_orbit(coords[i])
            disturb_equal_coords = np.array(equal_coords) + disturb[i]
            disturb_coords += disturb_equal_coords.tolist()
            disturb_types += [atom_type[i] for _ in range(len(equal_coords))]
        stru = Structure(latt, disturb_types, disturb_coords)
        return stru
    

class MultiGridTransfer(Transfer):
    #positoin, type, symmetry should be sorted in grid and sg
    def __init__(self):
        Transfer.__init__(self)
    
    def sort_by_grid_sg(self, grid, sg):
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
        tmp = np.stack((idx, grid, sg), axis=1).tolist()
        order = sorted(tmp, key=lambda x:(x[1], x[2]))
        idx = np.array(order)[:, 0]
        return idx
    
    def sort_by_sg_energy(self, sg, energys):
        """
        sort pos, type, symm in order of space group and energy
        
        Parameters
        ----------
        sg [int, 1d]: space group number 
        energys [float, 1d]: structure energys
        
        Returns
        ----------
        idx [int, 1d, np]: index of grid-sg order
        """
        idx = np.arange(len(sg))
        tmp = np.stack((idx, sg, energys), axis=1).tolist()
        order = sorted(tmp, key=lambda x:(x[1], x[2]))
        idx = np.array(order)[:, 0]
        return np.array(idx, dtype=int)
    
    def sort_by_grid_sg_energy(self, grid, sg, energys):
        """
        sort pos, type, symm in order of grid and space group and energy
        
        Parameters
        ----------
        grid [int, 1d]: name of grid
        sg [int, 1d]: space group number 
        energys [float, 1d]: structure energys
        
        Returns
        ----------
        idx [int, 1d, np]: index of grid-sg order
        """
        idx = np.arange(len(sg))
        tmp = np.stack((idx, grid, sg, energys), axis=1).tolist()
        order = sorted(tmp, key=lambda x:(x[1], x[2], x[3]))
        idx = np.array(order)[:, 0]
        return np.array(idx, dtype=int)
    
    def get_gnn_input_batch_general(self, atom_pos, atom_type, grid_name, grid_ratio, space_group, limit=20):
        """
        get input of GNN in different grids in multi-cores
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        limit [int, 0d]: limit of cores
        
        Returns
        ----------
        atom_fea_bh [float, 3d]: batch atom feature
        nbr_fea_bh [float, 4d]: batch bond feature
        nbr_idx_bh [int, 3d]: batch near index
        """
        #multi-cores
        elem_embed = self.import_data('elem')
        cores = max(1, min(limit, int(.5*pythonmp.cpu_count())))
        with pythonmp.get_context('fork').Pool(processes=cores) as pool:
            #initialize
            last_grid = grid_name[0]
            i, atom_fea_bh, nbr_fea_bh, nbr_idx_bh = 0, [], [], []
            #get input of gnn under different grids
            args_list = []
            for j, grid in enumerate(grid_name):
                if not grid == last_grid:
                    #divide jobs
                    latt_vec = self.import_data('latt', grid=last_grid)
                    args_list = self.divide_jobs_gnn_general(args_list, i, j, atom_pos, atom_type,
                                                             last_grid, grid_ratio, space_group, latt_vec, elem_embed)
                    last_grid = grid
                    i = j
            #divide jobs
            end = len(grid_name)
            latt_vec = self.import_data('latt', grid=last_grid)
            args_list = self.divide_jobs_gnn_general(args_list, i, end, atom_pos, atom_type,
                                                     last_grid, grid_ratio, space_group, latt_vec, elem_embed)
            #put atoms into grid with symmetry constrain
            jobs = [pool.apply_async(self.get_gnn_input_seq_general, args) for args in args_list]
            pool.close()
            pool.join()
            #get results
            jobs_pool = [p.get() for p in jobs]
            for atom_fea_seq, nbr_fea_seq, nbr_idx_seq in jobs_pool:
                atom_fea_bh += atom_fea_seq
                nbr_fea_bh += nbr_fea_seq
                nbr_idx_bh += nbr_idx_seq
        pool.close()
        del pool
        return atom_fea_bh, nbr_fea_bh, nbr_idx_bh
    
    def divide_jobs_gnn_general(self, args_list, start, end,
                                atom_pos, atom_type, grid, grid_ratio, space_group, 
                                latt_vec, elem_embed, limit=20):
        """
        divide parallel transfer jobs into smaller size for each core

        Parameters
        ----------
        args_list [list:tuple, 1d]: parameters of parallel jobs
        start [int, 0d]: start index
        end [int, 0d]: end index
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid [int, 1d]: grid name
        grid_ratio [int, 1d]: grid ratio
        space_group [int, 1d]: space group
        latt_vec [float, 2d, np]: lattice vector
        elem_embed [int, 2d, np]: embedding of elements
        limit [int, 0d]: number of structure per job
        
        Returns
        ----------
        args_list [list:tuple, 1d]: parameters of parallel jobs
        """
        a = start
        counter = 0
        for b in range(start, end):
            counter += 1
            if counter > limit:
                args = (atom_pos[a:b], atom_type[a:b], grid, grid_ratio[a:b], space_group[a:b], latt_vec, elem_embed)
                args_list.append(args)
                a = b
                counter = 0
        args = (atom_pos[a:end], atom_type[a:end], grid, grid_ratio[a:end], space_group[a:end], latt_vec, elem_embed)
        args_list.append(args)
        return args_list
    
    def get_gnn_input_batch_template(self, atom_pos, atom_type, grid_name, grid_ratio, space_group, limit=20):
        """
        get input of GNN in different grids in multi-cores
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        limit [int, 0d]: limit of cores
        
        Returns
        ----------
        atom_fea_bh [float, 3d]: batch atom feature
        nbr_fea_bh [float, 4d]: batch bond feature
        nbr_fea_idx_bh [int, 3d]: batch near index
        """
        #multi-cores
        cores = min(limit, pythonmp.cpu_count())
        with pythonmp.get_context('fork').Pool(processes=cores) as pool:
            #initialize
            last_grid = grid_name[0]
            i, atom_fea_bh, nbr_fea_bh, nbr_fea_idx_bh = 0, [], [], []
            #get input of gnn under different grids
            args_list = []
            for j, grid in enumerate(grid_name):
                if not grid == last_grid:
                    #divide jobs
                    args_list = self.divide_jobs_gnn_template(args_list, i, j, atom_pos, atom_type,
                                                              last_grid, grid_ratio, space_group)
                    last_grid = grid
                    i = j
            #divide jobs
            end = len(grid_name)
            args_list = self.divide_jobs_gnn_template(args_list, i, end, atom_pos, atom_type,
                                                      last_grid, grid_ratio, space_group)
            #put atoms into grid with symmetry constrain
            jobs = [pool.apply_async(self.get_gnn_input_seq_template, args) for args in args_list]
            pool.close()
            pool.join()
            #get results
            jobs_pool = [p.get() for p in jobs]
            for atom_fea_seq, nbr_fea_seq, nbr_fea_idx_seq in jobs_pool:
                atom_fea_bh += atom_fea_seq
                nbr_fea_bh += nbr_fea_seq
                nbr_fea_idx_bh += nbr_fea_idx_seq
        pool.close()
        del pool
        return atom_fea_bh, nbr_fea_bh, nbr_fea_idx_bh
    
    def divide_jobs_gnn_template(self, args_list, start, end,
                                 atom_pos, atom_type, grid, grid_ratio, space_group, limit=20):
        """
        divide parallel transfer jobs into smaller size for each core

        Parameters
        ----------
        args_list [list:tuple, 1d]: parameters of parallel jobs
        start [int, 0d]: start index
        end [int, 0d]: end index
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid [int, 1d]: grid name
        grid_ratio [int, 1d]: grid ratio
        space_group [int, 1d]: space group
        limit [int, 0d]: number of structure per job

        Returns
        ----------
        args_list [list:tuple, 1d]: parameters of parallel jobs
        """
        a = start
        counter = 0
        for b in range(start, end):
            counter += 1
            if counter > limit:
                args = (atom_pos[a:b], atom_type[a:b], grid, grid_ratio[a:b], space_group[a:b])
                args_list.append(args)
                a = b
                counter = 0
        args = (atom_pos[a:end], atom_type[a:end], grid, grid_ratio[a:end], space_group[a:end])
        args_list.append(args)
        return args_list
    
    def get_stru_batch(self, atom_pos, atom_type, grid_name, grid_ratio, space_group, angles, thicks):
        """
        get strucutre object in different grids
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        
        Returns
        ----------
        stru_bh [obj, 1d]: batch structure objects
        """
        last_grid = grid_name[0]
        i, stru_bh = 0, []
        #get structure object in different grids
        for j, grid in enumerate(grid_name):
            if not last_grid == grid:
                latt_vec = self.import_data('latt', grid=last_grid)
                stru_seq = self.get_stru_seq(atom_pos[i:j], atom_type[i:j], 
                                             last_grid, grid_ratio[i:j], space_group[i:j], angles[i:j], thicks[i:j], latt_vec)
                stru_bh += stru_seq
                last_grid = grid
                i = j
        latt_vec = self.import_data('latt', grid=last_grid)
        stru_seq = self.get_stru_seq(atom_pos[i:], atom_type[i:],
                                     last_grid, grid_ratio[i:], space_group[i:], angles[i:], thicks[i:], latt_vec)
        stru_bh += stru_seq
        return stru_bh
    
    def get_stru_batch_parallel(self, atom_pos, atom_type, grid_name, grid_ratio, space_group, angles, thicks, limit=0):
        """
        get strucutre object in different grids
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        limit [int, 0d]: limit of core number
        
        Returns
        ----------
        stru_bh [obj, 1d]: batch structure objects
        """
        if limit == 0:
            cores = pythonmp.cpu_count()
        else:
            cores = min(limit, pythonmp.cpu_count())
        with pythonmp.get_context('fork').Pool(processes=cores) as pool:
            #initialize
            last_grid = grid_name[0]
            i, stru_bh = 0, []
            #get input of gnn under different grids
            args_list = []
            for j, grid in enumerate(grid_name):
                if not grid == last_grid:
                    #divide jobs
                    latt_vec = self.import_data('latt', grid=last_grid)
                    args_list = self.divide_jobs_stru_general(args_list, i, j, atom_pos, atom_type,
                                                              last_grid, grid_ratio, space_group, angles, thicks, latt_vec)
                    last_grid = grid
                    i = j
            #divide jobs
            end = len(grid_name)
            latt_vec = self.import_data('latt', grid=last_grid)
            args_list = self.divide_jobs_stru_general(args_list, i, end, atom_pos, atom_type,
                                                      last_grid, grid_ratio, space_group, angles, thicks, latt_vec)
            #put atoms into grid with symmetry constrain
            jobs = [pool.apply_async(self.get_stru_seq, args) for args in args_list]
            pool.close()
            pool.join()
            #get results
            jobs_pool = [p.get() for p in jobs]
            for stru_seq in jobs_pool:
                stru_bh += stru_seq
        pool.close()
        del pool
        return stru_bh
    
    def divide_jobs_stru_general(self, args_list, start, end,
                                 atom_pos, atom_type, grid, grid_ratio, space_group, angles, thicks,
                                 latt_vec, limit=100):
        """
        divide parallel transfer jobs into smaller size for each core

        Parameters
        ----------
        args_list [list:tuple, 1d]: parameters of parallel jobs
        start [int, 0d]: start index
        end [int, 0d]: end index
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid [int, 1d]: grid name
        grid_ratio [int, 1d]: grid ratio
        space_group [int, 1d]: space group
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        latt_vec [float, 2d, np]: lattice vector
        limit [int, 0d]: number of structure per job
        
        Returns
        ----------
        args_list [list:tuple, 1d]: parameters of parallel jobs
        """
        a = start
        counter = 0
        for b in range(start, end):
            counter += 1
            if counter > limit:
                args = (atom_pos[a:b], atom_type[a:b], grid, grid_ratio[a:b], space_group[a:b], angles[a:b], thicks[a:b], latt_vec)
                args_list.append(args)
                a = b
                counter = 0
        args = (atom_pos[a:end], atom_type[a:end], grid, grid_ratio[a:end], space_group[a:end], angles[a:end], thicks[a:end], latt_vec)
        args_list.append(args)
        return args_list
    
    
if __name__ == "__main__":
    pass