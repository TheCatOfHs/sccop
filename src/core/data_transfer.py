import os, sys
import numpy as np

from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.utils import ListRWTools, system_echo


class Transfer(ListRWTools):
    #transfer data in different form
    def __init__(self, nbr=12, dmin=0, dmax=8, step=0.2, var=0.2):
        self.nbr = nbr
        self.var = var
        self.filter = np.arange(dmin, dmax+step, step)
    
    def get_atom_fea(self, atom_type, elem_embed):
        """
        initialize atom feature vectors

        Parameters
        ----------
        atom_type [int, 1d]: type of atoms
        elem_embed [int, 2d, np]: embedding of elements
        
        Returns
        ----------
        atom_fea [int, 2d, np]: atom feature vectors
        """
        atom_fea = elem_embed[atom_type]
        return atom_fea
    
    def get_nbr_fea(self, atom_pos, ratio, grid_idx, grid_dis):
        """
        neighbor bond features and index are cutoff by 12 atoms
        
        Parameters
        ----------
        atom_pos [int, 1d]: position of atoms
        ratio [float, 0d]: grid ratio
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        
        Returns
        ----------
        nbr_fea [float, 3d, np]: neighbor feature of atoms
        nbr_fea_idx [int, 2d, np]: neighbor index of atoms
        """
        #get index and distance of points
        atom_pos = np.array(atom_pos)
        point_idx = grid_idx[atom_pos]
        point_dis = grid_dis[atom_pos]
        point_dis *= ratio
        #initialize neighbor index and distance
        min_atom_num = len(atom_pos)
        nbr_idx = np.zeros((min_atom_num, self.nbr))
        nbr_dis = np.zeros((min_atom_num, self.nbr))
        for i, point in enumerate(point_idx):
            #find nearest atoms
            atom_idx = np.where(point==atom_pos[:, None])[-1]
            order = np.argsort(atom_idx)[:self.nbr]
            atom_idx = atom_idx[order]
            #get neighbor index and distancs
            nbr_idx[i] = point_idx[i, atom_idx]
            nbr_dis[i] = point_dis[i, atom_idx]
        #get bond features
        nbr_fea = self.expand(nbr_dis)
        nbr_fea_idx = self.idx_transfer(atom_pos, nbr_idx)
        return nbr_fea, nbr_fea_idx
    
    def idx_transfer(self, atom_pos, nbr_idx):
        """
        make nbr_idx consistent with nbr_dis
    
        Parameters
        ----------
        atom_pos [int, 1d]: position of atoms
        nbr_idx [int, 2d, np]: near index of atoms
        
        Returns
        ----------
        nbr_idx [int, 2d, np]: index start from 0
        """
        for i, idx in enumerate(atom_pos):
            nbr_idx[nbr_idx==idx] = i
        return nbr_idx
    
    def expand(self, distances):
        """
        near distance expanded in gaussian feature space
        
        Parameters
        ----------
        distances [float, 2d, np]: distance of near neighbors
        
        Returns
        ----------
        nbr_fea [float, 3d, np]: gaussian feature vector
        """
        return np.exp(-(distances[:, :, np.newaxis] - self.filter)**2 /
                      self.var**2)

    def get_ppm_input(self, atom_pos, atom_type, elem_embed,
                      ratio, grid_idx, grid_dis):
        """
        get input of property prediction model
        
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
        nbr_fea_idx [float, 2d, np]: neighbor index
        """
        #get atom features
        atom_fea = self.get_atom_fea(atom_type, elem_embed)
        #get bond features and neighbor index
        nbr_fea, nbr_fea_idx = \
            self.get_nbr_fea(atom_pos, ratio, grid_idx, grid_dis)
        return atom_fea, nbr_fea, nbr_fea_idx
        
    def get_stru(self, atom_pos, atom_type, latt_vec, ratio, grid_coords, sg):
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

        Returns
        ----------
        stru [obj, 0d]: structure object in pymatgen
        """
        if add_vacuum:
            latt = latt_vec*[[ratio], [ratio], [1]]
        else:
            latt = latt_vec*ratio
        coords = grid_coords[atom_pos]
        stru = Structure.from_spacegroup(sg, latt, atom_type, coords)
        return stru
    
    def get_ppm_input_seq(self, atom_pos, atom_type, grid, grid_ratio, space_group):
        """
        get input of ppm under same grid
        
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
        #import element embeddings and neighbor in all grid
        last_sg = space_group[0]
        elem_embed = self.import_data('elem', grid, last_sg)
        grid_idx, grid_dis = self.import_data('grid', grid, last_sg)
        atom_fea_seq, nbr_fea_seq, nbr_fea_idx_seq = [], [], []
        #transform pos, type into input of ppm under different space groups
        for i, sg in enumerate(space_group):
            if sg == last_sg:
                atom_fea, nbr_fea, nbr_fea_idx = \
                    self.get_ppm_input(atom_pos[i], atom_type[i], elem_embed,
                                       grid_ratio[i], grid_idx, grid_dis)
                atom_fea_seq.append(atom_fea)
                nbr_fea_seq.append(nbr_fea)
                nbr_fea_idx_seq.append(nbr_fea_idx)
            else:
                grid_idx, grid_dis = self.import_data('grid', grid, sg)
                atom_fea, nbr_fea, nbr_fea_idx = \
                    self.get_ppm_input(atom_pos[i], atom_type[i], elem_embed,
                                       grid_ratio[i], grid_idx, grid_dis)
                atom_fea_seq.append(atom_fea)
                nbr_fea_seq.append(nbr_fea)
                nbr_fea_idx_seq.append(nbr_fea_idx)
                last_sg = sg
        return atom_fea_seq, nbr_fea_seq, nbr_fea_idx_seq
    
    def get_stru_seq(self, atom_pos, atom_type, grid, grid_ratio, space_group):
        """
        get strucutre object in same lattice
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid [int, 0d]: grid number 
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number

        Returns
        ----------
        stru_seq [obj, 1d]: structure object in pymatgen
        """
        last_sg = space_group[0]
        latt_vec = self.import_data('latt', grid, last_sg)
        grid_coords = self.import_data('frac', grid, last_sg)
        stru_seq = []
        #get structure objects under different space groups
        for i, sg in enumerate(space_group):
            if sg == last_sg:
                stru = self.get_stru(atom_pos[i], atom_type[i],
                                     latt_vec, grid_ratio[i], grid_coords, sg)
                stru_seq.append(stru)
            else:
                grid_coords = self.import_data('frac', grid, sg)
                stru = self.get_stru(atom_pos[i], atom_type[i],
                                     latt_vec, grid_ratio[i], grid_coords, sg)
                stru_seq.append(stru)
                last_sg = sg
        return stru_seq
    
    def import_data(self, task, grid, sg):
        """
        import data according to task
        
        Parameters
        ----------
        task [str, 0d]: name of import data
        grid [int, 0d]: name of grid
        sg [int, 0d]: space group number
        """
        head = f'{grid_path}/{grid:03.0f}'
        #import element embedding file
        if task == 'elem':
            elem_embed = self.import_list2d(
                atom_init_file, int, numpy=True)
            return elem_embed
        #import grid index and distance file
        if task == 'grid':
            grid_idx = self.import_list2d(
                f'{head}_nbr_idx_{sg}.bin', int, binary=True)
            grid_dis =  self.import_list2d(
                f'{head}_nbr_dis_{sg}.bin', float, binary=True)
            return grid_idx, grid_dis
        #import mapping relationship
        if task == 'mapping':
            mapping = self.import_list2d(
                f'{head}_mapping_{sg}.bin', int, binary=True)
            return mapping
        #import lattice vector
        if task == 'latt':
            latt_vec = self.import_list2d(
                f'{head}_latt_vec.bin', float, binary=True)
            return latt_vec
        #import fraction coordinates of grid
        if task == 'frac':
            grid_coords = self.import_list2d(
                f'{head}_frac_coords_{sg}.bin', float, binary=True)
            return grid_coords
    
    
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
        grid_sg = np.stack((idx, grid, sg), axis=1).tolist()
        order = sorted(grid_sg, key=lambda x:(x[1], x[2]))
        idx = np.array(order)[:,0]
        return idx
    
    def get_ppm_input_bh(self, atom_pos, atom_type,
                         grid_name, grid_ratio, space_group):
        """
        get input of ppm in different grids
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        
        Returns
        ----------
        atom_fea_bh [float, 3d]: batch atom feature
        nbr_fea_bh [float, 4d]: batch bond feature
        nbr_fea_idx_bh [int, 3d]: batch near index
        """
        #initialize
        last_grid = grid_name[0]
        i, atom_fea_bh, nbr_fea_bh, nbr_fea_idx_bh = 0, [], [], []
        #get input of ppm under different grids
        for j, grid in enumerate(grid_name):
            if not grid == last_grid:
                atom_fea_seq, nbr_fea_seq, nbr_fea_idx_seq = \
                    self.get_ppm_input_seq(atom_pos[i:j], atom_type[i:j],
                                           last_grid, grid_ratio[i:j], space_group[i:j])
                atom_fea_bh += atom_fea_seq
                nbr_fea_bh += nbr_fea_seq
                nbr_fea_idx_bh += nbr_fea_idx_seq
                last_grid = grid
                i = j
        atom_fea_seq, nbr_fea_seq, nbr_fea_idx_seq = \
            self.get_ppm_input_seq(atom_pos[i:], atom_type[i:],
                                   grid, grid_ratio[i:], space_group[i:])
        atom_fea_bh += atom_fea_seq
        nbr_fea_bh += nbr_fea_seq
        nbr_fea_idx_bh += nbr_fea_idx_seq
        return atom_fea_bh, nbr_fea_bh, nbr_fea_idx_bh
    
    def get_stru_bh(self, atom_pos, atom_type, grid_name, grid_ratio, space_group):
        """
        get strucutre object in different grids
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        
        Returns
        ----------
        stru_bh [obj, 1d]: batch structure objects
        """
        last_grid = grid_name[0]
        i, stru_bh = 0, []
        #get structure object in different grids
        for j, grid in enumerate(grid_name):
            if not last_grid == grid:
                stru_seq = self.get_stru_seq(atom_pos[i:j], atom_type[i:j], 
                                             last_grid, grid_ratio[i:j], space_group[i:j])
                stru_bh += stru_seq
                last_grid = grid
                i = j
        stru_seq = self.get_stru_seq(atom_pos[i:], atom_type[i:],
                                     grid, grid_ratio[i:], space_group[i:])
        stru_bh += stru_seq
        return stru_bh


class DeleteDuplicates(MultiGridTransfer):
    #delete same structures
    def __init__(self):
        MultiGridTransfer.__init__(self)
    
    def delete_duplicates(self, atom_pos, atom_type,
                          grid_name, grid_ratio, space_group):
        """
        delete same structures by pos, type, symm, grid
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: grid of atoms
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        
        Returns
        ----------
        idx [int, 1d, np]: index of unique samples 
        """
        #convert to string list
        pos_str = self.list2d_to_str(atom_pos, '{0}')
        type_str = self.list2d_to_str(atom_type, '{0}')
        grid_str = self.list1d_to_str(grid_name, '{0}')
        ratio_str = self.list1d_to_str(grid_ratio, '{0:4.2f}')
        group_str = self.list1d_to_str(space_group, '{0}')
        label = [i+'-'+j+'-'+k+'-'+m+'-'+n for i, j, k, m, n in 
                 zip(pos_str, type_str, grid_str, ratio_str, group_str)]
        #delete same structure
        _, idx = np.unique(label, return_index=True)
        return idx

    def delete_duplicates_pymatgen(self, atom_pos, atom_type, 
                                   grid_name, grid_ratio, space_group):
        """
        delete same structures
        
        Parameters
        -----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: grid of atoms
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        
        Returns
        ----------
        idx [int, 1d, np]: index of different structures
        """
        strus = self.get_stru_bh(atom_pos, atom_type, grid_name, grid_ratio, space_group)
        strus_num = len(strus)
        strus_idx = np.arange(strus_num)
        idx, store, delet = [], [], []
        #compare structure by pymatgen
        while True:
            i = strus_idx[0]
            stru_1 = strus[i]
            for k in range(1, len(strus_idx)):
                j = strus_idx[k]
                stru_2 = strus[j]
                same = stru_1.matches(stru_2, ltol=0.2, stol=0.3, angle_tol=5, 
                                      primitive_cell=True, scale=False, 
                                      attempt_supercell=False, allow_subset=False)
                if same:
                    store.append(j)
                    delet.append(k)
            #update
            idx += [0] + store[:-1]
            strus_idx = np.delete(strus_idx, [0]+delet)
            store, delet = [], []
            if len(strus_idx) == 0:
                break
        all_idx = np.arange(strus_num)
        idx = np.setdiff1d(all_idx, idx)
        return idx
    
    def delete_duplicates_sg_pymatgen(self, atom_pos, atom_type, 
                                      grid_name, grid_ratio, space_group):
        """
        delete same structures in same space group
        
        Parameters
        -----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: grid of atoms
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        
        Returns
        ----------
        idx [int, 1d, np]: index of different structures
        """
        last_sg = space_group[0]
        i, idx = 0, []
        #delete in same space group
        for j, sg in enumerate(space_group):
            if not last_sg == sg:
                unique_idx = self.delete_duplicates_pymatgen(atom_pos[i:j], atom_type[i:j], 
                                                             grid_name[i:j], grid_ratio[i:j],
                                                             space_group[i:j])
                unique_idx = [i+k for k in unique_idx]
                idx += unique_idx
                last_sg = sg
                i = j
        unique_idx = self.delete_duplicates_pymatgen(atom_pos[i:], atom_type[i:],
                                                     grid_name[i:], grid_ratio[i:],
                                                     space_group[i:])
        unique_idx = [i+k for k in unique_idx]
        idx += unique_idx
        return idx
    
    def delete_duplicates_between_sg_pymatgen(self, atom_pos, atom_type, 
                                              grid_name, grid_ratio, space_group):
        """
        delete same structures in different space groups
        
        Parameters
        -----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: grid of atoms
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        
        Returns
        ----------
        idx [int, 1d, np]: index of different structures
        """
        strus = self.get_stru_bh(atom_pos, atom_type, grid_name, grid_ratio, space_group)
        strus_num = len(strus)
        strus_idx = np.arange(strus_num)
        idx, store, delet = [], [], []
        #compare structure by pymatgen
        while True:
            i = strus_idx[0]
            stru_1 = strus[i]
            #get start index
            sg = space_group[i]
            start, end = 0, len(strus_idx)
            for k in range(1, end):
                j = strus_idx[k]
                if sg != space_group[j]:
                    start = k
                    break  
            if start == 0:
                start = end
            #compare structures in different space groups
            for k in range(start, end):
                j = strus_idx[k]
                stru_2 = strus[j]
                same = stru_1.matches(stru_2, ltol=0.2, stol=0.3, angle_tol=5, 
                                      primitive_cell=True, scale=False, 
                                      attempt_supercell=False, allow_subset=False)
                if same:
                    store.append(j)
                    delet.append(k)
            #update
            idx += [0] + store[:-1]
            strus_idx = np.delete(strus_idx, [0]+delet)
            store, delet = [], []
            if len(strus_idx) == 0:
                break
        all_idx = np.arange(strus_num)
        idx = np.setdiff1d(all_idx, idx)
        return idx
    
    def delete_same_selected(self, pos_1, type_1, grid_1, ratio_1, sg_1,
                             pos_2, type_2, grid_2, ratio_2, sg_2):
        """
        delete common structures of set1 and set2
        return unique index of set1
        
        Parameters
        ----------
        pos_1 [int, 2d]: position of atoms in set1
        type_1 [int, 2d]: type of atoms in set1
        grid_1 [int, 1d]: name of grids in set1
        ratio_1 [float, 1d]: ratio of grids in set1
        sg_1 [int, 2d]: space group number in set1
        pos_2 [int, 2d]: position of atoms in set2
        type_2 [int, 2d]: type of atoms in set2
        grid_2 [int, 1d]: name of grids in set2
        ratio_2 [float, 1d]: ratio of grids in set2
        sg_2 [int, 1d]: space group number in set2
        
        Returns
        ----------
        idx [int, 1d]: index of sample in set 1
        """
        #convert to string list
        pos_str_1 = self.list2d_to_str(pos_1, '{0}')
        type_str_1 = self.list2d_to_str(type_1, '{0}')
        grid_str_1 = self.list1d_to_str(grid_1, '{0}')
        ratio_str_1 = self.list1d_to_str(ratio_1, '{0:4.2f}')
        sg_str_1 = self.list1d_to_str(sg_1, '{0}')
        pos_str_2 = self.list2d_to_str(pos_2, '{0}')
        type_str_2 = self.list2d_to_str(type_2, '{0}')
        grid_str_2 = self.list1d_to_str(grid_2, '{0}')
        ratio_str_2 = self.list1d_to_str(ratio_2, '{0:4.2f}')
        sg_str_2 = self.list1d_to_str(sg_2, '{0}')
        #find unique structures
        array_1 = [i+'-'+j+'-'+k+'-'+m+'-'+n for i, j, k, m, n in 
                   zip(pos_str_1, type_str_1, grid_str_1, ratio_str_1, sg_str_1)]
        array_2 = [i+'-'+j+'-'+k+'-'+m+'-'+n for i, j, k, m, n in 
                   zip(pos_str_2, type_str_2, grid_str_2, ratio_str_2, sg_str_2)]
        array = np.concatenate((array_1, array_2))
        _, idx, counts = np.unique(array, return_index=True, return_counts=True)
        #delete structures same as training set
        same = []
        for i, repeat in enumerate(counts):
            if repeat > 1:
                same.append(i)
        num = len(grid_1)
        idx = np.delete(idx, same)
        idx = [i for i in idx if i < num]
        return idx

    def delete_same_selected_pymatgen(self, strus_1, strus_2):
        """
        delete common structures of set1 and set2 by pymatgen
        return unique index of set1
        
        Parameters
        ----------
        strus_1 [obj, 1d]: structure objects in set1
        strus_2 [obj, 1d]: structure objects in set2

        Returns
        ----------
        idx [int, 1d]: index of sample in set 1
        """
        idx = []
        for i, stru_1 in enumerate(strus_1):
            for stru_2 in strus_2:
                same = stru_1.matches(stru_2, ltol=0.2, stol=0.3, angle_tol=5, 
                                      primitive_cell=True, scale=False, 
                                      attempt_supercell=False, allow_subset=False)
                if same:
                    idx.append(i)
                    break
        all_idx = np.arange(len(strus_1))
        idx = np.setdiff1d(all_idx, idx)
        return idx
    
    def delete_same_poscars(self, path):
        """
        delete same structures
        
        Parameters
        -----------
        path [str, 0d]: path of poscars
        """
        poscars = sorted(os.listdir(path))
        poscars_num = len(poscars)
        same_poscars = []
        for i in range(poscars_num):
            stru_1 = Structure.from_file(f'{path}/{poscars[i]}')
            for j in range(i+1, poscars_num):
                stru_2 = Structure.from_file(f'{path}/{poscars[j]}')
                same = stru_1.matches(stru_2, ltol=0.2, stol=0.3, angle_tol=5, 
                                      primitive_cell=True, scale=False, 
                                      attempt_supercell=False, allow_subset=False)
                if same:
                    same_poscars.append(poscars[i])
                    break
        for i in same_poscars:
            os.remove(f'{path}/{i}')
        same_poscars_num = len(same_poscars)
        system_echo(f'Delete same structures: {same_poscars_num}')
    
    def delete_same_strus_energy(self, strus, energys):
        """
        delete same structures and retain structure with lower energy
        
        Parameters
        ----------
        strus [obj, 1d]: structure objects in pymatgen
        energys [float, 1d]: corresponding energy
        
        Returns
        ----------
        idx [int, 1d, np]: index of different structures
        """
        strus_num = len(strus)
        idx = []
        #compare structure by pymatgen
        for i in range(strus_num):
            stru_1 = strus[i]
            for j in range(i+1, strus_num):
                stru_2 = strus[j]
                same = stru_1.matches(stru_2, ltol=0.2, stol=0.3, angle_tol=5, 
                                      primitive_cell=True, scale=False, 
                                      attempt_supercell=False, allow_subset=False)
                if same:
                    if energys[i] < energys[j]:
                        idx.append(j)
                    else:
                        idx.append(i)
        all_idx = np.arange(strus_num)
        idx = np.unique(idx)
        idx = np.setdiff1d(all_idx, idx)
        return idx
    
    def filter_samples(self, idx, atom_pos, atom_type, atom_symm, 
                       grid_name, grid_ratio, space_group):
        """
        filter samples by index
        
        Parameters
        ----------
        idx [int, 1d]: index of select samples or binary mask
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms after constrain
        atom_type [int, 2d]: type of atoms after constrain
        atom_symm [int, 2d]: symmetry of atoms after constrain
        grid_name [int, 1d]: name of grids after constrain
        grid_ratio [float, 1d]: ratio of grids after constrain
        space_group [int, 1d]: space group number after constrain
        """
        atom_pos = np.array(atom_pos, dtype=object)[idx].tolist()
        atom_type = np.array(atom_type, dtype=object)[idx].tolist()
        atom_symm = np.array(atom_symm, dtype=object)[idx].tolist()
        grid_name = np.array(grid_name, dtype=object)[idx].tolist()
        grid_ratio = np.array(grid_ratio, dtype=object)[idx].tolist()
        space_group = np.array(space_group, dtype=object)[idx].tolist()
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group

    
if __name__ == "__main__":
    pass