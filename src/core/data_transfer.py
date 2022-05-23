import os, sys
import numpy as np

from pymatgen.core.lattice import Lattice
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
    
    def get_nbr_dis(self, atom_pos, grid_idx, grid_dis):
        """
        neighbor distances are cutoff by 12 atoms
    
        Parameters
        ----------
        atom_pos [int, 1d]: position of atoms
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        
        Returns
        ----------
        nbr_dis [float, 2d, np]: neighbor distance of atoms
        """
        #get index and distance of points
        atom_pos = np.array(atom_pos)
        point_idx = grid_idx[atom_pos]
        point_dis = grid_dis[atom_pos]
        #initialize neighbor distance
        min_atom_num = len(atom_pos)
        nbr_dis = np.zeros((min_atom_num, self.nbr))
        for i, point in enumerate(point_idx):
            #find nearest atoms
            atom_idx = np.where(point==atom_pos[:,None])[-1]
            order = np.argsort(atom_idx)[:self.nbr]
            atom_idx = atom_idx[order]
            #get neighbor distance
            nbr_dis[i] = point_dis[i, atom_idx]
        return nbr_dis
    
    def get_nbr_fea(self, atom_pos, grid_idx, grid_dis):
        """
        neighbor bond features and index are cutoff by 12 atoms
        
        Parameters
        ----------
        atom_pos [int, 1d]: position of atoms
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
        #initialize neighbor index and distance
        min_atom_num = len(atom_pos)
        nbr_idx = np.zeros((min_atom_num, self.nbr))
        nbr_dis = np.zeros((min_atom_num, self.nbr))
        for i, point in enumerate(point_idx):
            #find nearest atoms
            atom_idx = np.where(point==atom_pos[:,None])[-1]
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

    def get_ppm_input(self, atom_pos, atom_type,
                      elem_embed, grid_idx, grid_dis):
        """
        get input of property prediction model
        
        Parameters
        ----------
        atom_pos [int, 1d]: position of atoms
        atom_type [int, 1d]: type of atoms
        elem_embed [int, 2d, np]: embedding of elements
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
            self.get_nbr_fea(atom_pos, grid_idx, grid_dis)
        return atom_fea, nbr_fea, nbr_fea_idx
        
    def get_stru(self, atom_pos, atom_type, latt, grid_coords, sg):
        """
        get structure object in pymatgen
        
        Parameters
        ----------
        atom_pos [int, 1d]: position of atoms
        atom_type [int, 1d]: type of atoms
        latt [obj, 0d]: lattice object in pymatgen
        grid_coords [float, 2d]: fraction coordinates of grid
        sg [int, 0d]: space group number

        Returns
        ----------
        stru [obj, 0d]: structure object in pymatgen
        """
        coords = grid_coords[atom_pos]
        stru = Structure.from_spacegroup(sg, latt, atom_type, coords)
        return stru
    
    def get_nbr_dis_seq(self, atom_pos, grid, space_group):
        """
        get neighbor distance under same grid
    
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        grid [int, 0d]: name of grid
        space_group [int, 1d]: space group number

        Returns
        ----------
        nbr_dis_bh [float, 3d]: neighbor distance
        """
        last_sg = space_group[0]
        grid_idx, grid_dis = self.import_data('grid', grid, last_sg)
        nbr_dis_seq = []
        #get neighbor distance under different space groups
        for i, sg in enumerate(space_group):
            if sg == last_sg:
                nbr_dis = self.get_nbr_dis(atom_pos[i], grid_idx, grid_dis)
                nbr_dis_seq.append(nbr_dis)
            else:
                grid_idx, grid_dis = self.import_data('grid', grid, sg)
                nbr_dis = self.get_nbr_dis(atom_pos[i], grid_idx, grid_dis)
                nbr_dis_seq.append(nbr_dis)
                last_sg = sg
        return nbr_dis_seq
    
    def get_ppm_input_seq(self, atom_pos, atom_type, grid, space_group):
        """
        get input of ppm under same grid
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid [int, 0d]: name of grid
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
                    self.get_ppm_input(atom_pos[i], atom_type[i],
                                       elem_embed, grid_idx, grid_dis)
                atom_fea_seq.append(atom_fea)
                nbr_fea_seq.append(nbr_fea)
                nbr_fea_idx_seq.append(nbr_fea_idx)
            else:
                grid_idx, grid_dis = self.import_data('grid', grid, sg)
                atom_fea, nbr_fea, nbr_fea_idx = \
                    self.get_ppm_input(atom_pos[i], atom_type[i],
                                       elem_embed, grid_idx, grid_dis)
                atom_fea_seq.append(atom_fea)
                nbr_fea_seq.append(nbr_fea)
                nbr_fea_idx_seq.append(nbr_fea_idx)
                last_sg = sg
        return atom_fea_seq, nbr_fea_seq, nbr_fea_idx_seq
    
    def get_stru_seq(self, atom_pos, atom_type, grid, space_group):
        """
        get strucutre object in same grid
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid [int, 0d]: grid number 
        space_group [int, 1d]: space group number

        Returns
        ----------
        stru_seq [obj, 1d]: structure object in pymatgen
        """
        last_sg = space_group[0]
        latt_vec = self.import_data('latt', grid, last_sg)
        grid_coords = self.import_data('frac', grid, last_sg)
        latt = Lattice(latt_vec)
        stru_seq = []
        #get structure objects under different space groups
        for i, sg in enumerate(space_group):
            if sg == last_sg:
                stru = self.get_stru(atom_pos[i], atom_type[i],
                                     latt, grid_coords, sg)
                stru_seq.append(stru)
            else:
                grid_coords = self.import_data('frac', grid, sg)
                stru = self.get_stru(atom_pos[i], atom_type[i],
                                     latt, grid_coords, sg)
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
    
    def sort_by_grid_and_sg(self,):
        pass
    
    def get_nbr_dis_bh(self, atom_pos, grid_name, space_group):
        """
        get neighbor distance in different grids
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        grid_name [int, 1d]: name of grids  
        space_group [int, 1d]: space group number
        
        Returns
        ----------
        nbr_dis_bh [float, 3d]: batch neighbor distance
        """
        #initialize
        last_grid = grid_name[0]
        i, nbr_dis_bh = 0, []
        #get neighbor distance under different grids
        for j, grid in enumerate(grid_name):
            if not grid == last_grid:
                nbr_dis_seq = self.get_nbr_dis_seq(atom_pos[i:j],
                                                   last_grid, space_group[i:j])
                nbr_dis_bh += nbr_dis_seq
                last_grid = grid
                i = j
        nbr_dis_seq = self.get_nbr_dis_seq(atom_pos[i:],
                                           grid, space_group[i:])
        nbr_dis_bh += nbr_dis_seq
        return nbr_dis_bh
    
    def get_ppm_input_bh(self, atom_pos, atom_type,
                         grid_name, space_group):
        """
        get input of ppm in different grids
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
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
                                           last_grid, space_group[i:j])
                atom_fea_bh += atom_fea_seq
                nbr_fea_bh += nbr_fea_seq
                nbr_fea_idx_bh += nbr_fea_idx_seq
                last_grid = grid
                i = j
        atom_fea_seq, nbr_fea_seq, nbr_fea_idx_seq = \
            self.get_ppm_input_seq(atom_pos[i:], atom_type[i:],
                                   grid, space_group[i:])
        atom_fea_bh += atom_fea_seq
        nbr_fea_bh += nbr_fea_seq
        nbr_fea_idx_bh += nbr_fea_idx_seq
        return atom_fea_bh, nbr_fea_bh, nbr_fea_idx_bh
    
    def get_stru_bh(self, atom_pos, atom_type, grid_name, space_group):
        """
        get strucutre object in different grids
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
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
                                             last_grid, space_group[i:j])
                stru_bh += stru_seq
                last_grid = grid
                i = j
        stru_seq = self.get_stru_seq(atom_pos[i:], atom_type[i:],
                                     grid, space_group[i:])
        stru_bh += stru_seq
        return stru_bh


class DeleteDuplicates(MultiGridTransfer):
    #delete same structures
    def __init__(self):
        MultiGridTransfer.__init__(self)
    
    def delete_duplicates(self, atom_pos, atom_type, atom_symm,
                          grid_name, space_group):
        """
        delete same structures by pos, type, symm, grid
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms 
        grid_name [int, 1d]: grid of atoms
        space_group [int, 1d]: space group number
        
        Returns
        ----------
        idx [int, 1d, np]: index of unique samples 
        """
        #different length of atoms convert to string
        pos_str = self.list2d_to_str(atom_pos, '{0}')
        type_str = self.list2d_to_str(atom_type, '{0}')
        symm_str = self.list2d_to_str(atom_symm, '{0}')
        grid_str = self.list1d_to_str(grid_name, '{0}')
        group_str = self.list1d_to_str(space_group, '{0}')
        label = [i+'-'+j+'-'+k+'-'+m+'-'+n for i, j, k, m, n in 
                 zip(pos_str, type_str, symm_str, grid_str, group_str)]
        #delete same structure accroding to pos, type and grid
        _, idx = np.unique(label, return_index=True)
        return idx
    
    def delete_duplicates_pymatgen(self, atom_pos, atom_type, 
                                   grid_name, space_group):
        """
        delete same structures
        
        Parameters
        -----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: grid of atoms
        space_group [int, 1d]: space group number
        
        Returns
        ----------
        idx [int, 1d, np]: index of different poscars
        """
        strus = self.get_stru_bh(atom_pos, atom_type, grid_name, space_group)
        strus_num = len(strus)
        idx = []
        #compare structure by pymatgen
        for i in range(strus_num):
            stru_1 = strus[i]
            for j in range(i+1, strus_num):
                stru_2 = strus[j]
                same = stru_1.matches(stru_2, ltol=0.1, stol=0.15, angle_tol=5, 
                                      primitive_cell=True, scale=False, 
                                      attempt_supercell=False, allow_subset=False)
                if same:
                    idx.append(i)
                    break
        all_idx = np.arange(strus_num)
        idx = np.setdiff1d(all_idx, idx)
        return idx
    
    def delete_same_poscars(self, path, poscars):
        """
        delete same structures
        
        Parameters
        -----------
        path [str, 0d]: path of poscars
        poscars [str, 1d]: name of poscars
        
        Returns
        ----------
        index [int, 1d, np]: index of different poscars
        """
        poscars_num = len(poscars)
        same_poscars = []
        for i in range(poscars_num):
            stru_1 = Structure.from_file(f'{path}/{poscars[i]}')
            for j in range(i+1, poscars_num):
                stru_2 = Structure.from_file(f'{path}/{poscars[j]}')
                same = stru_1.matches(stru_2, ltol=0.1, stol=0.15, angle_tol=5, 
                                      primitive_cell=True, scale=False, 
                                      attempt_supercell=False, allow_subset=False)
                if same:
                    same_poscars.append(i)
        same_poscars = np.unique(same_poscars)
        all_index = [i for i in range(poscars_num)]
        index = np.setdiff1d(all_index, same_poscars)
        return index, same_poscars
    
    def compare_poscars(self, poscar_1, poscar_2):
        """
        find common structures in poscar_1 
        
        Parameters
        ----------
        poscar_1 [str, 1d]: name of poscars
        poscar_2 [str, 1d]: name of poscars

        Returns
        ----------
        same_index [int, 1d]: index of common structures in poscar_1
        """
        same_index = []
        for index, i in enumerate(poscar_1):
            stru_1 = Structure.from_file(i)
            for j in poscar_2:
                stru_2 = Structure.from_file(j)
                same = stru_1.matches(stru_2, ltol=0.1, stol=0.15, angle_tol=5, 
                                      primitive_cell=True, scale=False, 
                                      attempt_supercell=False, allow_subset=False)
                if same:
                    same_index.append(index)
                    break
        return same_index

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
                same = stru_1.matches(stru_2, ltol=0.1, stol=0.15, angle_tol=5, 
                                      primitive_cell=True, scale=False, 
                                      attempt_supercell=False, allow_subset=False)
                if same:
                    same_poscars.append(poscars[i])
        same_poscars = np.unique(same_poscars)
        for i in same_poscars:
            os.remove(f'{path}/{i}')
        same_poscars_num = len(same_poscars)
        system_echo(f'Delete same structures: {same_poscars_num}')
    
    def filter_samples(self, idx, atom_pos, atom_type,
                       atom_symm, grid_name, space_group):
        """
        filter samples by index
        
        Parameters
        ----------
        idx [int, 1d]: index of select samples or binary mask
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        space_group [int, 1d]: space group number
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms after constrain
        atom_type [int, 2d]: type of atoms after constrain
        atom_symm [int, 2d]: symmetry of atoms after constrain
        grid_name [int, 1d]: name of grids after constrain
        space_group [int, 1d]: space group number after constrain
        """
        atom_pos = np.array(atom_pos, dtype=object)[idx].tolist()
        atom_type = np.array(atom_type, dtype=object)[idx].tolist()
        atom_symm = np.array(atom_symm, dtype=object)[idx].tolist()
        grid_name = np.array(grid_name, dtype=object)[idx].tolist()
        space_group = np.array(space_group, dtype=object)[idx].tolist()
        return atom_pos, atom_type, atom_symm, grid_name, space_group
    
    
if __name__ == "__main__":
    mul = MultiGridTransfer()