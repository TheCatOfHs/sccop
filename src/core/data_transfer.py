import os, sys
import numpy as np

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.utils import ListRWTools


class Transfer(ListRWTools):
    #transfer data in different form
    def __init__(self, nbr=12, dmin=0, dmax=8, step=0.2, var=0.2):
        self.nbr = nbr
        self.var = var
        self.filter = np.arange(dmin, dmax+step, step)
    
    def get_all_pos(self, min_pos, mapping):
        """
        get all equivalnent positions
        
        Parameters
        ----------
        min_pos [int, 1d]: position in minimum grid
        mapping [int, 2d]: mapping between min and all grid

        Returns
        ----------
        all_pos [int, 1d]: position in all grid
        """
        print(mapping)
        print(min_pos)
        all_pos = np.array(mapping, dtype=object)[min_pos]
        all_pos = np.concatenate(all_pos).tolist()
        return all_pos
    
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
        #build mask by atom position
        mask_len = grid_idx.shape[1]
        mask = np.full(mask_len, False)
        mask[atom_pos] = True
        point_idx = grid_idx[atom_pos]
        point_dis = grid_dis[atom_pos]
        #get neighbor distance
        nbr_dis = np.zeros((len(atom_pos), self.nbr), float)
        for i, (idx, dis) in enumerate(zip(point_idx, point_dis)):
            filter = mask[idx]
            nbr_dis[i,:] = np.compress(filter, dis, axis=0)[:self.nbr]
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
        #build mask by atom position
        mask_len = grid_idx.shape[1]
        mask = np.full(mask_len, False)
        mask[atom_pos] = True
        point_idx = grid_idx[atom_pos]
        point_dis = grid_dis[atom_pos]
        #get neighbor distance and index
        nbr_idx = np.zeros((len(atom_pos), self.nbr), int)
        nbr_dis = np.zeros((len(atom_pos), self.nbr), float)
        for i, (idx, dis) in enumerate(zip(point_idx, point_dis)):
            filter = mask[idx]
            nbr_idx[i,:] = np.compress(filter, idx, axis=0)[:self.nbr]
            nbr_dis[i,:] = np.compress(filter, dis, axis=0)[:self.nbr]
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
    
    def get_all_pos_seq(self, min_pos, grid):
        """
        get position in all grid under same grid
    
        Parameters
        ----------
        min_pos [int, 2d]: position in minimum grid
        grid [int, 0d]: name of grid

        Returns
        ----------
        all_pos_seq [int, 2d]: position in all grid
        """
        mapping = self.import_data('mapping', grid)
        all_pos_seq = [self.get_all_pos(pos, mapping) for pos in min_pos]
        return all_pos_seq
    
    def get_nbr_dis_seq(self, atom_pos, grid):
        """
        get neighbor distance under same grid
    
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        grid [int, 0d]: name of grid

        Returns
        ----------
        nbr_dis_bh [float, 3d]: neighbor distance
        """
        grid_idx, grid_dis = self.import_data('grid', grid)
        nbr_dis_seq = \
            [self.get_nbr_dis(pos, grid_idx, grid_dis) for pos in atom_pos]
        return nbr_dis_seq
    
    def get_ppm_input_seq(self, atom_pos, atom_type, grid):
        """
        get input of ppm under same grid
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid [int, 0d]: name of grid

        Returns
        ----------
        atom_fea_seq [float, 3d]: atom feature
        nbr_fea_seq [float, 4d]: bond feature
        nbr_fea_idx_seq [int, 3d]: near index
        """
        #import element embeddings and grid neighbors
        elem_embed = self.import_data('elem', grid)
        grid_idx, grid_dis = self.import_data('grid', grid)
        #transform pos, type into input of ppm
        atom_fea_seq, nbr_fea_seq, nbr_fea_idx_seq = [], [], []
        for pos, type in zip(atom_pos, atom_type):
            atom_fea, nbr_fea, nbr_fea_idx = \
                self.get_ppm_input(pos, type, elem_embed, grid_idx, grid_dis)
            atom_fea_seq.append(atom_fea)
            nbr_fea_seq.append(nbr_fea)
            nbr_fea_idx_seq.append(nbr_fea_idx)
        return atom_fea_seq, nbr_fea_seq, nbr_fea_idx_seq
    
    def import_data(self, task, grid):
        """
        import data according to task
        
        Parameters
        ----------
        task [str, 0d]: name of import data
        grid [int, 0d]: name of grid
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
                f'{head}_nbr_idx.bin', int, binary=True)
            grid_dis =  self.import_list2d(
                f'{head}_nbr_dis.bin', float, binary=True)
            return grid_idx, grid_dis
        #import mapping between minimum and all grid
        if task == 'mapping':
            mapping = self.import_list2d(
                f'{head}_mapping.bin', int, binary=True)
            return mapping

    
class MultiGridTransfer(Transfer):
    #positoin, type, symmetry should be sorted in grid order
    def __init__(self):
        Transfer.__init__(self)
    
    def get_all_pos_bh(self, min_pos, grid_name):
        """
        get position in all grid under different grids
        
        Parameters
        ----------
        min_pos [int, 2d]: position in minimum grid
        grid_name [int, 1d]: name of grid

        Returns
        ----------
        all_pos_bh [int, 2d]: batch position in all grid
        """
        last_grid = grid_name[0]
        i, all_pos_bh = 0, []
        #pos convert under different grids
        print(grid_name)
        for j, grid in enumerate(grid_name):
            if not grid == last_grid:
                all_pos_seq = self.get_all_pos_seq(min_pos[i:j], last_grid)
                all_pos_bh += all_pos_seq
                last_grid = grid
                i = j
        all_pos_seq = self.get_all_pos_seq(min_pos[i:], grid)
        all_pos_bh += all_pos_seq
        return all_pos_bh
    
    def get_nbr_dis_bh(self, atom_pos, grid_name):
        """
        calculate neighbor distance of structures
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms in all grid
        grid_name [int, 1d]: name of grids  

        Returns
        ----------
        nbr_dis_bh [float, 3d]: batch neighbor distance
        """
        last_grid = grid_name[0]
        i, nbr_dis_bh = 0, []
        #get neighbor distance under different grids
        for j, grid in enumerate(grid_name):
            if not grid == last_grid:
                nbr_dis_seq = self.get_nbr_dis_seq(atom_pos[i:], last_grid)
                nbr_dis_bh += nbr_dis_seq
                last_grid = grid
                i = j
        nbr_dis_seq = self.get_nbr_dis_seq(atom_pos[i:], grid)
        nbr_dis_bh += nbr_dis_seq
        return nbr_dis_bh
    
    def get_ppm_input_bh(self, atom_pos, atom_type, grid_name):
        """
        get input of ppm under different grids
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms in all grid
        atom_type [int, 2d]: type of atoms in all grid
        grid_name [int, 1d]: name of grids

        Returns
        ----------
        atom_fea_bh [float, 3d]: batch atom feature
        nbr_fea_bh [float, 4d]: batch bond feature
        nbr_fea_idx_bh [int, 3d]: batch near index
        """
        last_grid = grid_name[0]
        i, atom_fea_bh, nbr_fea_bh, nbr_fea_idx_bh = 0, [], [], []
        #get input of ppm under different grids
        for j, grid in enumerate(grid_name):
            if not grid == last_grid:
                atom_fea_seq, nbr_fea_seq, nbr_fea_idx_seq = \
                    self.get_ppm_input_seq(atom_pos[i:j], atom_type[i:j], last_grid)
                atom_fea_bh += atom_fea_seq
                nbr_fea_bh += nbr_fea_seq
                nbr_fea_idx_bh += nbr_fea_idx_seq
                last_grid = grid
                i = j
        atom_fea_seq, nbr_fea_seq, nbr_fea_idx_seq = \
            self.get_ppm_input_seq(atom_pos[i:], atom_type[i:], grid)
        atom_fea_bh += atom_fea_seq
        nbr_fea_bh += nbr_fea_seq
        nbr_fea_idx_bh += nbr_fea_idx_seq
        return atom_fea_bh, nbr_fea_bh, nbr_fea_idx_bh
        
        
if __name__ == "__main__":
    mul = MultiGridTransfer()