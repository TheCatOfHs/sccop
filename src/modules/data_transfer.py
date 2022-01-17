import os, sys
import numpy as np

sys.path.append(f'{os.getcwd()}/src')
from modules.global_var import *
from modules.utils import ListRWTools


class Transfer(ListRWTools):
    #transfer atoms_pos and atoms_type to the input of gragh network
    #all configurations with same grid
    def __init__(self, grid_name, 
                 nbr=12, dmin=0, dmax=8, step=0.2, var=0.2):
        """
        Parameters
        ----------
        grid_name [int, 0d]: name of grid 
        """
        self.nbr, self.var = nbr, var
        file_prefix = f'{grid_prop_path}/{grid_name:03.0f}'
        self.elem_embed = self.import_list2d(
            atom_init_file, int, numpy=True)
        self.latt_vec = self.import_list2d(
            f'{file_prefix}_latt_vec.bin', float, binary=True)
        self.frac_coor = self.import_list2d(
            f'{file_prefix}_frac_coor.bin', float, binary=True)
        self.grid_nbr_idx = self.import_list2d(
            f'{file_prefix}_nbr_idx.bin', int, binary=True)
        self.grid_nbr_dis = self.import_list2d(
            f'{file_prefix}_nbr_dis.bin', float, binary=True)
        self.filter = np.arange(dmin, dmax+step, step)
        self.grid_point_num = len(self.grid_nbr_dis)
        self.grid_point_array = np.arange(self.grid_point_num)
        
    def atom_initializer(self, atom_type):
        """
        initialize atom features

        Parameters
        ----------
        atom_type [int, 1d]: type of atoms
        
        Returns
        ----------
        atom_fea [int, 2d, np]: atom feature
        """
        return self.elem_embed[atom_type]

    def find_nbr_dis(self, atom_pos):
        """
        cutoff by n neighboring atoms
    
        Parameters
        ----------
        atom_pos [int, 1d]: atom list
        
        Returns
        ----------
        nbr_dis [float, 2d, np]: neighbor distance 
        """
        mask_len = self.grid_nbr_idx.shape[1]
        mask = np.full((mask_len,), False)
        mask[atom_pos] = True
        point_nbr_idx = self.grid_nbr_idx[atom_pos]
        point_nbr_dis = self.grid_nbr_dis[atom_pos]
        nbr_dis = np.zeros((len(atom_pos), self.nbr), float)
        for i, (idx, dis) in enumerate(zip(point_nbr_idx, point_nbr_dis)):
            filter = mask[idx]
            nbr_dis[i,:] = np.compress(filter, dis, axis=0)[:self.nbr]
        return nbr_dis
    
    def find_nbr(self, atom_pos):
        """
        cutoff by n neighboring atoms
    
        Parameters
        ----------
        atom_pos [int, 1d]: atom list
        
        Returns
        ----------
        nbr_fea [float, 3d, np]: neighbor feature of each atom
        nbr_fea_idx [int, 2d, np]: neighbor index of each atom
        """
        mask_len = self.grid_nbr_idx.shape[1]
        mask = np.full((mask_len,), False)
        mask[atom_pos] = True
        point_nbr_idx = self.grid_nbr_idx[atom_pos]
        point_nbr_dis = self.grid_nbr_dis[atom_pos]
        nbr_idx = np.zeros((len(atom_pos), self.nbr), int)
        nbr_dis = np.zeros((len(atom_pos), self.nbr), float)
        for i, (idx, dis) in enumerate(zip(point_nbr_idx, point_nbr_dis)):
            filter = mask[idx]
            nbr_idx[i,:] = np.compress(filter, idx, axis=0)[:self.nbr]
            nbr_dis[i,:] = np.compress(filter, dis, axis=0)[:self.nbr]
        nbr_fea = self.expand(nbr_dis)
        nbr_fea_idx = self.idx_transfer(atom_pos, nbr_idx)
        return nbr_fea, nbr_fea_idx
    
    def idx_transfer(self, atom_pos, nbr_idx):
        """
        make nbr_idx consistent with nbr_dis
    
        Parameters
        ----------
        atom_pos [int, 1d]: atom list
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

    def single(self, atom_pos, atom_type):
        """
        transfer configurations by single
                
        Parameters
        ----------
        atom_pos [int, 1d]: position of atoms in grid
        atom_type [int, 1d]: type of atoms in grid
        
        Returns
        ----------
        atom_fea [int, 2d, np]: atom feature
        nbr_fea [float, 3d, np]: neighbor feature
        nbr_fea_idx [float, 2d, np]: neighbor index
        """
        atom_fea = self.atom_initializer(atom_type)
        nbr_fea, nbr_fea_idx = self.find_nbr(atom_pos)
        return atom_fea, nbr_fea, \
                nbr_fea_idx
    
    def batch(self, atom_pos, atom_type):
        """
        transfer configurations by batch
                
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms in grid
        atom_type [int, 2d]: type of atoms in grid
        
        Returns
        ----------
        batch_atom_fea [int, 3d]: batch atom feature
        batch_nbr_fea [float, 4d]: batch neighbor feature
        batch_nbr_fea_idx [float, 3d]: batch neighbor index
        """
        batch_atom_fea, batch_nbr_fea, \
            batch_nbr_fea_idx = [], [], []
        for pos, type in zip(atom_pos, atom_type):
            atom_fea, nbr_fea, nbr_fea_idx = \
                self.single(pos, type)
            batch_atom_fea.append(atom_fea)
            batch_nbr_fea.append(nbr_fea)
            batch_nbr_fea_idx.append(nbr_fea_idx)
        return batch_atom_fea, batch_nbr_fea, \
                batch_nbr_fea_idx


class MultiGridTransfer:
    #transfer configurations in different grid into the input of PPM
    def __init__(self):
        pass
    
    def find_batch_nbr_dis(self, atom_pos, grid_name):
        """
        calculate near distance of configurations
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        grid_name [int, 1d]: name of grids  

        Returns
        ----------
        nbr_dis [float, 3d]: near distance of configurations
        """
        last_grid = grid_name[0]
        transfer = Transfer(last_grid)
        nbr_dis = []
        i = 0
        for j, grid in enumerate(grid_name):
            if not grid == last_grid:
                batch_nbr_dis = \
                    [transfer.find_nbr_dis(pos) for pos in atom_pos[i:j]]
                nbr_dis += batch_nbr_dis
                transfer = Transfer(grid)
                last_grid = grid
                i = j
        batch_nbr_dis = [transfer.find_nbr_dis(pos) for pos in atom_pos[i:]]
        nbr_dis += batch_nbr_dis
        return nbr_dis

    def batch(self, atom_pos, atom_type, grid_name):
        """
        transfer configurations into input of PPM in batch
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids

        Returns:
        atom_feas [float, 3d]: atom feature of configurations 
        nbr_feas [float, 4d]: bond feature of configurations
        nbr_fea_idxes [int, 3d]: near index of configurations
        """
        last_grid = grid_name[0]
        transfer = Transfer(last_grid)
        atom_feas, nbr_feas, nbr_fea_idxes = [], [], []
        i = 0
        for j, grid in enumerate(grid_name):
            if not grid == last_grid:
                atom_fea, nbr_fea, nbr_fea_idx = \
                    transfer.batch(atom_pos[i:j], atom_type[i:j])
                atom_feas += atom_fea
                nbr_feas += nbr_fea
                nbr_fea_idxes += nbr_fea_idx
                transfer = Transfer(grid)
                last_grid = grid
                i = j
        atom_fea, nbr_fea, nbr_fea_idx = \
            transfer.batch(atom_pos[i:], atom_type[i:])
        atom_feas += atom_fea
        nbr_feas += nbr_fea
        nbr_fea_idxes += nbr_fea_idx
        return atom_feas, nbr_feas, nbr_fea_idxes
    
    def put_into_grid(self, init_pos, init_frac, init_vec, mut_frac, mut_vec):
        """
        Approximate target configuration in grid, 
        return corresponding index of grid point
        
        Parameters
        ----------
        init_pos [int, 2d]: postion in old lattice
        init_frac [float, 2d]: fraction coordinate of old lattice
        init_vec [float, 2d]: lattice vector of old lattice
        mut_frac [float, 2d]: fraction coordinate of new lattice
        mut_vec [float, 2d]: lattice vector of new lattice
        
        Returns
        ----------
        pos [int, 1d]: postion of atoms in grid
        """
        init_frac = init_frac[init_pos]
        stru_coor = np.dot(init_frac, init_vec)
        grid_coor = np.dot(mut_frac, mut_vec)
        distance = np.zeros((len(stru_coor), len(grid_coor)))
        for i, atom_coor in enumerate(stru_coor):
            for j, point_coor in enumerate(grid_coor):
                distance[i, j] = np.sqrt(np.sum((atom_coor - point_coor)**2))
        pos = list(map(lambda x: np.argmin(x), distance))
        return pos


if __name__ == "__main__":
    grid_name = 1
    grid_len = 125
    grid_label = np.arange(grid_len)
    atom_pos = [[j for j in range(i)] for i in range(10, 20)]
    atom_type = [[j for j in range(i)] for i in range(10, 20)]
    targets = np.random.rand(10,)
    
    transfer = Transfer(grid_name)
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = transfer.batch(atom_pos, atom_type)