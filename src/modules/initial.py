import os, sys
import time
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from modules.global_var import *
from modules.utils import SSHTools, system_echo
from modules.predict import PPMData, PPModel
from modules.data_transfer import Transfer
from modules.grid_divide import GridDivide


class UpdateNodes(SSHTools):
    #make each node consistent with main node
    def __init__(self, sleep_time=1):
        self.num_node = len(nodes)
        self.sleep_time = sleep_time
    
    def update(self):
        """
        update cpu nodes
        """
        for node in nodes:
            self.copy_file_to_nodes(node)
        while not self.is_done(self.num_node):
            time.sleep(self.sleep_time)
        self.remove()
        system_echo('Each node consistent with main node')
    
    def copy_file_to_nodes(self, node):
        """
        SSH to target node and update ccop
        """
        ip = f'node{node}'
        shell_script = f'''
                        #!/bin/bash
                        cd /local
                        rm -rf ccop/
                        mkdir ccop/
                        cd ccop/
                        mkdir vasp/
                        
                        scp -r {gpu_node}:/local/ccop/data .
                        scp -r {gpu_node}:/local/ccop/libs .
                        scp -r {gpu_node}:/local/ccop/src .
                        
                        touch FINISH-{ip}
                        scp FINISH-{ip} {gpu_node}:/local/ccop/
                        rm FINISH-{ip}
                        '''
        self.ssh_node(shell_script, ip)
    
    def copy_latt_to_nodes(self, latt, node):
        """
        copy lattice vector file to each node
        
        Parameters
        ----------
        latt [str, 0d]: name of lattice vectors 
        """
        ip = f'node{node}'
        shell_script = f'''
                        #!/bin/bash
                        cd /local/ccop/{grid_prop_dir}
                        for i in {latt}
                        do
                            scp {gpu_node}:/local/ccop/$i .
                        done
                        
                        touch FINISH-{ip}
                        scp FINISH-{ip} {gpu_node}:/local/ccop/
                        rm FINISH-{ip}
                        '''
        self.ssh_node(shell_script, ip)
    
    def is_done(self, file_num):
        """
        If shell is completed, return True
        
        Returns
        ----------
        flag [bool, 0d]: whether job is done
        """
        command = f'ls -l | grep FINISH | wc -l'
        flag = self.check_num_file(command, file_num)
        return flag
    
    def remove(self): 
        os.system(f'rm FINISH*')


class Initial(GridDivide, UpdateNodes):
    #Generate initial samples of ccop
    def __init__(self):
        UpdateNodes.__init__(self)
    
    def generate(self, recyc, grid_store):
        """
        generate initial samples
        
        Parameters
        ----------
        recyc [int, 0d]: times of recycling
        grid_store [int, 1d]: name of grids
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_init [int, 1d]: new initial grids
        """
        num_grid = len(grid_store)
        initial_dir = f'{init_strs_path}_{recyc}'
        file = os.listdir(initial_dir)
        atom_pos, atom_type, grid_init, latt = [], [], [], []
        for i, poscar in enumerate(file):
            str = Structure.from_file(f'{initial_dir}/{poscar}', sort=True)
            latt_vec = str.lattice.matrix
            latt_file = f'{grid_prop_dir}/{num_grid+i:03.0f}_latt_vec.bin'
            self.write_list2d(latt_file, latt_vec, binary=True)
            type = self.get_atom_number(str)
            stru_frac = str.frac_coords
            grid_frac = self.fraction_coor(grain, latt_vec)
            pos = self.put_into_grid(stru_frac, grid_frac, latt_vec)
            atom_type.append(type)
            atom_pos.append(pos)
            latt.append(latt_file)
            grid_init.append(num_grid+i)
        latt = ' '.join(latt)
        for node in nodes:
            self.copy_latt_to_nodes(latt, node)
        while not self.is_done(self.num_node):
            time.sleep(self.sleep_time)
        self.remove()
        return atom_pos, atom_type, grid_init
    
    def put_into_grid(self, stru_frac, grid_frac, latt_vec):
        """
        Approximate target configuration in grid, 
        return corresponding index of grid point
        
        Parameters
        ----------
        stru_coor [float, 2d, np]: fraction coordinate of test configuration
        grid_coor [float, 2d. np]: fraction coordinate of grid point
        latt_vec [float, 2d, np]: lattice vector
        
        Returns
        ----------
        pos [int, 1d]: postion of atoms in grid
        """
        stru_coor = np.dot(stru_frac, latt_vec)
        grid_coor = np.dot(grid_frac, latt_vec)
        distance = np.zeros((len(stru_coor), len(grid_coor)))
        for i, atom_coor in enumerate(stru_coor):
            for j, point_coor in enumerate(grid_coor):
                distance[i, j] = np.sqrt(np.sum((atom_coor - point_coor)**2))
        pos = list(map(lambda x: np.argmin(x), distance))
        return pos
    
    def get_atom_number(self, stru):
        """
        get atom number of structure
        
        Parameters
        ----------
        stru [obj]: pymatgen structure object 

        Returns
        ----------
        atom_type [int, 1d]: start from 0 
        """
        return list(np.array(stru.atomic_numbers) - 1)
    

class Pretrain(Transfer):
    #
    def __init__(self, 
                 cutoff=8, nbr=12, dmin=0, dmax=8, step=0.2, var=0.2):
        self.cutoff = cutoff
        self.nbr, self.var = nbr, var
        self.filter = np.arange(dmin, dmax+step, step)
        self.elem_embed = self.import_list2d(
            atom_init_file, int, numpy=True)
    
    def pretrain(self,):
        train_df = pd.read_csv('database/mp_20/train.csv')
        valid_df = pd.read_csv('database/mp_20/val.csv')
        test_df = pd.read_csv('database/mp_20/test.csv')
        train_store = [train_df.iloc[idx] for idx in range(len(train_df))]
        train_cifs = [i['cif'] for i in train_store]
        train_energys = [i['formation_energy_per_atom'] for i in train_store]
        valid_store = [valid_df.iloc[idx] for idx in range(len(valid_df))]
        valid_cifs = [i['cif'] for i in valid_store]
        valid_energys = [i['formation_energy_per_atom'] for i in valid_store]
        test_store = [test_df.iloc[idx] for idx in range(len(test_df))]
        test_cifs = [i['cif'] for i in test_store]
        test_energys = [i['formation_energy_per_atom'] for i in test_store]
        train_atom_fea, train_nbr_fea, train_nbr_fea_idx = self.batch(train_cifs)
        valid_atom_fea, valid_nbr_fea, valid_nbr_fea_idx = self.batch(valid_cifs)
        test_atom_fea, test_nbr_fea, test_nbr_fea_idx = self.batch(test_cifs)
        train_data = PPMData(train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energys)
        valid_data = PPMData(valid_atom_fea, valid_nbr_fea, valid_nbr_fea_idx, valid_energys)
        test_data = PPMData(test_atom_fea, test_nbr_fea, test_nbr_fea_idx, test_energys)
        ppm = PPModel(100, train_data, valid_data, test_data, batch_size=128)
        ppm.train_epochs()
    
    def single(self, cif):
        """
        index and distance of near grid points
        
        Parameters
        ----------
        cif [str, 0d]: string of cif 
        
        Returns
        ----------
        atom_fea [int, 2d, np]: feature of atoms
        nbr_fea [float, 2d, np]: distance of near neighbor 
        nbr_idx [int, 2d, np]: index of near neighbor 
        """
        crystal = Structure.from_str(cif, fmt='cif')
        atom_type = np.array(crystal.atomic_numbers) - 1
        all_nbrs = crystal.get_all_neighbors(self.cutoff)
        all_nbrs = [sorted(nbrs, key = lambda x: x[1]) for nbrs in all_nbrs]
        num_near = min(map(lambda x: len(x), all_nbrs))
        nbr_idx, nbr_dis = [], []
        for nbr in all_nbrs:
            idx = list(map(lambda x: x[2], nbr[:num_near]))[:self.nbr]
            dis = list(map(lambda x: x[1], nbr[:num_near]))[:self.nbr]
            nbr_idx.append(idx)
            nbr_dis.append(dis)
        nbr_idx, nbr_dis = np.array(nbr_idx), np.array(nbr_dis)
        atom_fea = self.atom_initializer(atom_type)
        nbr_fea = self.expand(nbr_dis)
        return atom_fea, nbr_fea, nbr_idx
    
    def batch(self, cifs):
        """
        transfer cifs to input of predict model
        
        Parameters
        ----------
        cifs [str, 1d]: string of structure in cif

        Returns
        ----------
        batch_atom_fea [int, 3d]: batch atom feature
        batch_nbr_fea [float, 4d]: batch neighbor feature
        batch_nbr_fea_idx [float, 3d]: batch neighbor index
        """
        batch_atom_fea, batch_nbr_fea, \
            batch_nbr_fea_idx = [], [], []
        for cif in cifs:
            atom_fea, nbr_fea, nbr_fea_idx = \
                self.single(cif)
            batch_atom_fea.append(atom_fea)
            batch_nbr_fea.append(nbr_fea)
            batch_nbr_fea_idx.append(nbr_fea_idx)
        return batch_atom_fea, batch_nbr_fea, \
                batch_nbr_fea_idx
    
        
if __name__ == '__main__':
    init = Initial()
    print(init.generate('data/poscar/initial_strs_0'))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''
    def write_POSCAR(frac_coor, type, num, node, latt_vec, elements):
        """
        write POSCAR file of one configuration
        
        Parameters
        ----------
        pos [int, 1d]: atom position
        type [int, 1d]: atom type
        round [int, 0d]: searching round
        num [int, 0d]: configuration number
        node [int, 0d]: calculate node
        """
        head_str = ['E = -1', '1']
        latt_str = rwtools.list2d_to_str(latt_vec, '{0:8.4f}')
        compn = elements[type]
        compn_dict = dict(Counter(compn))
        compn_str = [' '.join(list(compn_dict.keys())),
                    ' '.join([str(i) for i in compn_dict.values()]),
                    'Direct']
        frac_coor_str = rwtools.list2d_to_str(frac_coor, '{0:8.4f}')
        file = f'{grid_poscar_dir}/POSCAR-{num:04.0f}-{node}'
        POSCAR = head_str + latt_str + compn_str + frac_coor_str
        with open(file, 'w') as f:
            f.write('\n'.join(POSCAR))
    
    init_sam = Initial()
    rwtools = ListRWTools()
    
    file = os.listdir('data/poscar/structure_0')
    for i, poscar in enumerate(file):
        a = Structure.from_file(f'data/poscar/structure_0/{poscar}', sort=True)
        rwtools.write_list2d(f'data/grid/property/{i:03.0f}_latt_vec.dat', a.lattice.matrix, style='{0:8.4f}')
    
    file = os.listdir('data/poscar/structure_0')
    print(file)
    elements = rwtools.import_list2d(elements_file, str, numpy=True).ravel()
    for i, poscar in enumerate(file):
        grid_frac = rwtools.import_list2d(f'data/grid/property/000_frac_coor.dat', float, numpy=True)
        test = Structure.from_file(f'data/poscar/structure_0/{poscar}', sort=True)
        pos = init_sam.put_into_grid(test.frac_coords, grid_frac, test.lattice.matrix)
        write_POSCAR(grid_frac[pos], np.array(test.atomic_numbers)-1, i, 1, test.lattice.matrix, elements)
    '''
