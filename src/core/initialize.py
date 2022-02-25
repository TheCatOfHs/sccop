import os, sys
import time
import random

import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure

from core.search import GeoCheck

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.utils import SSHTools, system_echo
from core.predict import PPMData, PPModel
from core.data_transfer import Transfer, MultiGridTransfer
from core.grid_divide import GridDivide, ParallelDivide
from core.search import GeoCheck


class UpdateNodes(SSHTools):
    #make cpu nodes consistent with gpu node
    def __init__(self, wait_time=0.1):
        self.num_node = len(nodes)
        self.wait_time = wait_time
    
    def update(self):
        """
        update cpu nodes
        """
        for node in nodes:
            self.copy_file_to_nodes(node)
        while not self.is_done('', self.num_node):
            time.sleep(self.wait_time)
        self.remove_flag('.')
        system_echo('CPU nodes are consistent with GPU node')
    
    def copy_file_to_nodes(self, node):
        """
        SSH to target node and copy necessary files
        """
        ip = f'node{node}'
        shell_script = f'''
                        cd /local
                        rm -rf ccop/
                        mkdir ccop/
                        cd ccop/
                        mkdir vasp/ libs/
                        
                        scp -r {gpu_node}:/local/ccop/data .
                        scp -r {gpu_node}:/local/ccop/src .
                        scp -r {gpu_node}:/local/ccop/libs/DPT libs/.
                        scp -r {gpu_node}:/local/ccop/libs/scripts libs/.
                        scp -r {gpu_node}:/local/ccop/libs/VASP_inputs libs/.
                        
                        touch FINISH-{ip}
                        scp FINISH-{ip} {gpu_node}:/local/ccop/
                        rm FINISH-{ip}
                        '''
        self.ssh_node(shell_script, ip)
    
    def copy_latt_to_nodes(self, node):
        """
        copy lattice vector file to each node
        
        Parameters
        ----------
        node [int, 0d]: cpu node
        """
        ip = f'node{node}'
        local_grid_prop_path = f'/local/ccop/{grid_prop_path}'
        shell_script = f'''
                        cd {local_grid_prop_path}
                        scp {gpu_node}:{local_grid_prop_path}/latt_vec.tar.gz .
                        tar -zxf latt_vec.tar.gz
                            
                        touch FINISH-{ip}
                        scp FINISH-{ip} {gpu_node}:/local/ccop/
                        rm FINISH-{ip} latt_vec.tar.gz
                        '''
        self.ssh_node(shell_script, ip)
    
    def zip_latt_vec(self, latt):
        """
        zip latt_vec.bin sent to nodes
        
        Parameters
        ----------
        latt [str, 1d]: name of lattice vectors 
        """
        latt_str = ' '.join(latt)
        shell_script = f'''
                        cd data/grid/property/
                        tar -zcf latt_vec.tar.gz {latt_str}
                        '''
        os.system(shell_script)
    
    def del_zip_latt(self):
        shell_script = f'''
                        cd data/grid/property/
                        rm latt_vec.tar.gz
                        '''
        os.system(shell_script)
        

class InitSampling(GridDivide, ParallelDivide, UpdateNodes, MultiGridTransfer, GeoCheck):
    #generate initial structures of ccop
    def __init__(self, component, ndensity, min_dis):
        ParallelDivide.__init__(self)
        self.create_dir()
        self.CSPD_generate(component, ndensity, min_dis)
    
    def generate(self, recyc):
        """
        initial samples from CSPD and random
        
        Parameters
        ----------
        recyc [int, 0d]: times of recyclings
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
        """
        #transfer CSPD structures
        atom_pos, atom_type, grid_name, point_num, latt_file = \
            self.structure_in_grid(recyc)
        self.zip_latt_vec(latt_file)
        for node in nodes:
            self.copy_latt_to_nodes(node)
        while not self.is_done('', self.num_node):
            time.sleep(self.wait_time)
        self.remove_flag('.')
        self.del_zip_latt()
        #structures generated randomly
        atom_pos_rand, atom_type_rand, grid_name_rand = \
            self.Rand_generate(atom_type, grid_name, point_num)
        #build grid
        grid_init = np.unique(grid_name)
        grid_origin = grid_init
        grid_mutate = grid_init
        self.assign_to_cpu(grid_origin, grid_mutate)
        system_echo('New grids have been built')
        #geometry check
        atom_pos_rand, atom_type_rand, grid_name_rand = \
            self.geo_constrain(atom_pos_rand, atom_type_rand, grid_name_rand)
        #initial structures
        atom_pos += atom_pos_rand
        atom_type += atom_type_rand
        grid_name += grid_name_rand
        system_echo(f'Sampling number: {len(atom_pos)}')    
        return atom_pos, atom_type, grid_name
    
    def geo_constrain(self, atom_pos, atom_type, grid_name):
        """
        geometry constrain to reduce structures
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids

        Returns:
        ----------
        atom_pos_right [int, 2d]: position of atoms after constrain
        atom_type_right [int, 2d]: type of atoms after constrain
        grid_name_right [int, 1d]: name of grids after constrain
        """
        nbr_dis = self.find_nbr_dis(atom_pos, grid_name)
        check_near = [self.near(i) for i in nbr_dis]
        atom_pos_right, atom_type_right, grid_name_right = [], [], []
        check_num = len(check_near)
        sample_num = np.ones(check_num)[check_near].sum()
        sample_idx = list(np.arange(check_num)[check_near])
        if sample_num > num_initial:
            sample_idx = random.sample(sample_idx, num_initial)
        for i in sample_idx:
            atom_pos_right.append(atom_pos[i])
            atom_type_right.append(atom_type[i])
            grid_name_right.append(grid_name[i])
        return atom_pos_right, atom_type_right, grid_name_right
    
    def CSPD_generate(self, component, ndensity, min_dis):
        """
        generate initial samples from CSPD
        
        Parameters
        ----------
        component [str, 0d]: component of searching system
        ndensity [float, 0d]: density of atoms
        min_dis [float, 0d]: minimal distance between atoms
        """
        options = f'--component {component} --ndensity {ndensity} --mindis {min_dis}'
        shell_script = f'''
                        cd libs/ASG
                        tar -zxf CSPD.tar.gz
                        python generate.py {options}
                        mv structure_folder ../../data/poscar/initial_strs_0
                        rm CSPD.db
                        '''
        os.system(shell_script)
    
    def Rand_generate(self, atom_type, grid_name, point_num):
        """
        generate initial samples randomly
        
        Parameters
        ----------
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
        point_num [int, 1d]: number of grid points
        
        Returns
        ----------
        atom_pos_rand [int, 2d]: position of atoms randomly
        atom_type_rand [int, 2d]: type of atoms
        grid_name_rand [int, 1d]: grid name
        """
        grid_num = len(grid_name)
        atom_pos_rand, atom_type_rand, grid_name_rand = [], [], []
        for i, grid in enumerate(grid_name):
            points = [i for i in range(point_num[i])]
            grid_name_rand += [grid for _ in range(num_rand)]
            for _ in range(num_rand):
                seed = random.randint(0, grid_num-1)
                atom_num = len(atom_type[seed])
                atom_pos_rand += [random.sample(points, atom_num)]
                atom_type_rand += [atom_type[seed]]
        return atom_pos_rand, atom_type_rand, grid_name_rand
    '''
    def Rand_generate(self, atom_type, grid_name, point_num):
        """
        generate initial samples randomly
        
        Parameters
        ----------
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
        point_num [int, 1d]: number of grid points
        
        Returns
        ----------
        atom_pos_rand [int, 2d]: position of atoms randomly
        atom_type_rand [int, 2d]: type of atoms
        grid_name_rand [int, 1d]: grid name
        """
        atom_pos_rand, atom_type_rand, grid_name_rand = [], [], []
        for i, grid in enumerate(grid_name):
            atom_num = len(atom_type[i])
            points = [j for j in range(point_num[i])]
            atom_pos_rand += [random.sample(points, atom_num) for _ in range(num_rand)]
            atom_type_rand += [atom_type[i] for _ in range(num_rand)]
            grid_name_rand += [grid for _ in range(num_rand)]
        return atom_pos_rand, atom_type_rand, grid_name_rand
    '''
    def structure_in_grid(self, recyc):
        """
        put structure into grid
        
        Parameters
        ----------
        recyc [int, 0d]: times of recycling
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: new initial grids
        point_num [int, 1d]: number of grid points
        latt_file [str, 1d]: name of lattice
        """
        grid_num = self.count_latt_num()
        init_path = f'{init_strs_path}_{recyc}'
        file_name = os.listdir(init_path)
        atom_pos, atom_type, grid_name, point_num, latt_file = [], [], [], [], []
        for i, poscar in enumerate(file_name):
            stru = Structure.from_file(f'{init_path}/{poscar}', sort=True)
            latt_vec = stru.lattice.matrix
            file = f'{grid_num+i:03.0f}_latt_vec.bin'
            self.write_list2d(f'{grid_prop_path}/{file}', latt_vec, binary=True)
            type = self.get_atom_number(stru)
            stru_frac = stru.frac_coords
            grid_frac = self.fraction_coor(grain, latt_vec)
            pos = self.put_into_grid(stru_frac, latt_vec, grid_frac, latt_vec)
            num = len(grid_frac)
            atom_type.append(type)
            atom_pos.append(pos)
            latt_file.append(file)
            grid_name.append(grid_num+i)
            point_num.append(num)
        return atom_pos, atom_type, grid_name, point_num, latt_file
    
    def count_latt_num(self):
        """
        count number of grids
        """
        command = f'ls -l {grid_prop_path} | grep latt_vec.bin | wc -l'
        num = os.popen(command)
        return int(num.read())
    
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
    
    def create_dir(self):
        """
        make directory
        """
        os.mkdir(poscar_path)
        os.mkdir(model_path)
        os.mkdir(search_path)
        os.mkdir(vasp_out_path)
        os.mkdir(grid_path)
        os.mkdir(grid_poscar_path)
        os.mkdir(grid_prop_path)
    
    
class Pretrain(Transfer):
    #Pretrain property predict model
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
    init = InitSampling(component, ndensity, min_dis_CSPD)
    cpu_nodes = UpdateNodes()
    cpu_nodes.update()
    (atom_pos, atom_type, grid_name) = init.generate(0)
    print([len(i) for i in atom_pos])
    print(len(grid_name))
    '''
    str = Structure.from_file(f'test/POSCAR-CCOP-0-0023-135', sort=True)
    latt_vec = str.lattice.matrix
    type = init.get_atom_number(str)
    stru_frac = str.frac_coords
    grid_frac = init.fraction_coor(grain, latt_vec)
    pos = init.put_into_grid(stru_frac, latt_vec, grid_frac, latt_vec)
    print(grid_frac[pos])
    init.write_list2d('test/coor.dat', grid_frac[pos])
    '''