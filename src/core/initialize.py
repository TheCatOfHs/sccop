import os, sys
import time
import random

import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure

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
    def __init__(self, number, component, ndensity, min_dis):
        ParallelDivide.__init__(self)
        self.create_dir()
        self.CSPD_generate(number, component, ndensity, min_dis)
    
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
        if recyc > 0:
            grain_loc = grain_fine
        else:
            grain_loc = grain_coarse
        #transfer CSPD structures
        atom_pos, atom_type, grid_name, point_num, latt_file = \
            self.structure_in_grid(recyc, grain_loc)
        self.zip_latt_vec(latt_file)
        for node in nodes:
            self.copy_latt_to_nodes(node)
        while not self.is_done('', self.num_node):
            time.sleep(self.wait_time)
        self.remove_flag('.')
        self.del_zip_latt()
        #structures generated randomly
        atom_pos, atom_type, grid_name, opt_idx= \
            self.add_random(recyc, atom_pos, atom_type, grid_name, point_num)
        #build grid
        grid_init = np.unique(grid_name)
        grid_origin = grid_init
        grid_mutate = grid_init
        self.assign_to_cpu(grain_loc, grid_origin, grid_mutate)
        system_echo('New grids have been built')
        #geometry check
        atom_pos, atom_type, grid_name = \
            self.geo_constrain(atom_pos, atom_type, grid_name, opt_idx)
        system_echo(f'Sampling number: {len(atom_pos)}')    
        return atom_pos, atom_type, grid_name, grid_init
    
    def geo_constrain(self, atom_pos, atom_type, grid_name, opt_idx):
        """
        geometry constrain to reduce structures
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
        opt_idx [int, 1d]: index of optimal samples
        
        Returns
        ----------
        atom_pos_right [int, 2d]: position of atoms after constrain
        atom_type_right [int, 2d]: type of atoms after constrain
        grid_name_right [int, 1d]: name of grids after constrain
        """
        #check geometry of structure
        nbr_dis = self.find_nbr_dis(atom_pos, grid_name)
        check_near = [self.near(i) for i in nbr_dis]
        check_overlay = [self.overlay(i, len(i)) for i in atom_pos]
        check = [i and j for i, j in zip(check_near, check_overlay)]
        #get correct sample index
        check_num = len(check)
        sample_idx = np.arange(check_num)[check]
        opt_idx = np.intersect1d(sample_idx, opt_idx)
        #sampling correct random samples
        sample_idx = np.setdiff1d(sample_idx, opt_idx)
        sample_num = len(sample_idx)
        if sample_num > num_initial:
            sample_idx = np.random.choice(sample_idx, num_initial, replace=False)
        sample_idx = np.concatenate((sample_idx, opt_idx))
        #add right samples into buffer
        atom_pos_right, atom_type_right, grid_name_right = [], [], []
        for i in sample_idx:
            atom_pos_right.append(atom_pos[i])
            atom_type_right.append(atom_type[i])
            grid_name_right.append(grid_name[i])
        return atom_pos_right, atom_type_right, grid_name_right
    
    def CSPD_generate(self, number, component, ndensity, min_dis):
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
        #delete extra samples
        dir = f'{poscar_path}/initial_strs_0'
        self.delete_same_poscars(dir)
        poscar = os.listdir(dir)
        poscar_num = len(poscar)
        if poscar_num > number:
            remove_num = poscar_num - number
            index = np.random.choice(np.arange(poscar_num), remove_num, replace=False)
            for i in index:
                os.remove(f'{dir}/{poscar[i]}')
        
    def add_random(self, recyc, atom_pos, atom_type, grid_name, point_num):
        """
        add random samples
        
        Parameters
        ----------
        recyc [int, 0d]: times of recyclings
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
        point_num [int, 1d]: number of grid points
        
        Returns
        ----------
        atom_pos_new [int, 2d]: position of atoms randomly
        atom_type_new [int, 2d]: type of atoms
        grid_name_new [int, 1d]: grid name
        opt_idx [int, 1d]: index of CSPD samples
        """
        #type pool with various number of atoms
        if recyc == 0:
            type_pool = atom_type
        else:
            type_pool = self.import_list2d(f'{record_path}/{recyc}/atom_type.dat', int)
        type_num = len(type_pool)
        #CSPD samples with random samples
        opt_idx, atom_pos_new, atom_type_new, grid_name_new = [], [], [], []
        for i, grid in enumerate(grid_name):
            atom_pos_new += [atom_pos[i]]
            atom_type_new += [atom_type[i]]
            grid_name_new += [grid_name[i]]
            opt_idx += [len(atom_pos_new)-1]
            #sampling on different grids
            points = [i for i in range(point_num[i])]
            grid_name_new += [grid for _ in range(num_rand)]
            for _ in range(num_rand):
                seed = random.randint(0, type_num-1)
                atom_num = len(type_pool[seed])
                atom_pos_new += [random.sample(points, atom_num)]
                atom_type_new += [type_pool[seed]]
        return atom_pos_new, atom_type_new, grid_name_new, opt_idx
    
    def structure_in_grid(self, recyc, grain):
        """
        put structure into grid
        
        Parameters
        ----------
        recyc [int, 0d]: times of recycling
        grain [float, 1d]: grain of grid
        
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
            #import structure
            stru = Structure.from_file(f'{init_path}/{poscar}', sort=True)
            latt_vec = stru.lattice.matrix
            type = self.get_atom_number(stru)
            stru_frac = stru.frac_coords
            grid_frac = self.fraction_coor(grain, latt_vec)
            pos = self.put_into_grid(stru_frac, latt_vec, grid_frac, latt_vec)
            #write lattice file
            file = f'{grid_num+i:03.0f}_latt_vec.bin'
            self.write_list2d(f'{grid_prop_path}/{file}', latt_vec, binary=True)
            #append to buffer
            atom_type.append(type)
            atom_pos.append(pos)
            latt_file.append(file)
            grid_name.append(grid_num+i)
            num = len(grid_frac)
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
        os.mkdir(record_path)
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
    init = InitSampling(num_CSPD, component, ndensity, min_dis_CSPD)
    init.generate(1)