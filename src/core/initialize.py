import os, sys
import re
import time
import random

import numpy as np
import pandas as pd
from collections import Counter
from pymatgen.core.lattice import Lattice
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
    def __init__(self, number, component):
        ParallelDivide.__init__(self)
        self.create_dir()
        self.RCSD_generate(number, component)
        
    def generate(self, recyc):
        """
        initial samples from RCSD and random
        
        Parameters
        ----------
        recyc [int, 0d]: times of recycle
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
        """
        if recyc > 0:
            grain = grain_fine
        else:
            grain = grain_coarse
        #transfer initial structures
        atom_pos, atom_type, grid_name, point_num, latt_file = \
            self.structure_in_grid(recyc, grain)
        #build grid
        grid_new = self.build_grid(grain, grid_name, latt_file)
        system_echo('New grids have been built')
        #structures generated randomly
        atom_pos_rand, atom_type_rand, grid_name_rand= \
            self.random_sampling(recyc, atom_type, grid_name, point_num)
        #geometry check
        atom_pos, atom_type, grid_name = \
            self.geo_constrain(num_RCSD, atom_pos, atom_type, grid_name)
        atom_pos_rand, atom_type_rand, grid_name_rand = \
            self.geo_constrain(num_Rand, atom_pos_rand, atom_type_rand, grid_name_rand)
        #add random samples
        atom_pos += atom_pos_rand
        atom_type += atom_type_rand
        grid_name += grid_name_rand
        system_echo(f'Sampling number: {len(atom_pos)}')    
        return atom_pos, atom_type, grid_name, grid_new
    
    def build_grid(self, grain, grid_name, latt_file):
        """
        build grid of structure

        Parameters
        ----------
        grain [float, 1d]: grain of grid
        grid_name [str, 1d]: name of grid
        latt_file [str, 1d]: lattice file

        Returns:
        grid_new [str, 1d]: name of grid
        """
        #copy lattice file to cpu nodes
        self.zip_latt_vec(latt_file)
        for node in nodes:
            self.copy_latt_to_nodes(node)
        while not self.is_done('', self.num_node):
            time.sleep(self.wait_time)
        self.remove_flag('.')
        self.del_zip_latt()
        #build grid
        grid_new = np.unique(grid_name)
        grid_origin = grid_new
        grid_mutate = grid_new
        self.assign_to_cpu(grain, grid_origin, grid_mutate)
        return grid_new

    def RCSD_generate(self, number, component):
        """
        generate initial samples randomly
        
        Parameters
        ----------
        number [int, 0d]: number of RCSD samples
        component [str, 0d]: component of searching system
        """
        system_echo(f'Generate Initial Data Randomly')
        #transfer component to atom type list
        elements= re.findall('[A-Za-z]+', component)
        ele_num = [int(i) for i in re.findall('[0-9]+', component)]
        species = []
        for ele, num in zip(elements, ele_num):
            for _ in range(num):
                species.append(ele)
        species = self.control_atom_number(species)
        species_num = len(species)
        #export poscars
        dir = f'{poscar_path}/initial_strs_0'
        if not os.path.exists(dir):
            os.mkdir(dir)
        for i in range(number):
            latt = self.lattice_generate()
            seed = random.randint(0, species_num-1)
            atom_type = species[seed]
            atom_num = len(atom_type)
            coors = np.random.rand(atom_num, 3)
            coors *=  free_aix
            stru = Structure(latt, atom_type, coors)
            stru.to(filename=f'{dir}/POSCAR-RCSD-{i:03.0f}', fmt='poscar')
    
    def lattice_generate(self):
        """
        generate lattice by crystal system
        
        Returns:
        latt [obj]: Lattice object of pymatgen
        """
        volume = 0
        system = np.arange(0, 7)
        while volume == 0:
            crystal_system = np.random.choice(system, p=system_weight)
            #cubic
            if crystal_system == 0:
                a = random.normalvariate(len_mu, len_sigma)
                b, c = a, a
                alpha, beta, gamma = 90, 90, 90
            #tetragonal
            if crystal_system == 1:
                a, c = np.random.normal(len_mu, len_sigma, 2)
                b = a
                alpha, beta, gamma = 90, 90, 90
            #orthorhombic
            if crystal_system == 2:
                a, b, c = np.random.normal(len_mu, len_sigma, 3)
                alpha, beta, gamma = 90, 90, 90
            #trigonal
            if crystal_system == 3:
                a = random.normalvariate(len_mu, len_sigma)
                b, c = a, a
                alpha = random.normalvariate(ang_mu, ang_sigma)
                beta, gamma = alpha, alpha
            #hexagonal
            if crystal_system == 4:
                a, c = np.random.normal(len_mu, len_sigma, 2)
                b = a
                alpha, beta, gamma = 90, 90, 120
            #monoclinic
            if crystal_system == 5:
                a, b, c = np.random.normal(len_mu, len_sigma, 3)
                alpha, gamma = 90, 90
                beta = random.normalvariate(ang_mu, ang_sigma)
            #triclinic
            if crystal_system == 6:
                a, b, c = np.random.normal(len_mu, len_sigma, 3)
                alpha, beta, gamma = np.random.normal(ang_mu, ang_sigma, 3)
            #check validity of lattice vector
            a, b, c = [i if 4 < i else 4 for i in (a, b, c)]
            a, b, c = [i if i < 6 else 6 for i in (a, b, c)]
            if add_vacuum:
                c = 20
            latt = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
            volume = latt.volume
        return latt
    
    def random_sampling(self, recyc, atom_type, grid_name, point_num):
        """
        add random samples
        
        Parameters
        ----------
        recyc [int, 0d]: times of recyclings
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
        point_num [int, 1d]: number of grid points
        
        Returns
        ----------
        atom_pos_new [int, 2d]: position of atoms randomly
        atom_type_new [int, 2d]: type of atoms
        grid_name_new [int, 1d]: grid name
        """
        #type pool with various number of atoms
        if recyc == 0:
            type_pool = self.atom_type_rand(atom_type)
        else:
            type_pool = self.import_list2d(f'{record_path}/{recyc-1}/atom_type.dat', int)
        type_num = len(type_pool)
        #sampling on different grids
        atom_pos_new, atom_type_new, grid_name_new = [], [], []
        for i, grid in enumerate(grid_name):
            counter = 0
            points = [i for i in range(point_num[i])]
            for _ in range(num_sampling):
                seed = random.randint(0, type_num-1)
                atom_num = len(type_pool[seed])
                if atom_num < len(points):
                    atom_pos_new += [random.sample(points, atom_num)]
                    atom_type_new += [type_pool[seed]]
                    counter += 1
            grid_name_new += [grid for _ in range(counter)]
        return atom_pos_new, atom_type_new, grid_name_new
    
    def atom_type_rand(self, atom_type):
        """
        generate various length of atom type
        
        Parameters
        ----------
        atom_type [int, 2d]: type of atoms

        Returns
        ----------
        types [int, 2d]: type of atoms 
        """
        count = Counter(atom_type[0])
        species = list(count.keys())
        type_num = list(count.values())
        #get greatest common divisor
        gcd = np.gcd.reduce(type_num)
        type_num = np.array(type_num)//gcd
        #get min length of atom type
        init_type = []
        for i, atom in enumerate(species):
            for _ in range(type_num[i]):
                init_type.append(atom)
        #get different length of atom type
        types = self.control_atom_number(init_type)
        return types
    
    def control_atom_number(self, atom_type):
        """
        control number of atoms
        
        Parameters
        ----------
        atom_type [int or str, 1d]: min number of atoms

        Returns
        ----------
        atom_types [int or str, 2d]: various number of atoms
        """
        times = 1
        atom_types = []
        type = atom_type
        while len(type) <= num_max_atom: 
            type = [i for i in atom_type for _ in range(times)]
            atom_types.append(type)
            times += 1
        atom_types = [i for i in atom_types if num_min_atom <= len(i)]
        return atom_types
    
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
    
    def geo_constrain(self, n, atom_pos, atom_type, grid_name):
        """
        geometry constrain to reduce structures
        
        Parameters
        ----------
        n [int, 0d]: number of samples
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
        
        Returns
        ----------
        atom_pos_right [int, 2d]: position of atoms after constrain
        atom_type_right [int, 2d]: type of atoms after constrain
        grid_name_right [int, 1d]: name of grids after constrain
        """
        #check overlay of atoms
        check_overlay = [self.overlay(i, len(i)) for i in atom_pos]
        check_num = len(check_overlay)
        sample_idx = np.arange(check_num)[check_overlay]
        #select samples that are not overlayed
        atom_pos_rand, atom_type_rand, grid_name_rand = [], [], []
        for i in sample_idx:
            atom_pos_rand.append(atom_pos[i])
            atom_type_rand.append(atom_type[i])
            grid_name_rand.append(grid_name[i])
        #check neighbor distance of atoms
        nbr_dis = self.find_nbr_dis(atom_pos_rand, grid_name_rand)
        check_near = [self.near(i) for i in nbr_dis]
        check_num = len(check_near)
        #sampling correct random samples
        sample_idx = np.arange(check_num)[check_near]
        atom_pos_right, atom_type_right, grid_name_right = [], [], []
        if len(sample_idx) > 0:
            grids = np.array(grid_name_rand)[check_near]
            sample_idx = self.balance_sampling(n, sample_idx, grids)
            #add right samples into buffer
            for i in sample_idx:
                atom_pos_right.append(atom_pos_rand[i])
                atom_type_right.append(atom_type_rand[i])
                grid_name_right.append(grid_name_rand[i])  
        return atom_pos_right, atom_type_right, grid_name_right
    
    def balance_sampling(self, n, index, grids):
        """
        select samples from different grids
        
        Parameters
        ----------
        n [int, 0d]: number of samples
        index [int, 1d]: index of samples
        grids [int, 1d]: grid of samples

        Returns
        ----------
        sample [int, 1d]: index of samples
        """
        array = np.stack((index, grids), axis=1)
        array = sorted(array, key=lambda x: x[1])
        index, grids = np.transpose(array)
        #group index by grids
        store, clusters = [], []
        last_grid = grids[0]
        for i, grid in enumerate(grids):
            if grid == last_grid:
                store.append(index[i])
            else:
                clusters.append(store)
                last_grid = grid
                store = []
                store.append(index[i])
        clusters.append(store)
        #get sampling number of each grid
        cluster_num = len(clusters)
        cluster_per_num = [len(i) for i in clusters]
        assign = [0 for _ in range(cluster_num)]
        flag = True
        while flag:
            for i in range(cluster_num):
                if sum(assign) == n:
                    flag = False
                    break
                if sum(cluster_per_num) == 0:
                    flag = False
                    break
                if cluster_per_num[i] > 0:
                    cluster_per_num[i] -= 1
                    assign[i] += 1
        #sampling on different grids
        sample = []
        for i, cluster in zip(assign, clusters):
            sample += random.sample(cluster, i)
        return sample
    
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
    #init = InitSampling(num_RCSD, component)
    #init.generate(1)
    print('ok')