import os, sys
import re
import time
import random
import numpy as np

from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from core.path import *
from core.global_var import *
from core.utils import *
from core.data_transfer import DeleteDuplicates
from core.grid_sampling import GridDivide, ParallelSampling
from core.sample_select import Select


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
                        #!/bin/bash
                        cd {CPU_local_path}
                        rm -rf sccop/
                        mkdir sccop/
                        cd sccop/
                        
                        mkdir vasp
                        scp -r {gpu_node}:{SCCOP_path}/data .
                        scp -r {gpu_node}:{SCCOP_path}/src .
                        scp -r {gpu_node}:{SCCOP_path}/libs .
                        
                        touch FINISH-{ip}
                        scp FINISH-{ip} {gpu_node}:{SCCOP_path}/
                        rm FINISH-{ip}
                        '''
        self.ssh_node(shell_script, ip)
    
    def copy_poscars_to_nodes(self, recyc, node):
        """
        copy initial poscars to each node
        
        Parameters
        ----------
        recyc [int, 0d]: recycle of sccop
        node [int, 0d]: cpu node
        """
        ip = f'node{node}'
        local_poscar_path = f'{SCCOP_path}/{init_strus_path}_{recyc}'
        shell_script = f'''
                        #!/bin/bash
                        mkdir {local_poscar_path}
                        cd {local_poscar_path}
                        scp {gpu_node}:{local_poscar_path}/poscars.tar.gz .
                        tar -zxf poscars.tar.gz
                            
                        touch FINISH-{ip}
                        scp FINISH-{ip} {gpu_node}:{SCCOP_path}/
                        rm FINISH-{ip} poscars.tar.gz
                        '''
        self.ssh_node(shell_script, ip)
    
    def zip_poscars(self, recyc):
        """
        zip poscars
        
        Parameters
        ----------
        recyc [int, 0d]: recycle of sccop
        """
        shell_script = f'''
                        #!/bin/bash
                        cd {init_strus_path}_{recyc}
                        tar -zcf poscars.tar.gz *
                        '''
        os.system(shell_script)
    
    def remove_poscars_zip(self, recyc):
        """
        remove zip of poscars
        """
        shell_script = f'''
                        #!/bin/bash
                        cd {init_strus_path}_{recyc}
                        rm poscars.tar.gz
                        '''
        os.system(shell_script)
    

class InitSampling(UpdateNodes, GridDivide, ParallelSampling,
                   DeleteDuplicates):
    #generate initial structures of sccop
    def __init__(self):
        UpdateNodes.__init__(self)
        ParallelSampling.__init__(self)
        DeleteDuplicates.__init__(self)
        self.select = Select(0)
        self.create_dir()
    
    def generate(self, recyc):
        """
        generate initial samples
        
        Parameters
        ----------
        recyc [int, 0d]: times of recycle
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        """
        #generate initial lattice
        self.initial_poscars(recyc, num_latt, component)
        #structures generated randomly
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.random_sampling(recyc)
        #select samples by ML
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.select.ml_sampling(recyc, atom_pos, atom_type, atom_symm,
                                    grid_name, grid_ratio, space_group)
        #delete duplicates
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.reduce_samples(atom_pos, atom_type, atom_symm,
                                grid_name, grid_ratio, space_group)
        #add random samples
        system_echo(f'Sampling number: {len(atom_pos)}')    
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group
    
    def initial_poscars(self, recyc, number, component):
        """
        generate initial poscars randomly
        
        Parameters
        ----------
        recyc [int, 0d]: times of recycle
        number [int, 0d]: number of poscars
        component [str, 0d]: component of searching system
        """
        #transfer component to atom type list
        elements= re.findall('[A-Za-z]+', component)
        ele_num = [int(i) for i in re.findall('[0-9]+', component)]
        atom_type = []
        for ele, num in zip(elements, ele_num):
            for _ in range(num):
                atom_type.append(ele)
        atom_types = self.control_atom_number(atom_type)
        #make directory of initial poscars
        dir = f'{poscar_path}/initial_strus_{recyc}'
        if not os.path.exists(dir):
            os.mkdir(dir)
        #generate poscars from crystal system randomly
        num = len(atom_types)
        for i in range(number):
            latt = self.lattice_generate()
            seed = random.randint(0, num-1)
            atom_type = atom_types[seed]
            atom_num = len(atom_type)
            coors = np.random.rand(atom_num, 3)
            stru = Structure(latt, atom_type, coors)
            stru.to(filename=f'{dir}/POSCAR-RCSD-{i:03.0f}', fmt='poscar')
        system_echo(f'Generate Initial Lattice')
    
    def random_sampling(self, recyc):
        """
        build grid of structure

        Parameters
        ----------
        recyc [int, 0d]: times of recycle

        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        """
        #copy initial poscars to cpu nodes
        self.zip_poscars(recyc)
        for node in nodes:
            self.copy_poscars_to_nodes(recyc, node)
        while not self.is_done('', self.num_node):
            time.sleep(self.wait_time)
        self.remove_flag('.')
        self.remove_poscars_zip(recyc)
        #sampling randomly
        start = self.get_latt_num()
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.sampling_on_grid(recyc, start)
        system_echo(f'Initial sampling: {len(grid_name)}')
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group
        
    def control_atom_number(self, atom_type):
        """
        control number of atoms
        
        Parameters
        ----------
        atom_type [str, 1d]: min number of atoms

        Returns
        ----------
        atom_types [str, 2d]: various number of atoms
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
    
    def reduce_samples(self, atom_pos, atom_type, atom_symm, 
                       grid_name, grid_ratio, space_group):
        """
        geometry constrain to reduce structures
        
        Parameters
        ----------
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
        #sampling random samples by space group
        idx = self.balance_sampling(2*num_Rand, space_group)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group)
        #sort structure in order of grid and space group
        idx = self.sort_by_grid_sg(grid_name, space_group)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group)
        #delete same samples under same space group
        idx = self.delete_duplicates_sg_pymatgen(atom_pos, atom_type,
                                                 grid_name, grid_ratio, space_group)
        system_echo(f'Delete duplicates: {len(idx)}')
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group)
        #sampling random samples by space group
        idx = self.balance_sampling(num_Rand, space_group)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group)
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group
    
    def balance_sampling(self, num, space_group):
        """
        select samples from different space groups
        
        Parameters
        ----------
        num [int, 0d]: number of samples
        space_group [int, 1d]: space group number

        Returns
        ----------
        sample [int, 1d]: index of samples
        """
        idx = np.arange(len(space_group))
        array = np.stack((idx, space_group), axis=1)
        array = sorted(array, key=lambda x: x[1])
        idx, space_group = np.transpose(array)
        #group index by space group
        store, clusters = [], []
        last_sg = space_group[0]
        for i, sg in enumerate(space_group):
            if sg == last_sg:
                store.append(idx[i])
            else:
                clusters.append(store)
                last_sg = sg
                store = []
                store.append(idx[i])
        clusters.append(store)
        #get sampling number of each space group
        cluster_num = len(clusters)
        per_num = [len(i) for i in clusters]
        assign = [0 for _ in range(cluster_num)]
        flag = True
        while flag:
            for i in range(cluster_num):
                if sum(assign) == num:
                    flag = False
                    break
                if sum(per_num) == 0:
                    flag = False
                    break
                if per_num[i] > 0:
                    per_num[i] -= 1
                    assign[i] += 1
        #sampling on different space groups
        sample = []
        flag = True
        while flag:
            for i in range(cluster_num):
                if sum(assign) == 0:
                    flag = False
                    break
                if assign[i] > 0:
                    idx = np.random.choice(clusters[i])
                    sample.append(idx)
                    assign[i] -= 1
                    clusters[i].remove(idx)
        return sample

    def create_dir(self):
        """
        make directory
        """
        if not os.path.exists(poscar_path):
            os.mkdir(poscar_path)
            os.mkdir(model_path)
            os.mkdir(search_path)
            os.mkdir(vasp_out_path)
            os.mkdir(grid_path)
            os.mkdir(json_path)
            os.mkdir(buffer_path)
    
    def get_latt_num(self):
        """
        count number of grids
        
        Returns
        ----------
        num [int, 0d]: number of grids
        """
        file = os.listdir(grid_path)
        grids = [i for i in file if i.endswith('latt_vec.bin')]
        if len(grids) == 0:
            num = 0
        else:
            grid = sorted(grids)[-1]
            num = int(grid.split('_')[0]) + 1
        return num
    
    
if __name__ == '__main__':
    pass