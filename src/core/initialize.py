import os, sys
import re
import time
import random
import copy
import numpy as np

from collections import Counter
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.utils import *
from core.data_transfer import DeleteDuplicates
from core.grid_divide import GridDivide, ParallelDivide
from core.search import ActionSpace, GeoCheck


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
                        cd /local
                        rm -rf ccop/
                        mkdir ccop/
                        cd ccop/
                        
                        mkdir vasp
                        scp -r {gpu_node}:/local/ccop/data .
                        scp -r {gpu_node}:/local/ccop/src .
                        scp -r {gpu_node}:/local/ccop/libs .
                        
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
        local_grid_path = f'/local/ccop/{grid_path}'
        shell_script = f'''
                        #!/bin/bash
                        cd {local_grid_path}
                        scp {gpu_node}:{local_grid_path}/latt_vec.tar.gz .
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
                        #!/bin/bash
                        cd data/grid/
                        tar -zcf latt_vec.tar.gz {latt_str}
                        '''
        os.system(shell_script)
    
    def del_zip_latt(self):
        shell_script = f'''
                        #!/bin/bash
                        cd data/grid/
                        rm latt_vec.tar.gz
                        '''
        os.system(shell_script)
    

class InitSampling(UpdateNodes, GridDivide, ParallelDivide,
                   ActionSpace, GeoCheck, DeleteDuplicates):
    #generate initial structures of ccop
    def __init__(self):
        UpdateNodes.__init__(self)
        ParallelDivide.__init__(self)
        DeleteDuplicates.__init__(self)
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
        space_group [int, 1d]: space group number
        """
        #generate initial lattice
        self.initial_poscars(recyc, num_latt, component)
        #structures generated randomly
        atom_pos, atom_type, atom_symm, grid_name, space_group = \
            self.random_sampling(recyc, grain)
        #build grid
        self.build_grid(grid_name)
        system_echo('New grids have been built')
        #geometry check
        atom_pos, atom_type, atom_symm, grid_name, space_group = \
            self.geo_constrain(atom_pos, atom_type, atom_symm,
                               grid_name, space_group)
        #add random samples
        system_echo(f'Sampling number: {len(atom_pos)}')    
        return atom_pos, atom_type, atom_symm, grid_name, space_group
    
    def initial_poscars(self, recyc, number, component):
        """
        generate initial poscars randomly
        
        Parameters
        ----------
        recyc [int, 0d]: times of recycle
        number [int, 0d]: number of poscars
        component [str, 0d]: component of searching system
        """
        system_echo(f'Generate Initial Data Randomly')
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
    
    def random_sampling(self, recyc, grain):
        """
        sampling randomly with symmetry
        
        Parameters
        ----------
        recyc [int, 0d]: times of recycling
        grain [float, 1d]: grain of grid
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        space_group [int, 1d]: space group number
        """
        grid_num = self.get_latt_num()
        init_path = f'{init_strus_path}_{recyc}'
        file_name = os.listdir(init_path)
        atom_pos, atom_type, atom_symm, grid_name, space_group = [], [], [], [], []
        for i, poscar in enumerate(file_name):
            #import lattice
            stru = Structure.from_file(f'{init_path}/{poscar}', sort=True)
            latt = stru.lattice
            atom_num = self.get_atom_number(stru)
            cry_system, params = self.judge_crystal_system(latt)
            latt = Lattice.from_parameters(*params)
            #sampling
            grid = grid_num + i
            pos_sample, type_sample, symm_sample, grid_sample, group_sample = \
                self.put_atoms_into_latt(atom_num, cry_system, grain, latt, grid)
            #append to buffer
            atom_pos += pos_sample
            atom_type += type_sample
            atom_symm += symm_sample
            grid_name += grid_sample
            space_group += group_sample
        return atom_pos, atom_type, atom_symm, grid_name, space_group
    
    def put_atoms_into_latt(self, atom_num, system, grain, latt, grid):
        """
        put atoms into lattice with different space groups
        
        Parameters
        ----------
        atom_num [dict, int:int]: number of different atoms\\
        system [int, 0d]: crystal system
        grain [float, 1d]: grain of grid
        latt [obj]: lattice object in pymatgen
        grid [int, 0d]: name of grid 
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        space_group [int, 1d]: space group number
        """
        #sampling space group randomly
        sgs = self.get_space_group(num_ave_sg, system)
        atom_pos, atom_type, atom_symm, grid_name, space_group = [], [], [], [], []
        for sg in sgs:
            #choose symmetry by space group
            all_grid, mapping = self.get_grid_points(sg, grain, latt)
            symm_site = self.group_symm_sites(mapping)
            assign_plan = self.assign_by_spacegroup(atom_num, symm_site)
            #put atoms into grid with symmetry constrain
            if len(assign_plan) > 0:
                grid_idx, grid_dis = self.import_data('grid', grid, sg)
                for assign in assign_plan:
                    type, symm = self.get_type_and_symm(assign)
                    for _ in range(num_per_sg):
                        pos, flag = self.get_pos(symm, symm_site, grid_idx, grid_dis)
                        if flag:
                            atom_pos.append(pos)
                    atom_type += [type for _ in range(num_per_sg)]
                    atom_symm += [symm for _ in range(num_per_sg)]
                    grid_name += [grid for _ in range(num_per_sg)]
                    space_group += [sg for _ in range(num_per_sg)]
                #export lattice file and mapping relationship
                head = f'{grid_path}/{grid:03.0f}'
                latt_file = f'{head}_latt_vec.bin'
                frac_file = f'{head}_frac_coords_{sg}.bin'
                mapping_file = f'{head}_mapping_{sg}.bin'
                if not os.path.exists(latt_file):
                    self.write_list2d(latt_file, latt.matrix, binary=True)
                if not os.path.exists(frac_file):
                    self.write_list2d(frac_file, all_grid, binary=True)
                    self.write_list2d(mapping_file, mapping, binary=True)
        return atom_pos, atom_type, atom_symm, grid_name, space_group
    
    def get_type_and_symm(self, assign):
        """
        get list of atom type and symmetry
        
        Parameters
        ----------
        assign [dict, int:list]: site assignment of atom
        e.g: assign {5:[1, 1, 2, 2], 6:[6, 12]}
        
        Returns
        ----------
        type [int, 1d]: type of atoms
        symm [int, 1d]: symmetry of atoms
        """
        type, symm = [], []
        for atom in assign.keys():
            value = assign[atom]
            symm += sorted(value)
            type += [atom for _ in range(len(value))]
        return type, symm

    def get_pos(self, symm, symm_site, grid_idx, grid_dis, trys=10):
        """
        sampling position of atoms by symmetry
        
        Parameters
        ----------
        symm [int, 1d]: symmetry of atoms\\
        symm_site [dict, int:list]: site position grouped by symmetry
        e.g. symm_site {1:[0], 2:[1, 2], 3:[3, 4, 5]}\\
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        
        Returns
        ----------
        pos [int, 1d]: position of atoms
        flag [bool, 0d]: whether get right initial position
        """
        flag = False
        counter = 0
        while counter < trys:
            pos = []
            for i in range(len(symm)):
                #get allowable sites
                allow = self.action_filter(self, i, pos, symm, symm_site,
                                           grid_idx, grid_dis, move=False)
                allow_num = len(allow)
                if allow_num == 0:
                    break
                #check distance of new generate symmetry atoms
                else:
                    for _ in range(allow_num):
                        point = np.random.choice(allow)
                        check_pos = pos.copy()
                        check_pos.append(point)
                        nbr_dis = self.get_nbr_dis(check_pos, grid_idx, grid_dis)
                        if self.near(nbr_dis):
                            pos.append(point)
                            break
            #check number of atoms
            if len(pos) == len(symm):
                flag = True
                break
            counter += 1
        return pos, flag
    
    def build_grid(self, grid_name):
        """
        build grid of structure
        
        Parameters
        ----------
        grid_name [int, 1d]: name of grid
        """
        #get lattice files
        latt_file = []
        grid_name = np.unique(grid_name)
        for grid in grid_name:
            latt_file.append(f'{grid:03.0f}_latt_vec.bin')
            latt_file.append(f'{grid:03.0f}_frac_coords_*.bin')
            latt_file.append(f'{grid:03.0f}_mapping_*.bin')
        #copy lattice file to cpu nodes
        self.zip_latt_vec(latt_file)
        for node in nodes:
            self.copy_latt_to_nodes(node)
        while not self.is_done('', self.num_node):
            time.sleep(self.wait_time)
        self.remove_flag('.')
        self.del_zip_latt()
        #build grid
        self.assign_to_cpu(grid_name)
    
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
    
    def geo_constrain(self, atom_pos, atom_type,
                      atom_symm, grid_name, space_group):
        """
        geometry constrain to reduce structures
        
        Parameters
        ----------
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
        #delete same samples roughly
        idx = self.delete_duplicates(atom_pos, atom_type, atom_symm,
                                     grid_name, space_group)
        atom_pos, atom_type, atom_symm, grid_name, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, space_group)
        #check overlay of atoms
        check_overlay = [self.overlay(i, len(i)) for i in atom_pos]
        check_num = len(check_overlay)
        idx = np.arange(check_num)[check_overlay]
        atom_pos, atom_type, atom_symm, grid_name, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm,
                                grid_name, space_group)
        #sort structure in order of grid and space group
        idx = self.sort_by_grid_sg(grid_name, space_group)
        atom_pos, atom_type, atom_symm, grid_name, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, space_group)
        #check neighbor distance of atoms
        nbr_dis = self.get_nbr_dis_bh(atom_pos, grid_name, space_group)
        mask = [self.near(i) for i in nbr_dis]
        atom_pos, atom_type, atom_symm, grid_name, space_group = \
            self.filter_samples(mask, atom_pos, atom_type, atom_symm,
                                grid_name, space_group)
        #sampling random samples by space group
        idx = self.balance_sampling(2*num_Rand, space_group)
        atom_pos, atom_type, atom_symm, grid_name, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm,
                                grid_name, space_group)
        #sort structure in order of grid and space group
        idx = self.sort_by_grid_sg(grid_name, space_group)
        atom_pos, atom_type, atom_symm, grid_name, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, space_group)
        #delete same samples by pymatgen
        idx = self.delete_duplicates_pymatgen(atom_pos, atom_type,
                                              grid_name, space_group)
        atom_pos, atom_type, atom_symm, grid_name, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm,
                                grid_name, space_group)
        #shuffle samples
        idx = np.arange(len(grid_name))
        random.shuffle(idx)
        idx = idx[:num_Rand]
        atom_pos, atom_type, atom_symm, grid_name, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm,
                                grid_name, space_group)
        return atom_pos, atom_type, atom_symm, grid_name, space_group
    
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
        cluster_per_num = [len(i) for i in clusters]
        assign = [0 for _ in range(cluster_num)]
        flag = True
        while flag:
            for i in range(cluster_num):
                if sum(assign) == num:
                    flag = False
                    break
                if sum(cluster_per_num) == 0:
                    flag = False
                    break
                if cluster_per_num[i] > 0:
                    cluster_per_num[i] -= 1
                    assign[i] += 1
        #sampling on different space groups
        sample = []
        for i, cluster in zip(assign, clusters):
            sample += random.sample(cluster, i)
        return sample
    
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
    
    def get_atom_number(self, stru):
        """
        get atom number of structure
        
        Parameters
        ----------
        stru [obj]: pymatgen structure object 

        Returns
        ----------
        atom_num [dict, int:int]: number of different atoms
        """
        atom_type = stru.atomic_numbers
        atom_num = dict(Counter(atom_type))
        return atom_num
    
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
    

if __name__ == '__main__':
    init = InitSampling()
    init_path = f'{init_strus_path}_{1}'
    file_name = os.listdir(init_path)
    for i, poscar in enumerate(file_name):
        #import lattice
        stru = Structure.from_file(f'{init_path}/{poscar}', sort=True)
        latt = stru.lattice
        atom_num = init.get_atom_number(stru)
        cry_system, params = init.judge_crystal_system(latt)
        latt = Lattice.from_parameters(*params)
        print(latt)