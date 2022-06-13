import os, sys
import re
import time
import random
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
                   GeoCheck, DeleteDuplicates):
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
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        """
        #generate initial lattice
        self.initial_poscars(recyc, num_latt, component)
        #build grid
        grids, sgs, assigns = self.build_grid(recyc, grain)
        #structures generated randomly
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.random_sampling(grids, sgs, assigns)
        #geometry check
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.geo_constrain(atom_pos, atom_type, atom_symm,
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
    
    def build_grid(self, recyc, grain):
        """
        build grid of structure

        Parameters
        ----------
        recyc [int, 0d]: times of recycle
        grain [float, 1d]: grain of grid

        Returns
        ----------
        grids [int, 1d]: number of grids
        sgs [int, 1d]: space groups
        assigns [dict, 2d, list]: assignments of atoms
        """
        #get start number of grid
        grid_num = self.get_latt_num()
        init_path = f'{init_strus_path}_{recyc}'
        file_name = os.listdir(init_path)
        grids, sgs, assigns = [], [], []
        for i, poscar in enumerate(file_name):
            #import lattice
            stru = Structure.from_file(f'{init_path}/{poscar}', sort=True)
            latt = stru.lattice
            atom_num = self.get_atom_number(stru)
            cry_system, params = self.judge_crystal_system(latt)
            latt = Lattice.from_parameters(*params)
            #get assignments
            grid = grid_num + i
            sgs_sample, assigns_sample =\
                self.get_assignment(atom_num, cry_system, grain, latt, grid)
            if len(sgs_sample) > 0:
                grids.append(grid)
                sgs.append(sgs_sample)
                assigns.append(assigns_sample)
        #get lattice files
        latt_file = []
        for grid in grids:
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
        self.assign_to_cpu(grids)
        system_echo('New grids have been built')
        return grids, sgs, assigns
        
    def get_assignment(self, atom_num_dict, system, grain, latt, grid):
        """
        put atoms into lattice with different space groups
        
        Parameters
        ----------
        atom_num_dict [dict, int:int]: number of different atoms\\
        system [int, 0d]: crystal system
        grain [float, 1d]: grain of grid
        latt [obj]: lattice object in pymatgen
        grid [int, 0d]: name of grid 
        
        Returns
        ----------
        sgs [int, 1d]: space groups
        assigns [dict, 2d, list]: assignments of atoms
        """
        #sampling space group randomly
        groups = self.get_space_group(num_ave_sg, system)
        sgs, assigns = [], []
        for sg in groups:
            atom_num = sum(atom_num_dict.values())
            assign_file = f'{json_path}/{grid}_{sg}_{atom_num}.json'
            if not os.path.exists(assign_file):
                #choose symmetry by space group
                all_grid, mapping = self.get_grid_points(sg, grain, latt)
                symm_site = self.group_symm_sites(mapping)
                assign_list = self.assign_by_spacegroup(atom_num_dict, symm_site)
                self.write_dict(assign_file, assign_list)
                #export lattice file and mapping relationship
                if len(assign_list) > 0:
                    head = f'{grid_path}/{grid:03.0f}'
                    latt_file = f'{head}_latt_vec.bin'
                    frac_file = f'{head}_frac_coords_{sg}.bin'
                    mapping_file = f'{head}_mapping_{sg}.bin'
                    if not os.path.exists(latt_file):
                        self.write_list2d(latt_file, latt.matrix, binary=True)
                    if not os.path.exists(frac_file):
                        self.write_list2d(frac_file, all_grid, binary=True)
                        self.write_list2d(mapping_file, mapping, binary=True)
                    sgs.append(sg)
                    assigns.append(assign_list) 
            #import existed assignments of atoms
            else:
                assign_list = self.import_dict(assign_file)
                if len(assign_list) > 0:
                    sgs.append(sg)
                    assigns.append(assign_list)
        return sgs, assigns

    def random_sampling(self, grids, sgs, assigns):
        """
        sampling randomly with symmetry
        
        Parameters
        ----------
        grids [int, 1d]: number of grids
        sgs [int, 1d]: space groups
        assigns [dict, 2d, list]: assignments of atoms
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        """
        #sampling space group randomly
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = [], [], [], [], [], []
        for i, grid in enumerate(grids):
            for sg, assign_list in zip(sgs[i], assigns[i]):
                mapping = self.import_data('mapping', grid, sg)
                symm_site = self.group_symm_sites(mapping)
                grid_idx, grid_dis = self.import_data('grid', grid, sg)
                #put atoms into grid with symmetry constrain
                for assign in assign_list:
                    counter = 0
                    type, symm = self.get_type_and_symm(assign)
                    for _ in range(num_per_sg):
                        pos, flag = self.get_pos(symm, symm_site, 1, grid_idx, grid_dis)
                        if flag:
                            atom_pos.append(pos)
                            counter += 1
                    atom_type += [type for _ in range(counter)]
                    atom_symm += [symm for _ in range(counter)]
                    grid_name += [grid for _ in range(counter)]
                    grid_ratio += [1 for _ in range(counter)]
                    space_group += [sg for _ in range(counter)]
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group
    
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
            plan = assign[atom]
            symm += sorted(plan)
            type += [atom for _ in range(len(plan))]
        return type, symm

    def get_pos(self, symm, symm_site, ratio, grid_idx, grid_dis, trys=5):
        """
        sampling position of atoms by symmetry
        
        Parameters
        ----------
        symm [int, 1d]: symmetry of atoms\\
        symm_site [dict, int:list]: site position grouped by symmetry
        e.g. symm_site {1:[0], 2:[1, 2], 3:[3, 4, 5]}\\
        ratio [float, 0d]: grid ratio
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
                allow = self.action_filter(i, pos, symm, symm_site,
                                           ratio, grid_idx, grid_dis, move=False)
                allow_num = len(allow)
                if allow_num == 0:
                    break
                #check distance of new generate symmetry atoms
                else:
                    for _ in range(10):
                        check_pos = pos.copy()
                        point = np.random.choice(allow)
                        check_pos.append(point)
                        if self.check_near(check_pos, ratio, grid_idx, grid_dis):
                            pos = check_pos
                            break
            #check number of atoms
            if len(pos) == len(symm):
                flag = True
                break
            counter += 1
        return pos, flag
    
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
    
    def geo_constrain(self, atom_pos, atom_type, atom_symm, 
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
        #delete same samples roughly
        idx = self.delete_duplicates(atom_pos, atom_type,
                                     grid_name, grid_ratio, space_group)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group)
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
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group)
        #delete same samples by pymatgen
        idx = self.delete_duplicates_between_sg_pymatgen(atom_pos, atom_type,
                                                         grid_name, grid_ratio, space_group)
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
            os.mkdir(json_path)


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