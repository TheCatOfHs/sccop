import os, sys
import time
import argparse
import numpy as np
import multiprocessing as mp

from collections import Counter
from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from core.path import *
from core.input import *
from core.space_group import *
from core.data_transfer import DeleteDuplicates
from core.grid_generate import GridGenerate
from core.search import ActionSpace
from core.utils import *


class ParallelSampling(ListRWTools, SSHTools):
    #divide grid by each node
    def __init__(self, wait_time=0.1):
        self.wait_time = wait_time
        self.local_grid_path = f'{SCCOP_path}/{grid_path}'
    
    def sampling_on_grid(self, recyc, start):
        """
        sampling radnomly on different grid

        Parameters
        ----------
        recyc [int, 0d]: recycle of sccop
        start [int, 0d]: start label of grid

        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        """
        init_path = f'{init_strus_path}_{recyc}'
        poscars = os.listdir(init_path)
        poscars_num = len(poscars)
        index = [i for i in range(poscars_num)]
        grids = [i for i in range(start, start+poscars_num)]
        node_assign = self.assign_node(poscars_num)
        #submit sampling job to each node
        work_node_num = self.sub_jobs(recyc, index, grids, node_assign)
        while not self.is_done(grid_path, work_node_num):
            time.sleep(self.wait_time)
        self.remove_flag_on_gpu()
        #update cpus
        for node in nodes[:work_node_num]:
            self.send_zip_to_cpu(node)
        while not self.is_done(grid_path, work_node_num):
            time.sleep(self.wait_time)
        self.remove_flag_on_gpu()
        self.remove_flag_on_cpu()
        system_echo(f'Unzip grid file on CPUs')
        #unzip on gpu
        self.unzip_on_gpu()
        while not self.is_done(grid_path, work_node_num):
            time.sleep(self.wait_time)
        self.remove_zip_on_gpu()
        self.remove_flag_on_gpu()
        system_echo(f'Unzip grid file on GPU')
        #collect samples
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.collect(recyc, grids)
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group
    
    def sub_jobs(self, recyc, index, grids, node_assign):
        """
        sub sampling to cpu nodes
        
        Parameters
        ----------
        recyc [int, 0d]: recycle of sccop
        index [int, 1d]: index of poscars
        grids [int, 1d]: name of grid
        node_assign [int, 1d]: node assign

        Returns
        ----------
        work_node_num [int, 0d]: number of work node
        """
        #grid grouped by nodes
        buffer_1, buffer_2 = [], []
        store_1, store_2 = [], []
        for node in nodes:
            for i, assign in enumerate(node_assign):
                if node == assign:
                    store_1.append(index[i])
                    store_2.append(grids[i])
            buffer_1.append(store_1)
            buffer_2.append(store_2)
            store_1, store_2 = [], []
        #submit sampling jobs to each node
        work_node_num = len(buffer_1)
        for i in range(work_node_num):
            self.sub_sampling(recyc, buffer_1[i], buffer_2[i], nodes[i])
        return work_node_num
    
    def sub_sampling(self, recyc, index, grids, node):
        """
        SSH to target node and sampling

        Parameters
        ----------
        recyc [int, 0d]: recycle of sccop
        index [int, 1d]: index of poscars
        grids [int, 1d]: name of grids
        node [int, 0d]: name of node
        """
        #generate sampling jobs
        ip = f'node{node}'
        sampling_jobs = []
        for i in range(len(index)):
            option = f'--recyc {recyc} --index {index[i]} --grid {grids[i]} '
            sampling_jobs.append(f'nohup python src/core/sampling.py {option} >& log&')
        sampling_jobs = ' '.join(sampling_jobs)
        #shell script of grid divide
        shell_script = f'''
                        #!/bin/bash
                        cd {SCCOP_path}/
                        rm -r data/grid/buffer
                        rm -r data/grid/json
                        mkdir data/grid/buffer
                        mkdir data/grid/json
                        scp {gpu_node}:{self.local_grid_path}/space_group_sampling.json data/grid/.
                        {sampling_jobs}
                        
                        while true;
                        do
                            num=`ps -ef | grep sampling.py | grep -v grep | wc -l`
                            if [ $num -eq 0 ]; then
                                rm log
                                break
                            fi
                            sleep 1s
                        done
                        
                        cd data/grid
                        touch FINISH-{node}
                        echo buffer >> record.dat
                        echo FINISH-{node} >> record.dat
                        cat record.dat | xargs tar -zcf {ip}.tar.gz
                        scp {ip}.tar.gz FINISH-{node} {gpu_node}:{self.local_grid_path}/
                        rm record.dat FINISH-{node} 
                        '''
        self.ssh_node(shell_script, ip)
    
    def send_zip_to_cpu(self, node):
        """
        update grid of each node
        """
        ip = f'node{node}'
        shell_script = f'''
                        #!/bin/bash
                        cd {self.local_grid_path}
                        scp {gpu_node}:{self.local_grid_path}/*.tar.gz .
                        
                        for i in *tar.gz
                        do
                            nohup tar -zxf $i >& log-$i &
                            rm log-$i
                        done
                        
                        touch FINISH-{node}
                        scp FINISH-{node} {gpu_node}:{self.local_grid_path}/
                        rm FINISH-{node} *.tar.gz
                        '''
        self.ssh_node(shell_script, ip)
    
    def unzip_on_gpu(self):
        """
        unzip grid property on gpu node
        """
        shell_script = f'''
                        #!/bin/bash
                        cd {self.local_grid_path}
                        for i in *.tar.gz
                        do
                            nohup tar -zxf $i >& log-$i &
                            rm log-$i
                        done
                        sleep 1
                        '''
        os.system(shell_script)
    
    def remove_flag_on_gpu(self): 
        """
        remove flag file on gpu
        """
        os.system(f'rm {grid_path}/FINISH*')
    
    def remove_flag_on_cpu(self):
        """
        remove FINISH flags on cpu
        """
        for node in nodes:
            ip = f'node{node}'
            shell_script = f'''
                            #!/bin/bash
                            cd {self.local_grid_path}
                            rm FINISH*
                            '''
            self.ssh_node(shell_script, ip)

    def remove_zip_on_gpu(self):
        """
        remove zip file
        """
        os.system(f'rm {grid_path}/*.tar.gz')

    def collect(self, recyc, grids):
        """
        collect data from each worker
        
        Parameters
        ----------
        recyc [int, 0d]: recycle of sccop
        grids [int, 1d]: name of grids

        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        """
        atom_pos, atom_type, atom_symm = [], [], []
        grid_name, grid_ratio, space_group =  [], [], []
        for grid in grids:
            #get name of path record files
            head = f'{buffer_path}/{recyc}'
            pos_file = f'{head}/atom_pos_{grid}.dat'
            type_file = f'{head}/atom_type_{grid}.dat'
            symm_file = f'{head}/atom_symm_{grid}.dat'
            grid_file = f'{head}/grid_name_{grid}.dat'
            ratio_file = f'{head}/grid_ratio_{grid}.dat'
            sg_file = f'{head}/space_group_{grid}.dat'
            #get searching results
            if os.path.exists(pos_file):
                pos = self.import_list2d(pos_file, int)
                type = self.import_list2d(type_file, int)
                symm = self.import_list2d(symm_file, int)
                grid = self.import_list2d(grid_file, int)
                ratio = self.import_list2d(ratio_file, float)
                sg = self.import_list2d(sg_file, int)
                atom_pos += pos
                atom_type += type
                atom_symm += symm
                grid_name += grid
                grid_ratio += ratio
                space_group += sg
        #convert to 1d list
        grid_name = np.array(grid_name).flatten().tolist()
        grid_ratio = np.array(grid_ratio).flatten().tolist()
        space_group = np.array(space_group).flatten().tolist()
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group
    

class AssignPlan(GridGenerate):
    #find assignment of atoms
    def get_assign(self, recyc, idx, grid, grain):
        """
        #get assignment of atoms in grid with different space group
        
        Parameters
        ----------
        recyc [int, 0d]: recycle of sccop
        idx [int, 0d]: index of poscar
        grid [int, 0d]: grid name
        grain [float, 1d]: grid of grid
        
        Returns
        ----------
        sgs [int, 1d]: space groups
        assigns [dict, 2d, list]: assignments of atoms
        """
        #get poscar
        init_path = f'{init_strus_path}_{recyc}'
        poscars = os.listdir(init_path)
        poscars = sorted([i for i in poscars if i.split('-')[1]=='Lattice'])
        poscar = f'{init_path}/{poscars[idx]}'
        poscar_split = poscars[idx].split('-')
        cry_system = int(poscar_split[2])
        #get lattice information
        stru = Structure.from_file(poscar, sort=True)
        latt = stru.lattice
        atom_num_dict = self.get_atom_number(stru)
        #get assignments
        sgs, assigns = self.get_plan(atom_num_dict, cry_system, grain, latt, grid)
        return sgs, assigns
        
    def get_plan(self, atom_num_dict, system, grain, latt, grid):
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
        groups = self.get_space_group(sg_per_latt, system)
        sgs, assigns = [], []
        for sg in groups:
            atom_num = atom_num_dict.values()
            assign_file = f'{json_path}/{grid}_{sg}_{sum(atom_num)}.json'
            if not os.path.exists(assign_file):
                #choose symmetry by space group
                if dimension == 2:
                    all_grid, mapping = self.get_grid_points_2d(sg, grain, latt, atom_num)
                elif dimension == 3:
                    all_grid, mapping = self.get_grid_points_3d(sg, grain, latt, atom_num)
                #check grid
                if len(all_grid) > 0:
                    symm_site = self.group_symm_sites(mapping)
                    assign_list = self.assign_by_spacegroup(atom_num_dict, symm_site)
                    self.write_dict(assign_file, assign_list)
                else:
                    assign_list = []
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


class RandomSampling(ActionSpace, AssignPlan, DeleteDuplicates):
    #sampling structure with distance constrain randomly
    def __init__(self):
        ActionSpace.__init__(self)
    
    def sampling(self, recyc, grid, sgs, assigns, parallel=20, num_per_sg=2):
        """
        sampling randomly with symmetry
        
        Parameters
        ----------
        recyc [int, 0d]: recycle of sccop
        grid [int, 0d]: number of grid
        sgs [int, 1d]: space groups
        assigns [dict, 2d, list]: assignments of atoms
        parallel [int, 0d]: number of parallel sampling
        num_per_sg [int, 0d]: sampling number each space group
        """
        #multi core
        pool = mp.Pool(processes=parallel)
        #sampling space group randomly
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            [], [], [], [], [], []
        for sg, assign_list in zip(sgs, assigns):
            mapping = self.import_data('mapping', grid, sg)
            symm_site = self.group_symm_sites(mapping)
            grid_idx, grid_dis = self.import_data('grid', grid, sg)
            #put atoms into grid with symmetry constrain
            for assign in assign_list:
                counter = 0
                type, symm = self.get_type_and_symm(assign)
                for _ in range(num_per_sg):
                    #run on multi cores
                    args_list = []
                    for _ in range(parallel):
                        args = (symm, symm_site, 1, grid_idx, grid_dis)
                        args_list.append(args)
                    pos_job = [pool.apply_async(self.get_pos, args) for args in args_list]
                    pos_pool = [p.get() for p in pos_job]
                    for pos, flag in pos_pool:
                        if flag:
                            atom_pos.append(pos)
                            counter += 1
                atom_type += [type for _ in range(counter)]
                atom_symm += [symm for _ in range(counter)]
                grid_name += [grid for _ in range(counter)]
                grid_ratio += [1 for _ in range(counter)]
                space_group += [sg for _ in range(counter)]
        #delete same structures of searching
        idx = self.delete_duplicates(atom_pos, atom_type,
                                     grid_name, grid_ratio, space_group)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group)
        #export random samples
        self.export_samples(recyc, grid, atom_pos, atom_type, atom_symm,
                            grid_name, grid_ratio, space_group)
        #close pool
        pool.close()
        
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
    
    def get_pos(self, symm, symm_site, ratio, grid_idx, grid_dis):
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
        while counter < 2:
            pos = []
            for i in range(len(symm)):
                #get allowable sites
                allow = self.action_filter(i, pos, symm, symm_site,
                                           ratio, grid_idx, grid_dis, move=False)
                if len(allow) == 0:
                    break
                #check distance of new generate symmetry atoms
                else:
                    for _ in range(5):
                        check_pos = pos.copy()
                        #choose allow position randomly
                        np.random.seed()
                        idx = np.random.randint(len(allow))
                        point = allow[idx]
                        del allow[idx]
                        check_pos.append(point)
                        #check distance
                        if self.check_near(check_pos, ratio, grid_idx, grid_dis):
                            pos = check_pos
                            break
                        if len(allow) == 0:
                            break
            #check number of atoms
            if len(pos) == len(symm):
                flag = True
                break
            counter += 1
        return pos, flag

    def export_samples(self, recyc, grid, atom_pos, atom_type, atom_symm,
                       grid_name, grid_ratio, space_group):
        """
        recyc [int, 0d]: 
        grid [int, 0d]: grid name
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        """
        path = f'{buffer_path}/{recyc}'
        if not os.path.exists(path):
            os.mkdir(path)
        self.write_list2d(f'{path}/atom_pos_{grid}.dat',
                          atom_pos, style='{0:4.0f}')
        self.write_list2d(f'{path}/atom_type_{grid}.dat',
                          atom_type, style='{0:4.0f}')
        self.write_list2d(f'{path}/atom_symm_{grid}.dat',
                          atom_symm, style='{0:4.0f}')
        self.write_list2d(f'{path}/grid_name_{grid}.dat',
                          np.transpose([grid_name]), style='{0:4.0f}')
        self.write_list2d(f'{path}/grid_ratio_{grid}.dat',
                          np.transpose([grid_ratio]), style='{0:8.4f}')
        self.write_list2d(f'{path}/space_group_{grid}.dat',
                          np.transpose([space_group]), style='{0:4.0f}')
        #record name of files
        shell_script = f'''
                        #!/bin/bash
                        cd {SCCOP_path}/{grid_path}
                        ls | grep {grid:03.0f} >> record.dat
                        '''
        os.system(shell_script)
        

if __name__ == '__main__':
    cutoff = 8
    min_bond = get_min_bond(composition)
    min_dis = min_bond/2
    grain = [min_dis, min_dis, min_dis]
    parser = argparse.ArgumentParser()
    parser.add_argument('--recyc', type=int)
    parser.add_argument('--index', type=int)
    parser.add_argument('--grid', type=int)
    args = parser.parse_args()
    
    recyc = args.recyc
    index = args.index
    grid = args.grid
    
    #get assignment
    ap = AssignPlan()
    sgs, assigns = ap.get_assign(recyc, index, grid, grain)
    if len(assigns) > 0:
        #build grid
        gg = GridGenerate()
        gg.build(grid, cutoff)
        #random sampling
        rs = RandomSampling()
        rs.sampling(recyc, grid, sgs, assigns)