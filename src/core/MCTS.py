import os, sys
import time
import argparse
import numpy as np
import multiprocessing as pythonmp

from collections import Counter
from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from core.log_print import *
from core.space_group import *
from core.GNN_tool import *
from core.grid_generate import GridGenerate
from core.multi_SA import ActionSpace
from core.utils import *


class ParallelSampling(ParallelPESUpdate):
    #monte carlo tree search in parallel
    def __init__(self, wait_time=0.1):
        ParallelPESUpdate.__init__(self)
        self.wait_time = wait_time
    
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
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: structure energy
        crys_vec [float, 2d]: cyrstal vector
        """
        init_path = f'{Init_Strus_Path}_{recyc:02.0f}'
        poscars = os.listdir(init_path)
        poscars = [i for i in poscars if len(i.split('-'))==4]
        poscars_num = len(poscars)
        index = [i for i in range(poscars_num)]
        grids = [i for i in range(start, start+poscars_num)]
        node_assign = self.assign_node(poscars_num)
        #submit sampling job to each node
        self.sub_mcts_jobs(recyc, index, grids, node_assign)
        while not self.is_done(Grid_Path, self.work_nodes_num):
            time.sleep(self.wait_time)
        self.remove_flag_on_host()
        system_echo(f'Initial sampling finish')
        #update work nodes
        for node in self.work_nodes:
            self.send_grid_between_work_nodes(node)
        while not self.is_done(Grid_Path, self.work_nodes_num, flag='GRID-SEND-DONE'):
            time.sleep(self.wait_time)
        for node in self.work_nodes:
            self.remove_flag_on_work_nodes(node)
        system_echo(f'Unzip grid file on work nodes')
        #collect samples
        self.unzip_mcts_samples_on_host()
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec = \
            self.collect_mcts_samples_all()
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec
    
    def sub_mcts_jobs(self, recyc, index, grids, node_assign):
        """
        submit sampling to work nodes
        
        Parameters
        ----------
        recyc [int, 0d]: recycle of sccop
        index [int, 1d]: index of poscars
        grids [int, 1d]: name of grid
        node_assign [int, 1d]: node assign
        """
        #submit sampling jobs to each work node
        last_node = node_assign[0]
        store_1, store_2 = [], []
        for i, node in enumerate(node_assign):
            if node == last_node:
                store_1.append(index[i])
                store_2.append(grids[i])
            else:
                self.sub_sampling(recyc, store_1, store_2, last_node)
                last_node = node
                store_1, store_2 = [], []
                store_1.append(index[i])
                store_2.append(grids[i])
        self.sub_sampling(recyc, store_1, store_2, last_node)
    
    def sub_sampling(self, recyc, index, grids, node, job_limit=20, job_monitor=3, core_limit=8, cpu_usage=20, wait_time=5, repeat_time=3):
        """
        SSH to target node and sampling
        
        Parameters
        ----------
        recyc [int, 0d]: recycle of sccop
        index [int, 1d]: index of poscars
        grids [int, 1d]: name of grids
        node [int, 0d]: name of node
        job_limit [int, 0d]: number of jobs in parallel
        job_monitor [int, 0d]: job number for monitoring sampling
        core_limit [int, 0d]: number of parallel core for each job
        cpu_usage [float, 0d]: low cpu usage for monitoring sampling
        wait_time [float, 0d]: waiting time for low cpu usage
        repeat_time [int, 0d]: repeat time for performing MCTS, at least 2
        """
        #generate sampling jobs
        job_num = len(index)
        assign_cores = self.assign_cores(job_num, core_limit)
        sampling_jobs = []
        for i in range(job_num):
            option = f'--flag 0 --node {node} --recyc {recyc} --index {index[i]} --grid {grids[i]} '
            sampling_jobs.append([f'src/core/MCTS.py {option}'])
        sampling_jobs.append([' '])
        job_file = f'sampling_jobs_{node}.dat'
        core_file = f'sampling_cores_{node}.dat'
        self.write_list2d(f'{self.local_grid_path}/{job_file}', sampling_jobs)
        self.write_list2d(f'{self.local_grid_path}/{core_file}', np.transpose([assign_cores]))
        #monitor MCTS process
        monitor_script = f'''
                          repeat=0
                          start=$(date +%s)
                          while true
                          do
                              end=$(date +%s)
                              time=$((end - start))
                              if [ $time -ge {Sampling_Time_Limit} ]; then
                                  rm {Grid_Path}/RUNNING*
                                  ps -ef | grep 'python src/core/MCTS.py' | grep -v grep | awk '{{print $2}}' | sort | uniq | xargs kill -9
                                  break
                              fi
                              
                              counter=`ls {Grid_Path} | grep RUNNING | wc -l`
                              cpu_sum=`ps aux | grep 'python src/core/MCTS.py' | grep -v grep | awk '{{sum += $3}} END {{print sum+0}}'`
                              if [ $counter -eq 0 ]; then
                                  cpu_ave=0
                              else
                                  cpu_ave=`echo "$cpu_sum / $counter" | bc -l`
                              fi
                              if [ $counter -le {job_monitor} ]; then
                                  is_cpu_low=`echo "$cpu_ave < {cpu_usage}" | bc -l`
                                  ((repeat++))
                                  if [ $is_cpu_low -eq 1 ]; then
                                      ((repeat++))
                                      if [ $repeat -ge {wait_time} ]; then
                                          rm {Grid_Path}/RUNNING*
                                          ps -ef | grep 'python src/core/MCTS.py' | grep -v grep | awk '{{print $2}}' | sort | uniq | xargs kill -9
                                          break
                                      fi
                                  fi
                                  sleep 1s
                              fi
                              sleep 1s
                              echo $cpu_ave >> data/grid/cpu_use
                              echo $counter >> data/grid/cpu_num
                          done
                          echo ---------- >> data/grid/cpu_use
                          echo ---------- >> data/grid/cpu_num
                          rm {Grid_Path}/RUNNING*
                          '''
        mcts_script = f'''
                       sub_counter=0
                       cat {job_file} | while read line
                       do
                           flag=`echo $line | awk '{{print $NF}}'`
                           cores=`echo $line | awk '{{print $1}}'`
                           params=`echo $line | awk '{{$1=""; print $0}}'`
                           if [ $sub_counter -le {job_limit} ]; then
                               taskset -c $cores python $params >> log&
                               touch {Grid_Path}/RUNNING_$flag
                               ((sub_counter++))
                               echo $sub_counter >> data/grid/counter_log
                           else
                               counter=`ls {Grid_Path} | grep RUNNING | wc -l`
                               echo $counter >> data/grid/counter_log
                               if [ $counter -eq 0 ]; then
                                   sleep {wait_time}s
                               else
                                   if [ $counter -gt {job_limit} ]; then
                                       while true
                                       do
                                           counter=`ls {Grid_Path} | grep RUNNING | wc -l`
                                           echo $counter >> data/grid/counter_log
                                           if [ $counter -le {job_limit} ]; then
                                               break
                                           fi
                                           sleep 1s
                                       done
                                   fi
                               fi
                               taskset -c $cores python $params >> log&
                               touch {Grid_Path}/RUNNING_$flag
                               ((sub_counter++))
                               echo $sub_counter >> data/grid/counter_log
                           fi
                       done
                       echo ---------- >> data/grid/counter_log
                       '''
        #shell script of grid divide
        shell_script = f'''
                        #!/bin/bash --login
                        {SCCOP_Env}
                        
                        cd {SCCOP_Path}/
                        rm -r data/grid/buffer
                        rm -r data/grid/json
                        mkdir data/grid/buffer
                        mkdir data/grid/json
                        scp {Host_Node}:{self.local_grid_path}/space_group_sampling.json data/grid/.
                        scp {Host_Node}:{self.local_grid_path}/{job_file} .
                        scp {Host_Node}:{self.local_grid_path}/{core_file} .
                        
                        for i in `seq 1 {repeat_time}`
                        do
                            paste -d ' ' {core_file} {job_file} | head -n $(wc -l < {job_file}) > {job_file}
                            {mcts_script}
                            {monitor_script}
                            if [ -f tmp_sampling_jobs.dat ]; then
                                mv tmp_sampling_jobs.dat {job_file}
                            else
                                break
                            fi
                        done
                        rm log {core_file} {job_file}
                        python src/core/MCTS.py --flag 3 --node {node}
                        
                        cd data/grid
                        cat grid_record.dat | xargs tar -zcf {node}.tar.gz
                        rm grid_record.dat
                        
                        cd buffer
                        scp {node}.tar.gz {Host_Node}:{self.local_grid_path}/buffer/.
                        
                        cd ../
                        touch FINISH-{node}
                        scp FINISH-{node} {Host_Node}:{self.local_grid_path}/.
                        '''
        self.ssh_node(shell_script, node)
    
    def send_grid_between_work_nodes(self, node):
        """
        update grid of each work node

        Parameters
        ----------
        node [str, 0d]: name of node
        """
        work_nodes = ' '.join([f'{i}' for i in self.work_nodes])
        shell_script = f'''
                        #!/bin/bash --login
                        cd {self.local_grid_path}
                        for i in {work_nodes}
                        do
                            if [ $i != {node} ]
                            then
                                scp $i:{self.local_grid_path}/$i.tar.gz .
                            fi
                        done
                        
                        for i in {work_nodes}
                        do
                            if [ $i != {node} ]
                            then
                                tar -zxf $i.tar.gz
                            fi
                        done
                        
                        touch GRID-SEND-DONE-{node}
                        scp GRID-SEND-DONE-{node} {Host_Node}:{self.local_grid_path}/.
                        '''
        self.ssh_node(shell_script, node)
    

class AssignPlan(GridGenerate):
    #find assignment of atoms
    def get_assign(self, recyc, idx, grid, grain):
        """
        get assignment of atoms in grid with different space group
        
        Parameters
        ----------
        recyc [int, 0d]: recycle of sccop
        idx [int, 0d]: index of poscar
        grid [int, 0d]: grid name
        grain [float, 1d]: grain of grid
        
        Returns
        ----------
        sgs [int, 1d]: space groups
        assigns [dict, 2d, list]: assignments of atoms
        job [str, 0d]: type of job
        """
        #get poscar
        init_path = f'{Init_Strus_Path}_{recyc:02.0f}'
        poscars = [i for i in os.listdir(init_path) if len(i.split('-'))==4]
        poscars = sorted(poscars, key=lambda x: x.split('-')[-1])
        ct = poscars[idx].split('-')
        job, cs = ct[1], int(ct[2])
        poscar = f'{init_path}/{poscars[idx]}'
        #generate lattice
        if General_Search or Cluster_Search:
            self.generate_latt(poscar, cs)
        #get lattice information
        stru = Structure.from_file(poscar)
        latt = stru.lattice
        latt_file = f'{Grid_Path}/{grid:03.0f}_latt_vec.bin'
        self.write_list2d(latt_file, latt.matrix, binary=True)
        #get assignments
        sgs, assigns = self.get_plan(stru, job, cs, grain, latt, grid)
        return sgs, assigns, job
    
    def get_plan(self, stru, job, crys_system, grain, latt, grid):
        """
        put atoms into lattice with different space groups
        
        Parameters
        ----------
        stru [obj, 0d]: pymatgen structure object
        job [str, 0d]: type of jobs
        crys_system [int, 0d]: crystal system
        grain [float, 1d]: grain of grid
        latt [obj]: lattice object in pymatgen
        grid [int, 0d]: name of grid 
        
        Returns
        ----------
        sgs [int, 1d]: space groups
        assigns [dict, 2d, list]: assignments of atoms
        """
        #sampling space group randomly
        if crys_system > 0:
            space_groups = self.get_space_group(SG_per_Latt, crys_system)
        else:
            space_groups = [1 for _ in range(SG_per_Latt)]
        sgs, assigns = [], []
        atom_num_dicts, max_num_dict = self.get_atom_number()
        for sg in space_groups:
            all_grid, mapping = self.get_grid_coord_mapping(stru, job, crys_system, sg, grain, latt, grid, max_num_dict)
            #get atom assignments
            for atom_num_dict in atom_num_dicts:
                atom_num = atom_num_dict.values()
                assign_file = f'{Json_Path}/{grid}_{sg}_{sum(atom_num)}.json'
                #check grid
                if len(all_grid) > 0:
                    symm_site = self.group_symm_sites(mapping)
                    assign_list = self.get_assign_parallel(atom_num_dict, symm_site)
                    self.write_dict(assign_file, assign_list)
                else:
                    assign_list = []
                #export lattice file and mapping relationship
                if len(assign_list) > 0:
                    sgs.append(sg)
                    assigns.append(assign_list)
        return sgs, assigns
    
    def get_grid_coord_mapping(self, stru, job, crys_system, sg, grain, latt, grid, max_num_dict):
        """
        generate grid fraction coordinate and DAU mapping files
        
        Parameters
        ----------
        stru [obj, 0d]: pymatgen structure object
        job [str, 0d]: type of jobs
        crys_system [int, 0d]: crystal system
        sg [int, 0d]: space group number
        grain [float, 1d]: grain of grid
        latt [obj]: lattice object in pymatgen
        grid [int, 0d]: name of grid 
        max_num_dict [list, dict, int:int]: number of maximum atoms
        
        Returns
        ----------
        all_grid [float, 2d, np]: fraction coordinates of grid points
        mapping [int, 2d]: mapping between min and all grid
        """
        head = f'{Grid_Path}/{grid:03.0f}'
        frac_file = f'{head}_frac_coords_{sg}.bin'
        mapping_file = f'{head}_mapping_{sg}.bin'
        atom_num = max_num_dict.values()
        #discrete space into grid
        if crys_system > 0:
            if Dimension == 2:
                all_grid, mapping = self.get_grid_points_2d(sg, grain, latt, atom_num)
            elif Dimension == 3:
                all_grid, mapping = self.get_grid_points_3d(sg, grain, latt, atom_num)
        else:
            if job == 'Template':
                all_grid, mapping = self.get_grid_points_seeds(stru)
        #export grid file
        if len(all_grid) > 0:
            self.write_list2d(frac_file, all_grid, binary=True)
            self.write_list2d(mapping_file, mapping, binary=True)
        else:
            all_grid, mapping = [], []
        return all_grid, mapping
    
    def get_atom_number(self, max_num=5):
        """
        get suitable atom number of structure
        
        Parameters
        ----------
        max_num [int, 0d]: maximum number of atoms

        Returns
        ----------
        atom_num_dicts [list, dict, int:int]: number of different atoms
        max_num_dict [list, dict, int:int]: number of maximum atoms
        """
        atom_types = self.import_list2d(f'{Grid_Path}/atom_types.dat', int)
        max_num_dict = dict(Counter(atom_types[-1]))
        #limit number of atom types
        type_num = len(atom_types)
        if type_num > max_num:
            idx = np.arange(type_num)
            np.random.shuffle(idx)
            idx = idx[:max_num]
            atom_types = np.array(atom_types, dtype=object)[idx]
        #transfer to dictionary
        atom_num_dicts = []
        for atom_type in atom_types:
            num_dict = dict(Counter(atom_type))
            atom_num_dicts.append(num_dict)
        return atom_num_dicts, max_num_dict


class RandomSampling(ActionSpace, AssignPlan, GNNSlice):
    #sampling structure with distance constrain randomly
    def __init__(self):
        ActionSpace.__init__(self)
        GNNSlice.__init__(self)
    
    def sampling(self, grid, sgs, assigns, job, repeat=5, core_limit=8):
        """
        sampling randomly with symmetry
        
        Parameters
        ----------
        grid [int, 0d]: number of grid
        sgs [int, 1d]: space groups
        assigns [dict, 2d, list]: assignments of atoms \\
        job [str, 0d]: type of job
        repeat [int, 0d]: repeat times of parallel sampling
        core_limit [int, 0d]: limit of core number
        """
        #sampling space group randomly
        atom_pos, atom_type, atom_symm = [], [], []
        grid_name, grid_ratio, space_group = [], [], []
        angles, thicks = [], []
        #get parallel sampling jobs
        last_sg = sgs[0]
        mapping = self.import_data('mapping', grid, last_sg)
        symm_site = self.group_symm_sites(mapping)
        grid_idx, grid_dis = self.import_data('grid', grid, last_sg)
        args_list = []
        for sg, assign_list in zip(sgs, assigns):
            for assign in assign_list:
                for _ in range(repeat):
                    if sg != last_sg:
                        mapping = self.import_data('mapping', grid, sg)
                        symm_site = self.group_symm_sites(mapping)
                        grid_idx, grid_dis = self.import_data('grid', grid, sg)
                        last_sg = sg
                    if General_Search or Cluster_Search:
                        ratio = 1 + 0.5*np.random.random()
                    elif Template_Search:
                        ratio = 1
                    args = (grid, ratio, sg, assign, symm_site, grid_idx, grid_dis, job)
                    args_list.append(args)
        #multi-cores
        with pythonmp.get_context('fork').Pool(processes=core_limit) as pool:
            #put atoms into grid with symmetry constrain
            jobs = [pool.apply_async(self.get_strus, args) for args in args_list]
            #get results
            pool.close()
            pool.join()
            jobs_pool = [p.get() for p in jobs]
            for pos, type, symm, grids, ratio, sg, angle, thick in jobs_pool:
                atom_pos += pos
                atom_type += type
                atom_symm += symm
                grid_name += grids
                grid_ratio += ratio
                space_group += sg
                angles += angle
                thicks += thick
        pool.close()
        del pool
        #export samples
        flag = False
        if len(atom_pos) > 0:
            flag = True
            self.export_sampling_samples(atom_pos, atom_type, atom_symm,
                                         grid_name, grid_ratio, space_group,
                                         angles, thicks)
        else:
            os.system(f'''
                      #!/bin/bash
                      cd {SCCOP_Path}/{Grid_Path}
                      rm {grid:03.0f}_*
                      
                      cd json/
                      rm {grid}_*
                      ''')
        return flag
    
    def get_strus(self, grid, ratio, sg, assign, symm_site, grid_idx, grid_dis, job, repeat=5):
        """
        get random structures for 1 core
        
        Parameters
        ----------
        grid [int, 0d]: number of grid
        ratio [float, 0d]: grid ratio
        sg [int, 0d]: space group
        assign [dict, 1d, list]: assignments of atoms
        symm_site [dict, int:list]: site position grouped by symmetry
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        job [str, 0d]: type of job
        repeat [int, 0d]: repeat times of sampling
        
        Returns
        ----------
        atom_pos [int, 2d]: atom positions
        atom_type [int, 2d]: atom types
        atom_symm [int, 2d]: atom symmetry
        grid_name [int, 1d]: grid name
        grid_ratio [int, 1d]: grid ratio
        space group [int, 1d]: space groups
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        """
        #put atoms into grid
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = [], [], [], [], [], []
        type_all, symm_all = self.get_type_and_symm(assign)
        for _ in range(repeat):
            if job == 'RCSD' or job == 'Latt':
                type, symm = type_all, symm_all
                pos_save, flag = self.get_pos_general(type, symm, symm_site, ratio, grid_idx, grid_dis)
            elif job == 'Template':
                fixed_pos, type, symm = self.fix_template_points(type_all, symm_all)
                pos_save, symm, flag = self.get_pos_template(type, symm, symm_site, ratio, grid_idx, grid_dis, fixed_pos)
            if flag:
                num = len(pos_save)
                atom_pos += pos_save
                atom_type = [type for _ in range(num)]
                atom_symm += [symm for _ in range(num)]
                grid_name += [grid for _ in range(num)]
                grid_ratio += [ratio for _ in range(num)]
                space_group += [sg for _ in range(num)]
        #adjust angles and thicks
        if len(atom_pos) > 0:
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
                self.sampling_angles_thicks(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group)
        else:
            angles, thicks = [], []
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks

    def get_fix_type(self):
        """
        get atom type from seed

        Returns
        ----------
        seed_type [int, 1d]:free_type
        """
        seed = os.listdir(f'{Seed_Path}')[0]
        stru = Structure.from_file(f'{Seed_Path}/{seed}')
        seed_type = list(stru.atomic_numbers)
        return seed_type
    
    def get_free_type(self, fixed_type, all_type):
        """
        get type of free atoms
        
        Parameters
        ----------
        fixed_type [int, 1d]: type of fixed atoms
        all_type [int, 1d]: type of all atoms

        Returns
        ----------
        free_type [int, 1d]: type of free atoms
        """
        del_idx = []
        store = fixed_type.copy()
        for idx_1, type_1 in enumerate(all_type):
            for idx_2, type_2 in enumerate(store):
                if type_1 == type_2:
                    del_idx.append(idx_1)
                    del store[idx_2]
                    break
        free_type = np.delete(all_type, del_idx)
        return free_type
    
    def fix_template_points(self, type, symm, low_ratio=.9, high_ratio=.9):
        """
        fix points of template
        
        Parameters
        ----------
        type [int, 1d]: all type of atoms
        symm [int, 1d]: symmetry of atoms
        low_ratio [float, 0d]: lower boundary of fixed ratio
        high_ratio [float, 0d]: higher boundary of fixed ratio
        
        Returns
        ----------
        fixed_pos [int, 1d]: position of fixed atoms
        type [int, 1d]: type of atoms after fixed
        symm [int, 1d]: symmetry of atoms after fixed
        """
        atom_num = len(symm)
        fixed_num, fixed_pos = 0, []
        if Num_Fixed_Temp > 0:
            if Disturb_Seeds:
                seeds_pos = self.import_list2d(f'{Seed_Path}/seeds_pos.dat', int)
                seeds_num = len(seeds_pos)
                idx = np.random.randint(low=0, high=seeds_num)
                seed_pos = seeds_pos[idx]
                fixed_pos = seed_pos[:Num_Fixed_Temp]
                #fixed sites of seed randomly
                left_num = atom_num - Num_Fixed_Temp
                lower_boundary = int(low_ratio*left_num)
                higher_boundary = max(lower_boundary+1, int(high_ratio*left_num))
                fixed_num = np.random.randint(low=lower_boundary, high=higher_boundary)
                fixed_pos += np.random.choice(seed_pos[Num_Fixed_Temp:], fixed_num, replace=False).tolist()
            else:
                fixed_pos = [i for i in range(Num_Fixed_Temp)]
            #label fixed points
            add_num = atom_num - Num_Fixed_Temp - fixed_num
            symm = [-1 for _ in range(Num_Fixed_Temp+fixed_num)] + [1 for _ in range(add_num)]
            #get atom type of points
            seed_type = self.get_fix_type()
            fixed_type = seed_type[:Num_Fixed_Temp+fixed_num]
            free_type = self.get_free_type(fixed_type, type)
            type = np.concatenate((fixed_type, free_type)).tolist()
        return fixed_pos, type, symm
    
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
    
    def get_pos_general(self, type, symm, symm_site, ratio, grid_idx, grid_dis,
                        fixed_pos=[], max_leaf_ratio=.8, leaf_num=5, max_num=100):
        """
        sampling position of atoms by symmetry
        
        Parameters
        ----------
        type [int, 1d]: type of atoms
        symm [int, 1d]: symmetry of atoms
        symm_site [dict, int:list]: site position grouped by symmetry
        e.g. symm_site {1:[0], 2:[1, 2], 3:[3, 4, 5]}
        ratio [float, 0d]: grid ratio
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        fixed_pos [int, 1d]: fixed positions
        max_leaf_ratio [float, 0d]: max leaf layer and the last leaf note as 1
        leaf_num [int, 0d]: maximum leaf number
        max_num [int, 0d]: maximum sampling number
        
        Returns
        ----------
        pos_save [int, 2d]: position of atoms
        flag [bool, 0d]: whether get right initial position
        """
        flag = False
        atom_num = len([i for i in symm if i > 0])
        #Monte Carlo Tree Search
        repeat = 0
        if Cluster_Search:
            deepth = 0
        else:
            deepth = max(1, int(max_leaf_ratio*atom_num))
        while repeat < 5:
            pos_store = [fixed_pos]
            for i in range(atom_num):
                store = []
                for pos in pos_store:
                    #get allowable sites
                    allow = self.action_filter(i, pos, type, symm, symm_site,
                                               ratio, grid_idx, grid_dis, move=False)
                    if len(allow) == 0:
                        pass
                    else:
                        #random sampling
                        np.random.seed()
                        np.random.shuffle(allow)
                        counter = 0
                        center_type, check_type = type[i], type[:i+1]
                        nbr_cutoff = self.get_nbr_cutoff(center_type, self.ele_types, self.bond_list)
                        for point in allow[:leaf_num]:
                            #check distance of new generate symmetry atoms
                            check_pos = pos.copy()
                            check_pos.append(point)
                            _, center_nbr_type, center_nbr_dis = \
                                self.get_atom_neighbors(point, check_pos, check_type, ratio, grid_idx, grid_dis, nbr_cutoff)
                            if len(center_nbr_type) > 0:
                                center_nbr_bond_list = \
                                    self.get_nbr_bond_list(center_type, center_nbr_type, self.ele_types, self.bond_list)
                                if self.check_nbr_dis(center_nbr_dis, center_nbr_bond_list):
                                    store.append(check_pos)
                                    counter += 1
                            else:
                                store.append(check_pos)
                                counter += 1
                            if i == atom_num - deepth:
                                pass
                            else:
                                break
                    if len(store) > max_num:
                        break
                pos_store = store
                if len(pos_store) == 0:
                    break
            #check number of atoms
            pos_save = []
            for pos in pos_store:
                if len(pos) == len(symm):
                    pos_save.append(pos)
            if len(pos_save) > 0:
                flag = True
                break
            repeat += 1
        return pos_save, flag
    
    def get_pos_template(self, type, symm, symm_site, ratio, grid_idx, grid_dis, fixed_pos=[], max_leaf_ratio=.5, leaf_num=5, max_num=100):
        """
        sampling position of atoms by symmetry
        
        Parameters
        ----------
        type [int, 1d]: type of atoms
        symm [int, 1d]: symmetry of atoms
        symm_site [dict, int:list]: site position grouped by symmetry
        e.g. symm_site {1:[0], 2:[1, 2], 3:[3, 4, 5]}\\
        ratio [float, 0d]: grid ratio
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        fixed_pos [int, 1d]: fixed positions
        max_leaf_ratio [float, 0d]: max leaf layer and the last leaf note as 1
        leaf_num [int, 0d]: maximum leaf number
        max_num [int, 0d]: maximum sampling number
        
        Returns
        ----------
        pos_save [int, 2d]: position of atoms
        symm [int, 1d]: symmetry of atoms
        flag [bool, 0d]: whether get right initial position
        """
        flag = False
        atom_num = len([i for i in symm if i > 0])
        #Monte Carlo Tree Search
        repeat = 0
        deepth = max(1, int(max_leaf_ratio*atom_num))
        while repeat < 5:
            pos_store = [fixed_pos]
            for i in range(atom_num):
                store = []
                for pos in pos_store:
                    #get allowable sites
                    allow = self.action_filter(i, pos, type, symm, symm_site,
                                               ratio, grid_idx, grid_dis, move=False)
                    if len(allow) == 0:
                        pass
                    else:
                        #check distance of new generate symmetry atoms
                        np.random.seed()
                        np.random.shuffle(allow)
                        for point in allow[:leaf_num]:
                            check_pos = pos.copy()
                            check_pos.append(point)
                            store.append(check_pos)
                            if i == atom_num - deepth:
                                pass
                            else:
                                break
                    if len(store) > max_num:
                        break
                pos_store = store
                if len(pos_store) == 0:
                    break
            #check number of atoms
            pos_save = []
            for pos in pos_store:
                if len(pos) == len(symm):
                    pos_save.append(pos)
            if len(pos_save) > 0:
                flag = True
                break
            repeat += 1
        if Disturb_Seeds:
            add_num = len(symm) - Num_Fixed_Temp
            symm = [-1 for _ in range(Num_Fixed_Temp)] + [1 for _ in range(add_num)]
        return pos_save, symm, flag
    
    def import_sampling_samples(self, grid):
        """
        import MCTS samples

        Parameters
        ----------
        grid [int, 0d]: grid name

        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [float, 2d]: atom displacement in z-direction
        """
        atom_pos = self.import_list2d(f'{Recyc_Store_Path}/atom_pos_{grid}.dat', int)
        atom_type = self.import_list2d(f'{Recyc_Store_Path}/atom_type_{grid}.dat', int)
        atom_symm = self.import_list2d(f'{Recyc_Store_Path}/atom_symm_{grid}.dat', int)
        grid_name = self.import_list2d(f'{Recyc_Store_Path}/grid_name_{grid}.dat', int)
        grid_ratio = self.import_list2d(f'{Recyc_Store_Path}/grid_ratio_{grid}.dat', float)
        space_group = self.import_list2d(f'{Recyc_Store_Path}/space_group_{grid}.dat', int)
        angles = self.import_list2d(f'{Recyc_Store_Path}/angles_{grid}.dat', int)
        thicks = self.import_list2d(f'{Recyc_Store_Path}/thicks_{grid}.dat', int)
        #flatten list
        grid_name = np.concatenate(grid_name).tolist()
        grid_ratio = np.concatenate(grid_ratio).tolist()
        space_group = np.concatenate(space_group).tolist()
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks
    
    def export_sampling_samples(self, atom_pos, atom_type, atom_symm,
                                grid_name, grid_ratio, space_group, angles, thicks, limit=5000):
        """
        export MCTS samples
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [float, 2d]: atom displacement in z-direction
        limit [int, 0d]: limit of sample number
        """
        #limit sampling number
        grid = grid_name[0]
        sample_num = min(len(grid_name), limit)
        idx = np.random.choice(np.arange(sample_num), sample_num, replace=False)
        idx = sorted(idx)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        #export samples
        self.write_list2d(f'{Recyc_Store_Path}/atom_pos_{grid}.dat',
                          atom_pos)
        self.write_list2d(f'{Recyc_Store_Path}/atom_type_{grid}.dat',
                          atom_type)
        self.write_list2d(f'{Recyc_Store_Path}/atom_symm_{grid}.dat',
                          atom_symm)
        self.write_list2d(f'{Recyc_Store_Path}/grid_name_{grid}.dat',
                          np.transpose([grid_name]))
        self.write_list2d(f'{Recyc_Store_Path}/grid_ratio_{grid}.dat',
                          np.transpose([grid_ratio]), style='{0:8.4f}')
        self.write_list2d(f'{Recyc_Store_Path}/space_group_{grid}.dat',
                          np.transpose([space_group]))
        self.write_list2d(f'{Recyc_Store_Path}/angles_{grid}.dat',
                          angles)
        self.write_list2d(f'{Recyc_Store_Path}/thicks_{grid}.dat',
                          thicks)
    
    def write_mcts_record(self, grid):
        """
        write record of mcts
        
        Parameters
        ----------
        grid [int, 0d]: grid name
        """
        #record name of files
        shell_script = f'''
                        #!/bin/bash --login
                        cd {SCCOP_Path}/{Grid_Path}
                        ls | grep {grid:03.0f}_ >> grid_record.dat
                        
                        cd {SCCOP_Path}/{Buffer_Path}
                        echo {grid} >> samples_record.dat
                        '''
        os.system(shell_script)
    
    def collect_mcts_samples_process(self, grid):
        """
        collect samples for one process
        
        Parameters
        ----------
        grid [int, 0d]: number of grid
        """
        #import samples
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = self.import_sampling_samples(grid)
        #delete same structures
        idx = self.delete_duplicates(atom_pos, atom_type, atom_symm,
                                     grid_name, grid_ratio, space_group, angles, thicks)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        #sorted by grid and space group
        idx = self.sort_by_grid_sg(grid_name, space_group)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        #calculate energy and crystal vector
        energys, crys_vec = self.update_PES(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks)
        #export mcts samples
        self.export_mcts_samples(grid, atom_pos, atom_type, atom_symm, grid_name,
                                 grid_ratio, space_group, angles, thicks, energys, crys_vec)
    
    def collect_mcts_samples_node(self):
        """
        collect samples for one node

        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: structure energy
        crys_vec [float, 2d]: crystal vectors
        """
        atom_pos, atom_type, atom_symm = [], [], []
        grid_name, grid_ratio, space_group, angles, thicks =  [], [], [], [], []
        energys, crys_vec = [], []
        grids = self.import_list2d(f'{Buffer_Path}/samples_record.dat', int)
        grids = np.concatenate(grids)
        for grid in grids:
            #get name of path record files
            pos_file = f'{Buffer_Path}/atom_pos_{grid}.dat'
            type_file = f'{Buffer_Path}/atom_type_{grid}.dat'
            symm_file = f'{Buffer_Path}/atom_symm_{grid}.dat'
            grid_file = f'{Buffer_Path}/grid_name_{grid}.dat'
            ratio_file = f'{Buffer_Path}/grid_ratio_{grid}.dat'
            sg_file = f'{Buffer_Path}/space_group_{grid}.dat'
            angle_file = f'{Buffer_Path}/angles_{grid}.dat'
            thick_file = f'{Buffer_Path}/thicks_{grid}.dat'
            energy_file = f'{Buffer_Path}/energy_{grid}.dat'
            crys_vec_file = f'{Buffer_Path}/crys_vec_{grid}.bin'
            #get searching results
            if os.path.exists(energy_file):
                pos = self.import_list2d(pos_file, int)
                type = self.import_list2d(type_file, int)
                symm = self.import_list2d(symm_file, int)
                grid = self.import_list2d(grid_file, int)
                ratio = self.import_list2d(ratio_file, float)
                sg = self.import_list2d(sg_file, int)
                angle = self.import_list2d(angle_file, int)
                thick = self.import_list2d(thick_file, int)
                energy = self.import_list2d(energy_file, float)
                vec = self.import_list2d(crys_vec_file, float, binary=True)
                atom_pos += pos
                atom_type += type
                atom_symm += symm
                grid_name += grid
                grid_ratio += ratio
                space_group += sg
                angles += angle
                thicks += thick
                energys += energy
                crys_vec += vec.tolist()
        #convert to 1d list
        grid_name = np.array(grid_name).flatten().tolist()
        grid_ratio = np.array(grid_ratio).flatten().tolist()
        space_group = np.array(space_group).flatten().tolist()
        energys = np.array(energys).flatten().tolist()
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec
    
    def export_mcts_samples_node(self, node, atom_pos, atom_type, atom_symm,
                                 grid_name, grid_ratio, sgs, angles, thicks, energys, crys_vec):
        """
        export monte carlo tree search samples
        
        Parameters
        ----------
        node [int, 0d]: node name
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: grid name
        grid_ratio [float, 1d]: ratio of grids
        sgs [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: structure energy
        crys_vec [float, 2d]: crystal vector
        """
        #export sampling results
        self.write_list2d(f'{Buffer_Path}/atom_pos_{node}.dat',
                          atom_pos)
        self.write_list2d(f'{Buffer_Path}/atom_type_{node}.dat',
                          atom_type)
        self.write_list2d(f'{Buffer_Path}/atom_symm_{node}.dat',
                          atom_symm)
        self.write_list2d(f'{Buffer_Path}/grid_name_{node}.dat',
                          np.transpose([grid_name]))
        self.write_list2d(f'{Buffer_Path}/grid_ratio_{node}.dat',
                          np.transpose([grid_ratio]), style='{0:8.4f}')
        self.write_list2d(f'{Buffer_Path}/space_group_{node}.dat',
                          np.transpose([sgs]))
        self.write_list2d(f'{Buffer_Path}/angles_{node}.dat',
                          angles)
        self.write_list2d(f'{Buffer_Path}/thicks_{node}.dat',
                          thicks)
        self.write_list2d(f'{Buffer_Path}/energy_{node}.dat',
                          np.transpose([energys]), style='{0:9.6f}')
        self.write_list2d(f'{Buffer_Path}/crys_vec_{node}.bin',
                          np.array(crys_vec), binary=True)
        #remove files and zip samples
        os.system(f'''
                  #!/bin/bash
                  cd {SCCOP_Path}/{Buffer_Path}
                  
                  cat samples_record.dat | while read line
                  do
                    rm atom_pos_$line.dat atom_type_$line.dat
                    rm atom_symm_$line.dat grid_name_$line.dat
                    rm grid_ratio_$line.dat space_group_$line.dat
                    rm angles_$line.dat thicks_$line.dat
                    rm energy_$line.dat crys_vec_$line.bin
                  done
                  
                  xargs tar -zcf {node}.tar.gz *
                  ''')


if __name__ == '__main__':
    torch.set_num_threads(1)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--flag', type=int, default=0)
    parser.add_argument('--node', type=str, default='0')
    parser.add_argument('--recyc', type=int, default=0)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--grid', type=int, default=0)
    args = parser.parse_args()
    
    flag = args.flag
    node = args.node
    recyc = args.recyc
    index = args.index
    grid = args.grid
    #monte carlo tree search
    if flag == 0:
        rw = ListRWTools()
        bond_dict = rw.import_dict(Bond_File)
        max_bond = bond_dict['max_bond']
        ave_bond = bond_dict['ave_bond']
        cutoff = max_bond
        min_dis = .5*ave_bond
        grain = [min_dis, min_dis, min_dis]
        #get assignment
        ap = AssignPlan()
        sgs, assigns, job = ap.get_assign(recyc, index, grid, grain)
        sample_flag = False
        if len(assigns) > 0:
            #build grid
            gg = GridGenerate()
            if Template_Search:
                gg.build(grid, 8)
            else:
                gg.build(grid, cutoff)
            #random sampling
            rs = RandomSampling()
            sample_flag = rs.sampling(grid, sgs, assigns, job)
        if sample_flag:
            tmp = f'src/core/MCTS.py --flag 1 --node {node} --recyc {recyc} --index {index} --grid {grid}'
            os.system(f'''
                    #!/bin/bash --login
                    cd {SCCOP_Path}
                    echo {tmp} >> tmp_sampling_jobs.dat
                    ''')
    #collect samples of each process
    elif flag == 1:
        rs = RandomSampling()
        rs.collect_mcts_samples_process(grid)
        tmp = f'src/core/MCTS.py --flag 2 --node {node} --recyc {recyc} --index {index} --grid {grid}'
        os.system(f'''
                  #!/bin/bash --login
                  cd {SCCOP_Path}
                  echo {tmp} >> tmp_sampling_jobs.dat
                  ''')
    #slice PES
    elif flag == 2:
        rs = RandomSampling()
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec = \
            rs.import_mcts_samples(grid)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec = \
            rs.gnn_slice_local(atom_pos, atom_type, atom_symm, grid_name,
                               grid_ratio, space_group, angles, thicks, energys, crys_vec)
        #export mcts samples
        rs.export_mcts_samples(grid, atom_pos, atom_type, atom_symm, grid_name,
                               grid_ratio, space_group, angles, thicks, energys, crys_vec)
        rs.write_mcts_record(grid)
    #collect samples on node
    elif flag == 3:
        rs = RandomSampling()
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec = \
            rs.collect_mcts_samples_node()
        rs.export_mcts_samples_node(node, atom_pos, atom_type, atom_symm, grid_name,
                                    grid_ratio, space_group, angles, thicks, energys, crys_vec)
    
    #remove running flag
    os.system(f'''
              #!/bin/bash --login
              cd {SCCOP_Path}/{Grid_Path}
              if [ -f RUNNING_{grid} ]; then
                  rm RUNNING_{grid}
              fi
              ''')