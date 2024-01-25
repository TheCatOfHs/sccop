import os, sys, time
import argparse
import itertools
import torch
import numpy as np

sys.path.append(f'{os.getcwd()}/src')
from core.log_print import *
from core.del_duplicates import DeleteDuplicates
from core.grid_generate import GridGenerate
from core.utils import *
from core.GNN_model import FeatureExtractNet, ReadoutNet
from core.GNN_tool import GNNPredict


class ParallelWorkers(SSHTools, DeleteDuplicates):
    #Assign sampling jobs to each node
    def __init__(self, sleep_time=1):
        SSHTools.__init__(self)
        self.sleep_time = sleep_time
    
    def search(self, iteration, init_pos, init_type, init_symm,
               init_grid, init_ratio, init_sg, init_angles, init_thicks):
        """
        optimize structure by ML on grid
        
        Parameters
        ----------
        iteration [int, 0d]: searching iteration
        init_pos [int, 2d]: initial position
        init_type [int, 2d]: initial atom type
        init_symm [int, 2d]: initial symmetry
        init_grid [int, 1d]: initial grid name
        init_ratio [float, 1d]: initial grid ratio
        init_sg [int, 1d]: initial space group
        init_angles [int, 2d]: initial angles
        init_thicks [int, 2d]: initial thicks
        """
        self.iteration = f'{iteration:02.0f}'
        self.sh_save_path = f'{Search_Path}/ml_{self.iteration}'
        self.model_save_path = f'{Model_Path}/{self.iteration}'
        self.generate_job(iteration, init_pos, init_type, init_symm,
                          init_grid, init_ratio, init_sg, init_angles, init_thicks)
        pos, type, symm, angle, thick, job = self.read_job()
        system_echo('Get the job node!')
        #sub searching job to each node
        work_node_num = self.sub_job_to_workers(pos, type, symm, angle, thick, job)
        system_echo('Successful assign works to workers!')
        while not self.is_done(self.sh_save_path, work_node_num):
            time.sleep(self.sleep_time)
        self.unzip()
        #collect searching path
        self.collect_all(iteration)
        self.remove_zip_files()
    
    def generate_job(self, iteration, init_pos, init_type, init_symm,
                     init_grid, init_ratio, init_sg, init_angles, init_thicks):
        """
        assign jobs by the following files
        initial_pos_XXX.dat: initial position
        initial_type_XXX.dat: initial type
        initial_symm_XXX.dat: initial symmetry
        worker_job_XXX.dat: node job
        e.g. iteration path node grid ratio sg
             1 1 131 1 1 183
        
        Parameters
        ----------
        iteration [int, 0d]: searching iteration
        init_pos [int, 2d]: initial position
        init_type [int, 2d]: initial atom type
        init_symm [int, 2d]: initial symmetry
        init_grid [int, 1d]: initial grid name
        init_ratio [float, 1d]: initial grid ratio
        init_sg [int, 1d]: initial space group
        init_angles [int, 2d]: initial angles
        init_thicks [int, 2d]: initial thicks
        """
        if not os.path.exists(self.sh_save_path):
            os.mkdir(self.sh_save_path)
        worker_job = []
        path_num = len(init_grid)
        node_assign = self.assign_node(path_num)
        for i, node in enumerate(node_assign):
            job = [iteration, i, node, init_grid[i], init_ratio[i], init_sg[i]]
            worker_job.append(job)
        #export searching files
        pos_file = f'{self.sh_save_path}/initial_pos_{self.iteration}.dat'
        type_file = f'{self.sh_save_path}/initial_type_{self.iteration}.dat'
        symm_file = f'{self.sh_save_path}/initial_symm_{self.iteration}.dat'
        angle_file = f'{self.sh_save_path}/initial_angles_{self.iteration}.dat'
        thick_file = f'{self.sh_save_path}/initial_thicks_{self.iteration}.dat'
        worker_file = f'{self.sh_save_path}/worker_job_{self.iteration}.dat'
        self.write_list2d(pos_file, init_pos)
        self.write_list2d(type_file, init_type)
        self.write_list2d(symm_file, init_symm)
        self.write_list2d(angle_file, init_angles)
        self.write_list2d(thick_file, init_thicks)
        self.write_list2d(worker_file, worker_job)
        
    def sub_job_to_workers(self, pos, type, symm, angle, thick, job):
        """
        sub searching jobs to nodes

        Parameters
        ----------
        pos [str, 1d, np]: initial pos
        type [str, 1d, np]: initial type
        symm [str, 1d, np]: initial symmetry
        angle [str, 1d, np]: initial angle
        thick [str, 1d, np]: initial thick
        job [str, 2d, np]: jobs assgined to nodes
        """
        #poscars grouped by nodes
        pos_node, type_node, symm_node, angles_node, thicks_node, job_node = [], [], [], [], [], []
        pos_assign, type_assign, symm_assign, angles_assign, thicks_assign, job_assign = [], [], [], [], [], []
        for node in self.work_nodes:
            for i, line in enumerate(job):
                #get node
                label = line[2]
                if label == node:
                    pos_assign.append(pos[i])
                    type_assign.append(type[i])
                    symm_assign.append(symm[i])
                    angles_assign.append(angle[i])
                    thicks_assign.append(thick[i])
                    job_assign.append(line)
            #assign jobs to node
            pos_node.append(pos_assign)
            type_node.append(type_assign)
            symm_node.append(symm_assign)
            angles_node.append(angles_assign)
            thicks_node.append(thicks_assign)
            job_node.append(np.transpose(job_assign))
            pos_assign, type_assign, symm_assign, angles_assign, thicks_assign, job_assign = [], [], [], [], [], []
        #sub job to target node
        for i, j, k, m, n, assign in zip(pos_node, type_node, symm_node, angles_node, thicks_node, job_node):
            self.sampling_with_ssh(i, j, k, m, n, *assign)
        work_node_num = len(pos_node)
        return work_node_num
    
    def sampling_with_ssh(self, atom_pos, atom_type, atom_symm, angles, thicks,
                          iteration, path, nodes, grid_name, grid_ratio, space_group, job_limit=100, wait_time=60):
        """
        SSH to target node and call workers for sampling

        Parameters
        ----------
        atom_pos [str, 1d]: initial atom position
        atom_type [str, 1d]: initial atom type
        atom_symm [str, 1d]: initial atom symmetry 
        angles [str, 1d]: initial angles
        thicks [str, 1d]: initial thicks
        iteration [str, 1d]: searching iteration
        path [str, 1d]: searching path
        nodes [str, 1d]: searching node
        grid_name [str, 1d]: name of grid
        grid_ratio [str, 1d]: ratio of grid
        space_group [str, 1d]: space group number
        job_limit [int, 0d]: limit of parallel jobs
        wait_time [float, 0d]: wait time for ML-SA
        """
        node = nodes[0]
        search_jobs = []
        for i in range(len(atom_pos)):
            option = f'--pos {atom_pos[i]} --type {atom_type[i]} --symm {atom_symm[i]} ' \
                     f'--angle {angles[i]} --thick {thicks[i]} ' \
                     f'--iteration {iteration[i]} --path {path[i]} --node {node} ' \
                     f'--grid {grid_name[i]} --ratio {grid_ratio[i]} --sg {space_group[i]}'
            search_jobs.append([f'src/core/multi_SA.py {option}'])
        search_jobs.append([' '])
        job_file = f'search_jobs_{node}.dat'
        self.write_list2d(f'{self.sh_save_path}/{job_file}', search_jobs)
        #ssh to target node then search from different start
        local_sh_save_path = f'{SCCOP_Path}/{self.sh_save_path}'
        job_num = f'`ls {self.sh_save_path} | grep RUNNING | wc -l`'
        update_script = f'''
                         if [ ! -d {self.sh_save_path} ]; then
                            mkdir {self.sh_save_path}
                         fi
                         scp {Host_Node}:{local_sh_save_path}/{job_file} .
                         '''
        sa_script = f'''
                     cat {job_file} | while read line
                     do
                         flag=`echo $line | grep -o -- "--path [0-9]\+" | awk '{{print $2}}'`
                         python $line >> log&
                         touch {self.sh_save_path}/RUNNING_$flag
                         counter={job_num}
                         while [ $counter -ge {job_limit} ]
                         do
                             counter={job_num}
                             sleep 0.1s
                         done
                     done
                     '''
        check_script = f'''
                        cd {self.sh_save_path}
                        for flag in RUNNING_*
                        do
                            number=$(echo $flag | grep -o '[0-9]\+')
                            echo $number >> tmp_finish.dat
                        done
                        rm RUNNING_*
                        
                        cd ../../../
                        mv {self.sh_save_path}/tmp_finish.dat .
                        awk 'NR==FNR {{a[$1]=1; next}} {{for(i=1;i<=NF;i++) {{if($i=="--path" && $(i+1) in a) {{print; break}}}}}}' tmp_finish.dat {job_file} > tmp_record.dat
                        mv tmp_record.dat {job_file}
                        rm tmp_finish.dat
                        '''
        shell_script = f'''
                        #!/bin/bash --login
                        {SCCOP_Env}
                        
                        cd {SCCOP_Path}/
                        {update_script}
                        {sa_script}
                        
                        counter={job_num}
                        repeat=0
                        while [ $counter -ge 1 ]
                        do
                            counter={job_num}
                            running=`ps -ef | grep 'python src/core/multi_SA.py' | grep -v grep | wc -l`
                            ((repeat++))
                            sleep 1s
                            if [ $repeat -ge {wait_time} ]; then
                                rm {self.sh_save_path}/RUNNING_*
                                ps -ef | grep 'python src/core/multi_SA.py' | grep -v grep | awk '{{print $2}}' | sort | uniq | xargs kill -9
                                break
                            fi
                            if [ $running -eq 0 ]; then
                                if [ $counter -ge 1 ]; then
                                    {check_script}
                                    break
                                fi
                            fi
                        done
                        
                        {sa_script}
                        
                        running=`ps -ef | grep 'python src/core/multi_SA.py' | grep -v grep | wc -l`
                        while [ $running -ge 1 ]
                        do
                            running=`ps -ef | grep 'python src/core/multi_SA.py' | grep -v grep | wc -l`
                            ((repeat++))
                            sleep 1s
                            if [ $repeat -ge {wait_time} ]; then
                                rm {self.sh_save_path}/RUNNING_*
                                ps -ef | grep 'python src/core/multi_SA.py' | grep -v grep | awk '{{print $2}}' | sort | uniq | xargs kill -9
                                break
                            fi
                        done
                        rm log {job_file}
                        
                        python src/core/multi_SA.py --flag 1 --iteration {iteration[0]}
                        
                        cd {self.sh_save_path}
                        rm RUNNING_*
                        touch FINISH-{node}
                        scp search-{node}.tar.gz FINISH-{node} {Host_Node}:{local_sh_save_path}
                        '''
        self.ssh_node(shell_script, node)
    
    def unzip(self):
        """
        unzip files of finish path
        """
        zip_file = os.listdir(self.sh_save_path)
        zip_file = [i for i in zip_file if i.endswith('gz')]
        zip_file = ' '.join(zip_file)
        shell_script = f'''
                        #!/bin/bash --login
                        cd {self.sh_save_path}
                        for i in {zip_file}
                        do
                            tar -zxf $i --warning=no-timestamp
                        done
                        sleep 1s
                        '''
        os.system(shell_script)
    
    def collect_all(self, iteration):
        """
        collect search results of each node
        
        Parameters
        ----------
        iteration [int, 0d]: search iteration
        """
        self.iteration = f'{iteration:02.0f}'
        self.sh_save_path = f'{Search_Path}/ml_{self.iteration}'
        atom_pos, atom_type, atom_symm = [], [], []
        grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec = [], [], [], [], [], [], []
        for node in self.work_nodes:
            #get name of path record files
            pos_file = f'{self.sh_save_path}/atom_pos_{node}.dat'
            type_file = f'{self.sh_save_path}/atom_type_{node}.dat'
            symm_file = f'{self.sh_save_path}/atom_symm_{node}.dat'
            grid_file = f'{self.sh_save_path}/grid_name_{node}.dat'
            ratio_file = f'{self.sh_save_path}/grid_ratio_{node}.dat'
            sg_file = f'{self.sh_save_path}/space_group_{node}.dat'
            angle_file = f'{self.sh_save_path}/angles_{node}.dat'
            thick_file = f'{self.sh_save_path}/thicks_{node}.dat'
            energy_file = f'{self.sh_save_path}/energy_{node}.dat'
            vec_file = f'{self.sh_save_path}/crys_vec_{node}.bin'
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
                vec = self.import_list2d(vec_file, float, binary=True)
                if len(energy) > 0:
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
            else:
                pass
        #delete same structures
        grid_name = np.concatenate(grid_name)
        grid_ratio = np.concatenate(grid_ratio)
        space_group = np.concatenate(space_group)
        idx = self.delete_duplicates(atom_pos, atom_type, atom_symm,
                                     grid_name, grid_ratio, space_group, angles, thicks)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = np.concatenate(energys)[idx]
        crys_vec = np.array(crys_vec)[idx]
        #sorted by energy
        idx = np.argsort(energys)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = energys[idx]
        crys_vec = crys_vec[idx]
        #delete duplicates by crystal vectors
        if Use_ML_Clustering:
            idx = self.delete_duplicates_crys_vec_parallel(crys_vec, energys)
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, grid_ratio, space_group, angles, thicks)
            energys = energys[idx]
            crys_vec = crys_vec[idx]
        #export optimization points
        self.export_all_results(atom_pos, atom_type, atom_symm,
                                grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec)
        system_echo(f'Number of samples: {len(grid_name)}')
    
    def export_all_results(self, atom_pos, atom_type, atom_symm,
                           grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec):
        """
        collect search results of all nodes

        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 2d]: prediction energys
        crys_vec [float, 2d]: crystal vectors
        """
        atom_pos = atom_pos[:Num_Sample_Limit]
        atom_type = atom_type[:Num_Sample_Limit]
        atom_symm = atom_symm[:Num_Sample_Limit]
        grid_name = np.transpose([grid_name])[:Num_Sample_Limit]
        grid_ratio = np.transpose([grid_ratio])[:Num_Sample_Limit]
        space_group = np.transpose([space_group])[:Num_Sample_Limit]
        angles = angles[:Num_Sample_Limit]
        thicks = thicks[:Num_Sample_Limit]
        energys = np.transpose([energys])[:Num_Sample_Limit]
        crys_vec = crys_vec[:Num_Sample_Limit]
        #export searching results
        self.write_list2d(f'{self.sh_save_path}/atom_pos.dat', atom_pos)
        self.write_list2d(f'{self.sh_save_path}/atom_type.dat', atom_type)
        self.write_list2d(f'{self.sh_save_path}/atom_symm.dat', atom_symm)
        self.write_list2d(f'{self.sh_save_path}/grid_name.dat', grid_name)
        self.write_list2d(f'{self.sh_save_path}/grid_ratio.dat', grid_ratio, style='{0:8.4f}')
        self.write_list2d(f'{self.sh_save_path}/space_group.dat', space_group)
        self.write_list2d(f'{self.sh_save_path}/angles.dat', angles)
        self.write_list2d(f'{self.sh_save_path}/thicks.dat', thicks)
        self.write_list2d(f'{self.sh_save_path}/energys.dat', energys, style='{0:9.6f}')
        self.write_list2d(f'{self.sh_save_path}/crys_vec.bin', crys_vec, binary=True)
        
    def collect_path(self, iteration):
        """
        collect search results of each worker
        
        Parameters
        ----------
        iteration [int, 0d]: search iteration
        """
        self.iteration = f'{iteration:02.0f}'
        self.sh_save_path = f'{Search_Path}/ml_{self.iteration}'
        jobs = self.import_list2d(f'{self.sh_save_path}/record.dat', dtype=str)
        atom_pos, atom_type, atom_symm = [], [], []
        grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec, lack = [], [], [], [], [], [], [], []
        for assign in jobs:
            #get name of path record files
            pos_file = self.get_file_name('pos', *assign)
            type_file = self.get_file_name('type', *assign)
            symm_file = self.get_file_name('symm', *assign)
            grid_file = self.get_file_name('grid', *assign)
            ratio_file = self.get_file_name('ratio', *assign)
            sg_file = self.get_file_name('sg', *assign)
            angle_file = self.get_file_name('angles', *assign)
            thick_file = self.get_file_name('thicks', *assign)
            energy_file = self.get_file_name('energy', *assign)
            vec_file = self.get_file_name('vec', *assign)
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
                vec = self.import_list2d(vec_file, float, binary=True)
                if len(energy) > 0:
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
                else:
                    lack.append(pos_file)
            else:
                lack.append(pos_file)
        #record lack files
        node = jobs[0][-1]
        self.export_lack_files(node, lack)
        #delete same structures
        grid_name = np.concatenate(grid_name)
        grid_ratio = np.concatenate(grid_ratio)
        space_group = np.concatenate(space_group)
        idx = self.delete_duplicates(atom_pos, atom_type, atom_symm,
                                     grid_name, grid_ratio, space_group, angles, thicks)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = np.concatenate(energys)[idx]
        crys_vec = np.array(crys_vec)[idx]
        #filter by energy
        idx = np.argsort(energys)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = energys[idx]
        crys_vec = crys_vec[idx]
        #delete duplicates by crystal vectors
        if Use_ML_Clustering:
            idx = self.delete_duplicates_crys_vec_parallel(crys_vec, energys)
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, grid_ratio, space_group, angles, thicks)
            energys = energys[idx]
            crys_vec = crys_vec[idx]
        #export search results
        self.export_worker_results(node, atom_pos, atom_type, atom_symm,
                                   grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec)
    
    def export_worker_results(self, node, atom_pos, atom_type, atom_symm,
                              grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec):
        """
        export search results
        
        Parameters
        ----------
        node [str, 0d]: computational node
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: prediction energys
        crys_vec [float, 2d]: crystal vectors
        """
        grid_name = np.transpose([grid_name])
        grid_ratio = np.transpose([grid_ratio])
        space_group = np.transpose([space_group])
        energys = np.transpose([energys])
        #export searching results
        self.write_list2d(f'{self.sh_save_path}/atom_pos_{node}.dat', atom_pos)
        self.write_list2d(f'{self.sh_save_path}/atom_type_{node}.dat', atom_type)
        self.write_list2d(f'{self.sh_save_path}/atom_symm_{node}.dat', atom_symm)
        self.write_list2d(f'{self.sh_save_path}/grid_name_{node}.dat', grid_name)
        self.write_list2d(f'{self.sh_save_path}/grid_ratio_{node}.dat', grid_ratio, style='{0:8.4f}')
        self.write_list2d(f'{self.sh_save_path}/space_group_{node}.dat', space_group)
        self.write_list2d(f'{self.sh_save_path}/angles_{node}.dat', angles)
        self.write_list2d(f'{self.sh_save_path}/thicks_{node}.dat', thicks)
        self.write_list2d(f'{self.sh_save_path}/energy_{node}.dat', energys, style='{0:9.6f}')
        self.write_list2d(f'{self.sh_save_path}/crys_vec_{node}.bin', crys_vec, binary=True)
        #record name of files
        shell_script = f'''
                        #!/bin/bash --login
                        cd {self.sh_save_path}
                        echo lack_{node}.dat >> tar_record.dat
                        echo atom_pos_{node}.dat >> tar_record.dat
                        echo atom_type_{node}.dat >> tar_record.dat
                        echo atom_symm_{node}.dat >> tar_record.dat
                        echo grid_name_{node}.dat >> tar_record.dat
                        echo grid_ratio_{node}.dat >> tar_record.dat
                        echo space_group_{node}.dat >> tar_record.dat
                        echo angles_{node}.dat >> tar_record.dat
                        echo thicks_{node}.dat >> tar_record.dat
                        echo energy_{node}.dat >> tar_record.dat
                        echo crys_vec_{node}.bin >> tar_record.dat
                        '''
        os.system(shell_script)
        #tar search files
        self.remove_all()
        os.system(f'''
                  #!/bin/bash
                  cd {self.sh_save_path}
                  cat tar_record.dat | xargs tar -zcf search-{node}.tar.gz 
                  ''')
    
    def export_lack_files(self, node, lack_files):
        """
        export name of lacked files
        
        Parameters
        ----------
        node [str, 0d]: computational node
        lack_files [str, 2d]: name of lacked files 
        """
        if len(lack_files) > 0:
            lack_files = np.transpose([lack_files])
        else:
            lack_files = []
        self.write_list2d(f'{self.sh_save_path}/lack_{node}.dat', lack_files)
    
    def get_file_name(self, name, iteration, path, node):
        """
        read result of each worker
        
        Parameters
        ----------
        name [str, 0d]: file name
        iteration [str, 0d]: iteration of searching
        path [str, 0d]: path of searching
        node [str, 0d]: computational node
        
        Returns
        ----------
        file [int, 2d]: name of search file
        """
        if name == 'vec':
            file = f'{self.sh_save_path}/' \
                   f'{name}-{iteration}-{path}-{node}.bin'
        else:
            file = f'{self.sh_save_path}/' \
                   f'{name}-{iteration}-{path}-{node}.dat'
        return file
    
    def read_job(self):
        """
        import initialize file
        
        Returns
        ----------
        pos [str, 1d, np]: initial position string list
        type [str, 1d, np]: initial type string list
        symm [str, 1d, np]: initial symmetry string list
        angle [str, 1d, np]: initial angle string list
        thick [str, 1d, np]: initial thick string list
        job [str, 2d, np]: job assign string list
        """
        pos = self.read_dat(f'initial_pos_{self.iteration}.dat')
        type = self.read_dat(f'initial_type_{self.iteration}.dat')
        symm = self.read_dat(f'initial_symm_{self.iteration}.dat')
        angle = self.read_dat(f'initial_angles_{self.iteration}.dat')
        thick = self.read_dat(f'initial_thicks_{self.iteration}.dat')
        job = self.read_dat(f'worker_job_{self.iteration}.dat', split=True)
        return pos, type, symm, angle, thick, job
    
    def read_dat(self, dat, split=False):
        """
        read initilize file as string list
        
        Parameters
        ----------
        dat [str, 0d]: name of initilize file
        split [bool, 0d]: whether split item
        
        Returns
        ----------
        list [str, 1d or 2d, np]: return string list
        """
        file = f'{self.sh_save_path}/{dat}'
        with open(file, 'r') as f:
            ct = f.readlines()
        if split:
            list = [item.split() for item in ct]
        else:
            list = [item.replace('\n','') for item in ct]
        return np.array(list)
    
    def remove_all(self):
        """
        remove file of each path
        """
        shell_script = f'''
                        #!/bin/bash --login
                        cd {self.sh_save_path}
                        rm pos-* type-* symm-*
                        rm grid-* ratio-* sg-* energy-*
                        rm angles-* thicks-* vec-*
                        '''
        os.system(shell_script)
    
    def remove_zip_files(self):
        """
        remove file of each path
        """
        shell_script = f'''
                        #!/bin/bash --login
                        cd {self.sh_save_path}
                        rm *.tar.gz
                        rm FINISH*
                        '''
        os.system(shell_script)
    

class ActionSpace(ListRWTools):
    #action space of optimizing position of atoms
    def __init__(self):
        bond_dict = self.import_dict(Bond_File)
        self.min_bond = bond_dict['min_bond']
        self.max_bond = bond_dict['max_bond']
        self.ele_types = bond_dict['ele_types']
        self.bond_list = bond_dict['bond_list']
        
    def action_filter(self, idx, pos, type, symm, symm_site,
                      ratio, grid_idx, grid_dis, move=True, limit=10):
        """
        distance between atoms should bigger than mininum bond length
        
        Parameters
        ----------
        idx [int, 0d]: index of select atom
        pos [int, 1d]: position of atoms
        type [int, 1d]: type of atoms
        symm [int, 1d]: symmetry of atoms
        symm_site [dict, int:list]: site position grouped by symmetry
        ratio [float, 0d]: grid ratio
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        move [bool, 0d]: move atom or add atom
        limit [int, 0d]: limit of available sites
        
        Returns
        ----------
        allow [int, 1d]: allowable actions
        """
        if move:
            obstacle = np.delete(pos, idx, axis=0).tolist()
        else:
            obstacle = pos
        #get forbidden sites
        equal_symm = np.abs(symm).tolist()
        symm_slt = equal_symm[idx]
        sites = symm_site[symm_slt]
        if len(obstacle) > 0:
            forbid = self.get_forbid(obstacle, type, ratio, grid_idx, grid_dis)
            occupy = self.get_occupy(symm_slt, pos, equal_symm)
        else:
            forbid, occupy = [], []
        #get available sites
        forbid = np.intersect1d(sites, forbid)
        vacancy = np.setdiff1d(sites, occupy)
        available = np.setdiff1d(vacancy, forbid)
        #check distance of self-symmetry sites
        np.random.seed()
        np.random.shuffle(available)
        available = available[:limit]
        center_idx = self.ele_types.index(type[idx])
        nbr_cutoff = self.bond_list[center_idx][center_idx]
        allow = self.check_self_symm_site(available, ratio, grid_idx, grid_dis, nbr_cutoff)
        return allow
    
    def get_forbid(self, points, type, ratio, grid_idx, grid_dis):
        """
        get position of forbidden area
        
        Parameters
        ----------
        points [int, 1d]: occupied points
        type [int, 1d]: type of target atoms
        ratio [float, 0d]: grid ratio
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid

        Returns
        ----------
        forbid [int, 1d, np]: forbidden area
        """
        nbr_idx = grid_idx[points]
        nbr_dis = grid_dis[points]*ratio
        actions = []
        for i, item in enumerate(nbr_dis):
            #get distance cutoff
            center_idx = self.ele_types.index(type[i])
            nbr_cutoff = np.min(self.bond_list[center_idx])
            #get forbidden area
            for j, dis in enumerate(item):
                if nbr_cutoff < dis:
                    break
            point_nbr_idx = nbr_idx[i]
            actions.append(point_nbr_idx[:j])
        forbid = np.unique(np.concatenate(actions))
        return forbid
    
    def get_occupy(self, symm_slt, atom_pos, atom_symm):
        """
        get occupied position by symmetry
        
        Parameters
        ----------
        symm_slt [int, 0d]: select symmetry
        atom_pos [int, 1d]: position of atoms
        atom_symm [int, 1d]: symmetry of atoms

        Returns
        ----------
        occupy [int, 1d]: position of occupied sites
        """
        occupy = []
        for i, pos in enumerate(atom_pos):
            if symm_slt == atom_symm[i]:
                occupy.append(pos)
        return occupy
    
    def check_self_symm_site(self, points, ratio, grid_idx, grid_dis, nbr_cutoff):
        """
        get allowable sites by checking self-symmetry sites
        
        Parameters
        ----------
        points [int, 1d]: potential allowable sites
        ratio [float, 0d]: grid ratio
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        nbr_cutoff [float, 0d]: cutoff distance of neighbors
        
        Returns
        ----------
        allow [int, 1d]: allowable sites
        """
        allow = []
        for point in points:
            nbr_idx = grid_idx[point]
            nbr_dis = grid_dis[point]*ratio
            #get forbidden sites
            tmp_idx = []
            for i, dis in enumerate(nbr_dis):
                if dis < nbr_cutoff:
                    tmp_idx.append(nbr_idx[i])
            #check neighbor sites
            atom_idx = np.where(np.array(tmp_idx)==point)[-1]
            if len(atom_idx) == 0:
                allow.append(point)
        return allow
    
    def exchange_action(self, idx, type, symm):
        """
        actions of exchanging atoms
        
        Parameters
        ----------
        idx [int, 0d]: index of select atom
        type [int, 1d]: type of atoms
        symm [int, 1d]: symmetry of atoms
        
        Returns
        ----------
        allow [int, 2d]: effective exchange actions
        """
        #get index of different elements of same symmetry
        ele_idx = self.get_ele_idx(idx, type, symm)
        #get allow exchange actions
        allow = []
        ele_num = len(ele_idx)
        buffer = np.arange(ele_num)
        if ele_num > 1:
            for i, j in itertools.combinations(buffer, 2):
                action = itertools.product(ele_idx[i], ele_idx[j])
                allow += [i for i in action]
        return allow
    
    def get_ele_idx(self, idx, atom_type, atom_symm):
        """
        get index of different elements of same symmetry
        
        Parameters
        ----------
        idx [int, 0d]: index of select atom in atom_pos
        atom_type [int, 1d]: type of atoms
        atom_symm [int, 1d]: symmetry of atoms

        Returns
        ----------
        ele_idx [int, 2d]: index of same symmetry elements
        """
        #get index with same symmetry
        symm_idx, type = [], []
        symm_slt = atom_symm[idx]
        for i, symm in enumerate(atom_symm):
            if symm == symm_slt:
                symm_idx.append(i)
                type.append(atom_type[i])    
        #get index of different atoms
        ele_idx = []
        for ele in np.unique(type):
            idx = [symm_idx[i] for i, j in enumerate(type) if j==ele]
            ele_idx.append(idx)
        return ele_idx
    
    def sampling_angles_thicks(self, atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, limit=2):
        """
        adjust cluster rotation angles or displace atoms in z-direction
        
        Parameters
        ----------
        atom_pos [int, 2d]: atom positions
        atom_type [int, 2d]: atom types
        atom_symm [int, 2d]: atom symmetry
        grid_name [int, 1d]: grid name
        grid_ratio [int, 1d]: grid ratio
        space group [int, 1d]: space groups
        grain [int, 0d]: grain in z-direction
        limit [int, 0d]: sampling number limit
        
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
        if Cluster_Search:
            cluster_angles = self.import_data('angles')
            angle_num = len(cluster_angles)
        #sampling angles and displacement in z-direction
        atom_pos_new, atom_type_new, atom_symm_new, grid_name_new, grid_ratio_new, space_group_new = [], [], [], [], [], []
        angles, thicks = [], []
        for i, type in enumerate(atom_type):
            type = np.array(type)
            atom_num = len(type)
            #initialize angles and thicks
            tmp_angles = [[0 for _ in range(atom_num)]]
            tmp_thicks = [[0 for _ in range(atom_num)]]
            #cluster search
            if Cluster_Search:
                mask = np.zeros(atom_num, dtype=int)
                mask[type<0] = 1
                tmp_angles += (mask*np.random.choice(angle_num, size=(limit, atom_num))).tolist()
                if Thickness == 0:
                    tmp_thicks += np.zeros((limit, atom_num), dtype=int).tolist()
                else:
                    tmp_thicks += np.random.randint(0, Z_Layers+1, size=(limit, atom_num)).tolist()
            #puckered structures search
            elif General_Search and Dimension == 2 and Thickness > 0:
                tmp_angles += np.zeros((limit, atom_num), dtype=int).tolist()
                tmp_thicks += np.random.randint(0, Z_Layers+1, size=(limit, atom_num)).tolist()
            #update
            num = min(len(tmp_angles), len(tmp_thicks))
            angles += tmp_angles[:num]
            thicks += tmp_thicks[:num]
            atom_pos_new += [atom_pos[i] for _ in range(num)]
            atom_type_new += [atom_type[i] for _ in range(num)]
            atom_symm_new += [atom_symm[i] for _ in range(num)]
            grid_name_new += [grid_name[i] for _ in range(num)]
            grid_ratio_new += [grid_ratio[i] for _ in range(num)]
            space_group_new += [space_group[i] for _ in range(num)]
        return atom_pos_new, atom_type_new, atom_symm_new, grid_name_new, grid_ratio_new, space_group_new, angles, thicks
    

class Search(ActionSpace, GridGenerate, GNNPredict):
    #Searching on PES by machine-learning potential
    def __init__(self, iteration):
        ActionSpace.__init__(self)
        GNNPredict.__init__(self)
        self.device = torch.device('cpu')
        self.iteration = f'{iteration:02.0f}'
        self.sh_save_path = f'{Search_Path}/ml_{self.iteration}'
        self.model_save_path = f'{Model_Path}/{self.iteration}'
    
    def gnn_SA_general(self, pos, type, symm, grid, ratio, sg, angle, thick, path, node, nbr_num=30, sample_limit=100):
        """
        simulated annealing for general search
        
        Parameters
        ----------
        pos [int, 1d]: position of atoms
        type [int, 1d]: type of atoms
        symm [int, 1d]: symmetry of atoms
        grid [int, 0d]: grid name
        ratio [float, 0d]: grid ratio
        sg [int, 0d]: space group number
        angle [int, 1d]: cluster angles
        thick [int, 1d]: displacement in z-direction
        path [int, 0d]: path number
        node [int, 0d]: node number
        nbr_num [int, 0d]: number of neighbors
        sample_limit [int, 0d]: limit of SA samples
        """
        np.random.seed()
        #load GNN model
        self.load_vec_out_model()
        #import embeddings and lattice vectors
        self.elem_embed = self.import_data('elem')
        self.latt_vec = self.import_data('latt', grid)
        self.grid_coords = self.import_data('frac', grid, sg)
        self.grid_idx, self.grid_dis = self.import_data('grid', grid, sg)
        #import cluster and angles
        self.cluster_angles, self.property_dict = [], []
        if Cluster_Search:
            self.cluster_angles = self.import_data('angles')
            self.property_dict = self.import_data('property')
        self.angles_num = len(self.cluster_angles)
        #group sites by symmetry
        mapping = self.import_data('mapping', grid, sg)
        self.symm_site = self.group_symm_sites(mapping)
        #initialize buffer
        atom_fea_gnn = self.get_atom_fea(type, self.elem_embed)
        self.nbr_idx, self.nbr_dis = self.get_nbr_general(pos, ratio, sg, self.latt_vec, self.grid_coords, nbr_num=nbr_num)
        nbr_idx_gnn, nbr_dis_gnn = self.cut_pad_neighbors(self.nbr_idx, self.nbr_dis, self.nbr)
        nbr_fea_gnn = self.expand(nbr_dis_gnn)
        energy, vec = self.predict_single(symm, atom_fea_gnn, nbr_fea_gnn, nbr_idx_gnn)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = [pos], [type], [symm], [grid], [ratio], [sg]
        angles, thicks, energys, crys_vec = [angle], [thick], [energy], [vec]
        #optimize position
        for _ in range(Restart_Times):
            tmp_pos, tmp_type, tmp_symm, tmp_grid, tmp_ratio, tmp_sg, tmp_angle, tmp_thick, tmp_energy, tmp_vec = \
                self.explore_pos_general(pos, type, symm, grid, ratio, sg, angle, thick, energy, vec)
            atom_pos += tmp_pos
            atom_type += tmp_type
            atom_symm += tmp_symm
            grid_name += tmp_grid
            grid_ratio += tmp_ratio
            space_group += tmp_sg
            angles += tmp_angle
            thicks += tmp_thick
            energys += tmp_energy
            crys_vec += tmp_vec
        #delete same structures
        idx = self.delete_duplicates(atom_pos, atom_type, atom_symm,
                                     grid_name, grid_ratio, space_group, angles, thicks)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = np.array(energys)[idx]
        crys_vec = np.array(crys_vec)[idx]
        #limit number of samples
        if len(energys) > sample_limit:
            num = max(1, min(sample_limit, int(SA_Path_Ratio*len(energys))))
            idx = np.argsort(energys)[:num]
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, grid_ratio, space_group, angles, thicks)
            energys = energys[idx]
            crys_vec = crys_vec[idx]
        #sample clustering
        idx = self.reduce_by_gnn(crys_vec, space_group, energys)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = energys[idx]
        crys_vec = crys_vec[idx]
        #optimize thicks and angles
        if Cluster_Search or (Dimension == 2 and Thickness > 0):
            if Thickness > 0:
                num = min(1, len(energys))
                #optimize thicks
                for i in range(num):
                    for _ in range(Restart_Times):
                        tmp_pos, tmp_type, tmp_symm, tmp_grid, tmp_ratio, tmp_sg, tmp_angle, tmp_thick, tmp_energy, tmp_vec = \
                            self.explore_thick_general(atom_pos[i], atom_type[i], atom_symm[i], grid_name[i], grid_ratio[i], 
                                                       space_group[i], angles[i], thicks[i])
                        atom_pos += tmp_pos
                        atom_type += tmp_type
                        atom_symm += tmp_symm
                        grid_name += tmp_grid
                        grid_ratio += tmp_ratio
                        space_group += tmp_sg
                        angles += tmp_angle
                        thicks += tmp_thick
                        energys = np.concatenate((energys, tmp_energy))
                        crys_vec = np.concatenate((crys_vec, tmp_vec))
                #sample clustering
                idx = self.reduce_by_gnn(crys_vec, space_group, energys)
                atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
                    self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                        grid_name, grid_ratio, space_group, angles, thicks)
                energys = energys[idx]
                crys_vec = crys_vec[idx]
            if Cluster_Search:
                num = min(1, len(energys))
                #optimize angles
                for i in range(num):
                    for _ in range(Restart_Times):
                        tmp_pos, tmp_type, tmp_symm, tmp_grid, tmp_ratio, tmp_sg, tmp_angle, tmp_thick, tmp_energy, tmp_vec = \
                            self.explore_angle_general(atom_pos[i], atom_type[i], atom_symm[i], grid_name[i], grid_ratio[i], 
                                                       space_group[i], angles[i], thicks[i])
                        atom_pos += tmp_pos
                        atom_type += tmp_type
                        atom_symm += tmp_symm
                        grid_name += tmp_grid
                        grid_ratio += tmp_ratio
                        space_group += tmp_sg
                        angles += tmp_angle
                        thicks += tmp_thick
                        energys = np.concatenate((energys, tmp_energy))
                        crys_vec = np.concatenate((crys_vec, tmp_vec))
                #sample clustering
                idx = self.reduce_by_gnn(crys_vec, space_group, energys)
                atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
                    self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                        grid_name, grid_ratio, space_group, angles, thicks)
                energys = energys[idx]
                crys_vec = crys_vec[idx]
        #delete duplicates by crystal vectors
        if Use_ML_Clustering:
            idx = self.delete_duplicates_crys_vec(crys_vec, energys)
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, grid_ratio, space_group, angles, thicks)
            energys = energys[idx]
            crys_vec = crys_vec[idx]
        self.save(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec, path, node)
    
    def explore_pos_general(self, pos, type, symm, grid, ratio, sg, angle, thick, energy, vec, T=1):
        """
        simulated annealing for general search for one core
        
        Parameters
        ----------
        pos [int, 1d]: position of atoms
        type [int, 1d]: type of atoms
        symm [int, 1d]: symmetry of atoms
        grid [int, 0d]: grid name
        ratio [float, 0d]: grid ratio
        sg [int, 0d]: space group number
        angle [int, 1d]: cluster angles
        thick [int, 1d]: displacement in z-direction
        energy [float, 0d]: prediction energy
        vec [float, 1d, np]: crystal vector
        T [float, 0d]: initial SA temperature
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: grid name
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: prediction energys
        crys_vec [float, 2d]: crystal vectors
        """
        sa_T = T
        pos_1, type_1, energy_1, vec_1 = pos, type, energy, vec
        nbr_idx_1, nbr_dis_1 = self.nbr_idx, self.nbr_dis
        nbr_idx_2, nbr_dis_2 = nbr_idx_1, nbr_dis_1
        energy_2, vec_2 = energy_1, vec_1
        nbr_idx_gnn, nbr_dis_gnn = self.cut_pad_neighbors(self.nbr_idx, self.nbr_dis, self.nbr)
        nbr_fea_gnn = self.expand(nbr_dis_gnn)
        #optimize order of atoms
        atom_pos, atom_type, energys, crys_vec = [pos], [type], [energy], [vec]
        for _ in range(SA_Steps):
            pos_2, type_2, point = self.atom_step_general(pos_1, type_1, symm, self.symm_site, ratio, self.grid_idx, self.grid_dis)
            #move atom
            if point >= 0:
                atom_fea_gnn = self.get_atom_fea(type_2, self.elem_embed)
                nbr_idx_2, nbr_dis_2 = self.update_neighbors(pos_1, pos_2, nbr_idx_1, nbr_dis_1, ratio, sg, self.latt_vec, self.grid_coords)
                nbr_idx_gnn, nbr_dis_gnn = self.cut_pad_neighbors(nbr_idx_2, nbr_dis_2, self.nbr)
                nbr_fea_gnn = self.expand(nbr_dis_gnn)
                energy_2, vec_2 = self.predict_single(symm, atom_fea_gnn, nbr_fea_gnn, nbr_idx_gnn)
            #exchange atoms
            elif point == -1:
                atom_fea_gnn = self.get_atom_fea(type_2, self.elem_embed)
                energy_2, vec_2 = self.predict_single(symm, atom_fea_gnn, nbr_fea_gnn, nbr_idx_gnn)
            #keep atoms
            elif point == -2:
                pass
            #metropolis criterion
            if self.metropolis(energy_1, energy_2, sa_T):
                pos_1, type_1, energy_1, vec_1 = pos_2, type_2, energy_2, vec_2
                nbr_idx_1, nbr_dis_1 = nbr_idx_2, nbr_dis_2
                atom_pos.append(pos_1)
                atom_type.append(type_1)
                energys.append(energy_1)
                crys_vec.append(vec_1)
            sa_T *= SA_Decay
        #get search results
        num = len(atom_pos)
        atom_symm = [symm for _ in range(num)] 
        grid_name = [grid for _ in range(num)] 
        grid_ratio = [ratio for _ in range(num)]
        space_group = [sg for _ in range(num)]
        angles = [angle for _ in range(num)]
        thicks = [thick for _ in range(num)]
        #delete same structures
        idx = self.delete_duplicates(atom_pos, atom_type, atom_symm,
                                     grid_name, grid_ratio, space_group, angles, thicks)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = np.array(energys)[idx].tolist()
        crys_vec = np.array(crys_vec)[idx].tolist()
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec
    
    def explore_thick_general(self, pos, type, symm, grid, ratio, sg, angle, thick, T=1):
        """
        simulated annealing for thick search for one core
        
        Parameters
        ----------
        pos [int, 1d]: position of atoms
        type [int, 1d]: type of atoms
        symm [int, 1d]: symmetry of atoms
        grid [int, 0d]: grid name
        ratio [float, 0d]: grid ratio
        sg [int, 0d]: space group number
        angle [int, 1d]: cluster angles
        thick [int, 1d]: displacement in z-direction
        T [float, 0d]: initial SA temperature
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: grid name
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: prediction energys
        crys_vec [float, 2d]: crystal vectors
        """
        sa_T = T
        thick_1 = thick
        stru = self.get_stru(pos, type, self.latt_vec, ratio, self.grid_coords, sg, self.cluster_angles, self.property_dict, angle, thick_1)
        atom_fea_gnn, nbr_fea_gnn, nbr_idx_gnn = self.get_gnn_input_from_stru(stru, self.elem_embed)
        symm_tmp = [1 for _ in range(len(atom_fea_gnn))]
        energy_1, vec_1 = self.predict_single(symm_tmp, atom_fea_gnn, nbr_fea_gnn, nbr_idx_gnn)
        #optimize thicks
        thicks, energys, crys_vec = [thick_1], [energy_1], [vec_1]
        for _ in range(SA_Steps):
            thick_2 = self.thick_step_general(thick_1)
            stru = self.get_stru(pos, type, self.latt_vec, ratio, self.grid_coords, sg, self.cluster_angles, self.property_dict, angle, thick_2)
            atom_fea_gnn, nbr_fea_gnn, nbr_idx_gnn = self.get_gnn_input_from_stru(stru, self.elem_embed)
            symm_tmp = [1 for _ in range(len(atom_fea_gnn))]
            energy_2, vec_2 = self.predict_single(symm_tmp, atom_fea_gnn, nbr_fea_gnn, nbr_idx_gnn)
            #metropolis criterion
            if self.metropolis(energy_1, energy_2, sa_T):
                thick_1, energy_1, vec_1 = thick_2, energy_2, vec_2
                thicks.append(thick_1)
                energys.append(energy_1)
                crys_vec.append(vec_1)
            sa_T *= SA_Decay
        #get search results
        num = len(thicks)
        atom_pos = [pos for _ in range(num)] 
        atom_type = [type for _ in range(num)] 
        atom_symm = [symm for _ in range(num)] 
        grid_name = [grid for _ in range(num)] 
        grid_ratio = [ratio for _ in range(num)] 
        space_group = [sg for _ in range(num)]
        angles = [angle for _ in range(num)]
        #delete same structures
        idx = self.delete_duplicates(atom_pos, atom_type, atom_symm,
                                     grid_name, grid_ratio, space_group, angles, thicks)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = np.array(energys)[idx]
        crys_vec = np.array(crys_vec)[idx]
        #sorted by energy
        idx = np.argsort(energys)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = energys[idx].tolist()
        crys_vec = crys_vec[idx].tolist()
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec
    
    def explore_angle_general(self, pos, type, symm, grid, ratio, sg, angle, thick, T=1):
        """
        simulated annealing for angle search for one core
        
        Parameters
        ----------
        pos [int, 1d]: position of atoms
        type [int, 1d]: type of atoms
        symm [int, 1d]: symmetry of atoms
        grid [int, 0d]: grid name
        ratio [float, 0d]: grid ratio
        sg [int, 0d]: space group number
        angle [int, 1d]: cluster angles
        thick [int, 1d]: displacement in z-direction
        T [float, 0d]: initial SA temperature
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: grid name
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: prediction energys
        crys_vec [float, 2d]: crystal vectors
        """
        sa_T = T
        angle_1 = angle
        stru = self.get_stru(pos, type, self.latt_vec, ratio, self.grid_coords, sg, self.cluster_angles, self.property_dict, angle_1, thick)
        atom_fea_gnn, nbr_fea_gnn, nbr_idx_gnn = self.get_gnn_input_from_stru(stru, self.elem_embed)
        symm_tmp = [1 for _ in range(len(atom_fea_gnn))]
        energy_1, vec_1 = self.predict_single(symm_tmp, atom_fea_gnn, nbr_fea_gnn, nbr_idx_gnn)
        #optimize angles
        angles, energys, crys_vec = [angle_1], [energy_1], [vec_1]
        for _ in range(SA_Steps):
            angle_2 = self.angle_step_general(symm, angle_1)
            stru = self.get_stru(pos, type, self.latt_vec, ratio, self.grid_coords, sg, self.cluster_angles, self.property_dict, angle_2, thick)
            atom_fea_gnn, nbr_fea_gnn, nbr_idx_gnn = self.get_gnn_input_from_stru(stru, self.elem_embed)
            symm_tmp = [1 for _ in range(len(atom_fea_gnn))]
            energy_2, vec_2 = self.predict_single(symm_tmp, atom_fea_gnn, nbr_fea_gnn, nbr_idx_gnn)
            #metropolis criterion
            if self.metropolis(energy_1, energy_2, sa_T):
                angle_1, energy_1, vec_1 = angle_2, energy_2, vec_2
                angles.append(angle_1)
                energys.append(energy_1)
                crys_vec.append(vec_1)
            sa_T *= SA_Decay
        #get search results
        num = len(angles)
        atom_pos = [pos for _ in range(num)] 
        atom_type = [type for _ in range(num)] 
        atom_symm = [symm for _ in range(num)] 
        grid_name = [grid for _ in range(num)] 
        grid_ratio = [ratio for _ in range(num)] 
        space_group = [sg for _ in range(num)]
        thicks = [thick for _ in range(num)]
        #delete same structures
        idx = self.delete_duplicates(atom_pos, atom_type, atom_symm,
                                     grid_name, grid_ratio, space_group, angles, thicks)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = np.array(energys)[idx]
        crys_vec = np.array(crys_vec)[idx]
        #sorted by energy
        idx = np.argsort(energys)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = energys[idx].tolist()
        crys_vec = crys_vec[idx].tolist()
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec
        
    def gnn_SA_template(self, pos, type, symm, grid, ratio, sg, angle, thick, path, node):
        """
        simulated annealing for template search
        
        Parameters
        ----------
        pos [int, 1d]: position of atoms
        type [int, 1d]: type of atoms
        symm [int, 1d]: symmetry of atoms
        grid [int, 0d]: grid name
        ratio [float, 0d]: grid ratio
        sg [int, 0d]: space group number
        angle [int, 1d]: cluster angles
        thick [int, 1d]: displacement in z-direction
        path [int, 0d]: path number
        node [int, 0d]: node number
        """
        np.random.seed()
        #load GNN model
        self.load_vec_out_model()
        #import embeddings and lattice vectors
        self.elem_embed = self.import_data('elem')
        self.grid_idx, self.grid_dis = self.import_data('grid', grid, sg)
        #group sites by symmetry
        mapping = self.import_data('mapping', grid, sg)
        self.symm_site = self.group_symm_sites(mapping)
        #initialize buffer
        energy, vec = self.gnn_template(pos, type, symm, ratio, self.grid_idx, self.grid_dis)
        atom_pos, energys, crys_vec = [pos], [energy], [vec]
        #optimize position
        for _ in range(Restart_Times):
            tmp_pos, tmp_energy, tmp_vec =\
                self.explore_template(pos, type, symm, ratio, energy, vec)
            atom_pos += tmp_pos
            energys += tmp_energy
            crys_vec += tmp_vec
        #save search results
        num = len(atom_pos)
        atom_type = [type for _ in range(num)] 
        atom_symm = [symm for _ in range(num)] 
        grid_name = [grid for _ in range(num)] 
        grid_ratio = [ratio for _ in range(num)]
        space_group = [sg for _ in range(num)] 
        angles = [angle for _ in range(num)]
        thicks = [thick for _ in range(num)]
        #sample clustering
        num = max(1, int(SA_Path_Ratio*len(energys)))
        if Use_ML_Clustering:
            crys_embedded = self.reduce(crys_vec)
            idx = self.cluster_by_labels(num, crys_embedded, space_group, energys)
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, grid_ratio, space_group, angles, thicks)
            energys = np.array(energys)[idx]
            crys_vec = np.array(crys_vec)[idx]
            #delete duplicates by crystal vectors
            idx = self.delete_duplicates_crys_vec(crys_vec, energys)
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, grid_ratio, space_group, angles, thicks)
            energys = energys[idx]
            crys_vec = crys_vec[idx]
        else:
            idx = np.argsort(energys)[:num]
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, grid_ratio, space_group, angles, thicks)
            energys = np.array(energys)[idx]
            crys_vec = np.array(crys_vec)[idx]
        self.save(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec, path, node)
    
    def explore_template(self, pos, type, symm, ratio, energy, vec, T=1):
        """
        simulated annealing for template search for one core
        
        Parameters
        ----------
        pos [int, 1d]: position of atoms
        type [int, 1d]: type of atoms
        symm [int, 1d]: symmetry of atoms
        ratio [float, 0d]: grid ratio
        energy [float, 0d]: prediction energy
        vec [float, 1d, np]: crystal vector
        T [float, 0d]: initial SA temperature
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        energys [float, 1d]: prediction energys
        crys_vec [float, 2d]: crystal vectors
        """
        sa_T = T
        pos_1, energy_1, vec_1 = pos, energy, vec
        #optimize order of atoms
        atom_pos, crys_vec, energys = [], [], []
        for _ in range(SA_Steps):
            pos_2 = self.atom_step_template(pos_1, type, symm, self.symm_site, ratio, self.grid_idx, self.grid_dis)
            energy_2, vec_2 = self.gnn_template(pos_2, type, symm, ratio, self.grid_idx, self.grid_dis)
            if self.metropolis(energy_1, energy_2, sa_T):
                pos_1, energy_1, vec_1 = pos_2, energy_2, vec_2
                atom_pos.append(pos_1)
                energys.append(energy_1)
                crys_vec.append(vec_1)
            sa_T *= SA_Decay
        return atom_pos, energys, crys_vec
    
    def atom_step_general(self, pos, type, symm, symm_site, ratio, grid_idx, grid_dis):
        """
        move atoms under the geometry constrain
        
        Parameters
        ----------
        pos [int, 1d]: inital position of atoms
        type [int, 1d]: initial type of atoms
        symm [int, 1d]: initial symm of atoms
        symm_site [dict, int:list]: site position grouped by symmetry
        ratio [float, 0d]: grid ratio
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        
        Returns
        ----------
        new_pos [int, 1d]: position of atoms
        new_type [int, 1d]: type of atoms
        point [int, 0d]: move point
        """
        new_pos = pos.copy()
        new_type = type.copy()
        atom_num = len(new_pos)
        pool = np.arange(0, atom_num)
        pool = [i for i, j in zip(pool, symm) if j > 0]
        if len(pool) > 0:
            #generate actions
            np.random.seed()
            idx = np.random.choice(pool)
            action_mv = self.action_filter(idx, new_pos, type, symm, symm_site,
                                           ratio, grid_idx, grid_dis)
            action_ex = self.exchange_action(idx, type, symm)
            mask_ex = [-1 for _ in action_ex]
            actions = action_mv + mask_ex
            if len(actions) == 0:
                actions += [-2]
            #optimize order of atoms
            actions = np.array(actions, dtype=int)
            point = np.random.choice(actions)
            #move atom
            if point >= 0:
                check_pos = new_pos.copy()
                check_type = new_type.copy()
                check_pos[idx] = point
                #get neighbors of atom
                center_type = check_type[idx]
                nbr_cutoff = self.get_nbr_cutoff(center_type, self.ele_types, self.bond_list)
                _, center_nbr_type, center_nbr_dis = \
                    self.get_atom_neighbors(point, check_pos, check_type, ratio, grid_idx, grid_dis, nbr_cutoff)
                center_nbr_bond_list = \
                    self.get_nbr_bond_list(center_type, center_nbr_type, self.ele_types, self.bond_list)
                #check distance
                if self.check_nbr_dis(center_nbr_dis, center_nbr_bond_list):
                    new_pos[idx] = point
                else:
                    point = -2
            #exchange atoms
            elif point == -1:
                action_num = len(action_ex)
                idx = np.random.randint(0, action_num)
                idx_1, idx_2 = action_ex[idx]
                #exchange atoms in trial
                check_type = new_type.copy()
                check_type[idx_1], check_type[idx_2] = \
                    check_type[idx_2], check_type[idx_1]
                #get neighbors of atom 1
                point_1, center_type_1 = new_pos[idx_1], check_type[idx_1]
                nbr_cutoff_1 = self.get_nbr_cutoff(center_type_1, self.ele_types, self.bond_list)
                _, center_nbr_type_1, center_nbr_dis_1 = \
                    self.get_atom_neighbors(point_1, new_pos, check_type, ratio, grid_idx, grid_dis, nbr_cutoff_1)
                center_nbr_bond_list_1 = \
                    self.get_nbr_bond_list(center_type_1, center_nbr_type_1, self.ele_types, self.bond_list)
                #check distance
                if self.check_nbr_dis(center_nbr_dis_1, center_nbr_bond_list_1):
                    #get neighbors of atom 2
                    point_2, center_type_2 = new_pos[idx_2], check_type[idx_2]
                    nbr_cutoff_2 = self.get_nbr_cutoff(center_type_2, self.ele_types, self.bond_list)
                    _, center_nbr_type_2, center_nbr_dis_2 = \
                        self.get_atom_neighbors(point_2, new_pos, check_type, ratio, grid_idx, grid_dis, nbr_cutoff_2)
                    center_nbr_bond_list_2 = \
                        self.get_nbr_bond_list(center_type_2, center_nbr_type_2, self.ele_types, self.bond_list)
                    #check distance
                    if self.check_nbr_dis(center_nbr_dis_2, center_nbr_bond_list_2):
                        new_type[idx_1], new_type[idx_2] = \
                            new_type[idx_2], new_type[idx_1]
                    else:
                        point = -2
                else:
                    point = -2
            #keep atoms
            elif point == -2:
                pass
        return new_pos, new_type, point
    
    def thick_step_general(self, thick):
        """
        move atoms under the geometry constrain

        Parameters
        ----------
        thick [int, 1d]: displacement in z-direction
        
        Returns
        ----------
        new_thick [int, 1d]: displacement in z-direction
        """
        new_thick = thick.copy()
        atom_num = len(thick)
        pool = np.arange(0, atom_num)
        if len(pool) > 0:
            #generate actions
            np.random.seed()
            idx = np.random.choice(pool)
            new_thick[idx] = np.random.randint(0, Z_Layers+1)
        return new_thick
    
    def angle_step_general(self, symm, angle):
        """
        move atoms under the geometry constrain

        Parameters
        ----------
        symm [int, 1d]: symmetry of atoms
        angle [int, 1d]: cluster angles

        Returns
        ----------
        new_angle [int, 1d]: cluster angles
        """
        new_angle = angle.copy()
        atom_num = len(symm)
        pool = np.arange(0, atom_num)
        pool = [i for i, j in zip(pool, symm) if j < 0]
        if len(pool) > 0:
            #generate actions
            np.random.seed()
            idx = np.random.choice(pool)
            new_angle[idx] = np.random.choice(self.angles_num)
        return new_angle
    
    def atom_step_template(self, pos, type, symm, symm_site, ratio, grid_idx, grid_dis):
        """
        move atoms under the geometry constrain
        
        Parameters
        ----------
        pos [int, 1d]: inital position of atoms
        type [int, 1d]: initial type of atoms
        symm [int, 1d]: initial symm of atoms
        symm_site [dict, int:list]: site position grouped by symmetry
        ratio [float, 0d]: grid ratio
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        
        Returns
        ----------
        new_pos [int, 1d]: position of atoms
        """
        new_pos = pos.copy()
        atom_num = len(new_pos)
        pool = np.arange(0, atom_num)
        pool = [i for i, j in zip(pool, symm) if j > 0]
        if len(pool) > 0:
            #generate actions
            np.random.seed()
            idx = np.random.choice(pool)
            action_mv = self.action_filter(idx, new_pos, type, symm, symm_site,
                                           ratio, grid_idx, grid_dis)
            action_ex = self.exchange_action(idx, type, symm)
            mask_ex = [-1 for _ in action_ex]
            actions = action_mv + mask_ex
            #exchange or move atoms
            actions = np.array(actions, dtype=int)
            point = np.random.choice(actions)
            if point >= 0:
                check_pos = new_pos.copy()
                check_type = type.copy()
                center_type = type[idx]
                nbr_cutoff = self.get_nbr_cutoff(center_type, self.ele_types, self.bond_list)
                _, center_nbr_type, center_nbr_dis = \
                    self.get_atom_neighbors(point, check_pos, check_type, ratio, grid_idx, grid_dis, nbr_cutoff)
                center_nbr_bond_list = \
                    self.get_nbr_bond_list(center_type, center_nbr_type, self.ele_types, self.bond_list)
                if self.check_nbr_dis(center_nbr_dis, center_nbr_bond_list):
                    new_pos[idx] = point
                else:
                    new_pos = new_pos
            elif point == -1:
                action_num = len(action_ex)
                idx = np.random.randint(0, action_num)
                idx_1, idx_2 = action_ex[idx]
                new_pos[idx_1], new_pos[idx_2] = \
                    new_pos[idx_2], new_pos[idx_1]
        return new_pos
    
    def metropolis(self, value_1, value_2, T):
        """
        metropolis criterion
        
        Parameters
        ----------
        value_1 [float, 0d]: current value
        value_2 [float, 0d]: next value
        T [float, 0d]: annealing temperature
        
        Returns
        ----------
        flag [bool, 0d]: whether do the action
        """
        delta = value_2 - value_1
        if np.exp(-delta/T) > np.random.rand():
            if delta == 0:
                return False
            else:
                return True
        else:
            return False
    
    def load_vec_out_model(self):
        """
        load feature extraction and readout model
        """
        self.vec_model = FeatureExtractNet()
        self.out_model = ReadoutNet()
        paras = torch.load(f'{self.model_save_path}/model_best.pth.tar', 
                           map_location=self.device)
        self.vec_model.load_state_dict(paras['state_dict'])
        self.out_model.load_state_dict(paras['state_dict'])
        self.normalizer.load_state_dict(paras['normalizer'])
    
    def gnn_template(self, pos, type, symm, ratio, grid_idx, grid_dis):
        """
        get crystal vector and energy of one structure for template search

        Parameters
        ----------
        pos [int, 1d]: position of atoms
        type [int ,1d]: type of atoms
        symm [int, 1d]: symmetry of atoms
        ratio [float, 0d]: grid ratio
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        
        Returns
        ----------
        energy [float, 0d]: predict energy
        crys_vec_np [float, 1d, np]: crystal vector
        """
        #transfer into input of GNN
        atom_fea, nbr_fea, nbr_idx = \
            self.get_gnn_input_template(pos, type, self.elem_embed, 
                                        ratio, grid_idx, grid_dis)
        #get crystal vector and prediction energy
        energy, crys_vec_np = self.predict_single(symm, atom_fea, nbr_fea, nbr_idx)
        return energy, crys_vec_np
    
    def save(self, atom_pos, atom_type, atom_symm, grid_name, grid_ratio,
             space_group, angles, thicks, energys, crys_vec, path, node):
        """
        save searching results
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: grid name
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: prediction energys
        crys_vec [float, 2d]: crystal vectors
        path [int, 0d]: path number
        node [int, 0d]: node number
        """
        #delete same structures
        idx = self.delete_duplicates(atom_pos, atom_type, atom_symm,
                                     grid_name, grid_ratio, space_group, angles, thicks)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = np.array(energys)[idx]
        crys_vec = np.array(crys_vec)[idx]
        #sorted structures by energy
        idx = np.argsort(energys)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = energys[idx]
        crys_vec = crys_vec[idx]
        #export results
        self.write_list2d(f'{self.sh_save_path}/'
                          f'pos-{self.iteration}-{path}-{node}.dat', 
                          atom_pos)
        self.write_list2d(f'{self.sh_save_path}/'
                          f'type-{self.iteration}-{path}-{node}.dat', 
                          atom_type)
        self.write_list2d(f'{self.sh_save_path}/'
                          f'symm-{self.iteration}-{path}-{node}.dat', 
                          atom_symm)
        self.write_list2d(f'{self.sh_save_path}/'
                          f'grid-{self.iteration}-{path}-{node}.dat', 
                          np.transpose([grid_name]))
        self.write_list2d(f'{self.sh_save_path}/'
                          f'ratio-{self.iteration}-{path}-{node}.dat', 
                          np.transpose([grid_ratio]), style='{0:8.4f}')
        self.write_list2d(f'{self.sh_save_path}/'
                          f'sg-{self.iteration}-{path}-{node}.dat', 
                          np.transpose([space_group]))
        self.write_list2d(f'{self.sh_save_path}/'
                          f'angles-{self.iteration}-{path}-{node}.dat', 
                          angles)
        self.write_list2d(f'{self.sh_save_path}/'
                          f'thicks-{self.iteration}-{path}-{node}.dat', 
                          thicks)
        self.write_list2d(f'{self.sh_save_path}/'
                          f'energy-{self.iteration}-{path}-{node}.dat', 
                          np.transpose([energys]), style='{0:8.4f}')
        self.write_list2d(f'{self.sh_save_path}/'
                          f'vec-{self.iteration}-{path}-{node}.bin', 
                          crys_vec, binary=True)
        #record files
        shell_script = f'''
                        #!/bin/bash --login
                        cd {SCCOP_Path}/{self.sh_save_path}
                        echo {self.iteration} {path} {node} >> record.dat
                        '''
        os.system(shell_script)
        

if __name__ == '__main__':
    torch.set_num_threads(1)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', type=int, nargs='+')
    parser.add_argument('--type', type=int, nargs='+')
    parser.add_argument('--symm', type=int, nargs='+')
    parser.add_argument('--grid', type=int)
    parser.add_argument('--ratio', type=float)
    parser.add_argument('--sg', type=int)
    parser.add_argument('--angle', type=int, nargs='+')
    parser.add_argument('--thick', type=int, nargs='+')
    parser.add_argument('--iteration', type=int)
    parser.add_argument('--path', type=int)
    parser.add_argument('--node', type=str)
    parser.add_argument('--flag', type=int, default=0)
    args = parser.parse_args()
    
    flag = args.flag
    iteration = args.iteration
    if flag == 0:
        #structure info
        pos = args.pos
        type = args.type
        symm = args.symm
        grid = args.grid
        ratio = args.ratio
        sg = args.sg
        angle = args.angle
        thick =args.thick
        #path label
        path = args.path
        node = args.node
        #Searching
        worker = Search(iteration)
        if General_Search or Cluster_Search:
            worker.gnn_SA_general(pos, type, symm, grid, ratio, sg, angle, thick, path, node)
        elif Template_Search:
            worker.gnn_SA_template(pos, type, symm, grid, ratio, sg, angle, thick, path, node)
        
        #remove flag
        os.system(f'''
                  #!/bin/bash --login
                  cd {worker.sh_save_path}
                  if [ -f RUNNING_{path} ]; then
                    rm RUNNING_{path}
                  fi
                  ''')
    
    elif flag == 1:
        worker = ParallelWorkers()
        worker.collect_path(iteration)