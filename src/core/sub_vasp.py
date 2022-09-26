import os, sys
import time
import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.path import *
from core.utils import ListRWTools, SSHTools, system_echo
from core.data_transfer import DeleteDuplicates


class ParallelSubVASP(ListRWTools, SSHTools):
    #submit vasp jobs
    def __init__(self, wait_time=0.1):
        self.wait_time = wait_time
    
    def sub_job(self, iteration):
        """
        calculate POSCARs and return energys
        
        POSCAR file notation: POSCAR-iteration-number-node
        e.g. POSCAR-001-136
        
        Parameters
        ----------
        iteration [int, 0d]: sccop iteration
        """
        poscars = os.listdir(f'{poscar_path}/{iteration:03.0f}')
        num_poscar = len(poscars)
        #make directory
        out_path = f'{vasp_out_path}/{iteration:03.0f}'
        os.mkdir(out_path)
        system_echo(f'Start VASP calculation---itersions: '
                    f'{iteration}, number: {num_poscar}')
        #vasp calculation
        work_node_num = self.sub_vasp_job(poscars, iteration)
        while not self.is_done(out_path, work_node_num):
            time.sleep(self.wait_time)
        self.remove_flag(out_path)
        #get energy of outputs
        self.get_energy(iteration)
    
    def sub_vasp_job(self, poscars, iteration):
        """
        submit vasp jobs to nodes

        Parameters
        ----------
        poscars [str, 1d]: name of poscars
        iteration [int, 0d]: sccop iteration
        
        Returns
        ----------
        work_node_num [int, 0d]: number of work node
        """
        #poscars grouped by nodes
        assign, jobs = [], []
        for node in nodes:
            for poscar in poscars:
                label = poscar.split('-')[-1]
                if label == str(node):
                    assign.append(poscar)
            jobs.append(assign)
            assign = []
        #sub job to target node
        for job in jobs:
            self.sub_job_with_ssh(job, iteration)
        work_node_num = len(jobs)
        return work_node_num
    
    def sub_job_with_ssh(self, job, iteration):
        """
        SSH to target node and call vasp for calculation
        
        Parameters
        ----------
        job [str, 1d]: name of poscars in same node
        iteration [int, 0d]: sccop iteration
        """
        node = job[0].split('-')[-1]
        poscar_str = ' '.join(job)
        ip = f'node{node}'
        job_num = f'`ps -ef | grep {VASP_3d_path} | grep -v grep | wc -l`'
        local_vasp_out_path = f'{SCCOP_path}/{vasp_out_path}/{iteration:03.0f}'
        shell_script = f'''
                        cd {SCCOP_path}/vasp
                        for p in {poscar_str}
                        do
                            mkdir $p
                            cd $p

                            cp ../../{vasp_files_path}/SinglePointEnergy/* .
                            scp {gpu_node}:{SCCOP_path}/{poscar_path}/{iteration:03.0f}/$p POSCAR
                            DPT -v potcar
                            
                            nohup {VASP_3d_exe} >& $p.out&
                            cd ../
                            
                            counter={job_num}
                            while [ $counter -ge 10 ]
                            do
                                counter={job_num}
                                sleep 1s
                            done
                        done
                        
                        while true;
                        do
                            num={job_num}
                            if [ $num -eq 0 ]; then
                                touch FINISH-{node}
                                break
                            fi
                            sleep 1s
                        done
                        
                        for p in {poscar_str}
                        do
                            cp $p/$p.out .
                        done
                        
                        scp *.out FINISH-{node} {gpu_node}:{local_vasp_out_path}/
                        rm -r *
                        '''
        self.ssh_node(shell_script, ip)
    
    def get_energy(self, iteration):
        """
        generate energy file of current vasp outputs directory 
        
        Parameters
        ----------
        iteration [int, 0d]: sccop iteration
        """
        true_E, energys = [], []
        vasp_out = os.listdir(f'{vasp_out_path}/{iteration:03.0f}')
        vasp_out = sorted(vasp_out)
        for out in vasp_out:
            VASP_output_file = f'{vasp_out_path}/{iteration:03.0f}/{out}'
            with open(VASP_output_file, 'r') as f:
                ct = f.readlines()
            energy = 1e6
            for line in ct[:15]:
                if 'POSCAR found :' in line:
                    atom_num = int(line.split()[-2])
            for line in ct[-15:]:
                if 'F=' in line:
                    energy = float(line.split()[2])
                    ave_E = energy/atom_num
            if energy == 1e6:
                ave_E = energy
                true_E.append(False)
                system_echo(' *WARNING* SinglePointEnergy is failed!')
            else:
                if -12 < ave_E < 0:
                    true_E.append(True)
                else:
                    true_E.append(False)
            energys.append([out, true_E[-1], ave_E])
            system_echo(f'{out}, {true_E[-1]}, {ave_E}')
        self.write_list2d(f'{vasp_out_path}/Energy-{iteration:03.0f}.dat', energys)
        system_echo(f'Energy file generated successfully!')


class VASPoptimize(SSHTools, DeleteDuplicates):
    #optimize structure by VASP
    def __init__(self, recycle, wait_time=1):
        self.wait_time = wait_time
        self.sccop_out_path = f'{sccop_out_path}-{recycle}'
        self.optim_strus_path = f'{init_strus_path}_{recycle+1}'
        self.energy_path = f'{vasp_out_path}/initial_strus_{recycle+1}'
        self.local_sccop_out_path = f'{SCCOP_path}/{self.sccop_out_path}'
        self.local_optim_strus_path = f'{SCCOP_path}/{self.optim_strus_path}'
        self.local_energy_path = f'{SCCOP_path}/{self.energy_path}'
        self.calculation_path = f'{SCCOP_path}/vasp'
        if not os.path.exists(self.optim_strus_path):
            os.mkdir(self.optim_strus_path)
            os.mkdir(self.energy_path)
    
    def run_optimization_low(self):
        '''
        optimize configurations at low level
        '''
        files = sorted(os.listdir(self.sccop_out_path))
        poscars = [i for i in files if i.startswith('POSCAR')]
        num_poscar = len(poscars)
        system_echo(f'Start VASP calculation --- Optimization')
        for poscar in poscars:
            node = poscar.split('-')[-1]
            ip = f'node{node}'
            shell_script = f'''
                            #!/bin/bash
                            cd {self.calculation_path}
                            p={poscar}
                            mkdir $p
                            cd $p
                            cp ../../{vasp_files_path}/Optimization/* .
                            scp {gpu_node}:{self.local_sccop_out_path}/$p POSCAR
                            
                            cp POSCAR POSCAR_0
                            DPT -v potcar
                            
                            for i in 1 2 3
                            do
                                cp INCAR_$i INCAR
                                cp KPOINTS_$i KPOINTS
                                date > vasp-$i.vasp
                                {VASP_2d_exe} >> vasp-$i.vasp
                                #{VASP_3d_exe} >> vasp-$i.vasp
                                date >> vasp-$i.vasp
                                cp CONTCAR POSCAR
                                cp CONTCAR POSCAR_$i
                                cp OUTCAR OUTCAR_$i
                                rm WAVECAR CHGCAR
                            done
                            line=`cat CONTCAR | wc -l`
                            fail=`tail -10 vasp-1.vasp | grep WARNING | wc -l`
                            if [ $line -ge 8 -a $fail -eq 0 ]; then
                                scp CONTCAR {gpu_node}:{self.local_optim_strus_path}/$p
                                scp vasp-3.vasp {gpu_node}:{self.local_energy_path}/out-$p
                            fi
                            cd ../
                            
                            touch FINISH-$p
                            scp FINISH-$p {gpu_node}:{self.local_optim_strus_path}/
                            rm -rf $p FINISH-$p
                            '''
            self.ssh_node(shell_script, ip)
        counter = 0
        while not self.is_done(self.optim_strus_path, num_poscar):
            time.sleep(self.wait_time)
            if counter > vasp_time_limit:
                self.kill_vasp_jobs()
                break
            counter += 1
        self.remove_flag(self.optim_strus_path)
        self.delete_same_poscars(self.optim_strus_path)
        self.delete_energy_files(self.optim_strus_path, self.energy_path)
        self.get_energy(self.energy_path)
        system_echo(f'All jobs are completed --- Optimization')
    
    def run_optimization_high(self):
        '''
        optimize configurations from low to high level
        '''
        #set path and make directory
        self.local_optim_strus_path = f'{SCCOP_path}/{optim_strus_path}'   
        self.local_sccop_out_path = f'{SCCOP_path}/{sccop_out_path}'
        self.local_energy_path = f'{SCCOP_path}/{energy_path}'
        if not os.path.exists(optim_strus_path):
            os.mkdir(optim_strus_path)
            os.mkdir(optim_vasp_path)
            os.mkdir(energy_path)
        #sub optimization script to each node
        files = sorted(os.listdir(sccop_out_path))
        poscars = [i for i in files if i.startswith('POSCAR')]
        poscar_num = len(poscars)
        system_echo(f'Start VASP calculation --- Optimization')
        for poscar in poscars:
            node = poscar.split('-')[-1]
            ip = f'node{node}'
            shell_script = f'''
                            #!/bin/bash
                            cd {self.calculation_path}
                            p={poscar}
                            mkdir $p
                            cd $p
                            cp ../../{vasp_files_path}/Optimization/* .
                            scp {gpu_node}:{self.local_sccop_out_path}/$p POSCAR
                            
                            cp POSCAR POSCAR_0
                            DPT -v potcar
                            
                            for i in 4
                            do
                                cp INCAR_$i INCAR
                                cp KPOINTS_$i KPOINTS
                                date > vasp-$i.vasp
                                {VASP_2d_exe} >> vasp-$i.vasp
                                #{VASP_3d_exe} >> vasp-$i.vasp
                                date >> vasp-$i.vasp
                                cp CONTCAR POSCAR
                                cp CONTCAR POSCAR_$i
                                cp OUTCAR OUTCAR_$i
                                rm WAVECAR CHGCAR
                            done
                            if [ `cat CONTCAR|wc -l` -ge 8 ]; then
                                scp CONTCAR {gpu_node}:{self.local_optim_strus_path}/$p
                                scp vasp-4.vasp {gpu_node}:{self.local_energy_path}/out-$p
                            fi
                            cd ../
                            
                            touch FINISH-$p
                            scp FINISH-$p {gpu_node}:{self.local_optim_strus_path}/
                            rm -rf $p FINISH-$p
                            '''
            self.ssh_node(shell_script, ip)
        counter = 0
        while not self.is_done(optim_strus_path, poscar_num):
            time.sleep(self.wait_time)
            if counter > vasp_time_limit:
                self.kill_vasp_jobs()
                break
            counter += 1
        self.add_symmetry_to_structure(optim_strus_path)
        self.remove_flag(optim_strus_path)
        self.delete_same_poscars(optim_strus_path)
        self.delete_energy_files(optim_strus_path, energy_path)
        self.get_energy(energy_path)
        system_echo(f'All jobs are completed --- Optimization')
    
    def kill_vasp_jobs(self):
        """
        kill vasp jobs that reach time limit
        """
        for node in nodes:
            ip = f'node{node}'
            shell_script = f'''
                            #!/bin/bash
                            ps -ef | grep vasp | cut -c 9-15 | xargs kill -9
                            '''
            self.ssh_node(shell_script, ip)
    
    def add_symmetry_to_structure(self, path):
        """
        find symmetry unit of structure

        Parameters
        ----------
        path [str, 0d]: structure save path 
        """
        files = sorted(os.listdir(path))
        poscars = [i for i in files if i.startswith('POSCAR')]
        for i in poscars:
            stru = Structure.from_file(f'{path}/{i}')
            anal_stru = SpacegroupAnalyzer(stru)
            sym_stru = anal_stru.get_refined_structure()
            sym_stru.to(filename=f'{path}/{i}', fmt='poscar')
    
    def delete_energy_files(self, poscar_path, energy_path):
        """
        delete duplicate energy files

        Parameters
        ----------
        poscar_path [str, 0d]: poscars path
        energy_path [str, 0d]: energy file path
        """
        poscars = os.listdir(poscar_path)
        out = [i[4:] for i in os.listdir(energy_path)]
        del_poscars = np.setdiff1d(out, poscars)
        for i in del_poscars:
            os.remove(f'{energy_path}/out-{i}')
        
    def get_energy(self, path):
        """
        generate energy file of vasp outputs directory
        
        Parameters
        ----------
        path [str, 0d]: energy file path
        """
        energys = []
        vasp_out = os.listdir(f'{path}')
        vasp_out_order = sorted(vasp_out)
        for out in vasp_out_order:
            energy = 1E6
            VASP_output_file = f'{path}/{out}'
            with open(VASP_output_file, 'r') as f:
                ct = f.readlines()
            for line in ct[:15]:
                if 'POSCAR found :' in line:
                    atom_num = int(line.split()[-2])
            for line in ct[-15:]:
                if 'F=' in line:
                    energy = float(line.split()[2])
            cur_E = energy/atom_num
            system_echo(f'{out}, {cur_E:18.9f}')
            energys.append([out, cur_E])
        self.write_list2d(f'{path}/Energy.dat', energys)
        system_echo(f'Energy file generated successfully!')
        
    
if __name__ == "__main__":
    pass