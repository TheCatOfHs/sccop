import os, sys
import shutil, time
import numpy as np

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.utils import ListRWTools, SSHTools, system_echo


class ParallelSubVASP(ListRWTools, SSHTools):
    #submit vasp jobs
    def __init__(self, dE=1e-3, repeat=2, wait_time=0.1):
        self.dE = dE
        self.repeat = repeat
        self.wait_time = wait_time
    
    def sub_job(self, round, vdW=False):
        """
        calculate POSCARs and return energys
        
        POSCAR file notation: POSCAR-round-number-node
        e.g., POSCAR-001-0001-136
        
        Parameters
        ----------
        round [int, 0d]: searching rounds
        vdW [bool, 0d]: whether add vdW modify
        """
        round = f'{round:03.0f}'
        poscars = os.listdir(f'{poscar_path}/{round}')
        poscars = sorted(poscars, key=lambda x: int(x.split('-')[2]))
        num_poscar = len(poscars)
        check_poscar = poscars
        for i in range(self.repeat):
            out_path = f'{vasp_out_path}/{round}-{i}'
            os.mkdir(out_path)
            system_echo(f'Start VASP calculation---itersions: '
                        f'{round}-{i}, number: {num_poscar}')
            #begin vasp calculation
            work_node_num = self.sub_vasp_job(check_poscar, round, i, vdW)
            while not self.is_done(out_path, work_node_num):
                time.sleep(self.wait_time)
            self.remove_flag(out_path)
            system_echo(f'All job are completed---itersions: '
                        f'{round}-{i}, number: {num_poscar}')
            if i > 0:
                self.copy_true_file(round, i, true_E, vasp_out)
            true_E, false_E, vasp_out = self.get_energy(round, i)
            check_poscar = np.array(poscars)[false_E]
            num_poscar = len(check_poscar)
            if num_poscar == 0:
                system_echo(f'VASP completed---itersions: '
                            f'{round}-{i}, number: {num_poscar}')
                break
    
    def sub_vasp_job(self, poscars, round, repeat, vdW=False):
        """
        submit vasp jobs to nodes

        Parameters
        ----------
        poscars [str, 1d]: name of poscars
        round [str, 0d]: searching rounds
        repeat [str, 0d]: repeat times 
        vdW [bool, 0d]: whether add vdW modify
        
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
            self.sub_job_with_ssh(job, round, repeat, vdW)
        work_node_num = len(jobs)
        return work_node_num
        
    def sub_job_with_ssh(self, job, round, repeat, vdW=False):
        """
        SSH to target node and call vasp for calculation
        
        Parameters
        ----------
        job [str, 1d]: name of poscars in same node
        round [str, 0d]: searching rounds
        repeat [str, 0d]: repeat times 
        vdW [bool, 0d]: whether add vdW modify
        """
        flag_vdW = 0
        if vdW:
            flag_vdW = 1
        node = job[0].split('-')[-1]
        poscar_str = ' '.join(job)
        ip = f'node{node}'
        job_num = '`ps -ef | grep /opt/intel/impi/4.0.3.008/intel64/bin/mpirun | grep -v grep | wc -l`'
        local_vasp_out_path = f'/local/ccop/{vasp_out_path}/{round}-{repeat}'
        shell_script = f'''
                        cd /local/ccop/vasp
                        for p in {poscar_str}
                        do
                            mkdir $p
                            cd $p

                            cp ../../{vasp_files_path}/SinglePointEnergy/* .
                            scp {gpu_node}:/local/ccop/{poscar_path}/{round}/$p POSCAR
                            DPT -v potcar
                            if [ {flag_vdW} -eq 1 ]; then
                                DPT --vdW DFT-D3
                            fi
                            
                            nohup /opt/intel/impi/4.0.3.008/intel64/bin/mpirun -np 48 vasp >& $p.out&
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
    
    def get_energy(self, round, repeat):
        """
        generate energy file of current vasp outputs directory 
        
        Returns
        ----------
        true_E [bool, 1d]: true vasp file notate as true
        false_E [bool, 1d]: false vasp file notate as true
        vasp_out_order [str, 1d]: name of vasp files
        """
        true_E, energys = [], []
        vasp_out = os.listdir(f'{vasp_out_path}/{round}-{repeat}')
        vasp_out_order = sorted(vasp_out)
        for out in vasp_out_order:
            VASP_output_file = f'{vasp_out_path}/{round}-{repeat}/{out}'
            with open(VASP_output_file, 'r') as f:
                ct = f.readlines()
            energy, state_line = 1e6, []
            for line in ct[:15]:
                if 'POSCAR found :' in line:
                    atom_num = int(line.split()[-2])
            for line in ct[-15:]:
                if 'F=' in line:
                    energy = float(line.split()[2])
                if 'DAV: ' in line:
                    state_line.append(line)
            if energy == 1e6:
                system_echo(' *WARNING* SinglePointEnergy is failed!')
                true_E.append(False)
                cur_E = energy
            else:
                if abs(float(state_line[-1].split()[3])) < self.dE:
                    true_E.append(True)
                else:
                    true_E.append(False)
                cur_E = energy/atom_num
            energys.append([out, true_E[-1], cur_E])
            system_echo(f'{out}, {true_E[-1]}, {cur_E}')
        self.write_list2d(f'{vasp_out_path}/Energy-{round}.dat', energys)
        system_echo(f'Energy file generated successfully!')
        false_E = [not i for i in true_E]
        return true_E, false_E, vasp_out_order
    
    def copy_true_file(self, round, repeat, true_E, vasp_out):
        """
        copy convergence vasp outputs to next repeatation

        Parameters
        ----------
        repeat [int, 0d]: repeat rounds
        true_E [bool, 1d]: binary mask of reliable energy structures
        """
        true_out = np.array(vasp_out)[true_E]
        if len(true_out) > 0:
            for out in vasp_out:
                last_true_file = f'{vasp_out_path}/{round}-{repeat-1}/{out}'
                current_true_file = f'{vasp_out_path}/{round}-{repeat}/{out}'
                shutil.copyfile(last_true_file, current_true_file)
            system_echo(f'Copy true vasp out to next---true number: {len(true_out)}.')
    
    
if __name__ == "__main__":
    vasp = ParallelSubVASP()
    vasp.sub_job(0)