import os, sys
import time

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.utils import ListRWTools, SSHTools, system_echo


class ParallelSubVASP(ListRWTools, SSHTools):
    #submit vasp jobs
    def __init__(self, wait_time=0.1):
        self.wait_time = wait_time
    
    def sub_job(self, round, vdW=False):
        """
        calculate POSCARs and return energys
        
        POSCAR file notation: POSCAR-round-number-node
        e.g., POSCAR-001-136
        
        Parameters
        ----------
        round [int, 0d]: ccop round
        vdW [bool, 0d]: whether add vdW modify
        """
        poscars = os.listdir(f'{poscar_path}/{round:03.0f}')
        num_poscar = len(poscars)
        #make directory
        out_path = f'{vasp_out_path}/{round:03.0f}'
        os.mkdir(out_path)
        system_echo(f'Start VASP calculation---itersions: '
                    f'{round}, number: {num_poscar}')
        #vasp calculation
        work_node_num = self.sub_vasp_job(poscars, round, vdW)
        while not self.is_done(out_path, work_node_num):
            time.sleep(self.wait_time)
        self.remove_flag(out_path)
        #get energy of outputs
        self.get_energy(round)
    
    def sub_vasp_job(self, poscars, round, vdW=False):
        """
        submit vasp jobs to nodes

        Parameters
        ----------
        poscars [str, 1d]: name of poscars
        round [int, 0d]: ccop round
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
            self.sub_job_with_ssh(job, round, vdW)
        work_node_num = len(jobs)
        return work_node_num
    
    def sub_job_with_ssh(self, job, round, vdW=False):
        """
        SSH to target node and call vasp for calculation
        
        Parameters
        ----------
        job [str, 1d]: name of poscars in same node
        round [int, 0d]: ccop round
        vdW [bool, 0d]: whether add vdW modify
        """
        flag_vdW = 0
        if vdW:
            flag_vdW = 1
        node = job[0].split('-')[-1]
        poscar_str = ' '.join(job)
        ip = f'node{node}'
        job_num = '`ps -ef | grep /opt/intel/impi/4.0.3.008/intel64/bin/mpirun | grep -v grep | wc -l`'
        local_vasp_out_path = f'/local/ccop/{vasp_out_path}/{round:03.0f}'
        shell_script = f'''
                        cd /local/ccop/vasp
                        for p in {poscar_str}
                        do
                            mkdir $p
                            cd $p

                            cp ../../{vasp_files_path}/SinglePointEnergy/* .
                            scp {gpu_node}:/local/ccop/{poscar_path}/{round:03.0f}/$p POSCAR
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
    
    def get_energy(self, round):
        """
        generate energy file of current vasp outputs directory 
        
        Parameters
        ----------
        round [int, 0d]: ccop round
        """
        true_E, energys = [], []
        vasp_out = os.listdir(f'{vasp_out_path}/{round:03.0f}')
        vasp_out = sorted(vasp_out)
        for out in vasp_out:
            VASP_output_file = f'{vasp_out_path}/{round:03.0f}/{out}'
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
        self.write_list2d(f'{vasp_out_path}/Energy-{round:03.0f}.dat', energys)
        system_echo(f'Energy file generated successfully!')


class VASPoptimize(SSHTools, ListRWTools):
    #optimize structure by VASP
    def __init__(self, recycle, wait_time=1):
        self.wait_time = wait_time
        self.ccop_out_path = f'{ccop_out_path}-{recycle}'
        self.optim_strus_path = f'{init_strus_path}_{recycle+1}'
        self.energy_path = f'{vasp_out_path}/initial_strus_{recycle+1}'
        self.local_ccop_out_path = f'/local/ccop/{self.ccop_out_path}'
        self.local_optim_strus_path = f'/local/ccop/{self.optim_strus_path}'
        self.local_energy_path = f'/local/ccop/{self.energy_path}'
        self.calculation_path = '/local/ccop/vasp'
        if not os.path.exists(self.optim_strus_path):
            os.mkdir(self.optim_strus_path)
            os.mkdir(self.energy_path)

    def run_optimization_low(self, vdW=False):
        '''
        optimize configurations at low level
        
        Parameters
        ----------
        vdW [bool, 0d]: whether add vdW modify
        '''
        flag_vdW = 0
        if vdW:
            flag_vdW = 1
        files = sorted(os.listdir(self.ccop_out_path))
        poscars = [i for i in files if re.match(r'POSCAR', i)]
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
                            scp {gpu_node}:{self.local_ccop_out_path}/$p POSCAR
                            
                            cp POSCAR POSCAR_0
                            DPT -v potcar
                            if [ {flag_vdW} -eq 1 ]; then
                                DPT --vdW DFT-D3
                            fi
                            
                            for i in 1 2 3
                            do
                                cp INCAR_$i INCAR
                                cp KPOINTS_$i KPOINTS
                                date > vasp-$i.vasp
                                /opt/intel/impi/4.0.3.008/intel64/bin/mpirun -np 48 vasp >> vasp-$i.vasp
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
        while not self.is_done(self.optim_strus_path, num_poscar):
            time.sleep(self.wait_time)
        self.remove_flag(self.optim_strus_path)
        self.add_symmetry_to_structure(self.optim_strus_path)
        self.delete_same_poscars(self.optim_strus_path)
        self.delete_energy_files(self.optim_strus_path, self.energy_path)
        self.get_energy(self.energy_path)
        system_echo(f'All jobs are completed --- Optimization')
        
    def add_symmetry_to_structure(self, path):
        """
        find symmetry unit of structure

        Parameters
        ----------
        path [str, 0d]: structure save path 
        """
        files = sorted(os.listdir(path))
        poscars = [i for i in files if re.match(r'POSCAR', i)]
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
    
    def run_optimization(self, vdW=False):
        '''
        optimize configurations from low to high level
        
        Parameters
        ----------
        vdW [bool, 0d]: whether add vdW modify
        '''
        flag_vdW = 0
        if vdW:
            flag_vdW = 1
        files = sorted(os.listdir(ccop_out_path))
        poscars = [i for i in files if re.match(r'POSCAR', i)]
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
                            scp {gpu_node}:{self.ccop_out_path}/$p POSCAR
                            
                            cp POSCAR POSCAR_0
                            DPT -v potcar
                            if [ {flag_vdW} -eq 1 ]; then
                                DPT --vdW DFT-D3
                            fi
                            
                            for i in 4 5
                            do
                                cp INCAR_$i INCAR
                                cp KPOINTS_$i KPOINTS
                                date > vasp-$i.vasp
                                /opt/intel/impi/4.0.3.008/intel64/bin/mpirun -np 48 vasp >> vasp-$i.vasp
                                date >> vasp-$i.vasp
                                cp CONTCAR POSCAR
                                cp CONTCAR POSCAR_$i
                                cp OUTCAR OUTCAR_$i
                                rm WAVECAR CHGCAR
                            done
                            if [ `cat CONTCAR|wc -l` -ge 8 ]; then
                                scp CONTCAR {gpu_node}:{self.optim_strus_path}/$p
                                scp vasp-5.vasp {gpu_node}:{self.energy_path}/out-$p
                            fi
                            cd ../
                            
                            touch FINISH-$p
                            scp FINISH-$p {gpu_node}:{self.optim_strus_path}/
                            rm -rf $p FINISH-$p
                            '''
            self.ssh_node(shell_script, ip)
        while not self.is_done(optim_strus_path, num_poscar):
            time.sleep(self.wait_time)
        self.add_symmetry_to_structure(optim_strus_path)
        self.remove_flag(optim_strus_path)
        self.delete_same_poscars(optim_strus_path)
        self.change_node_assign(optim_strus_path)
        self.get_energy(energy_path)
        system_echo(f'All jobs are completed --- Optimization')
    
    
    
if __name__ == "__main__":
    vasp = ParallelSubVASP()
    #vasp.sub_job(0)
    vasp.get_energy(0)