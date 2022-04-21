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
            os.mkdir(f'{vasp_out_path}/{round}-{i}')
            system_echo(f'Start VASP calculation---itersions: '
                        f'{round}-{i}, number: {num_poscar}')
            self.sub_vasp_job(check_poscar, round, i, vdW)
            while not self.is_VASP_done(round, i, num_poscar):
                time.sleep(self.wait_time)
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
        poscars [str, 1d]: name of poscar
        round [str, 0d]: searching rounds
        repeat [str, 0d]: repeat times 
        vdW [bool, 0d]: whether add vdW modify
        """
        for poscar in poscars:
            self.sub_job_with_ssh(poscar, round, repeat, vdW)
    
    def sub_job_with_ssh(self, poscar, round, repeat, vdW=False):
        """
        SSH to target node and call vasp for calculation
        
        Parameters
        ----------
        poscars [str, 1d]: name of poscar
        round [str, 0d]: searching rounds
        repeat [str, 0d]: repeat times 
        vdW [bool, 0d]: whether add vdW modify
        """
        flag_vdW = 0
        if vdW:
            flag_vdW = 1
        node = poscar.split('-')[-1]
        ip = f'node{node}'
        local_vasp_out_path = f'/local/ccop/{vasp_out_path}/{round}-{repeat}'
        shell_script = f'''
                        cd /local/ccop/vasp
                        p={poscar}
                        mkdir $p
                        cd $p
                        
                        cp ../../{vasp_files_path}/SinglePointEnergy/* .
                        scp {gpu_node}:/local/ccop/{poscar_path}/{round}/$p POSCAR
                        DPT -v potcar
                        if [ {flag_vdW} -eq 1 ]; then
                            DPT --vdW DFT-D3
                        fi
                        
                        date > $p.out
                        /opt/intel/impi/4.0.3.008/intel64/bin/mpirun -np 48 vasp >> $p.out
                        date >> $p.out
                        scp $p.out {gpu_node}:{local_vasp_out_path}/
                        cd ../
                        rm -r $p
                        '''
        self.ssh_node(shell_script, ip)
    
    def is_VASP_done(self, round, repeat, num_poscar):
        """
        if the vasp calculation is completed, return True
        
        Parameters
        ----------
        num_poscars [int, 0d]: number of POSCARs
        
        Returns
        ----------
        flag [bool, 0d]: whether all nodes are done
        """
        command = f'ls -l {vasp_out_path}/{round}-{repeat} | grep ^- | wc -l'
        flag = self.check_num_file(command, num_poscar)
        return flag
    
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