import os, sys
import shutil, time
import numpy as np

sys.path.append(f'{os.getcwd()}/src')
from modules.global_var import *
from modules.utils import ListRWTools, SSHTools, system_echo


class SubVASP(ListRWTools, SSHTools):
    #submit vasp jobs
    def __init__(self, dE=1e-3, repeat=2, sleep_time=1):
        self.dE = dE
        self.repeat = repeat
        self.sleep_time = sleep_time
    
    def sub_VASP_job(self, round):
        """
        calculate POSCARs and return energys
        
        POSCAR file notation: POSCAR-round-number-node
        e.g., POSCAR-001-001-136
        
        Parameters
        ----------
        round [int, 0d]: searching-fitting rounds
        """
        round = f'{round:03.0f}'
        poscar = os.listdir(f'{poscar_dir}/{round}')
        poscar = sorted(poscar, key=lambda x: int(x.split('-')[2]))
        num_poscar = len(poscar)
        batches, nodes = self.assign_job(poscar)
        for i in range(self.repeat):
            if not os.path.exists(f'{vasp_out_dir}/{round}-{i}'):
                os.mkdir(f'{vasp_out_dir}/{round}-{i}')
            system_echo(f'Start VASP calculation---itersions: '
                        f'{round}-{i}, numbers: {num_poscar}')
            self.sub_batch_job(batches, round, i, nodes)
            while not self.is_VASP_done(round, i, num_poscar):
                time.sleep(self.sleep_time)
            system_echo(f'All job are completed---itersions: '
                        f'{round}-{i}, numbers: {num_poscar}')
            if i > 0:
                self.copy_true_file(round, i, true_E, vasp_out)
            true_E, false_E, vasp_out = self.get_energy(round, i)
            check_poscar = np.array(poscar)[false_E]
            num_poscar = len(check_poscar)
            if not num_poscar == 0:
                batches, nodes = self.assign_job(check_poscar)
            else:
                system_echo(f'VASP completed---itersions: '
                            f'{round}-{i}, numbers: {num_poscar}')
                break
    
    def assign_job(self, poscar):
        """
        assign jobs to each node according to notation of POSCAR file

        Parameters
        ----------
        poscar [str, 1d]: name of POSCAR files
        
        Returns
        ----------
        batches [str, 1d]: string of jobs assigned to different nodes
        nodes [str, 1d]: job assigned nodes
        """
        store, batches, nodes = [], [], []
        last_node = poscar[0][-3:]
        nodes.append(last_node)
        for item in poscar:
            node = item[-3:]
            if node == last_node:
                store.append(item)
            else:
                batches.append(' '.join(store))
                last_node = node
                store = []
                store.append(item)
                nodes.append(last_node)
        batches.append(' '.join(store))
        return batches, nodes
    
    def sub_batch_job(self, batches, round, repeat, nodes):
        """
        submit vasp jobs to nodes

        Parameters
        ----------
        batches [str, 1d]: string of jobs assigned to different nodes
        round [str, 0d]: iteration rounds
        round_repeat [str, 0d]: repeat times of vasp calculation
        """
        for j, batch in enumerate(batches):
            self.sub_jobs_with_ssh(batch, round, repeat, nodes[j])
    
    def sub_jobs_with_ssh(self, batch, round, repeat, node):
        """
        SSH to target node and call vasp for calculation
        
        Parameters
        ----------
        batch [str, 0d]: POSCAR files assigned to this node
        node [str, 0d]: job assgined node
        """
        ip = f'node{node}'
        shell_script = f'''
                      #!/bin/bash
                      ulimit -s 262140
                      cd /local
                      rm -r VASP_calculations
                      if [ ! -d 'VASP_calculations' ]; then
                            mkdir VASP_calculations
                            cd VASP_calculations
                            mkdir VASP_inputs
                            cp ~/ccop/{sing_point_energy_dir}/* VASP_inputs/.
                            cd ..
                      fi
                      cd VASP_calculations
                      for i in {batch}
                      do
                            mkdir $i
                            cd $i
                            cp ../VASP_inputs/* . 
                            cp ~/ccop/{poscar_dir}/{round}/$i POSCAR
                            DPT -v potcar
                            date > $i.out
                            echo 'VASP-JOB-FINISH' >> $i.out
                            /opt/intel/impi/4.0.3.008/intel64/bin/mpirun -np 48 vasp >> $i.out
                            date >> $i.out
                            cp $i.out ~/ccop/{vasp_out_dir}/{round}-{repeat}/.
                            rm *
                            cd ..
                      done
                      rm -r POSCAR*
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
        command = f'ls -l {vasp_out_dir}/{round}-{repeat} | grep ^- | wc -l'
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
        vasp_out = os.listdir(f'{vasp_out_dir}/{round}-{repeat}')
        vasp_out_order = sorted(vasp_out, key=lambda x: x.split('-')[2])
        for out in vasp_out_order:
            VASP_output_file = f'{vasp_out_dir}/{round}-{repeat}/{out}'
            with open(VASP_output_file, 'r') as f:
                ct = f.readlines()
            energy, state_line = 1e6, []
            for line in ct[:10]:
                if 'POSCAR found :' in line:
                    atom_num = int(line.split()[-2])
            for line in ct[-10:]:
                if 'F=' in line:
                    energy = float(line.split()[2])
                if 'DAV: ' in line:
                    state_line.append(line)
            if energy == 1e6:
                system_echo(' *WARNING* SinglePointEnergy is failed!')
                true_E.append(False)
            else:
                if abs(float(state_line[-1].split()[3])) < self.dE:
                    true_E.append(True)
                else:
                    true_E.append(False)
            cur_E = energy/atom_num
            energys.append([out, true_E[-1], cur_E])
            system_echo(f'{out}, {true_E[-1]}, {cur_E}')
        self.write_list2d(f'{vasp_out_dir}/Energy-{round}.dat', energys, '{0}')
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
                last_true_file = f'{vasp_out_dir}/{round}-{repeat-1}/{out}'
                current_true_file = f'{vasp_out_dir}/{round}-{repeat}/{out}'
                shutil.copyfile(last_true_file, current_true_file)
            system_echo(f'Copy true vasp out to next---true numbers: {len(true_out)}.')
    
    
if __name__ == "__main__":
    sub_vasp = SubVASP()
    sub_vasp.sub_VASP_job(21)