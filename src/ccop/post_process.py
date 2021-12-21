import os, time
from ccop.utils import ListRWTools, SSHTools, system_echo

class PostProcess(ListRWTools, SSHTools):
    ''' process the crystals by VASP to relax the structures and calculate the properties'''
    def __init__(self, str_path, run_dir, repeat=2, sleep_time=1):
        self.repeat = repeat
        self.sleep_time = sleep_time
        self.str_path = str_path
        self.run_dir = run_dir # the vasp run dir of the POSCARs
        self.vasp_files_path = '../libs/VASP_inputs'
        self.optim_strs_path = './optim_strs'   # save all the optimizated structures
        self.poscars = os.listdir(self.str_path)    # all poscar names
        self.num_poscar = len(self.poscar)

    def run_optimization(self):
        ''' prepare the optimization files and submit the job'''
        cur_run_path = f'{self.run_dir}/optimization'
        batches, nodes = self.assign_job()
        if not os.path.exists(cur_run_path):
            system_echo(f'Calculation Dir: {cur_run_path} create!')
            os.makedirs(cur_run_path)
        
        for i in range(self.repeat):
            system_echo(f'Start VASP calculation --- Optimization')
            for j, batch in enumerate(batches):
                shell_script = f'''
                    #!/bin/bash
                    ulimit -s 262140
                    for p in {batch}
                    do
                        if [ ! -d "{cur_run_path}/$p" ]; then
                            mkdir {cur_run_path}/$p
                        fi
                        cd {cur_run_path}/$p
                        cp {self.vasp_files_path}/Optimization/* .
                        cp {self.str_path}/$p POSCAR

                        cp POSCAR POSCAR_0
                        rm FINISH
                        DPT -v potcar
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
                        touch ../FINISH-$p
                        cd ../
                    done
                    '''
                self.sub_batch_job_ssh(batch, i, nodes[j], shell_script)
            while not self.is_VASP_done(cur_run_path):
                time.sleep(self.sleep_time)
            system_echo(f'All job are completed --- Optimization')
            
            # ***note:*** I donot check if the VASP finish in the good result
            #             maybe some structures have the bad optimization
            
            # save the optimizated structure into the prepared dir, if all the structures have the good optimization
            self.save_optimization(cur_run_path)

    
    def run_phonon(self):
        # prepare the phonon spectrum files and submit the job
        pass
    
    def run_pbe_band(self):
        # prepare the electronic structure files by PBE method and submit the job
        pass

    def assign_job(self):
        """
        assign jobs, this function is equal to the assign_job() in sub_vasp.py file
        """
        store, batches, nodes = [], [], []
        last_node = self.poscar[0][-3:]
        nodes.append(last_node)
        for item in self.poscar:
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
    
    def sub_batch_job(self, batch, repeat, node):
        """
        submit vasp jobs to nodes
        """
        self.sub_jobs_with_ssh(batch, repeat, node)
    
    def sub_jobs_with_ssh(self, batch, repeat, node, shell_script):
        """
        SSH to target node and call vasp for calculation
        
        Parameters
        ----------
        batch [str, 0d]: POSCAR files assigned to this node
        node [str, 0d]: job assgined node
        """
        ip = f'node{node}'
        self.ssh_node(shell_script, ip)
        
    def is_VASP_done(self, cur_run_path):
        """
        if the vasp calculation is completed, return True
        
        Parameters
        ----------
        num_poscars [int, 0d]: number of POSCARs
        
        Returns
        ----------
        flag [bool, 0d]: whether all nodes are done
        """
        command = f'ls -l {cur_run_path} | grep FINISH | wc -l'
        flag = self.check_num_file(command, self.num_poscar)
        return flag
    
    def save_optimization(self, cur_run_path):
        os.system(f'for p in {self.poscars} ; do cp {cur_run_path}/$p/CONTCAR {self.optim_strs_path}/$p ; done')
        
        
    
if __name__ == '__main__':
    a = PostProcess('../test/Optim', './VASP_calculations', 1)