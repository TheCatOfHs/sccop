import os, time
from modules.utils import SSHTools, system_echo

class PostProcess(SSHTools):
    #process the crystals by VASP to relax the structures and calculate the properties
    def __init__(self, str_path, run_dir, sleep_time=1):
        self.sleep_time = sleep_time
        self.str_path = str_path
        self.run_dir = run_dir # the vasp run dir of the POSCARs
        self.libs_path = '../libs'  # the absolute path is the best
        self.vasp_files_path = f'{self.libs_path}/VASP_inputs'
        self.optim_strs_path = './optim_strs'   # save all the optimizated structures
        self.poscars = os.listdir(self.str_path)    # all poscar names
        self.num_poscar = len(self.poscars)
    
    def run_optimization(self):
        ''' 
        prepare the optimization files and submit the job.
        in this process, we copy the vasp files from libs/VASP_inputs/Optimization to self.run_dir/optimization/{each poscar}
        and then perform the calculations in each dir in turn
        here, we count the FINISH files' number to make sure all the calculations down
        after the calculations, save the optimized structures into the dir self.optim_strs_path
        '''
        cur_run_path, batches, nodes = self.prepare_data('Optimization')
        
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
            self.sub_jobs_with_ssh(nodes[j], shell_script)
        while not self.is_VASP_done(cur_run_path):
            time.sleep(self.sleep_time)
        system_echo(f'All job are completed --- Optimization')
            
            # ***note:*** I donot check if the VASP finish in the good result,
            #             maybe some structures have the bad optimization
            
            # save the optimizated structure into the prepared dir, if all the structures have the good optimization
        self.save_optimization(cur_run_path)

    
    def run_phonon(self):
        ''' 
        prepare the phonon spectrum files and submit the job
        same as the run_optimization(), but the phonopy package is used
        we save each phonon spectrum data into self.run_dir/Phonon/phonon-{poscar}.dat
        '''
        cur_run_path, batches, nodes = self.prepare_data('Phonon')
        for i in range(self.repeat):
            system_echo(f'Start VASP calculation --- Phonon spectrum')
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
                        cp {self.vasp_files_path}/Phonon/* .
                        cp {self.optim_strs_path}/$p POSCAR
                        
                        phonopy -d --dim="2 2 2"
                        n=`ls | grep POSCAR- | wc -l`
                        b=`expr $n + 1000`
                        c=${"{b:1:3}"}

                        for i in {"{001..$c}"}
                        do
                            mkdir disp-$i
                            cp INCAR KPOINTS POTCAR vdw* disp-$i/
                            cp POSCAR-$i disp-$i/POSCAR

                            cd disp-$i/
                                /opt/intel/impi/4.0.3.008/intel64/bin/mpirun -np 48 vasp >> vasp.out 2>>err.vasp
                                rm CHG* WAVECAR
                                cp vasprun.xml ../vasprun.xml-$i
                            cd ..
                        done
                        phonopy -f vasprun.xml-*
                        phonopy band.conf
                        phonopy-bandplot --gnuplot --legacy band.yaml > ../phonon-$p.dat
                        python {self.libs_path}/scripts/plot-phonon-band.py ../phonon-$p.dat
                        cp PHON.png ../phonon-$p.png
                        
                        touch ../FINISH-$p
                        cd ../
                    done
                    '''
                self.sub_jobs_with_ssh(nodes[j], shell_script)
            while not self.is_VASP_done(cur_run_path):
                time.sleep(self.sleep_time)
            system_echo(f'All job are completed --- Optimization')
    
    def run_pbe_band(self):
        ''' 
        prepare the electronic structure files by PBE method and submit the job.
        same as the run_phonon()
        the electronic structure data is generated by DPT and save into self.run_dir/ElectronicStructure/band-{poscar}.dat
        '''
        cur_run_path, batches, nodes = self.prepare_data('ElectronicStructure')
        for i in range(self.repeat):
            system_echo(f'Start VASP calculation --- Electronic Structure')
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
                        cp {self.vasp_files_path}/ElectronicStructure/* .
                        cp {self.optim_strs_path}/$p POSCAR
                        
                        for i in 1 2
                        do
                            cp INCAR_$i INCAR
                            cp KPOINTS_$i KPOINTS
                            /opt/intel/impi/4.0.3.008/intel64/bin/mpirun -np 48 vasp >> vasp.out 2>>err.vasp
                            cp OUTCAR OUTCAR_$i
                            cp IBZKPT IBZKPT_$i
                            cp EIGENVAL EIGENVAL_$i
                        done

                        DPT -b
                        cp DPT.BAND.dat ../band-$p.dat
                        python {self.libs_path}/scripts/plot-energy-band.py
                        cp DPT.band.png ../band-$p.png
                        
                        touch ../FINISH-$p
                        cd ../
                    done
                    '''
                self.sub_jobs_with_ssh(nodes[j], shell_script)
            while not self.is_VASP_done(cur_run_path):
                time.sleep(self.sleep_time)
            system_echo(f'All job are completed --- Optimization')

    def prepare_data(self, data_type):
        '''
        create the current run dir, assign the poscars and nodes by assign_job() and return
        data_type: the name of the run dir, such as Optimization, Phonon. It is free to set.
        '''
        cur_run_path = f'{self.run_dir}/{data_type}'
        if not os.path.exists(cur_run_path):
            system_echo(f'Calculation Dir: {cur_run_path} create!')
            os.makedirs(cur_run_path)
        batches, nodes = self.assign_job()
        return cur_run_path, batches, nodes
    
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
    
    def sub_jobs_with_ssh(self, node, shell_script):
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
    a = PostProcess('../test/Optim', './VASP_calculations')
    a.run_optimization()
    #a.run_phonon()
    #a.run_pbe_band()