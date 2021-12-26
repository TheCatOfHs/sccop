import os, sys
import time
import re
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.symmetry.kpath import KPathSetyawanCurtarolo

sys.path.append(f'{os.getcwd()}/src')
from modules.global_var import *
from modules.utils import ListRWTools, SSHTools, system_echo


class PostProcess(SSHTools, ListRWTools):
    #process the crystals by VASP to relax the structures and calculate the properties
    def __init__(self, sleep_time=1):
        self.sleep_time = sleep_time
        self.ccop_out_dir = f'~/ccop/{ccop_out_dir}'
        self.optim_strs_path = f'~/ccop/{optim_strs_path}'
        self.dielectric_path = f'~/ccop/{dielectric_path}'
        self.elastic_path = f'~/ccop/{elastic_path}'
        self.energy_path = f'~/ccop/{energy_path}'
        self.pbe_band_path = f'~/ccop/{pbe_band_path}'
        self.phonon_path = f'~/ccop/{phonon_path}'
        self.KPOINTS = '~/ccop/vasp/KPOINTS'
        self.phonon_ks = '~/ccop/vasp/phonon_ks'
        self.calculation_path = '/local/ccop/vasp'
        if not os.path.exists('vasp'):
            os.mkdir('vasp')
            os.mkdir('vasp/KPOINTS')
            os.mkdir('vasp/phonon-ks')
        if not os.path.exists(optim_strs_path):
            os.mkdir(optim_strs_path)
        if not os.path.exists(optim_vasp_path):
            os.mkdir(optim_vasp_path)
            os.mkdir(dielectric_path)
            os.mkdir(elastic_path)
            os.mkdir(energy_path)
            os.mkdir(pbe_band_path)
            os.mkdir(phonon_path)
        
    def run_optimization(self):
        '''
        optimize configurations from low to high level
        '''
        files = sorted(os.listdir(ccop_out_dir))
        self.poscars = [i for i in files if re.match(r'POSCAR', i)]
        self.num_poscar = len(self.poscars)
        batches, nodes = self.assign_job(self.poscars)
        system_echo(f'Start VASP calculation --- Optimization')
        for j, batch in enumerate(batches):
            shell_script = f'''
                            #!/bin/bash
                            cd {self.calculation_path}
                            for p in {batch}
                            do
                                mkdir $p
                                cd $p
                                cp ../../{vasp_files_path}/Optimization/* .
                                cp {self.ccop_out_dir}/$p POSCAR
                                
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
                                if [ `cat CONTCAR|wc -l` -ne 0 ]; then
                                    cp CONTCAR {self.optim_strs_path}/$p
                                    cp vasp-3.vasp {self.energy_path}/out-$p
                                fi
                                cd ../
                                touch FINISH-$p
                                mv FINISH-$p {self.optim_strs_path}/
                                rm -r $p
                            done
                            '''
            self.sub_jobs_with_ssh(nodes[j], shell_script)
        while not self.is_done():
            time.sleep(self.sleep_time)
        system_echo(f'All job are completed --- Optimization')
        self.remove()

    def run_pbe_band(self):
        '''
        calculate energy band of optimied configurations
        '''
        self.poscars = sorted(os.listdir(optim_strs_path))
        self.num_poscar = len(self.poscars)
        batches, nodes = self.assign_job(self.poscars)
        self.get_k_points(self.poscars, format='band')
        system_echo(f'Start VASP calculation --- Electronic structure')
        for j, batch in enumerate(batches):
            shell_script = f'''
                            #!/bin/bash
                            cd {self.calculation_path}
                            for p in {batch}
                            do
                                mkdir $p
                                cd $p
                                cp ../../{vasp_files_path}/ElectronicStructure/* .
                                cp {self.optim_strs_path}/$p POSCAR
                                cp {self.KPOINTS}/KPOINTS-$p KPOINTS_2

                                for i in 1 2
                                do
                                    DPT -v potcar
                                    cp INCAR_$i INCAR
                                    cp KPOINTS_$i KPOINTS
                                    /opt/intel/impi/4.0.3.008/intel64/bin/mpirun -np 48 vasp >> vasp.out 2>>err.vasp
                                    cp OUTCAR OUTCAR_$i
                                    cp IBZKPT IBZKPT_$i
                                    cp EIGENVAL EIGENVAL_$i
                                done
                            
                                DPT -b
                                python ../../libs/scripts/plot-energy-band.py
                                cp DPT.BAND.dat {self.pbe_band_path}/band-$p.dat
                                cp DPT.band.png {self.pbe_band_path}/band-$p.png
                                cd ../
                                touch FINISH-$p
                                mv FINISH-$p {self.optim_strs_path}/
                                rm -r $p
                            done
                            '''
            self.sub_jobs_with_ssh(nodes[j], shell_script)
        while not self.is_done():
            time.sleep(self.sleep_time)
        system_echo(f'All job are completed --- PBE band')
        self.remove()
        
    def run_phonon(self):
        '''
        calculate phonon spectrum of optimized configurations
        '''
        self.poscars = sorted(os.listdir(optim_strs_path))
        self.num_poscar = len(self.poscars)
        batches, nodes = self.assign_job(self.poscars)
        self.get_k_points(self.poscars, format='phonon')
        system_echo(f'Start VASP calculation --- Phonon spectrum')
        for j, batch in enumerate(batches):
            shell_script = f'''
                            #!/bin/bash
                            cd {self.calculation_path}
                            for p in {batch}
                            do
                                mkdir $p
                                cd $p
                                cp ../../{vasp_files_path}/Phonon/* .
                                cp {self.optim_strs_path}/$p POSCAR

                                phonopy -d --dim="2 2 2"
                                n=`ls | grep POSCAR- | wc -l`
                                for i in `seq -f%03g 1 $n`
                                do
                                    mkdir disp-$i
                                    cp INCAR KPOINTS POTCAR vdw* disp-$i/
                                    cp POSCAR-$i disp-$i/POSCAR
                                
                                    cd disp-$i/
                                        DPT -v potcar
                                        /opt/intel/impi/4.0.3.008/intel64/bin/mpirun -np 48 vasp >> vasp.out 2>>err.vasp
                                        rm CHG* WAVECAR
                                        cp vasprun.xml ../vasprun.xml-$i
                                    cd ..
                                done
                                
                                phonopy -f vasprun.xml-*
                                phonopy band.conf
                                phonopy-bandplot --gnuplot --legacy band.yaml > phonon-$p.dat
                                python ../../libs/scripts/plot-phonon-band.py phonon-$p.dat
                                cp phonon-$p.dat {self.phonon_path}/phonon-$p.dat
                                cp PHON.png {self.phonon_path}/phonon-$p.png
                                cd ../
                                touch FINISH-$p
                                mv FINISH-$p {self.optim_strs_path}/
                                rm -r $p
                            done
                            '''
            self.sub_jobs_with_ssh(nodes[j], shell_script)
        while not self.is_done():
            time.sleep(self.sleep_time)
        system_echo(f'All job are completed --- Phonon')
        self.remove()
    
    def run_elastic(self):
        pass
    
    def run_dielectric(self):
        pass
    
    def sub_jobs_with_ssh(self, node, shell_script):
        """
        SSH to target node and call vasp for calculation
        
        Parameters
        ----------
        node [int, 1d]: POSCAR files assigned to nodes
        shell_script [str, 0d]: job assgined to nodes
        """
        ip = f'node{node}'
        self.ssh_node(shell_script, ip)
        
    def is_done(self):
        """
        if the vasp calculation is completed, return True
        
        Parameters
        ----------
        num_poscars [int, 0d]: number of POSCARs
        
        Returns
        ----------
        flag [bool, 0d]: whether all nodes are done
        """
        command = f'ls -l {optim_strs_path} | grep FINISH | wc -l'
        flag = self.check_num_file(command, self.num_poscar)
        return flag
    
    def remove(self):
        """
        remove FINISH flags
        """
        os.system(f'rm {optim_strs_path}/FINISH*')

    def get_energy(self):
        """
        generate energy file of vasp outputs directory
        """
        energys = []
        vasp_out = os.listdir(f'{energy_path}')
        vasp_out_order = sorted(vasp_out)
        for out in vasp_out_order:
            VASP_output_file = f'{energy_path}/{out}'
            with open(VASP_output_file, 'r') as f:
                ct = f.readlines()
            for line in ct[:10]:
                if 'POSCAR found :' in line:
                    atom_num = int(line.split()[-2])
            for line in ct[-10:]:
                if 'F=' in line:
                    energy = float(line.split()[2])
            cur_E = energy/atom_num
            system_echo(f'{out}, {cur_E:18.9f}')
            energys.append([out, cur_E])
        self.write_list2d(f'{energy_path}/Energy.dat', energys, '{0:18.9f}')
        system_echo(f'Energy file generated successfully!')

    def get_k_points(self, poscars, format):
        for poscar in poscars:
            k_path = KPathSetyawanCurtarolo(Structure.from_file(f'{optim_strs_path}/{poscar}'))
            if format == 'band':
                k_points = list(k_path.get_kpoints(line_density=10, coords_are_cartesian=False))
                weights = [[1/len(k_points[0])] for _ in k_points[0]]
                labels = [f'\Gamma' if item == '\\Gamma' else item for item in k_points[1]]
                labels = [['  !'] if item == '' else [f'  ! ${item}$'] for item in labels]
                k_points.insert(1, labels)
                k_points.insert(1, weights)
                self.write_list2d_columns(f'vasp/KPOINTS/KPOINTS-{poscar}', k_points,
                                          ['{0:8.4f}', '{0:8.4f}', '{0:<16s}'], 
                                          head = ['Automatically generated mesh', str(len(k_points[0])), 'Reciprocal lattice'])
            elif format == 'phonon':
                k_points = list(k_path.get_kpoints(line_density=1, coords_are_cartesian=False))
            else:
                system_echo(' Error: illegal parameter')
                exit(0)

    def test(self):
        self.poscars = sorted(os.listdir(optim_strs_path))
        self.num_poscar = len(self.poscars)
        batches, nodes = self.assign_job(self.poscars)
        self.get_k_points(self.poscars, format='band')
    
    
if __name__ == '__main__':
    from modules.pretrain import Initial
    init = Initial()
    init.update()
    post = PostProcess()
    #post.run_optimization()
    #post.get_energy()
    post.run_pbe_band()
    #post.run_phonon()
    #post.run_elastic()
    #post.run_dielectric()
    #post.test()
    
    '''
    from pymatgen.core.structure import Structure
    from pymatgen.symmetry.kpath import KPathSetyawanCurtarolo
    crystal = Structure.from_file('test/GaN_ZnO_2/optim_strs/POSCAR-CCOP-001-131')
    kpath = KPathSetyawanCurtarolo(crystal)
    kpts = kpath.get_kpoints(line_density=10, coords_are_cartesian=False)
    print(kpts)
    '''