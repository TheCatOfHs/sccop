import os, sys
import time
import re
import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.symmetry.kpath import KPathSeek
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.search import GeoCheck
from core.utils import ListRWTools, SSHTools, system_echo


class VASPoptimize(SSHTools, ListRWTools, GeoCheck):
    #optimize structure by VASP
    def __init__(self, recycle, wait_time=1):
        self.wait_time = wait_time
        self.ccop_out_path = f'{ccop_out_path}-{recycle}'
        self.optim_strs_path = f'{init_strs_path}_{recycle+1}'
        self.energy_path = f'{vasp_out_path}/initial_strs_{recycle+1}'
        self.local_ccop_out_path = f'/local/ccop/{self.ccop_out_path}'
        self.local_optim_strs_path = f'/local/ccop/{self.optim_strs_path}'
        self.local_energy_path = f'/local/ccop/{self.energy_path}'
        self.calculation_path = '/local/ccop/vasp'
        if not os.path.exists(self.optim_strs_path):
            os.mkdir(self.optim_strs_path)
            os.mkdir(self.energy_path)

    def run_optimization_low(self):
        '''
        optimize configurations at low level
        '''
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
                            #DPT --vdW DFT-D3
                            for i in 1
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
                                scp CONTCAR {gpu_node}:{self.local_optim_strs_path}/$p
                                scp vasp-1.vasp {gpu_node}:{self.local_energy_path}/out-$p
                            fi
                            cd ../
                            
                            touch FINISH-$p
                            scp FINISH-$p {gpu_node}:{self.local_optim_strs_path}/
                            rm -rf $p FINISH-$p
                            '''
            self.ssh_node(shell_script, ip)
        while not self.is_done(self.optim_strs_path, num_poscar):
            time.sleep(self.wait_time)
        self.remove_flag(self.optim_strs_path)
        self.delete_same_poscars(self.optim_strs_path)
        self.delete_energy_files(self.optim_strs_path, self.energy_path)
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
            sym_stru.to(filename=f'{optim_strs_sym_path}/{i}', fmt='poscar')
    
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

    
class PostProcess(VASPoptimize, GeoCheck):
    #process the crystals by VASP to relax the structures and calculate properties
    def __init__(self, wait_time=1):
        self.wait_time = wait_time
        self.ccop_out_path = f'/local/ccop/{ccop_out_path}'
        self.optim_strs_path = f'/local/ccop/{optim_strs_path}'
        self.dielectric_path = f'/local/ccop/{dielectric_path}'
        self.elastic_path = f'/local/ccop/{elastic_path}'
        self.energy_path = f'/local/ccop/{energy_path}'
        self.pbe_band_path = f'/local/ccop/{pbe_band_path}'
        self.phonon_path = f'/local/ccop/{phonon_path}'
        self.thermalconductivity_path = f'/local/ccop/{thermalconductivity_path}'
        self.KPOINTS = f'/local/ccop/{KPOINTS_file}'
        self.bandconf = f'/local/ccop/{bandconf_file}'
        self.calculation_path = '/local/ccop/vasp'
        if not os.path.exists(optim_strs_path):
            os.mkdir(optim_strs_path)
            os.mkdir(optim_strs_sym_path)
        if not os.path.exists('vasp'):
            os.mkdir('vasp')
            os.mkdir(KPOINTS_file)
            os.mkdir(bandconf_file)
        if not os.path.exists(optim_vasp_path):
            os.mkdir(optim_vasp_path)
            os.mkdir(dielectric_path)
            os.mkdir(elastic_path)
            os.mkdir(energy_path)
            os.mkdir(pbe_band_path)
            os.mkdir(phonon_path)
            os.mkdir(thermalconductivity_path)
    
    def run_optimization(self):
        '''
        optimize configurations from low to high level
        '''
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
                            #DPT --vdW DFT-D3
                            for i in 2 3
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
                                scp CONTCAR {gpu_node}:{self.optim_strs_path}/$p
                                scp vasp-3.vasp {gpu_node}:{self.energy_path}/out-$p
                            fi
                            cd ../
                            
                            touch FINISH-$p
                            scp FINISH-$p {gpu_node}:{self.optim_strs_path}/
                            rm -rf $p FINISH-$p
                            '''
            self.ssh_node(shell_script, ip)
        while not self.is_done(optim_strs_path, num_poscar):
            time.sleep(self.wait_time)
        self.add_symmetry_to_structure(optim_strs_path)
        self.remove_flag(optim_strs_path)
        self.delete_same_poscars(optim_strs_path)
        self.change_node_assign(optim_strs_path)
        self.get_energy(energy_path)
        system_echo(f'All jobs are completed --- Optimization')
    
    def run_pbe_band(self):
        '''
        calculate energy band of optimied configurations
        '''
        poscars = sorted(os.listdir(optim_strs_path))
        num_poscar = len(poscars)
        self.get_k_points(poscars, task='band')
        system_echo(f'Start VASP calculation --- Electronic structure')
        for poscar in poscars:
            node = poscar.split('-')[-1]
            ip = f'node{node}'
            shell_script = f'''
                            #!/bin/bash
                            cd {self.calculation_path}
                            p={poscar}
                            
                            mkdir $p
                            cd $p
                            cp ../../{vasp_files_path}/ElectronicStructure/* .
                            scp {gpu_node}:{self.optim_strs_path}/$p POSCAR
                            scp {gpu_node}:{self.KPOINTS}/KPOINTS-$p KPOINTS_2

                            DPT -v potcar
                            #DPT --vdW DFT-D3
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
                            python ../../libs/scripts/plot-energy-band.py
                            scp DPT.BAND.dat {gpu_node}:{self.pbe_band_path}/band-$p.dat
                            scp DPT.band.png {gpu_node}:{self.pbe_band_path}/band-$p.png
                            cd ../
                            touch FINISH-$p
                            scp FINISH-$p {gpu_node}:{self.optim_strs_path}/
                            rm -rf $p FINISH-$p
                            '''
            self.ssh_node(shell_script, ip)
        while not self.is_done(optim_strs_path, num_poscar):
            time.sleep(self.wait_time)
        system_echo(f'All jobs are completed --- Electronic structure')
        self.remove_flag(optim_strs_path)
    
    def run_phonon(self):
        '''
        calculate phonon spectrum of optimized configurations
        '''
        poscars = sorted(os.listdir(optim_strs_path))
        num_poscar = len(poscars)
        self.get_k_points(poscars, task='phonon')
        system_echo(f'Start VASP calculation --- Phonon spectrum')
        for poscar in poscars:
            node = poscar.split('-')[-1]
            ip = f'node{node}'
            shell_script = f'''
                            #!/bin/bash
                            cd {self.calculation_path}
                            p={poscar}
                            
                            mkdir $p
                            cd $p
                            cp ../../{vasp_files_path}/Phonon/* .
                            scp {gpu_node}:{self.optim_strs_path}/$p POSCAR
                            scp {gpu_node}:{self.bandconf}/band.conf-$p band.conf
                            #DPT --vdW DFT-D3
                            
                            phonopy -d --dim="3 3 1"
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
                            phonopy --full-fc band.conf
                            phonopy-bandplot --gnuplot --legacy band.yaml > phonon-$p.dat
                            python ../../libs/scripts/plot-phonon-band.py phonon-$p.dat band.conf
                            scp phonon-$p.dat {gpu_node}:{self.phonon_path}/phonon-$p.dat
                            scp PHON.png {gpu_node}:{self.phonon_path}/phonon-$p.png
                            scp FORCE_CONSTANTS {gpu_node}:{self.phonon_path}/FORCE_CONSTANTS_2ND-$p
                            cd ../
                            touch FINISH-$p
                            scp FINISH-$p {gpu_node}:{self.optim_strs_path}/
                            rm -rf $p FINISH-$p
                            '''
            self.ssh_node(shell_script, ip)
        while not self.is_done(optim_strs_path, num_poscar):
            time.sleep(self.wait_time)
        system_echo(f'All jobs are completed --- Phonon spectrum')
        self.remove_flag(optim_strs_path)
    
    def run_3RD(self):
        '''
        calculate third order force constants of optimized configurations
        '''
        poscars = sorted(os.listdir(optim_strs_path))
        num_poscar = len(poscars)
        system_echo(f'Start VASP calculation --- Third Order Force Constants')
        for poscar in poscars:
            node = poscar.split('-')[-1]
            ip = f'node{node}'
            shell_script = f'''
                            #!/bin/bash
                            cd {self.calculation_path}
                            p={poscar}
                            
                            mkdir $p
                            cd $p
                            cp ../../{vasp_files_path}/ThirdOrder/* .
                            scp {gpu_node}:{self.optim_strs_path}/$p POSCAR
                            tar -zxf thirdorder-files.tar.gz
                            cp thirdorder-files/* .
                            #DPT --vdW DFT-D3
                            
                            python2 thirdorder_vasp.py sow 2 2 2 -5
                            file=`ls | grep 3RD.POSCAR.`
                            for i in $file
                            do
                                mkdir disp-$i
                                cp INCAR KPOINTS vdw* disp-$i/
                                cp $i disp-$i/POSCAR
                                
                                cd disp-$i/
                                    DPT -v potcar
                                    /opt/intel/impi/4.0.3.008/intel64/bin/mpirun -np 48 vasp >> vasp.out 2>>err.vasp
                                    rm CHG* WAVECAR
                                cd ../
                            done
                            
                            find disp-* -name vasprun.xml | sort -n | python2 thirdorder_vasp.py reap 2 2 2 -5 > log
                            scp FORCE_CONSTANTS_3RD {gpu_node}:{self.phonon_path}/FORCE_CONSTANTS_3RD-$p
                            cd ../
                            touch FINISH-$p
                            scp FINISH-$p {gpu_node}:{self.optim_strs_path}/
                            rm -rf $p FINISH-$p
                            '''
            self.ssh_node(shell_script, ip)
        while not self.is_done(optim_strs_path, num_poscar):
            time.sleep(self.wait_time)
        system_echo(f'All jobs are completed --- Third Order Force Constants')
        self.remove_flag(optim_strs_path)
    
    def run_elastic(self):
        """
        calculate elastic matrix
        """
        poscars = sorted(os.listdir(optim_strs_path))
        num_poscar = len(poscars)
        system_echo(f'Start VASP calculation --- Elastic modulous')
        for poscar in poscars:
            node = poscar.split('-')[-1]
            ip = f'node{node}'
            shell_script = f'''
                            #!/bin/bash
                            cd {self.calculation_path}
                            p={poscar}
        
                            mkdir $p
                            cd $p
                            cp ../../{vasp_files_path}/Elastic/* .
                            scp {gpu_node}:{self.optim_strs_path}/$p POSCAR
                                
                            DPT -v potcar
                            cp INCAR_$i INCAR
                            cp KPOINTS_$i KPOINTS
                            /opt/intel/impi/4.0.3.008/intel64/bin/mpirun -np 48 vasp >> vasp.out

                            DPT --elastic
                            python ../../libs/scripts/plot-poisson-ratio.py
                            scp DPT.poisson.png {gpu_node}:{self.elastic_path}/poisson-$p.png
                            scp DPT.elastic_constant.dat {gpu_node}:{self.elastic_path}/elastic_constant-$p.dat
                            scp DPT.modulous.dat {gpu_node}:{self.elastic_path}/modulous-$p.dat
                            cd ../
                            touch FINISH-$p
                            scp FINISH-$p {gpu_node}:{self.optim_strs_path}/
                            rm -rf $p FINISH-$p
                            '''
            self.ssh_node(shell_script, ip)
        while not self.is_done(optim_strs_path, num_poscar):
            time.sleep(self.wait_time)
        system_echo(f'All jobs are completed --- Elastic modulous')
        self.remove_flag(optim_strs_path)
    
    def run_dielectric(self):
        """
        calculate dielectric matrix
        """
        poscars = sorted(os.listdir(optim_strs_path))
        num_poscar = len(poscars)
        system_echo(f'Start VASP calculation --- Dielectric tensor')
        for poscar in poscars:
            node = poscar.split('-')[-1]
            ip = f'node{node}'
            shell_script = f'''
                            #!/bin/bash
                            cd {self.calculation_path}
                            p={poscar}
                            
                            mkdir $p
                            cd $p
                            cp ../../{vasp_files_path}/Dielectric/* .
                            scp {gpu_node}:{self.optim_strs_path}/$p POSCAR
                                
                            DPT -v potcar
                            cp INCAR_$i INCAR
                            cp KPOINTS_$i KPOINTS
                            /opt/intel/impi/4.0.3.008/intel64/bin/mpirun -np 48 vasp >> vasp.out
                                
                            DPT --diele
                            scp dielectric.dat {gpu_node}:{self.dielectric_path}/dielectric-$p.dat
                            scp born_charges.dat {gpu_node}:{self.dielectric_path}/born_charges-$p.dat
                            cd ../
                            touch FINISH-$p
                            scp FINISH-$p {gpu_node}:{self.optim_strs_path}/
                            rm -rf $p FINISH-$p
                            '''
            self.ssh_node(shell_script, ip)
        while not self.is_done(optim_strs_path, num_poscar):
            time.sleep(self.wait_time)
        system_echo(f'All jobs are completed --- Dielectric tensor')
        self.remove_flag(optim_strs_path)
    
    def run_thermal_conductivity(self):
        """
        calculate boltzmann transport equation (BTE) to solve thermal conductivity
        """
        poscars = sorted(os.listdir(optim_strs_path))
        num_poscar = len(poscars)
        system_echo(f'Start VASP calculation --- Thermal Conductivity')
        for poscar in poscars:
            node = poscar.split('-')[-1]
            ip = f'node{node}'
            shell_script = f'''
                            #!/bin/bash
                            cd {self.calculation_path}
                            p={poscar}
                            
                            mkdir $p
                            cd $p
                            cp ../../{vasp_files_path}/ThermalConductivity/* .
                            scp {gpu_node}:{self.optim_strs_path}/$p POSCAR
                            scp {gpu_node}:{self.dielectric_path}/dielectric-$p.dat dielectric.dat
                            scp {gpu_node}:{self.dielectric_path}/born_charges-$p.dat born_charges.dat
                            scp {gpu_node}:{self.phonon_path}/FORCE_CONSTANTS_3RD-$p FORCE_CONSTANTS_3RD
                            scp {gpu_node}:{self.phonon_path}/FORCE_CONSTANTS_2ND-$p FORCE_CONSTANTS_2ND
                            
                            python create-CONTROL.py
                            mkdir files
                            mv FORCE_CONSTANTS_2ND FORCE_CONSTANTS_3RD CONTROL files/.
                            for i in `seq 100 20 920`
                            do
                                mkdir $i
                                cd $i
                                cp ../files/* .
                                sed -i "s|T-place|$i|g" CONTROL
                                ~/opt/openmpi-2.0.2/bin/mpirun -np 48 ShengBTE
                                cd ../
                            done
                            
                            python get-kappa.py
                            python ../../libs/scripts/plot-thermal-conductivity.py
                            scp kappa.dat {gpu_node}:{self.thermalconductivity_path}/kappa-$p.dat
                            scp 300/BTE.v {gpu_node}:{self.thermalconductivity_path}/BTE.v-$p.dat
                            scp 300/BTE.qpoints {gpu_node}:{self.thermalconductivity_path}/BTE.qpoints-$p.dat
                            scp 300/T300K/BTE.w {gpu_node}:{self.thermalconductivity_path}/BTE.w-$p.dat
                            scp Thermal.png {gpu_node}:{self.thermalconductivity_path}/thermal-$p.png
                            scp Scatter.png {gpu_node}:{self.thermalconductivity_path}/scatter-$p.png
                            scp Velocity.png {gpu_node}:{self.thermalconductivity_path}/velocity-$p.png
                            cd ../
                            touch FINISH-$p
                            scp FINISH-$p {gpu_node}:{self.optim_strs_path}/
                            rm -rf $p FINISH-$p
                            '''
            self.ssh_node(shell_script, ip)
        while not self.is_done(optim_strs_path, num_poscar):
            time.sleep(self.wait_time)
        system_echo(f'All jobs are completed --- Thermal Conductivity')
        self.remove_flag(optim_strs_path)

    def get_k_points(self, poscars, task):
        """
        search k path by Hinuma, Y., Pizzi, G., Kumagai, Y., Oba, F., & Tanaka, I. (2017)
        
        Parameters
        ----------
        poscars [str, 1d]: string list of poscars
        task [str, 0d]: task name
        """
        for poscar in poscars:
            structure = Structure.from_file(f'{optim_strs_path}/{poscar}')
            k_path = KPathSeek(structure)
            if task == 'band':
                k_points = list(k_path.get_kpoints(line_density=40, coords_are_cartesian=False))
                weights = [[1/len(k_points[0])] for _ in k_points[0]]
                labels = self.convert_special_k_labels(k_points[1], ['GAMMA', 'SIGMA_0', 'LAMBDA_0', 'DELTA_0'], 
                                                       ['\Gamma', '\Sigma', '\Lambda', '\Delta'])
                labels = [['  !'] if item == '' else [f'  ! ${item}$'] for item in labels]
                k_points.insert(1, labels)
                k_points.insert(1, weights)
                self.write_list2d_columns(f'{KPOINTS_file}/KPOINTS-{poscar}', k_points,
                                          ['{0:8.4f}', '{0:8.4f}', '{0:<16s}'], 
                                          head = ['Automatically generated mesh', str(len(k_points[0])), 'Reciprocal lattice'])
            elif task == 'phonon':
                points, labels = k_path.get_kpoints(line_density=1, coords_are_cartesian=False)
                labels = self.convert_special_k_labels(labels, ['GAMMA', 'SIGMA_0', 'LAMBDA_0', 'DELTA_0'], 
                                                       ['\Gamma', '\Sigma', '\Lambda', '\Delta'])
                while '' in labels:
                    points.pop(labels.index(''))
                    labels.remove('')
                phonon_points = [[[points[0], labels[0]]]]
                for i in range(1, len(labels)-2, 2):    # find the continuous bands
                    phonon_points[-1].append([points[i], labels[i]])
                    if labels[i] != labels[i+1]:
                        phonon_points.append([[points[i+1], labels[i+1]]])
                phonon_points[-1].append([points[-1], labels[-1]])
                band, band_label = '', ''
                for i, continuous_path in enumerate(phonon_points): # convert each continuous band to the required format
                    for point in continuous_path:   # e.g. 0.0 0.5 0.5  0.5 0.5 0.5  0.0 0.5 0.0,  0.0 0.0 0.0  0.5 0.0 0.0
                        band += '  {0}'.format(' '.join([f'{item:.3f}' for item in point[0]]))
                        band_label += ' ${0}$'.format(point[1])
                    band = band if i == len(phonon_points)-1 else band + ','
                band_conf = [[['ATOM_NAME'], ['DIM'], ['BAND'], ['BAND_LABEL'], ['FORCE_CONSTANTS']], 
                                [[' = '] for i in range(5)], 
                                [['XXX'], ['3 3 1'], [band], [band_label], ['write']]] # output the file by columns
                self.write_list2d_columns(f'{bandconf_file}/band.conf-{poscar}', band_conf, ['{0}', '{0}', '{0}'])
            else:
                system_echo(' Error: illegal parameter')
                exit(0)
        
    def convert_special_k_labels(self, labels, sp, std):
        """
        convert special labels to LaTeX style
        
        Parameters
        ----------
        labels [str, 1d]: list of labels
        sp [str, 1d]: special characters
        std [str, 1d]: LaTeX characters

        Returns
        ----------
        labels [str, 1d]: LaTeX labels
        """
        for i in range(len(std)):
            labels = [std[i] if item == sp[i] else item for item in labels]
        return labels
    
    
if __name__ == '__main__':
    #vasp = VASPoptimize(0)
    #vasp.run_optimization_low()
    #vasp.get_energy()
    post = PostProcess()
    #post.get_energy()
    #post.run_pbe_band()
    #post.run_phonon()
    #post.run_elastic()
    #post.run_dielectric()
    #post.run_3RD()
    #post.run_thermal_conductivity()
    #post.add_symmetry_to_structure('test/initial_strs_3')
    #post.rotate_axis('test/initial_strs_3')
    post.get_k_points(['POSCAR-CCOP-1-0030-132'], 'phonon')