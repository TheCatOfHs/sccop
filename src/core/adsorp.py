import os, sys
import time
import re

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.utils import ListRWTools, SSHTools, system_echo

from pymatgen.core.structure import Molecule, Structure
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.surface import SlabGenerator


class Adsorp(ListRWTools, SSHTools):
    #
    def __init__(self, wait_time=1):
        self.wait_time = wait_time
        self.calculation_path = '/local/ccop/vasp'
        self.local_adsorp_strs_path = f'/local/ccop/{adsorp_strs_path}'
        self.local_adsorp_energy_path = f'/local/ccop/{adsorp_energy_path}'
        if not os.path.exists(adsorp_strs_path):
            os.mkdir(adsorp_strs_path)
        
    def calculate(self, file, atom, miller_index, scalling_mat):
        """
        calculate adsorp energy 
        
        Parameters
        ----------
        file []: 
        atom []: 
        miller_index []: 
        scalling_mat []: 

        Returns
        ----------
        []: 
        """
        self.generate_poscar(file, atom, miller_index, scalling_mat)
        self.run_optimization()
        while not self.is_done(num, path):
            time.sleep(self.wait_time)
        self.remove_flag()
        energy = self.get_energy(adsorp_energy_path)
        energy = self.adsorp_energy()
        return energy
        
    def generate_poscar(self, file, atom, miller_index, scalling_mat):
        """
        generate adsorption POSCARs according to 
        Montoya, J.H., Persson, K.A. A npj Comput Mater 3, 14 (2017). 
        
        Parameters
        ----------
        file [str, 0d]: file name of adsorption structure
        atom [int, 1d]: atom number of adsorbate
        miller_index [tuple, 1d]: miller_index
        scalling_mat [int, 2d]: size of supercell
        """
        stru = Structure.from_file(file)
        slab = SlabGenerator(stru, miller_index=miller_index,
                             min_slab_size=2.0, min_vacuum_size=15.0)
        surface = slab.get_slab()
        asf = AdsorbateSiteFinder(surface)
        adsorbate = Molecule(atom, [[0, 0, 0]])
        ads_structs = asf.generate_adsorption_structures(adsorbate, scalling_mat)
        for i, poscar in enumerate(ads_structs):
            poscar.to(filename=f'{adsorp_strs_path}/POSCAR-Adsorp-{i:03.0f}', fmt='poscar')
    
    def run_optimization(self):
        '''
        optimize configurations
        '''
        files = sorted(os.listdir(adsorp_strs_path))
        num_poscar = len(files)
        for poscar in files:
            node = poscar.split('-')[-1]
            ip = f'node{node}'
            shell_script = f'''
                            #!/bin/bash
                            cd {self.calculation_path}
                            p={poscar}
                            mkdir $p
                            cd $p
                            cp ../../{vasp_files_path}/Optimization/* .
                            scp {gpu_node}:{self.local_adsorp_strs_path}/$p POSCAR
                            
                            cp POSCAR POSCAR_0
                            DPT -v potcar
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
                                scp CONTCAR {gpu_node}:{self.local_adsorp_strs_path}/$p-Relax
                                scp vasp-1.vasp {gpu_node}:{self.local_adsorp_energy_path}/out-$p
                            fi
                            cd ../
                            
                            touch FINISH-$p
                            scp FINISH-$p {gpu_node}:{self.local_adsorp_strs_path}/
                            rm -rf $p FINISH-$p
                            '''
            self.ssh_node(shell_script, ip)
        while not self.is_done(adsorp_strs_path, num_poscar):
            time.sleep(self.wait_time)
        self.get_energy(adsorp_energy_path)
        system_echo(f'All job are completed --- Optimization')
        self.remove_flag(adsorp_strs_path)
        
    def get_energy(self, path, ):
        """
        generate energy file of vasp outputs directory
        
        Parameters
        ----------
        path [str, 0d]: energy file path
        """
        energys = []
        vasp_out = os.listdir(f'{path}')
        vasp_out = [i for i in vasp_out if re.match(r'out', i)]
        vasp_out_order = sorted(vasp_out)
        for out in vasp_out_order:
            VASP_output_file = f'{path}/{out}'
            with open(VASP_output_file, 'r') as f:
                ct = f.readlines()
            for line in ct[-10:]:
                if 'F=' in line:
                    energy = float(line.split()[2])
            cur_E = energy
            energys.append([out, cur_E])
        self.write_list2d(f'{path}/Energy.dat', energys)
        
    def adsorp_energy(self, num, energy_single, energy_compound):
        energy_adsorp = self.import_list2d(adsorp_energy_path, str)
        energys = []
        for i, e in enumerate(energy_adsorp):
            formation = e - (num*energy_single + energy_compound)
            energys.append(formation)
            system_echo(f'Formation Energy of POSCAR-{i:03.0f}:{formation}')
        self.write_list2d(f'{adsorp_energy_path}/Energy_Adsorp.dat')
        
    
    
    
class Battery(SSHTools):
    #
    def __init__(self):
        pass
    
    def coverage(self,):
        pass
    
    def ion_transfer_rate(self,):
        pass

    def diffuse_barrier(self,):
        pass
    
    def open_circuit_voltage(self,):
        pass
    
    
if __name__ == '__main__':
    ads = Adsorp()
    ads.generate_poscar('POSCAR-CCOP-0-0008-132', [11], (1,0,0), [[1,0,0],[0,1,0],[0,0,1]])