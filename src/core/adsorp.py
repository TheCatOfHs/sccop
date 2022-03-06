import os, sys, shutil
import time
import re
import numpy as np

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.utils import ListRWTools, SSHTools, system_echo

from pymatgen.core.structure import Molecule, Structure
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.surface import SlabGenerator


class Adsorp(ListRWTools, SSHTools):
    #Find unequal adsorption sites and calculate formation energy
    def __init__(self, wait_time=1):
        self.wait_time = wait_time
        self.calculation_path = '/local/ccop/vasp'
        self.local_adsorp_strs_path = f'/local/ccop/{adsorp_strs_path}'
        self.local_adsorp_energy_path = f'/local/ccop/{adsorp_energy_path}'
        if not os.path.exists(anode_strs_path):
            os.mkdir(anode_strs_path)
            os.mkdir(adsorp_strs_path)
    
    def formation_energy(self, num_atom, energy_single, energy_compound):
        """
        calculate formation energy
        
        Parameters
        ----------
        num_atom []: 
        energy_single []: 
        energy_compound []: 
        """
        energys = []
        energy_adsorp = self.import_list2d(f'{adsorp_energy_path}/Energy.dat', str)
        for i, e in enumerate(energy_adsorp):
            formation = e - (num_atom*energy_single + energy_compound)
            energys.append([formation])
            system_echo(f'Formation Energy of POSCAR-{i:03.0f}:{formation}')
        self.write_list2d(f'{adsorp_energy_path}/Energy_Formation.dat', energys)
    
    def relax(self, atom, scaling_mat):
        """
        relax adsorped structures
        
        Parameters
        ----------
        atom [int, 1d]: adsorbate atom
        scaling_mat [int, 2d]: saclling matrix
        """
        self.select_poscar()
        files = os.listdir(anode_strs_path)
        for file in files:
            adsorp_names = self.generate_poscar(file, atom, scaling_mat)
            system_echo(f'adsorp poscar generate---{file}')
            self.run_optimization(adsorp_names)
            system_echo(f'adsorp structure relaxed---{file}')
        self.get_energy(adsorp_energy_path)
    
    def select_poscar(self):
        """
        select poscar whose atoms in xy plane
        """
        files = os.listdir(optim_strs_path)
        for i in files:
            if self.judge_frequency(f'{phonon_path}/{i}'):
                stru = Structure.from_file(f'{optim_strs_path}/{i}')
                coor = stru.frac_coords
                coor_std = np.std(coor, axis=0)
                axis = np.argsort(coor_std)[0]
                if axis == 2:
                    shutil.copy(f'{optim_strs_path}/{i}', 
                                f'{anode_strs_path}')
    
    def judge_frequency(self, file):
        """

        Parameters
        ----------
        file [str, 0d]: name of file

        Returns
        ----------
        flag [bool, 0d]: whether stable
        """
        flag = True
        counter = 0
        with open(file, 'r') as f:
            for _ in range(2):
                next(f)
            for _ in range(500):
                line = f.readline().strip()
                ct = line.split()
                if ct != []:
                    if float(ct[1]) < 0:
                        counter += 1
                        if counter > 10:
                            flag = False
                            break
        return flag
    
    def generate_poscar(self, file, atom, scaling_mat):
        """
        generate adsorp POSCARs according to 
        Montoya, J.H., Persson, K.A. A npj Comput Mater 3, 14 (2017). 
        
        Parameters
        ----------
        file [str, 0d]: name of adsorped file
        atom [int, 1d]: atom number of adsorbate
        scaling_mat [int, 2d]: size of supercell
        
        Returns
        ---------
        adsorp_names [str, 1d]:
        """
        #
        stru = Structure.from_file(f'{anode_strs_path}/{file}')
        slab = SlabGenerator(stru, miller_index=(0,0,1),
                             min_slab_size=2.0, min_vacuum_size=15.0, center_slab=True)
        surface = slab.get_slab()
        asf = AdsorbateSiteFinder(surface)
        adsorbate = Molecule(atom, [[0, 0, 0]])
        ads_strus = asf.generate_adsorption_structures(adsorbate, scaling_mat)
        #
        job_num = len(ads_strus)
        node_assign = self.assign_node(job_num)
        adsorp_names = []
        for i, poscar in enumerate(ads_strus):
            node = node_assign[i]
            name = f'{file}-Adsorp-{i:03.0f}-{node}'
            poscar.to(filename=f'{adsorp_strs_path}/{name}', fmt='poscar')
            adsorp_names.append(name)
        return adsorp_names
        
    def run_optimization(self, files):
        '''
        optimize configurations
        
        Parameters
        ----------
        files [str, 1d]:
        '''
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
                            #DPT --vdW optPBE
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
        system_echo(f'All jobs are completed --- Optimization')
        self.remove_flag(adsorp_strs_path)
        
    def get_energy(self, path):
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

    
class Battery(SSHTools):
    #calculate properties of battery
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
    #ads = Adsorp()
    #ads.relax([11], [[1,0,0],[0,1,0],[0,0,1]])
    counter = 0
    with open('test/phonon/phonon-POSCAR-01-131-131.dat', 'r') as f:
        for i in range(2):
            next(f)
        for i in range(100):
            a = f.readline().strip()
            b = a.split()
            if b != []:
                print(float(b[1]))
                if float(b[1]) < 0:
                    counter += 1
                    if counter > 5:
                        break
    
    '''
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    import numpy as np
    stru = Structure.from_file('POSCAR-12-134-134')
    anal_stru = SpacegroupAnalyzer(stru)
    sym_stru = anal_stru.get_refined_structure()
    coor = sym_stru.frac_coords
    same_level = []
    for i in np.transpose(coor):
        same_level.append(len(np.unique(i)))
    axis = np.argsort(same_level)[0]
    miller_index = np.identity(3, dtype=int)[axis]
    slab = SlabGenerator(sym_stru, miller_index=miller_index,
                         min_slab_size=0.1, min_vacuum_size=15.0, center_slab=True)
    surface = slab.get_slab()
    print(stru)
    print(surface)
    surface.to(filename='POSCAR-Slab-12-134-134', fmt='poscar')
    '''
    