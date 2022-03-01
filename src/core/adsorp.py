import os, sys
import time

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.utils import SSHTools, system_echo

from pymatgen.core.structure import Molecule
from pymatgen.core.structure import Structure
from pymatgen.analysis.adsorption import *
from pymatgen.core.surface import generate_all_slabs


class Adsorp(SSHTools):
    #
    def __init__(self):
        pass
    
    def run_optimization(self):
        '''
        optimize configurations at low level
        '''
        files = sorted(os.listdir(self.ccop_out_path))
        num_poscar = len(poscars)
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
        self.get_energy(self.energy_path)
        system_echo(f'All job are completed --- Optimization')
        self.remove_flag(self.optim_strs_path)

    def generate(self,):
        pass
        
    def adsorp_energy(self, path):
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
            for line in ct[:10]:
                if 'POSCAR found :' in line:
                    atom_num = int(line.split()[-2])
            for line in ct[-10:]:
                if 'F=' in line:
                    energy = float(line.split()[2])
            cur_E = energy/atom_num
            system_echo(f'{out}, {cur_E:18.9f}')
            energys.append([out, cur_E])
        self.write_list2d(f'{path}/Energy.dat', energys)
        system_echo(f'Energy file generated successfully!')
    

if __name__ == '__main__':
    stru = Structure.from_file('POSCAR-CCOP-0-0008-132')
    slabs = generate_all_slabs(stru, max_index=1, min_slab_size=2.0, min_vacuum_size=15.0)
    bc_100 = [slab for slab in slabs if slab.miller_index==(1,0,0)][0]

    asf_bc_100 = AdsorbateSiteFinder(bc_100)
    ads_sites = asf_bc_100.find_adsorption_sites()

    adsorbate = Molecule([11], [[0, 0, 0]])
    ads_structs = asf_bc_100.generate_adsorption_structures(adsorbate)
    for i, poscar in enumerate(ads_structs):
        poscar.to(filename=f'POSCAR-{i}', fmt='poscar')