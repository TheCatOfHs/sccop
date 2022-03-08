import os, sys, shutil
import time
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from torch import inverse

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.utils import ListRWTools, SSHTools, system_echo

from pymatgen.core.structure import Molecule, Structure
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


class Adsorp(ListRWTools, SSHTools):
    #Find unequal adsorption sites and calculate formation energy
    def __init__(self, wait_time=1):
        self.wait_time = wait_time
        self.calculation_path = '/local/ccop/vasp'
        self.local_adsorp_strs_path = f'/local/ccop/{adsorp_strs_path}'
        self.local_adsorp_energy_path = f'/local/ccop/{adsorp_energy_path}'
        if not os.path.exists(adsorp_path):
            os.mkdir(anode_strs_path)
            os.mkdir(adsorp_strs_path)
            os.mkdir(adsorp_energy_path)
            os.mkdir(adsorp_analysis_path)
    
    def adsorp_sites_analysis(self, atom_num, single):
        """

        
        Parameters
        ----------
        atom_num []: 
        single []: 
        """
        files = sorted(os.listdir(adsorp_energy_path))
        poscars = np.unique([i[4:21] for i in files])
        for i in poscars:
            compound = self.get_energy(f'{energy_path}/out-{i[:-4]}')
            sites = self.adsorp_sites_coor(i)
            form = self.formation_energy(i, atom_num, single, compound)
            result = np.concatenate((sites, form), axis=1)
            self.write_list2d(f'{adsorp_analysis_path}/{i}.dat', result)
            system_echo(f'Adsorp sites analysis finished: {i}')
    
    def adsorp_sites_coor(self, poscar):
        """
        get cartesian coordinate of adsorp
        
        Parameters
        ----------
        poscar [str, 0d]: name of relax structure 

        Returns
        ----------
        sites [float, 2d, np]: position of adsorbates
        """
        files = sorted(os.listdir(adsorp_strs_path))
        pattern = f'{poscar}-[\w]*-[\w]*-[\w]*-Relax'
        poscars = [i for i in files if re.match(pattern, i)]
        poscars = [i[:-6] for i in poscars]
        sites = []
        for i in poscars:
            stru = Structure.from_file(f'{adsorp_strs_path}/{i}')
            sites.append(stru.cart_coords[-1])
        return np.array(sites)
    
    def formation_energy(self, poscar, atom_num, single, compound):
        """
        calculate formation energy
        
        Parameters
        ----------
        poscar [str, 0d]: name of poscar
        atom_num [int, 0d]: number of adsorp atoms
        energy_single [float, 0d]: energy of simple substance
        energy_compound [float, 0d]: energy of compound
        
        Returns
        ----------
        formations [float, 2d, np]: formation energy
        """
        files = sorted(os.listdir(adsorp_energy_path))
        poscars = [i for i in files if re.match(f'out-{poscar}', i)]
        formations = []
        for i in poscars:
            energy = self.get_energy(f'{adsorp_energy_path}/{i}')
            delta_E = energy - (atom_num*single + compound)
            formations.append(delta_E)
        return np.transpose([formations])
    
    def relax(self, atom, scaling_mat):
        """
        relax adsorped structures
        
        Parameters
        ----------
        atom [int, 1d]: adsorbate atom
        scaling_mat [int, 2d]: saclling matrix
        """
        self.select_poscar()
        files = sorted(os.listdir(anode_strs_path))
        for file in files:
            adsorp_names = self.generate_poscar(file, atom, scaling_mat)
            system_echo(f'adsorp poscar generate---{file}')
            self.run_optimization(adsorp_names)
            system_echo(f'adsorp structure relaxed---{file}')
        
    def select_poscar(self):
        """
        select poscar whose atoms in xy plane
        """
        files = sorted(os.listdir(optim_strs_path))
        for i in files:
            if self.judge_frequency(f'{phonon_path}/phonon-{i}.dat'):
                shutil.copy(f'{optim_strs_path}/{i}', anode_strs_path)
    
    def judge_frequency(self, file):
        """
        judge negative frequency
        
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
                    freq = float(ct[1])
                    if  freq < -1:
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
        adsorp_names [str, 1d]: name of adsorp poscars
        """
        #get adsorp structure
        surface = Structure.from_file(f'{anode_strs_path}/{file}')
        asf = AdsorbateSiteFinder(surface)
        adsorbate = Molecule(atom, [[0, 0, 0]])
        ads_strus = asf.generate_adsorption_structures(adsorbate, scaling_mat)
        #get name of adsorp job
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
        files [str, 1d]: name of relax poscars
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
        system_echo(f'All jobs are completed --- Optimization')
        self.remove_flag(adsorp_strs_path)
    
    def get_energy(self, file):
        """
        get energy of vasp output
        
        Parameters
        ----------
        file [str, 0d]: name of energy file
        """
        with open(file, 'r') as f:
            ct = f.readlines()
        for line in ct[-10:]:
            if 'F=' in line:
                energy = float(line.split()[2])
        return energy

    def adsorp_sites_plot(self,):
        
        files = sorted(os.listdir(adsorp_analysis_path))
        poscars = [i[:-4] for i in files]
        for i, dat in enumerate(files):
            result = self.import_list2d(f'{adsorp_analysis_path}/{dat}', float, numpy=True)
            slab = Structure.from_file(f'{anode_strs_path}/{poscars[i]}')
            x, y, _, v = np.transpose(result)
            figure = plt.figure()
            ax = figure.add_subplot(1, 1, 1)
            plt.scatter(x, y, c=v, s=50)
            clb = plt.colorbar()
            clb.ax.set_title('$\Delta$E/eV')
            plot_slab(slab, ax, adsorption_sites=True)
            plt.title(f'{poscars[i]}', fontsize=16)
            ax.set_xlabel('x direction')
            ax.set_ylabel('y direction')
            plt.scatter(x, y, c=v, s=50)
            plt.savefig(f'{adsorp_analysis_path}/{poscars[i]}.png', dpi=600)
            plt.close('all')
            break

    
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
    from pymatgen.analysis.adsorption import plot_slab
    ads = Adsorp()
    #ads.relax([11], [[1,0,0],[0,1,0],[0,0,1]])
    #ads.adsorp_sites_analysis(1, -1.3156)
    ads.adsorp_sites_plot()
    #stru = Structure.from_file(f'{anode_strs_path}/POSCAR-04-131-131')
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plot_slab(stru, ax, adsorption_sites=False)
    #plt.show()
    '''
    from pymatgen.core.lattice import Lattice
    from pymatgen.core.surface import SlabGenerator
    from sklearn.cluster import DBSCAN
    
    def rotate_axis(path):
        """
        rotate axis to make atoms in xy plane
        
        Parameters
        ----------
        path [str, 0d]: structure save path
        """
        files = sorted(os.listdir(path))
        poscars = [i for i in files if re.match(r'POSCAR', i)]
        for i in poscars:
            stru = Structure.from_file(f'{path}/{i}')
            latt = Lattice(stru.lattice.matrix)
            miller_index = latt.get_miller_index_from_coords(stru.cart_coords, round_dp=0)
            #
            if sum(np.abs(miller_index)) > 10:
                cluster_num = []
                for coor in np.transpose(stru.cart_coords):
                    clustering = DBSCAN(eps=.5, min_samples=1).fit(coor.reshape(-1,1))
                    num = len(np.unique(clustering.labels_))
                    cluster_num.append(num)
                axis = np.argsort(cluster_num)[0]
                miller_index = np.identity(3, dtype=int)[axis]
            #
            slab = SlabGenerator(stru, miller_index=miller_index,
                                 min_slab_size=.1, min_vacuum_size=0.0, center_slab=True)
            surface = slab.get_slab()
            surface.to(filename=f'{i}', fmt='poscar')
    rotate_axis('test')
    '''
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