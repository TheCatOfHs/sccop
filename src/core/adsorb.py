import os, sys, shutil
import time
import re
import itertools
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from collections import Counter
from sklearn.cluster import KMeans

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder, plot_slab

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.utils import *
from core.post_process import PostProcess
from core.search import GeoCheck


class AdsorbSites(ListRWTools, SSHTools, ClusterTools):
    #Find unequal adsorbtion sites and calculate formation energy
    def __init__(self, wait_time=1):
        self.wait_time = wait_time
        self.calculation_path = '/local/ccop/vasp'
        self.local_optim_strs_path = f'/local/ccop/{optim_strs_path}'
        self.local_anode_energy_path = f'/local/ccop/{anode_energy_path}'
        self.local_adsorb_strs_path = f'/local/ccop/{adsorb_strs_path}'
        self.local_adsorb_energy_path = f'/local/ccop/{adsorb_energy_path}'
        self.post = PostProcess()
        if not os.path.exists(adsorb_path):
            os.mkdir(adsorb_path)
            os.mkdir(anode_strs_path)
            os.mkdir(anode_energy_path)
            os.mkdir(adsorb_strs_path)
            os.mkdir(adsorb_energy_path)
            os.mkdir(adsorb_analysis_path)
    
    def get_slab(self):
        """
        generate slab structure
        """
        #calculate phonon spectrum
        self.rotate_axis(optim_strs_path, 0, 2)
        #self.post.run_phonon()
        #get and optimize slab structures
        poscars = sorted(os.listdir(optim_strs_path))
        self.rotate_axis(optim_strs_path, 15)
        self.run_optimization(poscars, 1, optim_strs_path, 
                              self.local_optim_strs_path, 
                              self.local_anode_energy_path)
        #choose dynamic stable structures
        '''
        for i in poscars:
            os.remove(f'{optim_strs_path}/{i}')
        poscars_relax = sorted(os.listdir(optim_strs_path))
        for i in poscars_relax:
            os.rename(f'{optim_strs_path}/{i}',
                      f'{optim_strs_path}/{i[:-6]}')
        self.rotate_axis(optim_strs_path, 0)
        self.select_poscar()
        '''
        
    def rotate_axis(self, path, vacuum_size, repeat=1):
        """
        rotate axis to make atoms in xy plane and add vaccum layer
        
        Parameters
        ----------
        path [str, 0d]: optimized structure save path
        vaccum_size [float, 0d]: size of vaccum layer
        repeat [int, 0d]: repeat times of rotate axis
        """
        files = sorted(os.listdir(path))
        poscars = [i for i in files if re.match(r'POSCAR', i)]
        for _ in range(repeat):
            for i in poscars:
                #import structure
                stru = Structure.from_file(f'{path}/{i}')
                latt = Lattice(stru.lattice.matrix)
                cart_coor = stru.cart_coords
                frac_coor = stru.frac_coords
                #calculate miller index
                miller_index = self.get_miller_index(latt, cart_coor, frac_coor)
                #generate slab poscar
                slab = SlabGenerator(stru, miller_index=miller_index, min_slab_size=.1,
                                     min_vacuum_size=vacuum_size, center_slab=True)
                surface = slab.get_slab()
                surface.to(filename=f'{path}/{i}', fmt='poscar')
    
    def get_miller_index(self, lattice, cart_coor, frac_coor):
        """
        find coplanar atoms
        
        Parameters
        ----------
        lattice [obj]: lattice object of structure
        cart_coor [float, 2d, np]: cartesian coordinate
        frac_coor [float, 2d, np]: fraction coordinate

        Returns
        ----------
        miller_index [float, 1d]: miller index of atom plane
        """
        #assume all atoms are coplanar
        volume = 0
        vectors = cart_coor - cart_coor[0]
        vec1, vec2 = vectors[1:3]
        normal = np.cross(vec1, vec2)
        for i in vectors:
            volume += np.abs(np.dot(normal, i))
        if volume < .2:
            flag = False
            miller_index = lattice.get_miller_index_from_coords(cart_coor, round_dp=0)
        else:
            flag = True
        #if atoms are distributed on two plane
        if flag:
            stds = []
            for coor in np.transpose(frac_coor):
                kmeans = KMeans(2, random_state=0).fit(coor.reshape(-1, 1))
                clusters = kmeans.labels_
                order = np.argsort(clusters)
                sort_coor = coor[order]
                a_num = dict(Counter(clusters))[0]
                a_std = np.std(sort_coor[:a_num])
                b_std = np.std(sort_coor[a_num:])
                stds.append(a_std + b_std)
            std_min = np.argsort(stds)[0]
            miller_index = np.identity(3, dtype=int)[std_min]
        return miller_index
    
    def select_poscar(self):
        """
        select poscar that is dynamic stable
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
        if os.path.exists(file):
            with open(file, 'r') as f:
                for _ in range(2):
                    next(f)
                for _ in range(1000):
                    line = f.readline().strip()
                    ct = line.split()
                    if len(ct) != 0:
                        freq = float(ct[1])
                        if freq < -.5:
                            counter += 1
                            if counter > 10:
                                flag = False
                                break
        else:
            flag = False
        return flag
    
    def relax(self, atom, repeat):
        """
        optimize adsorbate structures
        
        Parameters
        ----------
        atom [int, 1d]: adsorbate atom
        repeat [int, tuple]: size of supercell
        """
        files = sorted(os.listdir(anode_strs_path))
        for file in files:
            #generate adsorb poscars
            adsorbs = self.generate_poscar(file, atom, repeat)
            system_echo(f'adsorb poscar generate --- {file}')
            #optimize adsorb poscars
            monitor_path = f'{adsorb_strs_path}/{file}'
            local_strs_path = f'{self.local_adsorb_strs_path}/{file}' 
            local_energy_path = f'{self.local_adsorb_energy_path}/{file}'
            os.mkdir(f'{adsorb_energy_path}/{file}')
            self.run_optimization(adsorbs, 1, monitor_path, 
                                  local_strs_path, local_energy_path)
            system_echo(f'adsorb structure relaxed --- {file}')
    
    def generate_poscar(self, file, atom, repeat):
        """
        generate adsorb POSCARs according to 
        Montoya, J.H., Persson, K.A. A npj Comput Mater 3, 14 (2017). 
        
        Parameters
        ----------
        file [str, 0d]: name of adsorbed file
        atom [int, 1d]: atom number of adsorbate
        repeat [int, tuple]: size of supercell
        
        Returns
        ---------
        adsorb_names [str, 1d]: name of adsorb poscars
        """
        #get adsorb structure
        surface = Structure.from_file(f'{anode_strs_path}/{file}')
        slab = SlabGenerator(surface, miller_index=(0,0,1),
                             min_slab_size=.1, min_vacuum_size=0, center_slab=True)
        slab = slab.get_slab()
        asf = AdsorbateSiteFinder(slab)
        adsorbate = Molecule([atom], [[0, 0, 0]])
        ads_strus_one_site = asf.generate_adsorption_structures(adsorbate, repeat=repeat)
        ads_strus_two_site = asf.adsorb_both_surfaces(adsorbate, repeat=repeat)
        #get name of adsorb job
        one_num = len(ads_strus_one_site)
        two_num = len(ads_strus_two_site)
        node_assign = self.assign_node(one_num+two_num)
        adsorb_names = []
        #adsorb from one side
        os.mkdir(f'{adsorb_strs_path}/{file}')
        for i, poscar in enumerate(ads_strus_one_site):
            node = node_assign[i]
            name = f'{file}-adsorb-One-{i:03.0f}-{node}'
            poscar.to(filename=f'{adsorb_strs_path}/{file}/{name}', fmt='poscar')
            adsorb_names.append(name)
        #adsorb from both sides
        for i, poscar in enumerate(ads_strus_two_site):
            node = node_assign[i+one_num]
            name = f'{file}-adsorb-Two-{i:03.0f}-{node}'
            poscar.to(filename=f'{adsorb_strs_path}/{file}/{name}', fmt='poscar')
            adsorb_names.append(name)
        return adsorb_names
    
    def run_optimization(self, files, times, monitor_path, 
                         local_strs_path, local_energy_path):
        """
        optimize configurations
        
        Parameters
        ----------
        files [str, 1d]: name of relax poscars
        times [int, 0d]: number of optimize times
        monitor_path [str, 0d]: path of FINISH flags
        local_strs_path [str, 0d]: structure path in GPU node
        local_energy_path [str, 0d]: energy path in GPU node
        """
        opt_times = ' '.join([str(i) for i in range(1, times+1)])
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
                            cp ../../{vasp_files_path}/Adsorb/* .
                            scp {gpu_node}:{local_strs_path}/$p POSCAR
                            
                            cp POSCAR POSCAR_0
                            DPT -v potcar
                            DPT --vdW optPBE
                            for i in {opt_times}
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
                                scp CONTCAR {gpu_node}:{local_strs_path}/$p-Relax
                                scp vasp-{opt_times}.vasp {gpu_node}:{local_energy_path}/out-$p
                            fi
                            cd ../
                            
                            touch FINISH-$p
                            scp FINISH-$p {gpu_node}:{local_strs_path}/
                            rm -rf $p FINISH-$p
                            '''
            self.ssh_node(shell_script, ip)
        while not self.is_done(monitor_path, num_poscar):
            time.sleep(self.wait_time)
        system_echo(f'All jobs are completed --- Optimization')
        self.remove_flag(monitor_path)
    
    def sites_analysis(self, single, repeat):
        """
        position of adsorb sites and formation energy
        
        Parameters
        ----------
        single [float, 0d]: energy of simple substance
        repeat [int, tuple]: size of supercell 
        """
        ratio = reduce(lambda x, y: x*y, repeat)
        poscars = sorted(os.listdir(adsorb_energy_path))
        pattern = 'POSCAR-[\w]*-[\w]*-[\w]*'
        poscars = [i for i in poscars if not re.match(pattern, i)]
        for side in ['One', 'Two']:
            if side == 'One':
                atom_num = 1
            else:
                atom_num = 2
            for i in poscars:
                compound = ratio*self.get_energy(f'{anode_energy_path}/out-{i}')
                sites = self.sites_coor(side, i)
                form = self.formation_energy(side, i, atom_num, single, compound)
                result = np.concatenate((sites, form), axis=1)
                self.write_list2d(f'{adsorb_analysis_path}/{i}-{side}.dat', result)
                system_echo(f'adsorb sites analysis finished: {i}')
    
    def sites_coor(self, side, poscar):
        """
        get cartesian coordinate of adsorb
        
        Parameters
        ----------
        side [str, 0d]: adsorb side
        poscar [str, 0d]: name of relax structure 

        Returns
        ----------
        sites [float, 2d, np]: position of adsorbates
        """
        files = sorted(os.listdir(f'{adsorb_strs_path}/{poscar}'))
        pattern = f'{poscar}-[\w]*-{side}-[\w]*-[\w]*-Relax'
        poscars = [i for i in files if re.match(pattern, i)]
        #poscars = [i[:-6] for i in poscars]
        names, sites = [], []
        for i in poscars:
            stru = Structure.from_file(f'{adsorb_strs_path}/{poscar}/{i}')
            sites.append(stru.cart_coords[-1])
            names.append([i])
        return np.concatenate((names, sites), axis=1)

    def formation_energy(self, side, poscar, atom_num, single, compound):
        """
        calculate formation energy
        
        Parameters
        ----------
        side [str, 0d]: adsorb side
        poscar [str, 0d]: name of poscar
        atom_num [int, 0d]: number of adsorb atoms
        energy_single [float, 0d]: energy of simple substance
        energy_compound [float, 0d]: energy of compound
        
        Returns
        ----------
        formations [float, 2d, np]: formation energy
        """
        files = sorted(os.listdir(f'{adsorb_energy_path}/{poscar}'))
        pattern = f'out-{poscar}-[\w]*-{side}-[\w]*-[\w]*'
        poscars = [i for i in files if re.match(pattern, i)]
        formations = []
        for i in poscars:
            energy = self.get_energy(f'{adsorb_energy_path}/{poscar}/{i}')
            delta_E = energy - (atom_num*single + compound)
            formations.append(delta_E)
        return np.transpose([formations])
    
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

    def cluster_sites(self, max_sites=4):
        """
        cluster adsorb sites by cartesian coordinates 
        and choose the lowest energy sites as adsorb site
        
        Parameters
        ----------
        max_sites [int, 0d]: max number of clusters
        """
        poscars = sorted(os.listdir(adsorb_energy_path))
        pattern = 'POSCAR-[\w]*-[\w]*-[\w]*'
        poscars = [i for i in poscars if not re.match(pattern, i)]
        for side in ['One', 'Two']:
            for poscar in poscars:
                #cluster adsorb sites
                file = f'{adsorb_analysis_path}/{poscar}-{side}.dat'
                sites = self.import_list2d(file, str, numpy=True)
                #delete duplicates
                path = f'{adsorb_strs_path}/{poscar}'
                index, _ = self.delete_same_names(path, sites[:,0])
                sites = sites[index]
                #cluster sites
                coors = np.array(sites[:,1:3], dtype=float)
                if len(coors) < max_sites:
                    max_sites = len(coors)
                kmeans = KMeans(max_sites, random_state=0).fit(coors)
                clusters = kmeans.labels_
                idx_all = np.arange(len(clusters))
                values = np.array(sites[:,-1], dtype=float)
                idx = self.min_in_cluster(idx_all, values, clusters)
                cluster_sites = sites[idx]
                #export adsorb sites
                v = values[idx]
                energy_order = np.argsort(v)
                cluster_sites_order = cluster_sites[energy_order]
                file = f'{adsorb_analysis_path}/{poscar}-{side}-Cluster.dat'
                self.write_list2d(file, cluster_sites_order)

    def sites_plot(self, cluster=True):
        """
        plot formation energy of each adsorb site
        """
        if cluster:
            notation = '-Cluster'
        else:
            notation = ''
        poscars = sorted(os.listdir(adsorb_energy_path))
        pattern = 'POSCAR-[\w]*-[\w]*-[\w]*'
        poscars = [i for i in poscars if not re.match(pattern, i)]
        for side in ['One', 'Two']:
            for poscar in poscars:
                #import data
                file = f'{adsorb_analysis_path}/{poscar}-{side}{notation}.dat'
                sites = self.import_list2d(file, str, numpy=True)[:,1:]
                x, y, _, v = np.transpose(np.array(sites, dtype=float))
                #plot adsorb sites
                figure = plt.figure()
                ax = figure.add_subplot(1, 1, 1)
                cm = plt.cm.get_cmap('jet')
                plt.scatter(x, y, c=v, cmap=cm, s=160, marker='x', zorder=1000)
                #plot adsorbate
                slab = Structure.from_file(f'{anode_strs_path}/{poscar}')
                plot_slab(slab, ax, adsorption_sites=False, repeat=3)
                #appearance of figure
                plt.title(f'{poscar}', fontsize=16)
                ax.set_xlabel('x direction')
                ax.set_ylabel('y direction')
                clb = plt.colorbar()
                clb.ax.set_title('$\Delta$E/eV')
                #export figure
                file = f'{adsorb_analysis_path}/{poscar}-{side}{notation}.png'
                plt.savefig(file, dpi=600)
                plt.close('all')


class MultiAdsorbSites(AdsorbSites, GeoCheck):
    #
    def __init__(self):
        AdsorbSites.__init__(self)
    
    def generate_poscar(self, file, side, atom, repeat):
        """
        generate filling poscars
        
        Parameters
        ----------
        file [str, 0d]: file name of poscar
        side [str, 0d]: adsorb side
        atom [int, 0d]: atomic number of adsorbate
        repeat [int, tuple]: size of supercell
        """
        #get filling order
        dat = f'{adsorb_analysis_path}/{file}-{side}-Cluster.dat'
        cluster_sites = self.import_list2d(dat, str, numpy=True)
        poscars = cluster_sites[:,0]
        poscars_seq = self.filling_order(poscars)
        ratio = reduce(lambda x, y: x*y, repeat)
        #make directory of filled poscar
        fill_path = f'{adsorb_strs_path}/{file}-{side}-Fill'
        if not os.path.exists(fill_path):
            os.mkdir(fill_path)
        for path, poscars in enumerate(poscars_seq):
            slab = Structure.from_file(f'{anode_strs_path}/{file}')
            slab.make_supercell(repeat)
            #add atom by filling order
            sites_seq = []
            for poscar in poscars:
                adsorb_coor = []
                stru = Structure.from_file(f'{adsorb_strs_path}/{file}/{poscar}')
                for i, number in enumerate(stru.atomic_numbers):
                    if number == atom:
                        adsorb_coor.append(stru.frac_coords[i])
                seq = self.filling_sites(adsorb_coor, repeat)
                sites_seq += seq
            #export filling poscars
            for sites in sites_seq:
                for site in sites:
                    slab.append(atom, site)
                atom_num = dict(Counter(slab.atomic_numbers))[atom]
                filename = f'{fill_path}/{file[:-4]}-{side}-{path}-{atom_num}-{ratio}{file[-4:]}'
                slab.to(filename=filename, fmt='poscar')
        self.delete_same_poscars(f'{fill_path}')
        self.change_node_assign(f'{fill_path}')
        return os.listdir(f'{fill_path}')

    def filling_order(self, poscars):
        """
        filling from lowest energy sites
        finish one periodic site then begin next
        
        Parameters
        ----------
        poscars [str, 1d]: poscar of adsorb sites

        Returns
        ----------
        path [str, 2d]: order of poscars
        """
        poscar_0 = poscars[0]
        path = [list(i) for i in itertools.permutations(poscars[1:])]
        path = [[poscar_0]+i for i in path]
        return path
        
    def filling_sites(self, frac_coor, repeat):
        """
        filling periodic sites

        Parameters
        ----------
        frac_coor [float, 2d]: fraction coordinates
        repeat [int, tuple]: size of supercell

        Returns
        ----------
        sites_seq [float, 3d]: sequence of sites added 
        """
        divide = [1/i for i in repeat]
        a, b, c = [[j for j in np.arange(0, 1, i)] for i in divide]
        dispalce = [i for i in itertools.product(a, b, c)]
        #filling sites
        sites_seq = []
        for dis in dispalce:
            seq = []
            for coor in frac_coor:
                seq.append(coor + dis)
            sites_seq.append(seq)
        return sites_seq
        
    def relax(self, atom, repeat):
        """
        optimize adsorbate structures
        
        Parameters
        ----------
        atom [int, 1d]: adsorbate atom
        repeat [int, tuple]: size of supercell
        """
        files = sorted(os.listdir(anode_strs_path))
        for file in files:
            for side in ['One', 'Two']:
                #generate adsorb poscars
                adsorbs = self.generate_poscar(file, side, atom, repeat)
                system_echo(f'adsorb poscar generate --- {file}')
                #optimize adsorb poscars
                monitor_path = f'{adsorb_strs_path}/{file}-{side}-Fill'
                local_strs_path = f'{self.local_adsorb_strs_path}/{file}-{side}-Fill' 
                local_energy_path = f'{self.local_adsorb_energy_path}/{file}-{side}-Fill'
                os.mkdir(f'{adsorb_energy_path}/{file}-{side}-Fill')
                self.run_optimization(adsorbs, 1, monitor_path, 
                                      local_strs_path, local_energy_path)
            system_echo(f'adsorb structure relaxed --- {file}')
    
    def analysis(self, single, repeat):
        """
        position of adsorb sites and formation energy
        
        Parameters
        ----------
        single [float, 0d]: energy of simple substance 
        repeat [int, tuple]: size of supercell
        """
        ratio = reduce(lambda x, y: x*y, repeat)
        poscars = sorted(os.listdir(adsorb_energy_path))
        pattern = 'POSCAR-[\w]*-[\w]*-[\w]*'
        poscars = [i for i in poscars if not re.match(pattern, i)]
        for side in ['One', 'Two']:
            for i in poscars:
                compound = ratio*self.get_energy(f'{anode_energy_path}/out-{i}')
                form = self.formation_energy(side, i, single, compound)
                self.write_list2d(f'{adsorb_analysis_path}/{i}-{side}-Coverage.dat', form)
                system_echo(f'adsorb sites analysis finished: {i}')
    
    def formation_energy(self, side, poscar, single, compound):
        """
        calculate formation energy
        
        Parameters
        ----------
        side [str, 0d]: adsorb side
        poscar [str, 0d]: name of poscar
        atom_num [int, 0d]: number of adsorb atoms
        energy_single [float, 0d]: energy of simple substance
        energy_compound [float, 0d]: energy of compound
        
        Returns
        ----------
        formations [float, 2d, np]: formation energy
        """
        poscars = sorted(os.listdir(f'{adsorb_energy_path}/{poscar}-{side}-Fill'))
        atom_ratios, formations, capacities, ocvs = [], [], [], []
        for i in poscars:
            #
            atom_num, ratio = i.split('-')[-3:-1]
            atom_num = float(atom_num)
            ratio = float(ratio)
            atom_ratio = atom_num/ratio/2
            atom_ratios.append([atom_ratio])
            #
            energy = self.get_energy(f'{adsorb_energy_path}/{poscar}-{side}-Fill/{i}')
            delta_E = energy - (atom_num*single + compound)
            formations.append([delta_E])
            #
            capacity = (atom_ratio*26.81*1e3)/46.8/2
            capacities.append([capacity])
            #
            ocv = -delta_E/atom_num
            ocvs.append([ocv])
        result = np.concatenate((atom_ratios, formations, capacities, ocvs), axis=1)
        result = sorted(result, key=lambda x: x[0])
        return result


class NEBSolver():
    #
    def __init__(self):
        pass 
    

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
    repeat = (2,2,1)
    adsorb = AdsorbSites()
    adsorb.get_slab()
    '''
    if len(os.listdir(anode_strs_path)) > 0:
        #adsorb.relax(11, repeat)
        adsorb.sites_analysis(-1.3156, repeat)
        adsorb.cluster_sites()
        adsorb.sites_plot()
        adsorb.sites_plot(cluster=False)
        
        multi_adsorb = MultiAdsorbSites()
        multi_adsorb.relax(11, repeat)
        multi_adsorb.analysis(-1.3156, repeat)
    else:
        system_echo('No suitable adsorbates are found')
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''
    from pymatgen.core.structure import Structure
    from pymatgen.io.vasp.outputs import Chgcar, Outcar
    from pymatgen.analysis.path_finder import ChgcarPotential, NEBPathfinder
    from pymatgen.analysis.transition_state import NEBAnalysis
    
    poscar = Structure.from_file('test/POSCAR')
    start = Structure.from_file('test/POSCAR_1')
    end = Structure.from_file('test/POSCAR_2')
    chgcar = Chgcar(poscar, {'total':np.array([[[1]]]), 'diff':np.array([[0,0,0]])})
    potential = ChgcarPotential(chgcar.from_file('test/CHGCAR')).get_v()
    neb = NEBPathfinder(start, end, [-1], potential, n_images=20)
    for i, image in enumerate(neb.images):
        image.to(filename=f'test/NEB_path/POSCAR-NEB-{i:02.0f}', fmt='poscar')
    neb.plot_images('test/NEB_path/POSCAR-NEB')
    
    file = os.listdir('test/NEB_output')
    contcars = [i for i in file if i[0]=='C']
    outcars = [i for i in file if i[0]=='O']
    strus, outs = [], []
    for i in contcars:
        strus.append(Structure.from_file(f'test/NEB_output/{i}'))
    for i in outcars:
        outs.append(Outcar(f'test/NEB_output/{i}'))
    neb_analyzer = NEBAnalysis.from_outcars(outs, strus)
    neb_plot = neb_analyzer.get_plot()
    neb_plot.savefig('test/NEB_output/NEB.png', dpi=300)
    '''