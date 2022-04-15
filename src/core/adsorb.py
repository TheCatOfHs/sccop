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
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder, plot_slab
from pymatgen.io.vasp.outputs import Chgcar, Outcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.path_finder import ChgcarPotential, NEBPathfinder
from pymatgen.analysis.transition_state import NEBAnalysis

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
        self.rotate_axis(optim_strs_path, 15)
        self.post.run_phonon()
        #get and optimize slab structures
        poscars = sorted(os.listdir(optim_strs_path))
        self.run_SinglePointEnergy(poscars, optim_strs_path, 
                                   self.local_optim_strs_path, 
                                   self.local_anode_energy_path,
                                   CHGCAR=True)
        #choose dynamic stable structures
        self.select_poscar()
    
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
            self.run_optimization(adsorbs, 2, monitor_path, 
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
    
    def run_optimization(self, poscars, times, monitor_path, 
                         local_strs_path, local_energy_path,
                         out=True):
        """
        optimize configurations
        
        Parameters
        ----------
        poscars [str, 1d]: name of poscars
        times [int, 0d]: number of optimize times
        monitor_path [str, 0d]: path of FINISH flags
        local_strs_path [str, 0d]: structure path in GPU node
        local_energy_path [str, 0d]: energy path in GPU node
        """
        flag_out = 0
        if out:
            flag_out = 1
        opt_times = ' '.join([str(i) for i in range(1, times+1)])
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
                            cp ../../{vasp_files_path}/Adsorb/Optimization/* .
                            scp {gpu_node}:{local_strs_path}/$p POSCAR
                            
                            cp POSCAR POSCAR_0
                            DPT -v potcar
                            DPT --vdW DFT-D3
                            
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
                            fail=`tail -10 vasp-{opt_times}.vasp | grep WARNING | wc -l`
                            if [ $line -ge 8 -a $fail -eq 0 ]; then
                                scp CONTCAR {gpu_node}:{local_strs_path}/$p-Relax
                                if [ {flag_out} -eq 1 ]; then
                                    scp vasp-{times}.vasp {gpu_node}:{local_energy_path}/out-$p
                                fi
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
    
    def run_SinglePointEnergy(self, poscars, monitor_path, 
                              local_strs_path, local_energy_path,
                              out=True, CHGCAR=False, OUTCAR=False):
        """
        calculate single point energy
        
        Parameters
        ----------
        poscars [str, 1d]: name of poscars
        monitor_path [str, 0d]: path of FINISH flags
        local_strs_path [str, 0d]: structure path in GPU node
        local_energy_path [str, 0d]: energy path in GPU node
        out [bool, 0d]: cp out file to local directory
        CHGCAR [bool, 0d]: cp CHGCAR to local directory
        OUTCAR [bool, 0d]: cp OUTCAR to local directory
        """
        flag_out = 0
        flag_CHGCAR = 0
        flag_OUTCAR = 0
        if out:
            flag_out = 1
        if CHGCAR:
            flag_CHGCAR = 1
        if OUTCAR:
            flag_OUTCAR = 1
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
                            cp ../../{vasp_files_path}/Adsorb/SinglePointEnergy/* .
                            scp {gpu_node}:{local_strs_path}/$p POSCAR
                            
                            DPT -v potcar
                            DPT --vdW DFT-D3
                            date > vasp-0.vasp
                            /opt/intel/impi/4.0.3.008/intel64/bin/mpirun -np 48 vasp >> vasp-0.vasp
                            date >> vasp-0.vasp
                            
                            fail=`tail -10 vasp-0.vasp | grep WARNING | wc -l`
                            if [ $fail -eq 0 ]; then
                                if [ {flag_out} -eq 1 ]; then
                                    scp vasp-0.vasp {gpu_node}:{local_energy_path}/out-$p
                                fi
                                if [ {flag_CHGCAR} -eq 1 ]; then
                                    scp CHGCAR {gpu_node}:{local_energy_path}/CHGCAR-$p
                                fi
                                if [ {flag_OUTCAR} -eq 1 ]; then
                                    scp OUTCAR {gpu_node}:{local_energy_path}/OUTCAR-$p
                                fi
                            fi
                            cd ../
                            
                            touch FINISH-$p
                            scp FINISH-$p {gpu_node}:{local_strs_path}/
                            rm -rf $p FINISH-$p
                            '''
            self.ssh_node(shell_script, ip)
        while not self.is_done(monitor_path, num_poscar):
            time.sleep(self.wait_time)
        system_echo(f'All jobs are completed --- Energy calculate')
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

    def cluster_sites(self, max_sites=3):
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

    def sites_plot(self, repeat, cluster=True):
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
                slab.make_supercell(repeat)
                plot_slab(slab, ax, adsorption_sites=False, window=0.9, repeat=1)
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
        dat = f'{adsorb_analysis_path}/{file}-One-Cluster.dat'
        cluster_sites = self.import_list2d(dat, str, numpy=True)
        poscars = cluster_sites[:,0][:2]
        poscars_seq = self.filling_order(poscars)
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
                seq = self.filling_sites(atom, file, poscar, side, repeat)
                sites_seq += seq
            #export filling poscars
            for sites in sites_seq:
                for site in sites:
                    slab.append(atom, site)
                atom_num = dict(Counter(slab.atomic_numbers))[atom]
                filename = f'{fill_path}/{file[:-4]}-{side}-{path}-{atom_num}{file[-4:]}'
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
    
    def filling_sites(self, atom, file, poscar, side, repeat):
        """
        filling sites by adsorb energy order
        
        Parameters
        ----------
        atom [int, 0d]: atomic number of adsorbate
        file [str, 0d]: file name of poscar
        poscar [str, 0d]: poscar of adsorb site
        side [str, 0d]: adsorb side
        repeat [int, tuple]: size of supercell

        Returns
        ----------
        seq [float, 2d]: sequence of filling sites
        """
        stru = Structure.from_file(f'{anode_strs_path}/{file}')
        stru.make_supercell(repeat)
        latt = stru.lattice
        adsorb_stru = Structure.from_file(f'{adsorb_strs_path}/{file}/{poscar}')
        adsorb_coor = adsorb_stru.frac_coords[-1]
        #get all equivalent sites
        analy = SpacegroupAnalyzer(stru)
        sym_op = analy.get_symmetry_operations()
        sym_coor = [i.operate(adsorb_coor) for i in sym_op]
        #delete same sites
        sym_coor = [PeriodicSite(atom, i, latt, to_unit_cell=True) for i in sym_coor]
        sym_coor = self.remove_same_periodic_sites(sym_coor)
        sym_coor = [i.frac_coords for i in sym_coor]
        sym_coor = [i for i in sym_coor if i[-1] > 0.5]
        #adsorb on one or both sides
        if side == 'One':
            seq = [[i] for i in sym_coor]
        elif side == 'Two':
            seq = [[i, np.abs(np.array([0,0,1])-i)] for i in sym_coor]
        return seq
    
    def remove_same_periodic_sites(self, sites):
        """
        delete duplicates of periodic sites
        
        Parameters
        ----------
        sites [obj, 2d]: periodic sites

        Returns
        ----------
        sites [obj, 2d]: unique periodic sites
        """
        point_num = len(sites)
        for i in range(point_num):
            same_points = []
            coor_1 = sites[i]
            for j in range(i+1, point_num):
                coor_2 = sites[j]
                dis = coor_1.distance(coor_2)
                if dis < 0.1:
                    same_points.append(j)
            sites = np.delete(sites, same_points, axis=0)
            point_num = len(sites)
            if i+1 == point_num:
                break
        return sites
    
    def relax(self, atom, repeat, sides):
        """
        optimize adsorbate structures
        
        Parameters
        ----------
        atom [int, 1d]: adsorbate atom
        repeat [int, tuple]: size of supercell
        sides [str, 1d]: adsorb from which side
        """
        files = sorted(os.listdir(anode_strs_path))
        for file in files:
            for side in sides:
                #generate adsorb poscars
                adsorbs = self.generate_poscar(file, side, atom, repeat)
                system_echo(f'adsorb poscar generate --- {file}')
                #optimize adsorb poscars
                monitor_path = f'{adsorb_strs_path}/{file}-{side}-Fill'
                local_strs_path = f'{self.local_adsorb_strs_path}/{file}-{side}-Fill' 
                local_energy_path = f'{self.local_adsorb_energy_path}/{file}-{side}-Fill'
                os.mkdir(f'{adsorb_energy_path}/{file}-{side}-Fill')
                self.run_optimization(adsorbs, 2, monitor_path, 
                                      local_strs_path, local_energy_path)
            system_echo(f'adsorb structure relaxed --- {file}')
    
    def analysis(self, single, repeat, sides):
        """
        position of adsorb sites and formation energy
        
        Parameters
        ----------
        single [float, 0d]: energy of simple substance 
        repeat [int, tuple]: size of supercell
        sides [str, 1d]: adsorb from which side
        """
        ratio = reduce(lambda x, y: x*y, repeat)
        poscars = sorted(os.listdir(adsorb_energy_path))
        pattern = 'POSCAR-[\w]*-[\w]*-[\w]*'
        poscars = [i for i in poscars if not re.match(pattern, i)]
        for side in sides:
            for i in poscars:
                compound = ratio*self.get_energy(f'{anode_energy_path}/out-{i}')
                result = self.get_result(side, i, single, compound)
                self.write_list2d(f'{adsorb_analysis_path}/{i}-{side}-Coverage.dat', result)
                capacity, ocv = np.transpose(result[:,-2:])
                self.plot_ocv(f'{adsorb_analysis_path}/{i}-{side}-OCV.png', capacity, ocv)
                system_echo(f'adsorb sites analysis finished: {i}')
    
    def get_result(self, side, poscar, single, compound):
        """
        
        
        Parameters
        ----------
        side [str, 0d]: adsorb side
        poscar [str, 0d]: name of poscar
        energy_single [float, 0d]: energy of simple substance
        
        Returns
        ----------
        formations [float, 2d, np]: formation energy
        """
        poscars = sorted(os.listdir(f'{adsorb_energy_path}/{poscar}-{side}-Fill'))
        atom_nums, energys = [], []
        for i in poscars:
            #
            atom_num = i.split('-')[-2]
            atom_num = float(atom_num)
            atom_nums.append(atom_num)
            #
            energy = self.get_energy(f'{adsorb_energy_path}/{poscar}-{side}-Fill/{i}')
            energys.append(energy)
        #
        order = np.argsort(atom_nums)
        atom_nums = np.array(atom_nums)[order]
        energys = np.array(energys)[order]
        poscars = np.array(poscars)[order]
        energy_buffer, index_buffer, cluster = [], [], []
        label = atom_nums[0]
        for i, atom_num in enumerate(atom_nums):
            if atom_num == label:
                index_buffer.append(i)
                energy_buffer.append(energys[i])
            else:
                label = atom_num
                min_index = np.argsort(energy_buffer)[0]
                cluster.append(index_buffer[min_index])
                index_buffer = []
                energy_buffer = []
                index_buffer.append(i)
                energy_buffer.append(energys[i])
        min_index = np.argsort(energy_buffer)[0]
        cluster.append(index_buffer[min_index])
        atom_nums = atom_nums[cluster]
        energys = energys[cluster]
        poscars = poscars[cluster]
        #
        ocvs = []
        for i in range(len(energys)):
            if i == 0:
                num = atom_nums[i]
                ocv = (-(energys[i] - compound) + (num*single))/num
            else:
                num = atom_nums[i] - atom_nums[i-1]
                ocv = (-(energys[i] - energys[i-1]) + (num*single))/num
            ocvs.append(ocv)
        #
        capacity = (atom_nums*26.81*1e3)/(46.8*8)
        result = np.concatenate(([poscars], [atom_nums], [energys], [capacity], [ocvs]), axis=0)
        result = np.transpose(result)
        return result
    
    def plot_ocv(self, file, capacity, ocv):
        capacity = np.array(capacity, dtype=float)
        ocv = np.array(ocv, dtype=float)
        plt.step(capacity, ocv, marker='o', where='post')
        plt.xlabel('Capacity (mAh/g)')
        plt.ylabel('Open Circuit Voltage (V)')
        plt.savefig(file, dpi=500)
        plt.close()
        

class NEBSolver(MultiAdsorbSites):
    #diffusion path between lowest energy sites
    def __init__(self):
        MultiAdsorbSites.__init__(self)
        if not os.path.exists(neb_path):
            os.mkdir(neb_path) 
            os.mkdir(neb_analysis_path)
    
    def calculate(self, atom, repeat):
        """
        NEB calculation of structure in anode_strs_path
        
        Parameters
        ----------
        atom [int, 0d]: atomic number
        repeat [int, tuple]: size of supercell
        """
        hosts = os.listdir(anode_strs_path)
        for host in hosts:
            path = f'{neb_path}/{host}-path-barrier'
            #generate diffusion path
            self.generate_path(atom, path, host, repeat)
            #calculate points in path
            local_neb_path = f'/local/ccop/{path}'
            poscars = os.listdir(path)
            pattern = 'POSCAR-NEB-[\w]*-[\w]*'
            poscars = [i for i in poscars if re.match(pattern, i)]
            self.run_SinglePointEnergy(poscars, path,
                                       local_neb_path, 
                                       local_neb_path,
                                       out=False, OUTCAR=True)
            #plot diffusion barrier
            self.plot_barrier(host, path)
            
    def generate_path(self, atom, path, host, repeat):
        """
        generate diffusion path between lowest energy sites
        
        Parameters
        ----------
        atom [int, 0d]: atomic number
        path [str, 0d]: path of NEB poscars
        host [str, 0d]: name of host structure
        repeat [int, tuple]: size of supercell
        """
        if not os.path.exists(path):
            os.mkdir(path)
        #get lowest adsorb site
        dat = f'{adsorb_analysis_path}/{host}-One-Cluster.dat'
        cluster_sites = self.import_list2d(dat, str, numpy=True)
        poscar = cluster_sites[0, 0]
        sites = self.filling_sites(atom, host, poscar, 'One', repeat)
        #get endpoints of diffusion path
        endpoints = self.find_endpoint(atom, path, host, sites, repeat)
        #get CHGCAR
        self.get_CHGCAR(path, endpoints)
        #interpolate points between sites
        self.interpolate(0, atom, path, endpoints)
        
    def find_endpoint(self, atom, path, host, sites, repeat, self_define=False):
        """
        find all periodic coordinates of lowest adsorb sites
        
        Parameters
        ----------
        atom [int, 0d]: atomic number of adsorb atom
        path [str, 0d]: path of NEB poscars
        host [str, 0d]: name of host structure
        sites [float, 3d]: equivalent lowest adsorb sites
        repeat [int, tuple]: size of supercell
        self_define [bool, 0d]: self define the near sites
        
        Returns
        ----------
        near_sites [str, tuple]: name of relaxed start and end
        """
        stru = Structure.from_file(f'{anode_strs_path}/{host}')
        stru.make_supercell(repeat)
        stru.to(filename=f'{path}/POSCAR-supercell', fmt='poscar')
        #get nearest sites
        if self_define:
            near_sites = sites
        else:
            latt = stru.lattice
            near_sites = self.nearest_sites(atom, sites, latt)
        #export site poscars
        sites_num = len(near_sites)
        node_assign = self.assign_node(sites_num)
        for i, point in enumerate(near_sites):
            stru = Structure.from_file(f'{path}/POSCAR-supercell')
            stru.append(atom, point)
            file_name = f'{path}/POSCAR-Site-{i:02.0f}-{node_assign[i]}'
            stru.to(filename=file_name, fmt='poscar')
            if self_define:
                if i == 1:
                    self.fix_atom(file_name)
            else:
                self.fix_atom(file_name)
        #optimize sites
        local_sites_path = f'/local/ccop/{path}'
        files = os.listdir(path)
        pattern = 'POSCAR-Site-[\w]*-[\w]*'
        near_sites = [i for i in files if re.match(pattern, i)]
        self.run_optimization(near_sites, 3, path, 
                              local_sites_path, local_sites_path,
                              out=False)
        files = os.listdir(path)
        pattern = f'POSCAR-Site-[\w]*-[\w]*-Relax'
        near_sites = sorted([i for i in files if re.match(pattern, i)])
        return near_sites
    
    def get_CHGCAR(self, path, endpoints):
        """
        get CHGCAR of host structure

        Parameters
        ----------
        path [str, 0d]: path of NEB poscars
        endpoints [str, 0d]: name of host structure
        """
        #generate host structures
        file_name = []
        if len(endpoints) == 2:
            shutil.copy(f'{path}/POSCAR-supercell',
                        f'{path}/POSCAR-supercell-0')
            file_name.append('POSCAR-supercell-0')
        else:
            poscars = np.delete(endpoints, 1)
            for i, host in enumerate(poscars):
                stru = Structure.from_file(f'{path}/{host}')
                sites = stru.sites[:-1]
                stru._sites = sites
                name = f'POSCAR-supercell-{i}'
                stru.to(filename=f'{path}/{name}', fmt='poscar')
                file_name.append(name)
        #calculate charge density of host structure
        file_num = len(file_name)
        assign = self.assign_node(file_num)
        poscar_name = [i+f'-{j}' for i, j in zip(file_name, assign)]
        for i, j in zip(file_name, poscar_name):
            os.rename(f'{path}/{i}', f'{path}/{j}')
        local_neb_potential = f'/local/ccop/{path}'
        self.run_SinglePointEnergy(poscar_name, path,
                                   local_neb_potential,
                                   local_neb_potential,
                                   out=False, CHGCAR=True)
        #rename POSCAR and CHGCAR
        for i, j in zip(poscar_name, file_name):
            os.rename(f'{path}/{i}', f'{path}/{j}')
            os.rename(f'{path}/CHGCAR-{i}', f'{path}/CHGCAR-{j}')
        
    def fix_atom(self, poscar):
        """
        fix atoms of host structure
        
        Parameters
        ----------
        poscar [str, 0d]: name of poscar
        """
        with open(poscar, 'r') as obj:
            ct = obj.readlines()
        head = ct[:7] + ['Selective\n', 'direct\n']
        #fix atom
        coors = ct[8:]
        coors_num = len(ct[8:])
        new_coor = []
        for i, row in enumerate(coors):
            coor = row.split()[:3]
            if i == coors_num-1:
                new_coor.append(' '.join(coor) + ' T T T\n')
            else:
                new_coor.append(' '.join(coor) + ' F F F\n')
        #export poscar
        with open(poscar, 'w') as obj:
            obj.write(''.join(head) + ''.join(new_coor))
    
    def nearest_sites(self, atom, sites, latt):
        """
        find nearest sites of adsorb site
        
        Parameters
        ----------
        atom [int, 0d]: atomic number of adsorb atom
        sites [float, 3d]: equivalent lowest adsorb sites
        latt [obj]: lattice object in pymatgen

        Returns
        ----------
        near_sites [float, 2d]: fraction coordinates of start and end
        """
        coors = [PeriodicSite(atom, i[0], latt) for i in sites]
        start = coors[0]
        period_dis = []
        for i in coors:
            period_dis.append(start.distance(i))
        index = np.argsort(period_dis)[1]
        end = coors[index]
        near_sites = [start.frac_coords, end.frac_coords]
        return near_sites
    
    def interpolate(self, index, atom, path, endpoints, n_images=7):
        """
        interpolate betwen two endpoints of diffusion path
        
        Ziqin Rong, Daniil Kitchaev, Pieremanuele Canepa, Wenxuan Huang, Gerbrand
        Ceder, The Journal of Chemical Physics 145 (7), 074112
        
        Parameters
        ----------
        index [int, 0d]: index of NEB path
        atom [int, 0d]: atomic number of adsorb atom
        path [str, 0d]: path of NEB poscars
        endpoints [str, 1d]: endpoints of NEB path
        n_images [int, 0d]: image number of NEB path
        
        Returns
        ----------
        names [str, 1d]: full path name of images
        """
        #import potential
        host = f'POSCAR-supercell-{index}'
        poscar = Structure.from_file(f'{path}/{host}')
        chgcar_name = f'{path}/CHGCAR-{host}'
        chgcar = Chgcar(poscar, {'total':np.array([[[0]]]), 'diff':np.array([[0,0,0]])})
        potential = ChgcarPotential(chgcar.from_file(chgcar_name)).get_v()
        #interpolate between endpoints
        start, end = endpoints
        start = Structure.from_file(f'{path}/{start}')
        end = Structure.from_file(f'{path}/{end}')
        start_coor = start.frac_coords[-1]
        end_coor = end.frac_coords[-1]
        points = NEBPathfinder.string_relax(start_coor, end_coor, potential,
                                            n_images=n_images, h=0.1)
        #export poscars
        names = []
        image_num = len(points)
        assign = self.assign_node(image_num)
        for i, point in enumerate(points):
            file_name = f'{path}/POSCAR-NEB-{index}-{i:02.0f}-{assign[i]}'
            image = Structure.from_file(f'{path}/POSCAR-supercell-{index}')
            image.append(atom, point)
            image.to(filename=file_name, fmt='poscar')
            names.append(file_name)
        return names
        
    def three_points_interpolate(self, atom, path, endpoints):
        """
        interpolate betwen three endpoints of diffusion path
        
        Parameters
        ----------
        atom [int, 0d]: atomic number of adsorb atom
        path [str, 0d]: path of NEB poscars
        endpoints [str, 1d]: endpoints of NEB path
        """
        #generate NEB path
        poscar_1 = self.interpolate(0, atom, path, endpoints[0:2], n_images=6)
        poscar_2 = self.interpolate(1, atom, path, endpoints[1:3][::-1], n_images=6)
        #remove end point of each path
        os.remove(poscar_1[-1])
        os.remove(poscar_2[-1])
        poscar_1 = poscar_1[:-1]
        poscar_2 = poscar_2[-2::-1]
        #collect NEB path
        mid_point = f'{path}/POSCAR-mid'
        shutil.copy(f'{path}/{endpoints[1]}', mid_point)
        poscars = poscar_1 + [mid_point] + poscar_2
        #assign nodes
        image_num = len(poscars)
        assign = self.assign_node(image_num)
        for i, poscar in enumerate(poscars):
            file_name = f'{path}/POSCAR-NEB-{i:02.0f}-{assign[i]}'
            os.rename(poscar, file_name)
        
    def plot_barrier(self, host, path):
        """
        export figure of NEB path
        
        Parameters
        ----------
        host [str, 0d]: name of host structure
        path [str, 0d]: path of NEB poscars
        """
        file = os.listdir(path)
        pattern = 'POSCAR-NEB-[\w]*-[\w]*'
        poscars = sorted([i for i in file if re.match(pattern, i)])
        strus, outs = [], []
        for i in poscars:
            strus.append(Structure.from_file(f'{path}/{i}'))
            outs.append(Outcar(f'{path}/OUTCAR-{i}'))         
        #plot diffusion barrier
        neb_analyzer = NEBAnalysis.from_outcars(outs, strus)
        neb_plot = neb_analyzer.get_plot()
        neb_plot.savefig(f'{neb_analysis_path}/{host}.png', dpi=500)
        #export extreams dat
        min, max = neb_analyzer.get_extrema()
        extreams = min + max
        extreams = sorted(extreams, key=lambda x: -x[1])
        file_name = f'{neb_analysis_path}/{host}-extreams.dat'
        self.write_list2d(file_name, extreams, '{0:8.6f}')
    
    def self_define_calculate(self, index, atom, host, sites, repeat):
        """
        NEB calculation of structure according to self define sites
        
        Parameters
        ----------
        index [int, 0d]: index of NEB path
        atom [int, 0d]: atomic number
        sites [float, 2d]: sites on NEB path
        repeat [int, tuple]: size of supercell
        """
        path = f'{neb_path}/{host}-path-{index:02.0f}'
        if not os.path.exists(path):
            os.mkdir(path)
        #get endpoints of NEB path
        endpoints = self.find_endpoint(atom, path, host, sites, repeat, self_define=True)
        #get CHGCAR
        self.get_CHGCAR(path, endpoints)
        #interpolate points between sites
        self.three_points_interpolate(atom, path, endpoints)
        #calculate points in path
        local_neb_path = f'/local/ccop/{path}'
        poscars = os.listdir(path)
        pattern = 'POSCAR-NEB-[\w]*-[\w]*'
        poscars = [i for i in poscars if re.match(pattern, i)]
        self.run_SinglePointEnergy(poscars, path,
                                   local_neb_path, 
                                   local_neb_path,
                                   out=False, OUTCAR=True)
        #plot diffusion barrier
        self.plot_barrier(host+f'-path-{index:02.0f}', path)
        

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
    atom = 13
    repeat = (2, 2, 1)
    sides = ['One']
    adsorb = AdsorbSites()
    #adsorb.get_slab()
    if len(os.listdir(anode_strs_path)) > 0:
        #adsorb.relax(atom, repeat)
        #adsorb.sites_analysis(-1.35, repeat)
        #adsorb.cluster_sites()
        #adsorb.sites_plot(repeat)
        #adsorb.sites_plot(repeat, cluster=False)
        
        #multi_adsorb = MultiAdsorbSites()
        #multi_adsorb.relax(atom, repeat, sides)
        #multi_adsorb.analysis(-1.31116, repeat, sides)
        
        neb = NEBSolver()
        neb.calculate(atom, repeat)
        sites = [[0.25, 0., 0.597], [0.5, 0., 0.597], [.75, 0., 0.597]]
        neb.self_define_calculate(0, atom, 'POSCAR-04-131', sites, repeat)
    else:
        system_echo('No suitable adsorbates are found')
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    slab = Structure.from_file(f'{anode_strs_path}/POSCAR-04-131')
    slab.make_supercell((2,2,1))
    plot_slab(slab, ax, adsorption_sites=True, window=0.9, repeat=1)
    plt.xlabel('x direction')
    plt.ylabel('y direction')
    plt.title('POSCAR-04-131', fontsize=16)
    plt.savefig('adsorb_sites.png', dpi=500)
    '''