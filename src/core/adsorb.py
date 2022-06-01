import os, sys
import time
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from collections import Counter
from sklearn.cluster import KMeans
from scipy.optimize import fsolve

from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.adsorption import AdsorbateSiteFinder, plot_slab
from pymatgen.io.vasp.outputs import Outcar
from pymatgen.analysis.diffusion.neb.pathfinder import IDPPSolver
from pymatgen.analysis.transition_state import NEBAnalysis

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.utils import *
from core.post_process import PostProcess
from core.search import GeoCheck
from core.sample_select import Select


class AdsorbSites(Select):
    #Find unequal adsorbtion sites and calculate formation energy
    def __init__(self, wait_time=1):
        self.wait_time = wait_time
        self.calculation_path = '/local/ccop/vasp'
        self.local_optim_strus_path = f'/local/ccop/{optim_strus_path}'
        self.local_anode_strus_path = f'/local/ccop/{anode_strus_path}'
        self.local_anode_energy_path = f'/local/ccop/{anode_energy_path}'
        self.local_adsorb_strus_path = f'/local/ccop/{adsorb_strus_path}'
        self.local_adsorb_energy_path = f'/local/ccop/{adsorb_energy_path}'
        self.post = PostProcess()
        if not os.path.exists(adsorb_path):
            os.mkdir(adsorb_path)
            os.mkdir(anode_strus_path)
            os.mkdir(anode_energy_path)
            os.mkdir(adsorb_strus_path)
            os.mkdir(adsorb_energy_path)
            os.mkdir(adsorb_analysis_path)
    
    def get_slab(self):
        """
        generate slab structure
        """
        #rotate monolayer to xy plane
        self.rotate_axis(optim_strus_path, 0, repeat=2)
        self.add_vacuum(optim_strus_path, 15)
        #assign node
        poscars = os.listdir(optim_strus_path)
        poscar_num = len(poscars)
        assign = self.assign_node(poscar_num)
        for i, poscar in enumerate(poscars):
            file_name = f'{optim_strus_path}/{poscar}-{assign[i]}'
            os.rename(f'{optim_strus_path}/{poscar}', file_name)
        #optimize slab
        poscars = os.listdir(optim_strus_path)
        self.run_optimization(poscars, 5, optim_strus_path, 
                              self.local_optim_strus_path, 
                              self.local_optim_strus_path,
                              out=False, cover=True)
        
    def rotate_axis(self, path, vacuum_size, repeat=1):
        """
        rotate axis to make atoms in xy plane and add vaccum layer
        
        Parameters
        ----------
        path [str, 0d]: optimized structure save path
        vaccum_size [float, 0d]: size of vaccum layer
        repeat [int, 0d]: repeat times of rotate axis
        """
        poscars = sorted(os.listdir(path))
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
                surface = Structure(latt, surface.species, surface.frac_coords)
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
        volume = self.get_volume(cart_coor)
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
            std_min = np.argmin(stds)
            miller_index = np.identity(3, dtype=int)[std_min]
        return miller_index
    
    def get_volume(self, coors):
        """
        volume of points that are coplane
        
        Parameters
        ----------
        cart_coor [float, 2d, np]: cartesian coordinate
        
        Returns
        ----------
        volume [float, 0d]: volume of points 
        """
        volume = 0
        if len(coors) > 3:
            vectors = coors - coors[0]
            vec1, vec2 = vectors[1:3]
            normal = np.cross(vec1, vec2)
            for i in vectors:
                volume += np.abs(np.dot(normal, i))
        return volume
    
    def add_vacuum(self, path, vacuum_size):
        """
        add vacuum layer
        
        Parameters
        ----------
        path [str, 0d]: optimized structure save path
        vaccum_size [float, 0d]: size of vaccum layer
        """
        poscars = sorted(os.listdir(path))
        for poscar in poscars:
            #import structure
            stru = Structure.from_file(f'{path}/{poscar}')
            latt = stru.lattice
            a, b, c, alpha, beta, gamma = latt.parameters
            c = vacuum_size
            alpha, beta = 90, 90
            latt = Lattice.from_parameters(a=a, b=b, c=c, 
                                           alpha=alpha,
                                           beta=beta,
                                           gamma=gamma)
            stru = Structure(latt, stru.species, stru.frac_coords)
            stru.to(filename=f'{path}/{poscar}', fmt='poscar')
    
    def relax(self, atom, repeat):
        """
        optimize adsorbate structures
        
        Parameters
        ----------
        atom [int, 1d]: adsorbate atom
        repeat [int, tuple]: size of supercell
        """
        #energy of host structure
        hosts = sorted(os.listdir(anode_strus_path))
        self.run_SinglePointEnergy(hosts, anode_strus_path,
                                   self.local_anode_strus_path, 
                                   self.local_anode_energy_path)
        #find stable adsorption sites
        for host in hosts:
            #generate adsorb poscars
            adsorbs = self.generate_poscar(host, atom, repeat)
            system_echo(f'adsorb poscar generate --- {host}')
            #optimize adsorb poscars
            monitor_path = f'{adsorb_strus_path}/{host}'
            local_strs_path = f'{self.local_adsorb_strus_path}/{host}' 
            local_energy_path = f'{self.local_adsorb_energy_path}/{host}'
            os.mkdir(f'{adsorb_energy_path}/{host}')
            self.run_optimization(adsorbs, 3, monitor_path, 
                                  local_strs_path, local_energy_path)
            system_echo(f'adsorb structure relaxed --- {host}')
    
    def generate_poscar(self, host, atom, repeat):
        """
        generate adsorb POSCARs according to 
        Montoya, J.H., Persson, K.A. A npj Comput Mater 3, 14 (2017). 
        
        Parameters
        ----------
        host [str, 0d]: name of host structure
        atom [int, 1d]: atom number of adsorbate
        repeat [int, tuple]: size of supercell
        
        Returns
        ---------
        adsorb_names [str, 1d]: name of adsorb poscars
        """
        #get adsorb structure
        surface = Structure.from_file(f'{anode_strus_path}/{host}')
        slab = SlabGenerator(surface, miller_index=(0,0,1),
                             min_slab_size=.1, min_vacuum_size=0, center_slab=True)
        slab = slab.get_slab()
        asf = AdsorbateSiteFinder(slab)
        adsorbate = Molecule([atom], [[0, 0, 0]])
        ads_strus_site = asf.generate_adsorption_structures(adsorbate, repeat=repeat)
        #get name of adsorb job
        site_num = len(ads_strus_site)
        node_assign = self.assign_node(site_num)
        adsorb_names = []
        #adsorb from one side
        os.mkdir(f'{adsorb_strus_path}/{host}')
        for i, poscar in enumerate(ads_strus_site):
            node = node_assign[i]
            name = f'{host}-adsorb-{i:03.0f}-{node}'
            poscar.to(filename=f'{adsorb_strus_path}/{host}/{name}', fmt='poscar')
            adsorb_names.append(name)
        return adsorb_names
    
    def run_optimization(self, poscars, times, monitor_path, 
                         local_strs_path, local_energy_path,
                         out=True, cover=False):
        """
        optimize configurations
        
        Parameters
        ----------
        poscars [str, 1d]: name of poscars
        times [int, 0d]: number of optimize times
        monitor_path [str, 0d]: path of FINISH flags
        local_strs_path [str, 0d]: structure path in GPU node
        local_energy_path [str, 0d]: energy path in GPU node
        out [bool, 0d]: whether copy vasp.out
        cover [bool, 0d]: whether cover POSCAR
        """
        flag_out, flag_cover = 0, 0
        if out:
            flag_out = 1
        if cover:
            flag_cover = 1
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
                                contcar=$p-Relax
                                if [ {flag_cover} -eq 1 ]; then
                                    contcar=$p
                                fi
                                scp CONTCAR {gpu_node}:{local_strs_path}/$contcar
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
                              out=True, OUTCAR=False):
        """
        calculate single point energy
        
        Parameters
        ----------
        poscars [str, 1d]: name of poscars
        monitor_path [str, 0d]: path of FINISH flags
        local_strs_path [str, 0d]: structure path in GPU node
        local_energy_path [str, 0d]: energy path in GPU node
        out [bool, 0d]: cp out file to local directory
        OUTCAR [bool, 0d]: cp OUTCAR to local directory
        """
        flag_out = 0
        flag_OUTCAR = 0
        if out:
            flag_out = 1
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
        for i in poscars:
            compound = ratio*self.get_energy(f'{anode_energy_path}/out-{i}')
            sites = self.sites_coor(i)
            form = self.formation_energy(i, single, compound)
            result = np.concatenate((sites, form), axis=1)
            self.write_list2d(f'{adsorb_analysis_path}/{i}.dat', result)
            system_echo(f'adsorb sites analysis finished: {i}')
    
    def sites_coor(self, poscar):
        """
        get cartesian coordinate of adsorb
        
        Parameters
        ----------
        poscar [str, 0d]: name of relax structure 

        Returns
        ----------
        sites [float, 2d, np]: position of adsorbates
        """
        files = os.listdir(f'{adsorb_strus_path}/{poscar}')
        poscars = sorted([i for i in files if i.endswith('Relax')])
        #get stable adsorption sites
        names, sites = [], []
        for i in poscars:
            stru = Structure.from_file(f'{adsorb_strus_path}/{poscar}/{i}')
            sites.append(stru.cart_coords[-1])
            names.append([i])
        return np.concatenate((names, sites), axis=1)

    def formation_energy(self, poscar, single, compound):
        """
        calculate formation energy
        
        Parameters
        ----------
        poscar [str, 0d]: name of poscar
        single [float, 0d]: energy of simple substance
        compound [float, 0d]: energy of compound
        
        Returns
        ----------
        formations [float, 2d, np]: formation energy
        """
        poscars = sorted(os.listdir(f'{adsorb_energy_path}/{poscar}'))
        formations = []
        for i in poscars:
            energy = self.get_energy(f'{adsorb_energy_path}/{poscar}/{i}')
            delta_E = energy - (single + compound)
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

    def cluster_sites(self, max_sites=8):
        """
        cluster adsorb sites by cartesian coordinates 
        and choose the lowest energy sites as adsorb site
        
        Parameters
        ----------
        max_sites [int, 0d]: max number of clusters
        """
        poscars = sorted(os.listdir(adsorb_energy_path))
        for poscar in poscars:
            #cluster adsorb sites
            file = f'{adsorb_analysis_path}/{poscar}.dat'
            sites = self.import_list2d(file, str, numpy=True)
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
            file = f'{adsorb_analysis_path}/{poscar}-Cluster.dat'
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
        for poscar in poscars:
            #import data
            file = f'{adsorb_analysis_path}/{poscar}{notation}.dat'
            sites = self.import_list2d(file, str, numpy=True)[:,1:]
            x, y, _, v = np.transpose(np.array(sites, dtype=float))
            #plot adsorb sites
            figure = plt.figure()
            ax = figure.add_subplot(1, 1, 1)
            cm = plt.cm.get_cmap('jet')
            plt.scatter(x, y, c=v, cmap=cm, s=160, marker='x', zorder=1000)
            #plot adsorbate
            slab = Structure.from_file(f'{anode_strus_path}/{poscar}')
            slab.make_supercell(repeat)
            plot_slab(slab, ax, adsorption_sites=False, window=0.9, repeat=1)
            #appearance of figure
            plt.title(f'{poscar}', fontsize=16)
            ax.set_xlabel('x direction')
            ax.set_ylabel('y direction')
            clb = plt.colorbar()
            clb.ax.set_title('$\Delta$E/eV')
            #export figure
            file = f'{adsorb_analysis_path}/{poscar}{notation}.png'
            plt.savefig(file, dpi=600)
            plt.close('all')


class Arrangement(ListRWTools):
    #disperse atoms on surface as much as possible
    def search(self, atom, host, repeat, max_sites=8, path_num=100):
        """
        optimize arrangement of adsorbates by SA
        
        Parameters
        ----------
        atom [int, 0d]: atomic number
        host [str, 0d]: name of host structure
        repeat [int, tuple]: size of supercell
        max_sites [int, 0d]: max equivalent adsorption sites
        path_num [int, 0d]: number of SA path
        
        Returns
        ----------
        sites_buffer [obj, 3d]: optimal sites of different adsorbates
        """
        #cluster adsorb sites
        file = f'{adsorb_analysis_path}/{host}-Cluster.dat'
        sites_num = len(self.import_list2d(file, str, numpy=True))
        if sites_num > max_sites:
            sites_num = max_sites
        #get PeriodicSites and distance matrix
        sites, site_energys = [], []
        for i in range(sites_num):
            equal_sites = self.find_equal_sites(atom, host, repeat, order=i)
            sites += equal_sites[0]
            site_energys += equal_sites[1]
        pos_num = len(sites)
        pos_seq = [i for i in range(pos_num)]
        dis_mat = self.get_distance_matrix(sites)
        #get radiu of adsorbate atom
        self.radiu = Element(sites[0].specie).atomic_radius.real
        #perform n SA paths
        sites_buffer = []
        for i in range(1, pos_num):
            pos_buffer, value_buffer, flag_buffer = [], [], []
            for _ in range(path_num):
                pos, value, flag = self.explore(i, dis_mat, site_energys, pos_seq)
                pos_buffer.append(pos)
                value_buffer.append(value)
                flag_buffer.append(flag)
            #delete duplicates
            pos_buffer, idx = np.unique(pos_buffer, axis=0, return_index=True)
            value_buffer = np.array(value_buffer)[idx]
            flag_buffer = np.array(flag_buffer)[idx]
            #get optimal position
            flags, store = [], []
            index = np.argsort(value_buffer)[:10]
            for idx in index:
                flag = flag_buffer[idx]
                if flag:
                    opt_pos = pos_buffer[idx]
                    opt_sites = np.array(sites)[opt_pos]
                    store.append(opt_sites)
                    flags.append(flag)
            if len(flags) == 0:
                break
            else:
                sites_buffer.append(store)
        return sites_buffer
    
    def explore(self, number, dis_mat, site_energys, pos_seq):
        """
        simulated annealing

        Parameters
        ----------
        number [int, 0d]: number of adsorbates
        dis_mat [float, 2d]: periodic distance matrix
        site_energys [float, 1d]: energy of each site
        pos_seq [int, 1d]: position of sites
        
        Returns
        ----------
        total_dis [float, 0d]: total distance between adsorbates
        opt_pos [int, 1d]: optimal positions
        flag [bool, 0d]: whether satisfy distance constrain
        """
        self.T = T
        pos_1 = random.sample(pos_seq, number)
        dis_1, flag_1 = self.total_distance(pos_1, dis_mat)
        energy_1 = self.total_energy(pos_1, site_energys)
        value_1 = -0.01*dis_1 + energy_1
        pos_buffer, value_buffer = [], []
        pos_buffer.append(pos_1)
        value_buffer.append(value_1)
        #SA optimize
        for _ in range(steps):
            pos_2 = self.step(pos_1, pos_seq)
            dis_2, flag_2 = self.total_distance(pos_2, dis_mat)
            energy_2 = self.total_energy(pos_2, site_energys)
            value_2 = -0.01*dis_2 + energy_2
            if self.metropolis(value_1, value_2, self.T):
                pos_1 = pos_2
                value_1 = value_2
                flag_1 = flag_2
                pos_buffer.append(pos_1)
                value_buffer.append(value_1)
            self.T *= decay
        #get optimal position
        index = np.argmin(value_buffer)
        value = value_buffer[index]
        opt_pos = pos_buffer[index]
        return sorted(opt_pos), value, flag_1
    
    def step(self, pos, pos_seq):
        """
        move one atom to vacancy
        
        Parameters
        ----------
        pos [int, 1d]: inital position of atom
        pos_seq [int, 1d]: sequence of positions
        
        Returns
        ----------
        new_pos [int, 1d]: position of atom after 1 SA step
        """
        vacancy = np.setdiff1d(pos_seq, pos)
        new_pos = pos.copy()
        idx_1 = np.random.randint(len(new_pos))
        idx_2 = np.random.randint(len(vacancy))
        new_pos[idx_1] = vacancy[idx_2]
        return new_pos
    
    def metropolis(self, value_1, value_2, T):
        """
        metropolis criterion
        
        Parameters
        ----------
        value_1 [float, 0d]: current value
        value_2 [float, 0d]: next value
        T [float, 0d]: annealing temperature
        
        Returns
        ----------
        flag [bool, 0d]: whether do the action
        """
        delta = value_2 - value_1
        if np.exp(-delta/T) > np.random.rand():
            return True
        else:
            return False
    
    def total_energy(self, pos, site_energys):
        """
        sum up energy of sites
        
        Parameters
        ----------
        pos [int, 1d]: occupied sites
        site_energys [float, 1d]: energy of each sites

        Returns
        ----------
        energy [float, 0d]: total energy 
        """
        energy = np.array(site_energys)[pos].sum()
        return energy
    
    def total_distance(self, pos, dis_mat):
        """
        calculate total inner distance
        
        Parameters
        ----------
        pos [int, 1d]: position of adsorbates
        dis_mat [float, 2d]: periodic distance matrix 

        Returns
        ----------
        total_dis [float, 0d]: total inner distance
        flag [bool, 0d]: inner distance should bigger than atom radiu
        """
        flag = True
        total_dis = 0.
        point_num = len(pos)
        for i in range(point_num):
            for j in range(i+1, point_num):
                dis = dis_mat[pos[i], pos[j]]
                if dis < self.radiu:
                     dis = -100
                     flag = False
                total_dis += dis
        return total_dis, flag
    
    def get_distance_matrix(self, sites):
        """
        distance matrix of PeriodicSite
        
        Parameters
        ----------
        sites [obj, 1d]: PeriodicSite object 

        Returns
        ----------
        dis_mat [float, 2d]: periodic distance matrix 
        """
        #get distance matrix
        sites_num = len(sites)
        dis_mat = np.zeros((sites_num, sites_num))
        for i in range(sites_num):
            for j in range(i+1, sites_num):
                point_1 = sites[i]
                point_2 = sites[j]
                dis = point_1.distance(point_2)
                dis_mat[i,j] = dis
        dis_mat += np.transpose(dis_mat)
        return dis_mat
    
    def find_equal_sites(self, atom, host, repeat, order=0):
        """
        get equivalent sites in PeriodicSite form

        Parameters
        ----------
        atom [int, 0d]: atomic number
        host [str, 0d]: name of host structure
        repeat [int, tuple]: size of supercell
        order [int, 0d]: energy order of adsorption sites
        
        Returns
        ----------
        sites [obj, 1d]: PeriodicSite object 
        sites_energy [float, 1d]: adsorption energy of each site
        """
        #get equivalent sites
        dat = f'{adsorb_analysis_path}/{host}-Cluster.dat'
        cluster_sites = self.import_list2d(dat, str, numpy=True)
        poscar = cluster_sites[:,0][order]
        energy = cluster_sites[:,-1][order]
        sites = self.symmetry_sites(atom, host, poscar, repeat)
        #sites energy
        sites_energy = [float(energy) for _ in range(len(sites))]
        return sites, sites_energy
    
    def symmetry_sites(self, atom, host, poscar, repeat):
        """
        get equivalent sites by symmetry operations
        
        Parameters
        ----------
        atom [int, 0d]: atomic number of adsorbate
        host [str, 0d]: name of host poscar
        poscar [str, 0d]: poscar of adsorb site
        repeat [int, tuple]: size of supercell

        Returns
        ----------
        equal_sites [obj, 1d]: equivalent adsorb sites
        """
        stru = Structure.from_file(f'{anode_strus_path}/{host}')
        stru.make_supercell(repeat)
        latt = stru.lattice
        adsorb_stru = Structure.from_file(f'{adsorb_strus_path}/{host}/{poscar}')
        adsorb_coor = adsorb_stru.frac_coords[-1]
        #get all equivalent sites
        analy = SpacegroupAnalyzer(stru)
        sym_op = analy.get_symmetry_operations()
        sym_coor = [i.operate(adsorb_coor) for i in sym_op]
        #delete same sites
        equal_sites = []
        sym_coor = [PeriodicSite(atom, i, latt, to_unit_cell=True) for i in sym_coor]
        for i in sym_coor:
            if i.frac_coords[-1] > 0.5:
                equal_sites.append(i)
        equal_sites = self.remove_same_periodic_sites(equal_sites)
        return equal_sites
    
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
        sites_num = len(sites)
        for i in range(sites_num):
            same_points = []
            point_1 = sites[i]
            for j in range(i+1, sites_num):
                point_2 = sites[j]
                dis = point_1.distance(point_2)
                if dis < .5:
                    same_points.append(j)
            sites = np.delete(sites, same_points, axis=0)
            sites_num = len(sites)
            if i+1 == sites_num:
                break
        return sites.tolist()
    

class MultiAdsorbSites(AdsorbSites, GeoCheck, Arrangement):
    #calculate coverage of host structure
    def __init__(self):
        AdsorbSites.__init__(self)
    
    def relax(self, atom, repeat, sides):
        """
        optimize adsorbate structures
        
        Parameters
        ----------
        atom [int, 1d]: atomic number of adsorbate atom
        repeat [int, tuple]: size of supercell
        sides [str, 1d]: adsorb from which side
        """
        hosts = sorted(os.listdir(anode_strus_path))
        for host in hosts:
            #optimize arrangement
            sites = self.search(atom, host, repeat)
            for side in sides:
                #generate adsorb poscars
                side_name = f'{host}-Fill-{side}'
                fill_path = f'{adsorb_strus_path}/{side_name}'
                energy_path = f'{adsorb_energy_path}/{side_name}'
                if not os.path.exists(fill_path):
                    os.mkdir(fill_path)
                if not os.path.exists(energy_path):
                    os.mkdir(energy_path)
                system_echo(f'adsorb poscar generate --- {host}')
                poscars = self.generate_poscar(fill_path, atom, sites,
                                               host, repeat, side)
                #optimize adsorb poscars
                local_strs_path = f'{self.local_adsorb_strus_path}/{side_name}' 
                local_energy_path = f'{self.local_adsorb_energy_path}/{side_name}'
                #self.run_optimization(poscars, 3, fill_path,
                #                      local_strs_path, local_energy_path)
            system_echo(f'adsorb structure relaxed --- {host}')
    
    def generate_poscar(self, path, atom, sites, host, repeat, side):
        """
        generate filling poscars
        
        Parameters
        ----------
        path [str, 0d]: path of filling adsorbates
        atom [int, 1d]: adsorbate atom
        sites [obj, 3d]: optimal sites of adsorbates
        host [str, 0d]: name of host structure
        repeat [int, tuple]: size of supercell
        side [str, 0d]: adsorb side
        
        Returns
        ----------
        poscars [str, 1d]: name of poscars
        """
        poscar_num = len(sites)
        node_assign = self.assign_node(poscar_num, order=False)
        for i, comp in enumerate(sites):
            #put adsorbates upon base structure
            strus = []
            for j, opt_sites in enumerate(comp):
                stru = Structure.from_file(f'{anode_strus_path}/{host}')
                stru.make_supercell(repeat)
                for site in opt_sites:
                    coor = site.frac_coords
                    stru.append(atom, coor)
                    if side == 'Two':
                        coor = np.abs(np.array([0,0,1])-coor)
                        stru.append(atom, coor)
                strus.append(stru)
            #export poscars
            index = self.delete_same_strus(strus)
            for j, idx in enumerate(index):
                if j < 5:
                    file_name = f'{path}/{host}-{side}-{i+1:02.0f}-{j}-{node_assign[i]}'
                    strus[idx].to(filename=file_name, fmt='poscar')
        poscars = os.listdir(path)
        return poscars
    
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
        hosts = sorted(os.listdir(anode_strus_path))
        for host in hosts:
            for side in sides:
                compound = ratio*self.get_energy(f'{anode_energy_path}/out-{host}')
                result = self.get_result(side, host, single, compound, repeat)
                self.write_list2d(f'{adsorb_analysis_path}/{host}-{side}-Coverage.dat', result)
                capacity, ocv = np.transpose(result[:,-2:])
                self.plot_ocv(f'{adsorb_analysis_path}/{host}-{side}-OCV.png', capacity, ocv)
                system_echo(f'adsorb sites analysis finished: {host}')
    
    def get_result(self, side, host, single, compound, repeat):
        """
        
        
        Parameters
        ----------
        side [str, 0d]: adsorb side
        host [str, 0d]: name of host structure
        single [float, 0d]: energy of simple substance
        compound [float, 0d]:
        repeat [int, tuple]: size of supercell
        
        Returns
        ----------
        result []:
        """
        #
        side_name = f'{host}-Fill-{side}'
        out_path = f'{adsorb_energy_path}/{side_name}'
        outs = sorted(os.listdir(out_path))
        #
        strs_path = f'{adsorb_strus_path}/{side_name}'
        poscars = os.listdir(strs_path)
        poscars = sorted([i for i in poscars if i.endswith('Relax')])
        coplane = self.check_coplane(strs_path, poscars)
        #
        atom_nums, energys = [], []
        for i, out in enumerate(outs):
            if coplane[i]:
                atom_num = out.split('-')[-2]
                atom_num = float(atom_num)
                atom_nums.append(atom_num)
                energy = self.get_energy(f'{out_path}/{out}')
                energys.append(energy)
        #
        ocvs = []
        for i in range(len(energys)):
            num = atom_nums[i]
            ocv = (-(energys[i] - compound) + (num*single))/num
            ocvs.append(ocv)
        #
        stru = Structure.from_file(f'{anode_strus_path}/{host}')
        stru.make_supercell(repeat)
        mass = stru.composition.weight.real
        #
        capacity = (np.array(atom_nums)*26.81*1e3)/mass
        result = np.concatenate(([atom_nums], [energys], [capacity], [ocvs]), axis=0)
        result = np.transpose(result)
        return result
    
    def plot_ocv(self, file, capacity, ocv):
        #
        capacity = np.array(capacity, dtype=float)
        ocv = np.array(ocv, dtype=float)
        '''
        #
        x = np.linspace(capacity[0], capacity[-1], 100)
        func = interpolate.interp1d(capacity, ocv, kind='slinear')
        v = [func(i) for i in x]
        index = np.argmin(np.abs(v))
        zero = fsolve(func, x[index], maxfev=1000)
        #
        print(zero)
        '''
        plt.plot(capacity, ocv, marker='o')
        plt.hlines(0, 0, capacity[-1], color='r')
        #plt.plot(zero, 0., marker='*', color='g', ms=15)
        plt.xlabel('Capacity (mAh/g)', fontsize=18)
        plt.ylabel('Open Circuit Voltage (V)', fontsize=18)
        plt.savefig(file, dpi=500)
        plt.close()
    
    def check_coplane(self, path, poscars):
        """
        check coplane of poscars in path
        
        Parameters
        ----------
        path [str, 0d]: path of poscars
        poscars [str, 1d]: name of poscars
        
        Returns
        ----------
        coplane [bool, 1d]: coplane of poscars
        """
        coplane = []
        for poscar in poscars:
            stru = Structure.from_file(f'{path}/{poscar}')
            sites = stru.sites
            atom = sites[-1].species_string
            coors = []
            for site in sites:
                if site.species_string == atom:
                    coors.append(site.coords)
            coors = np.array(coors)
            volume = self.get_volume(coors)
            if volume < 15:
                coplane.append(True)
            else:
                coplane.append(False)
        return coplane
    

class NEBSolver(MultiAdsorbSites):
    #diffusion path between lowest energy sites
    def __init__(self):
        MultiAdsorbSites.__init__(self)
        if not os.path.exists(neb_path):
            os.mkdir(neb_path) 
            os.mkdir(neb_analysis_path)
    
    def calculate(self, atom, repeat):
        """
        NEB calculation of structures
        
        Parameters
        ----------
        atom [int, 0d]: atomic number
        repeat [int, tuple]: size of supercell
        """
        hosts = os.listdir(anode_strus_path)
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
        #get all equivalent lowest adsorb sites
        sites, _ = self.find_equal_sites(atom, host, repeat)
        #get endpoints of diffusion path
        endpoints = self.find_endpoint(atom, path, host, sites, repeat)
        #interpolate points between sites
        self.interpolate(0, path, endpoints)
        
    def find_endpoint(self, atom, path, host, sites, repeat, self_define=False):
        """
        find all periodic coordinates of lowest adsorb sites
        
        Parameters
        ----------
        atom [int, 0d]: atomic number of adsorb atom
        path [str, 0d]: path of NEB poscars
        host [str, 0d]: name of host structure
        sites [float, 2d]: equivalent lowest adsorb sites
        repeat [int, tuple]: size of supercell
        self_define [bool, 0d]: self define the near sites
        
        Returns
        ----------
        near_sites [str, tuple]: name of relaxed start and end
        """
        #get nearest sites
        if self_define:
            near_sites = sites
        else:
            near_sites = self.nearest_sites(sites)
        #export site poscars
        sites_num = len(near_sites)
        node_assign = self.assign_node(sites_num)
        for i, point in enumerate(near_sites):
            stru = Structure.from_file(f'{anode_strus_path}/{host}')
            stru.make_supercell(repeat)
            stru.append(atom, point)
            file_name = f'{path}/POSCAR-Site-{i:02.0f}-{node_assign[i]}'
            stru.to(filename=file_name, fmt='poscar')
        #optimize sites
        local_sites_path = f'/local/ccop/{path}'
        files = os.listdir(path)
        pattern = 'POSCAR-Site-[\w]*-[\w]*'
        near_sites = [i for i in files if re.match(pattern, i)]
        self.run_optimization(near_sites, 3, path, 
                              local_sites_path, local_sites_path,
                              out=False)
        files = os.listdir(path)
        near_sites = sorted([i for i in files if i.endswith('Relax')])
        return near_sites
    
    def nearest_sites(self, sites):
        """
        find nearest sites of adsorb site
        
        Parameters
        ----------
        sites [obj, 1d]: equivalent lowest adsorb sites

        Returns
        ----------
        near_sites [float, 2d]: fraction coordinates of start and end
        """
        #get start point
        latt = sites[0].lattice
        atom = Element(sites[0].specie)
        coor = (.5, .5, sites[0].frac_coords[-1])
        center = PeriodicSite(atom, coor, latt)
        period_dis = []
        for site in sites:
            period_dis.append(center.distance(site))
        index = np.argsort(period_dis)[0]
        start = sites[index]
        #get end point
        period_dis = []
        radiu = atom.atomic_radius.real
        for site in sites:
            period_dis.append(start.distance(site))
        for i in np.argsort(period_dis):
            if period_dis[i] > radiu:
                index = i
                break
        end = sites[index]
        near_sites = [start.frac_coords, end.frac_coords]
        return near_sites
    
    def interpolate(self, index, path, endpoints, n_images=3):
        """
        interpolate betwen two endpoints of diffusion path
        
        Deng, Z.; Zhu, Z.; Chu, I.H.; Ong, S. P. Data-Driven First-Principles
        Methods for the Study and Design of Alkali Superionic Conductors,
        Chem. Mater., 2016, acs.chemmater.
        
        Parameters
        ----------
        index [int, 0d]: index of NEB path
        path [str, 0d]: path of NEB poscars
        endpoints [str, 1d]: endpoints of NEB path
        n_images [int, 0d]: image number of NEB path
        
        Returns
        ----------
        names [str, 1d]: full path name of images
        """
        #interpolate between endpoints
        start, end = endpoints
        start = Structure.from_file(f'{path}/{start}')
        end = Structure.from_file(f'{path}/{end}')
        atom = start.sites[-1].specie
        neb_string = IDPPSolver.from_endpoints(endpoints=[start, end], nimages=n_images)
        images = neb_string.run(maxiter=5000, tol=1e-5, gtol=1e-3, species=[atom])
        #export poscars
        names = []
        image_num = len(images)
        assign = self.assign_node(image_num)
        for i, image in enumerate(images):
            file_name = f'{path}/POSCAR-NEB-{index}-{i:02.0f}-{assign[i]}'
            image.to(filename=file_name, fmt='poscar')
            names.append(file_name)
        return names
        
    def multi_points_interpolate(self, path, endpoints):
        """
        interpolate betwen three endpoints of diffusion path
        
        Parameters
        ----------
        path [str, 0d]: path of NEB poscars
        endpoints [str, 1d]: endpoints of NEB path
        """
        poscars = []
        #generate NEB path
        for i in range(len(endpoints)-1):
            poscar = self.interpolate(i, path, endpoints[i:i+2])
            if i > 0:
                os.remove(poscar[0])
                poscar = poscar[1:]
            poscars += poscar
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
        host [str, 0d]: name of host structure
        sites [float, 2d]: sites on NEB path
        repeat [int, tuple]: size of supercell
        """
        path = f'{neb_path}/{host}-path-{index:02.0f}'
        if not os.path.exists(path):
            os.mkdir(path)
        #get endpoints of NEB path
        endpoints = self.find_endpoint(atom, path, host, sites, repeat, self_define=True)
        #interpolate points between sites
        self.multi_points_interpolate(path, endpoints)
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
        

class ThermalConductivity(PostProcess):
    #calculate lattice thermal conductivity
    def __init__(self):
        PostProcess.__init__(self)    
    
    def calculate(self):
        """
        calculate property of structures
        """
        #dielectric matrix
        self.run_dielectric()
        #calculate phonon spectrum 
        self.run_phonon(vdW=False, dimension=2)
        #3 order force constant
        self.run_3RD()
        #lattice thermal conductivity
        self.run_thermal_conductivity()
        
    
if __name__ == '__main__':
    atom = 11
    repeat = (2, 2, 1)
    sides = ['One']
    adsorb = AdsorbSites()
    #adsorb.get_slab()
    #adsorb.relax(atom, repeat)
    #adsorb.sites_analysis(-1.3113, repeat)
    #adsorb.cluster_sites()
    #adsorb.sites_plot(repeat)
    #adsorb.sites_plot(repeat, cluster=False)
    
    #neb = NEBSolver()
    #neb.calculate(atom, repeat)
    #sites = [[0.25, 0., 0.597], [0.5, 0., 0.597], [.75, 0., 0.597]]
    #neb.self_define_calculate(0, atom, 'POSCAR-04-131', sites, repeat)
    
    multi_adsorb = MultiAdsorbSites()
    multi_adsorb.relax(atom, repeat, sides)
    #multi_adsorb.analysis(-1.3113, repeat, sides)
    
    
    #thermal = ThermalConductivity()
    #thermal.calculate()