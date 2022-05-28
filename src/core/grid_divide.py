import os, sys
import time
import random
import argparse
import numpy as np

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.space_group import PlanarSpaceGroup
from core.utils import ListRWTools, SSHTools, system_echo


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=int)
args = parser.parse_args()


class ParallelDivide(ListRWTools, SSHTools):
    #divide grid by each node
    def __init__(self, wait_time=0.1):
        self.wait_time = wait_time
        self.local_grid_path = f'/local/ccop/{grid_path}'
    
    def assign_to_cpu(self, grid_name):
        """
        send divide jobs to nodes and update nodes
        
        Parameters
        ----------
        grid_name [int, 1d]: name of grid
        """
        grid_num = len(grid_name)
        node_assign = self.assign_node(grid_num)
        #generate grid and send back to gpu node
        work_node_num = self.sub_jobs(grid_name, node_assign)
        while not self.is_done(grid_path, work_node_num):
            time.sleep(self.wait_time)
        self.remove_flag_on_gpu()
        system_echo(f'Lattice generate number: {grid_num}')
        #update cpus
        for node in nodes[:work_node_num]:
            self.send_grid_to_cpu(node)
        while not self.is_done(grid_path, work_node_num):
            time.sleep(self.wait_time)
        self.remove_flag_on_gpu()
        self.remove_flag_on_cpu()
        system_echo(f'Unzip grid file on CPUs')
        #unzip on gpu
        self.unzip_grid_on_gpu()
        while not self.is_done(grid_path, work_node_num):
            time.sleep(self.wait_time)
        self.remove_zip_on_gpu()
        self.remove_flag_on_gpu()
        system_echo(f'Unzip grid file on GPU')
    
    def sub_jobs(self, grid_name, node_assign):
        """
        sub grid divide to cpu nodes
        
        Parameters
        ----------
        grid_name [int, 1d]: name of grid
        node_assign [int, 1d]: node assign

        Returns
        ----------
        work_node_num [int, 0d]: number of work node
        """
        #grid grouped by nodes
        jobs, store = [], []
        for node in nodes:
            for i, assign in enumerate(node_assign):
                if node == assign:
                    store.append(grid_name[i])
            jobs.append(store)
            store = []
        #submit grid divide jobs to each node
        for i, grids in enumerate(jobs):
            self.sub_divide(grids, nodes[i])
        work_node_num = len(jobs)
        return work_node_num
    
    def sub_divide(self, grids, node):
        """
        SSH to target node and divide grid

        Parameters
        ----------
        grids [int, 1d]: name of grid
        node [int, 0d]: name of node
        """
        ip = f'node{node}'
        file = [] 
        for grid in grids:
            file.append(f'{grid:03.0f}_nbr_dis_*.bin')
            file.append(f'{grid:03.0f}_nbr_idx_*.bin')
        file_str = ' '.join(file)
        grid_str = ' '.join([str(i) for i in grids])
        #condition of job finish
        frac, nbr = [], []
        for grid in grids:
            frac.append(f'{grid:03.0f}_frac')
            nbr.append(f'{grid:03.0f}_nbr_dis')
        frac_str = '|'.join(frac)
        nbr_str = '|'.join(nbr)
        frac_num = f'`ls -l | grep -E \"{frac_str}\" | wc -l`'
        nbr_num = f'`ls -l | grep -E \"{nbr_str}\" | wc -l`'
        #shell script of grid divide
        shell_script = f'''
                        #!/bin/bash
                        cd /local/ccop/
                        for i in {grid_str}
                        do
                            nohup python src/core/grid_divide.py --name $i >& log&
                        done
                        rm log
                        
                        cd {grid_path}
                        while true;
                        do
                            num1={frac_num}
                            num2={nbr_num}
                            if [ $num1 -eq $num2 ]; then
                                sleep 1s
                                break
                            fi
                            sleep 1s
                        done
                        
                        touch FINISH-{node}            
                        tar -zcf {ip}.tar.gz {file_str} FINISH-{node}
                        scp {ip}.tar.gz FINISH-{node} {gpu_node}:{self.local_grid_path}/
                        rm {ip}.tar.gz FINISH-{node}
                        '''
        self.ssh_node(shell_script, ip)
    
    def send_grid_to_cpu(self, node):
        """
        update grid of each node
        """
        ip = f'node{node}'
        shell_script = f'''
                        #!/bin/bash
                        cd {self.local_grid_path}
                        scp {gpu_node}:{self.local_grid_path}/*.tar.gz .
                        
                        for i in *tar.gz
                        do
                            nohup tar -zxf $i >& log-$i &
                            rm log-$i
                        done
                        
                        touch FINISH-{node}
                        scp FINISH-{node} {gpu_node}:{self.local_grid_path}/
                        rm FINISH-{node} *.tar.gz
                        '''
        self.ssh_node(shell_script, ip)
    
    def unzip_grid_on_gpu(self):
        """
        unzip grid property on gpu node
        """
        shell_script = f'''
                        #!/bin/bash
                        cd {self.local_grid_path}
                        for i in *.tar.gz
                        do
                            nohup tar -zxf $i >& log-$i &
                            rm log-$i
                        done
                        '''
        os.system(shell_script)
    
    def remove_flag_on_gpu(self): 
        """
        remove flag file on gpu
        """
        os.system(f'rm {grid_path}/FINISH*')
    
    def remove_flag_on_cpu(self):
        """
        remove FINISH flags on cpu
        """
        for node in nodes:
            ip = f'node{node}'
            shell_script = f'''
                            #!/bin/bash
                            cd {self.local_grid_path}
                            rm FINISH*
                            '''
            self.ssh_node(shell_script, ip)

    def remove_zip_on_gpu(self):
        """
        remove zip file
        """
        os.system(f'rm {grid_path}/*.tar.gz')
    

class GridDivide(ListRWTools, PlanarSpaceGroup):
    #Build the grid
    def __init__(self):
        pass
    
    def build(self, grid, cutoff):
        """
        save nbr_idx, nbr_dis of grid
        
        Parameters
        ----------
        grid [int, 0d]: name of grid
        cutoff [float, 0d]: cutoff distance
        """
        head = f'{grid_path}/{grid:03.0f}'
        latt_vec = self.import_list2d(f'{head}_latt_vec.bin',
                                      float, binary=True)
        #add vacuum layer to structure
        if add_vacuum:
            latt = self.add_vacuum(latt_vec)
        #get near neighbors and distance
        space_group = self.grid_space_group(grid)
        for sg in space_group:
            coords = self.import_list2d(f'{head}_frac_coords_{sg}.bin',
                                        float, binary=True)
            mapping = self.import_list2d(f'{head}_mapping_{sg}.bin',
                                        int, binary=True)
            atoms = [1 for _ in range(len(coords))]
            stru = Structure.from_spacegroup(1, latt, atoms, coords)
            #export neighbor index and distance in min area
            nbr_idx, nbr_dis = self.near_property(stru, cutoff, mapping)
            self.write_list2d(f'{head}_nbr_idx_{sg}.bin', 
                              nbr_idx, binary=True)
            self.write_list2d(f'{head}_nbr_dis_{sg}.bin', 
                              nbr_dis, binary=True)
    
    def grid_space_group(self, grid):
        """
        get space groups of grid

        Parameters
        ----------
        grid [int, 0d]: name of grid
        
        Returns
        ---------
        space_group [str, 1d]: space group number
        """
        grid_file = os.listdir(grid_path)
        coords_file = [i for i in grid_file 
                       if i.startswith(f'{grid:03.0f}_frac')]
        name = [i.split('.')[0] for i in coords_file]
        space_group = [i.split('_')[-1] for i in name]
        return space_group
        
    def add_vacuum(self, latt_vec):
        """
        add vacuum layer
        
        Parameters
        ----------
        latt_vec [float, 2d, np]: lattice vector

        Returns
        ----------
        latt [obj]: lattice object in pymatgen 
        """
        latt = Lattice(latt_vec)
        a, b, c, alpha, beta, gamma = latt.parameters
        if c < vacuum_space:
            c = vacuum_space
        latt = Lattice.from_parameters(a=a, b=b, c=c, 
                                       alpha=alpha,
                                       beta=beta,
                                       gamma=gamma)
        return latt
    
    def near_property(self, stru, cutoff, mapping, near=0):
        """
        index and distance of near grid points
        
        Parameters
        ----------
        stru [obj]: structure object in pymatgen
        cutoff [float, 0d]: cutoff distance
        
        Returns
        ----------
        nbr_idx [int, 2d, np]: index of near neighbor in min
        nbr_dis [float, 2d, np]: distance of near neighbor in min
        """
        all_nbrs = stru.get_all_neighbors(cutoff)
        all_nbrs = [sorted(nbrs, key = lambda x: x[1]) for nbrs in all_nbrs]
        if near == 0:
            num_near = min(map(lambda x: len(x), all_nbrs))
        else:
            num_near = near
        nbr_idx, nbr_dis = [], []
        for nbr in all_nbrs:
            nbr_idx.append(list(map(lambda x: x[2], nbr[:num_near])))
            nbr_dis.append(list(map(lambda x: x[1], nbr[:num_near])))
        nbr_idx, nbr_dis = self.reduce_to_min(nbr_idx, nbr_dis, mapping)
        return nbr_idx, nbr_dis
    
    def reduce_to_min(self, nbr_idx, nbr_dis, mapping):
        """
        reduce neighbors to min unequal area
        
        Parameters
        ----------
        nbr_idx [int, 2d]: index of near neighbor 
        nbr_dis [float, 2d]: distance of near neighbor 
        mapping [int, 2d]: mapping between min and all grid

        Returns
        ----------
        nbr_idx [int, 2d, np]: index of near neighbor in min
        nbr_dis [float, 2d, np]: distance of near neighbor in min
        """
        min_atom_num = len(mapping)
        nbr_idx = np.array(nbr_idx)[:min_atom_num]
        nbr_dis = np.array(nbr_dis)[:min_atom_num]
        #reduce to min area
        for line in mapping:
            if len(line) > 1:
                min_atom = line[0]
                for atom in line[1:]:
                    nbr_idx[nbr_idx==atom] = min_atom
        return nbr_idx, nbr_dis
    
    def lattice_generate(self):
        """
        generate lattice by crystal system
        
        Returns
        ----------
        latt [obj]: Lattice object of pymatgen
        crystal_system [int, 0d]: crystal system number
        """
        volume = 0
        system = np.arange(0, 7)
        while volume == 0:
            crystal_system = np.random.choice(system, p=system_weight)
            #triclinic
            if crystal_system == 0:
                a, b, c = np.random.normal(len_mu, len_sigma, 3)
                alpha, beta, gamma = np.random.normal(ang_mu, ang_sigma, 3)
            #monoclinic
            if crystal_system == 1:
                a, b, c = np.random.normal(len_mu, len_sigma, 3)
                alpha, gamma = 90, 90
                beta = random.normalvariate(ang_mu, ang_sigma)
            #orthorhombic
            if crystal_system == 2:
                a, b, c = np.random.normal(len_mu, len_sigma, 3)
                alpha, beta, gamma = 90, 90, 90
            #tetragonal
            if crystal_system == 3:
                a, c = np.random.normal(len_mu, len_sigma, 2)
                b = a
                alpha, beta, gamma = 90, 90, 90
            #trigonal
            if crystal_system == 4:
                a = random.normalvariate(len_mu, len_sigma)
                b, c = a, a
                alpha = random.normalvariate(ang_mu, ang_sigma)
                beta, gamma = alpha, alpha
            #hexagonal
            if crystal_system == 5:
                a, c = np.random.normal(len_mu, len_sigma, 2)
                b = a
                alpha, beta, gamma = 90, 90, 120
            #cubic
            if crystal_system == 6:
                a = random.normalvariate(len_mu, len_sigma)
                b, c = a, a
                alpha, beta, gamma = 90, 90, 90
            #check validity of lattice vector
            a, b, c = [i if len_lower < i else len_lower for i in (a, b, c)]
            a, b, c = [i if i < len_upper else len_upper for i in (a, b, c)]
            latt = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
            volume = latt.volume
        return latt
    
    def get_space_group(self, num, system):
        """
        get space group number according to crystal system
        
        Parameters
        ----------
        num [int, 0d]: number of space groups
        system [int, 0d]: crystal system

        Returns
        ----------
        space_group [int, 1d, np]: international number of space group
        """
        #space group of 2-dimensional structure
        if num_dim == 2:
            if system == 0:
                groups = [1, 2]
            if system == 2:
                groups = [3, 4, 5, 25, 28, 32, 35]
            if system == 3:
                groups = [75, 99, 100]
            if system == 5:
                groups = [143, 156, 157, 168, 183]
        #space group of 3-dimensional structure
        if num_dim == 3:
            if system == 0:
                groups = np.arange(1, 3)
            if system == 1:
                groups = np.arange(3, 16)
            if system == 2:
                groups = np.arange(16, 75)
            if system == 3:
                groups = np.arange(75, 143)
            if system == 4:
                groups = np.arange(143, 168)
            if system == 5:
                groups = np.arange(168, 195)
            if system == 6:
                groups = np.arange(195, 231)
        space_group = np.sort(np.random.choice(groups, num*len(groups)))
        return space_group
    

if __name__ == '__main__':
    name = args.name
    
    #Build grid
    grid = GridDivide()
    grid.build(name, cutoff)