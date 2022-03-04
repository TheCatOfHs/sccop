import os, sys
import time
import argparse
import numpy as np

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.utils import ListRWTools, SSHTools, system_echo
from pymatgen.core.structure import Structure


parser = argparse.ArgumentParser()
parser.add_argument('--grain', type=float, nargs='+')
parser.add_argument('--origin', type=int)
parser.add_argument('--mutate', type=int)
args = parser.parse_args()


class ParallelDivide(ListRWTools, SSHTools):
    #divide grid by each node
    def __init__(self, wait_time=0.1):
        self.num_node = len(nodes)
        self.wait_time = wait_time
        self.prop_path = f'/local/ccop/{grid_prop_path}'
    
    def assign_to_cpu(self, grain, grid_origin, grid_mutate):
        """
        send divide jobs to nodes and update nodes
        
        Parameters
        ----------
        grain [int, 1d]: grain of grid
        grid_origin [int, 1d]: name of origin grid 
        grid_mutate [int, 1d]: name of mutate grid 
        """
        num_grid = len(grid_mutate)
        node_assign = self.assign_node(num_grid)
        #generate grid and send back to gpu node
        for origin, mutate, node in zip(grid_origin, grid_mutate, node_assign):
            self.sub_divide(grain, origin, mutate, node)
        while not self.is_done(grid_prop_path, num_grid):
            time.sleep(self.wait_time)
        self.zip_file_name(grid_mutate)
        self.remove_flag_on_gpu()
        system_echo(f'Lattice generate: {grid_mutate}')
        #update cpus
        for node in nodes:
            self.send_grid_to_cpu(node)
        while not self.is_done(grid_prop_path, self.num_node*num_grid):
            time.sleep(self.wait_time)
        self.remove_flag_on_gpu()
        self.remove_flag_on_cpu()
        system_echo(f'Unzip grid file on CPUs')
        #unzip on gpu
        self.unzip_grid_on_gpu()
        while not self.is_done(grid_prop_path, num_grid):
            time.sleep(self.wait_time)
        self.remove_zip_on_gpu()
        self.remove_flag_on_gpu()
        system_echo(f'Unzip grid file on GPU')
    
    def sub_divide(self, grain, origin, mutate, node):
        """
        SSH to target node and divide grid

        Parameters
        ----------
        grain [float, 1d]: grain of grid
        origin [int, 0d]: name of origin grid
        mutate [int, 0d]: name of mutate grid
        node [int, 0d]: name of node
        """
        ip = f'node{node}'
        file = [f'{mutate:03.0f}_frac_coor.bin',
                f'{mutate:03.0f}_latt_vec.bin',
                f'{mutate:03.0f}_nbr_dis.bin',
                f'{mutate:03.0f}_nbr_idx.bin']
        file = ' '.join(file)
        grain_str = ' '.join([str(i) for i in grain])
        options = f'--grain {grain_str} --origin {origin} --mutate {mutate}'
        shell_script = f'''
                        cd /local/ccop/
                        python src/core/grid_divide.py {options}
                        cd {grid_prop_path}
                        
                        touch FINISH-{mutate}                        
                        tar -zcf {mutate}.tar.gz {file} FINISH-{mutate}
                        scp {mutate}.tar.gz {gpu_node}:{self.prop_path}/
                        
                        scp FINISH-{mutate} {gpu_node}:{self.prop_path}/
                        rm {file} {mutate}.tar.gz FINISH-{mutate}
                        '''
        self.ssh_node(shell_script, ip)
    
    def zip_file_name(self, grid_name):
        """
        zip file is used to transport between nodes
        
        Parameters
        ----------
        grid_name [int, 1d]: name of new grid 
        """
        file = []
        for grid in grid_name:
            file.append(f'{grid}.tar.gz')
        self.zip_file = file
    
    def send_grid_to_cpu(self, node):
        """
        update grid of each node
        """
        for i, file in enumerate(self.zip_file):
            ip = f'node{node}'
            shell_script = f'''
                            cd /local/ccop/{grid_prop_path}
                            scp {gpu_node}:{self.prop_path}/{file} .
                            tar -zxf {file}
                            rm {file}
                            
                            touch FINISH-{node}-{i}
                            scp FINISH-{node}-{i} {gpu_node}:{self.prop_path}/
                            rm FINISH-{node}-{i}
                            '''
            self.ssh_node(shell_script, ip)
    
    def unzip_grid_on_gpu(self):
        """
        unzip grid property on gpu node
        """
        zip_file = ' '.join(self.zip_file)
        shell_script = f'''
                        cd {grid_prop_path}/
                        for i in {zip_file}
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
        os.system(f'rm {grid_prop_path}/FINISH*')
        
    def remove_flag_on_cpu(self):
        """
        remove FINISH flags on cpu
        """
        for node in nodes:
            ip = f'node{node}'
            shell_script = f'''
                            cd /local/ccop/{grid_prop_path}
                            rm FINISH*
                            '''
            self.ssh_node(shell_script, ip)

    def remove_zip_on_gpu(self):
        """
        remove zip file
        """
        os.system(f'rm {grid_prop_path}/*.tar.gz')
    
    
class GridDivide(ListRWTools):
    #Build the grid
    def __init__(self):
        self.mu = latt_mu
        self.sigma = latt_sigma
    
    def build_grid(self, grid_name, latt_vec, grain, cutoff, mutate=False):
        """
        save POSCAR, latt_vec, frac_coor, nbr_idx, nbr_dis of grid
        
        Parameters
        ----------
        grid_name [str, 0d]: name of grid
        latt_vec [float, 2d, np]: lattice vector of grid
        grain [float, 1d]: fraction coordinate of grid points
        cutoff [float, 0d]: cutoff distance
        mutate [bool, 0d]: whether mutate lattice vector
        """
        self.grid = f'{grid_name:03.0f}'
        self.prop_path = f'{grid_prop_path}/{self.grid}'
        self.poscar = f'{grid_poscar_path}/POSCAR_{self.grid}'
        if mutate:
            delta = np.identity(3) + self.strain_mat()
            latt_vec = np.dot(delta, latt_vec)
            system_echo('Lattice mutate!')
        frac_coor = self.fraction_coor(grain, latt_vec)
        self.write_grid_POSCAR(latt_vec, frac_coor)
        stru = Structure.from_file(self.poscar)
        nbr_idx, nbr_dis = self.near_property(stru, cutoff)
        self.write_list2d(f'{self.prop_path}_latt_vec.bin', 
                          latt_vec, binary=True)
        self.write_list2d(f'{self.prop_path}_frac_coor.bin',
                          frac_coor, binary=True)
        self.write_list2d(f'{self.prop_path}_nbr_idx.bin',
                          nbr_idx, binary=True)
        self.write_list2d(f'{self.prop_path}_nbr_dis.bin',
                          nbr_dis, binary=True)
    
    def strain_mat(self):
        """
        symmetry stain matrix
        
        Returns
        ----------
        strain [float, 2d, np]: strain matrix
        """
        gauss_mat = np.random.normal(self.mu, self.sigma, (3, 3))
        gauss_sym = 0.25*(gauss_mat + np.transpose(gauss_mat)) + \
            0.5*np.identity(3)*gauss_mat
        strain = np.clip(gauss_sym, -1, 1)
        return strain
    
    def fraction_coor(self, grain, latt_vec):
        """
        fraction coordinate of grid
        
        Parameters
        ----------
        bond_len [float, 1d]: fine grain of grid
        
        Returns
        ----------
        coor [float, 2d]: fraction coordinate of grid
        """
        norm = [np.linalg.norm(i) for i in latt_vec]
        n = [norm[i]//grain[i] for i in range(3)]
        n = [1 if i==0 else i for i in n]
        grain_a, grain_b, grain_c = [1/i for i in n]
        coor = [[i, j, k] for i in np.arange(0, 1, grain_a)
                for j in np.arange(0, 1, grain_b)
                for k in np.arange(0, 1, grain_c)]
        return np.array(coor)
    
    def near_property(self, stru, cutoff, near=0):
        """
        index and distance of near grid points
        
        Parameters
        ----------
        stru [obj]: pymatgen object
        cutoff [float, 0d]: cutoff distance
        
        Returns
        ----------
        nbr_idx [int, 2d]: index of near neighbor 
        nbr_dis [float, 2d]: distance of near neighbor 
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
        return np.array(nbr_idx), np.array(nbr_dis)
    
    def write_grid_POSCAR(self, latt_vec, frac_coor):
        """
        write POSCAR of grid
        
        Parameters
        ----------
        latt_vec [float, 2d]: lattice vector
        frac_coor [float, 2d]: fraction coordinate of grid
        """
        head = ['E = -1', '1']
        latt_vec_str = self.list2d_to_str(latt_vec, '{0:8.4f}')
        compn = ['H', f'{len(frac_coor)}', 'Direct']
        frac_coor_str = self.list2d_to_str(frac_coor, '{0:8.4f}')
        POSCAR = head + latt_vec_str + compn + frac_coor_str
        with open(self.poscar, 'w') as f:
            f.write('\n'.join(POSCAR))
        

if __name__ == '__main__':
    grain = args.grain
    grid_origin = args.origin
    grid_mutate = args.mutate
    
    #Build grid
    grid = GridDivide()
    rwtools = ListRWTools()
    latt_file = f'{grid_prop_path}/{grid_origin:03.0f}_latt_vec.bin'
    latt_vec = rwtools.import_list2d(latt_file, float, binary=True)
    if grid_origin == grid_mutate:
        mutate = False
    else:
        mutate = True
    grid.build_grid(grid_mutate, latt_vec, grain, cutoff, mutate=mutate)