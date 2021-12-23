import os, sys
import time
import argparse
import numpy as np

sys.path.append(f'{os.getcwd()}/src')
from modules.global_var import *
from modules.utils import ListRWTools, SSHTools, system_echo
from pymatgen.core.structure import Structure


parser = argparse.ArgumentParser()
parser.add_argument('--origin', type=int)
parser.add_argument('--mutate', type=int)
args = parser.parse_args()


class MultiDivide(ListRWTools, SSHTools):
    #divide grid by each node
    def __init__(self, sleep_time=1):
        self.sleep_time = sleep_time
    
    def assign(self, grid_origin, grid_mutate):
        """
        send divide jobs to nodes and update nodes
        
        Parameters
        ----------
        grid_origin [int, 1d]: name of origin grid 
        grid_mutate [int, 1d]: name of mutate grid 
        """
        num_node = len(nodes)
        num_grid = len(grid_mutate)
        node_assign = self.assign_node(num_grid, num_node)
        for origin, mutate, node in zip(grid_origin, grid_mutate, node_assign):
            self.sub_divide(origin, mutate, node)
        while not self.is_done(num_grid, 'grid'):
            time.sleep(self.sleep_time)
        system_echo(f'Lattice mutate: {grid_mutate}')
        self.remove('grid')
        for node in nodes:
            self.update_grid(grid_mutate, node)
        while not self.is_done(num_node, 'node'):
            time.sleep(self.sleep_time)
        system_echo(f'Update grid of each node!')
        self.remove('node')
    
    def assign_node(self, num_grid, num_node):
        """
        assign divide jobs to nodes

        Returns
        ----------
        node_assign [int, 1d]: vasp job list of nodes
        """
        num_assign, node_assign = 0, []
        while not num_assign == num_grid:
            left = num_grid - num_assign
            assign = left//num_node
            if assign == 0:
                node_assign = node_assign + nodes[:left]
            else:
                node_assign = [i for i in nodes for _ in range(assign)]
            num_assign = len(node_assign)
        return sorted(node_assign)
    
    def sub_divide(self, origin, mutate, node):
        """
        SSH to target node and divide grid

        Parameters
        ----------
        origin [int, 0d]: name of origin grid
        mutate [int, 0d]: name of mutate grid
        node [int, 0d]: name of node
        """
        ip = f'node{node}'
        frac_coor = f'{mutate:03.0f}_frac_coor.dat'
        latt_vec = f'{mutate:03.0f}_latt_vec.dat'
        nbr_dis = f'{mutate:03.0f}_nbr_dis.dat'
        nbr_idx = f'{mutate:03.0f}_nbr_idx.dat'
        local_prop_dir = f'~/ccop/{grid_prop_dir}'
        options = f'--origin {origin} --mutate {mutate}'
        shell_script = f'''
                        #!/bin/bash
                        cd /local/ccop/
                        python src/modules/grid_divide.py {options}
                        cd {grid_prop_dir}
                        mv {frac_coor} {local_prop_dir}/
                        mv {latt_vec} {local_prop_dir}/
                        mv {nbr_dis} {local_prop_dir}/
                        mv {nbr_idx} {local_prop_dir}/
                        > grid{mutate}
                        mv grid{mutate} {local_prop_dir}/
                        '''
        self.ssh_node(shell_script, ip)
    
    def update_grid(self, grid_name, node):
        """
        SSH to target node and update grid
        """
        ip = f'node{node}'
        file = []
        for grid in grid_name:
            file.append(f'{grid:03.0f}_frac_coor.dat '
                        f'{grid:03.0f}_latt_vec.dat '
                        f'{grid:03.0f}_nbr_dis.dat '
                        f'{grid:03.0f}_nbr_idx.dat')
        file = ' '.join(file)
        shell_script = f'''
                        #!/bin/bash
                        cd /local/ccop/{grid_prop_dir}
                        for i in {file}
                        do
                            cp ~/ccop/{grid_prop_dir}/$i .
                        done
                        > {ip}
                        mv {ip} ~/ccop/{grid_prop_dir}/
                        '''
        self.ssh_node(shell_script, ip)
    
    def is_done(self, file_num, key):
        """
        If shell is completed, return True
        
        Returns
        ----------
        file_num [int, 0d]: number of file
        """
        command = f'ls -l {grid_prop_dir} | grep {key} | wc -l'
        flag = self.check_num_file(command, file_num)
        return flag
    
    def remove(self, key): 
        """
        remove key file
        """
        os.system(f'rm {grid_prop_dir}/{key}*')
        

class GridDivide(ListRWTools):
    #Build the grid
    def __init__(self, mu=0., sigma=.2):
        self.mu = mu
        self.sigma = sigma
    
    def build_grid(self, grid_name, latt_vec, grain, cutoff, mutate=False):
        """
        save POSCAR, latt_vec, frac_coor, nbr_idx, nbr_dis of grid
        
        Parameters
        ----------
        grid_name [str, 0d]: name of grid
        latt_vec [float, 2d]: lattice vector of grid
        grain [float, 1d]: fraction coordinate of grid points
        cutoff [float, 0d]: cutoff distance
        mutate [bool, 0d]: whether mutate lattice vector
        """
        self.grid = f'{grid_name:03.0f}'
        self.prop_dir = f'{grid_prop_dir}/{self.grid}'
        self.poscar = f'{grid_poscar_dir}/POSCAR_{self.grid}'
        if mutate:
            delta = np.identity(3) + self.strain_mat()
            latt_vec = np.dot(delta, latt_vec)
            system_echo('Lattice mutate!')
        frac_coor = self.fraction_coor(grain)
        self.write_grid_POSCAR(latt_vec, frac_coor)
        nbr_idx, nbr_dis = self.near_property(cutoff)
        self.write_list2d(f'{self.prop_dir}_latt_vec.dat', 
                          latt_vec, '{0:4.4f}')
        self.write_list2d(f'{self.prop_dir}_frac_coor.dat',
                          frac_coor, '{0:4.4f}')
        self.write_list2d(f'{self.prop_dir}_nbr_idx.dat',
                          nbr_idx, '{0:4.0f}')
        self.write_list2d(f'{self.prop_dir}_nbr_dis.dat',
                          nbr_dis, '{0:8.6f}')

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
    
    def fraction_coor(self, grain):
        """
        fraction coordinate of grid
        
        Parameters
        ----------
        bond_len [float, 1d]: fine grain of grid
        
        Returns
        ----------
        coor [float, 2d]: fraction coordinate of grid
        """
        grain_a, grain_b, grain_c = grain
        coor = [[i, j, k] for i in np.arange(0, 1, grain_a)
                for j in np.arange(0, 1, grain_b)
                for k in np.arange(0, 1, grain_c)]
        return coor
    
    def near_property(self, cutoff):
        """
        index and distance of near grid points
        
        Parameters
        ----------
        cutoff [float, 0d]: cutoff distance
        
        Returns
        ----------
        nbr_idx [int, 2d]: index of near neighbor 
        nbr_dis [float, 2d]: distance of near neighbor 
        """
        crystal = Structure.from_file(self.poscar)
        all_nbrs = crystal.get_all_neighbors(cutoff)
        all_nbrs = [sorted(nbrs, key = lambda x: x[1]) for nbrs in all_nbrs]
        num_near = min(map(lambda x: len(x), all_nbrs))
        nbr_idx, nbr_dis = [], []
        for nbr in all_nbrs:
            nbr_idx.append(list(map(lambda x: x[2], nbr[:num_near])))
            nbr_dis.append(list(map(lambda x: x[1], nbr[:num_near])))
        nbr_idx, nbr_dis = list(nbr_idx), list(nbr_dis)
        return nbr_idx, nbr_dis
    
    def write_grid_POSCAR(self, latt_vec, frac_coor):
        """
        write POSCAR of grid
        
        Parameters
        ----------
        latt_vec [float, 2d]: lattice vector
        frac_coor [float, 2d]: fraction coordinate of grid
        """
        head = ['E = -1', '1']
        latt_vec_str = self.list2d_to_str(latt_vec, '{0:4.4f}')
        compn = ['H', f'{len(frac_coor)}', 'Direct']
        frac_coor_str = self.list2d_to_str(frac_coor, '{0:4.4f}')
        POSCAR = head + latt_vec_str + compn + frac_coor_str
        with open(self.poscar, 'w') as f:
            f.write('\n'.join(POSCAR))
    

if __name__ == '__main__':
    grid_origin = args.origin
    grid_mutate = args.mutate
    
    #Build grid
    grid = GridDivide()
    rwtools = ListRWTools()
    latt_file = f'{grid_prop_dir}/{grid_origin:03.0f}_latt_vec.dat'
    latt_vec = rwtools.import_list2d(latt_file, float, numpy=True)
    grid.build_grid(grid_mutate, latt_vec, grain, cutoff, mutate=True)