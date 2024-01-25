import os, sys
import re
import paramiko
import pickle
import json
import numpy as np
from collections import Counter

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element

sys.path.append(f'{os.getcwd()}/src')
from core.log_print import *


def convert_composition_into_atom_type():
    """
    convert composition into list of atom types

    Returns
    ----------
    atom_type [int, 1d, np]: type of atoms
    """
    elements= re.findall('[A-Za-z]+', Composition)
    ele_num = [int(i) for i in re.findall('[0-9]+', Composition)]
    atom_type = []
    for ele, num in zip(elements, ele_num):
        for _ in range(num):
            atom_type.append(ele)
    #convert to atomic number
    elements = [Element(i) for i in atom_type]
    atom_type = [i.Z for i in elements]
    #sorted by atom radiu
    radius = [i.atomic_radius.real for i in elements]
    order = np.argsort(radius)[::-1]
    atom_type = np.array(atom_type)[order]
    return atom_type

def count_atoms(atom_types):
    """
    count number of each atom
    
    Parameters
    ----------
    atom_types [int, 1d]: atom type
    
    Returns
    ----------
    types_num [int, 2d, np]: number of each type
    """
    tmp = Counter(atom_types)
    types_num = [[i, j] for i, j in tmp.items()]
    return np.array(types_num)

def get_seeds_pos(seeds, template):
    """
    get position of seed in template

    Parameters
    ----------
    seeds [str, 1d]: name of seeds
    template [str, 0d]: name of template

    Returns
    ----------
    seed_pos [int, 2d]: position of seeds in template
    """
    stru_temp = Structure.from_file(f'{Seed_Path}/{template}')
    coord_temp = stru_temp.frac_coords
    seed_pos = []
    for seed in seeds:
        stru_seed = Structure.from_file(f'{Seed_Path}/{seed}')
        coord_seed = stru_seed.frac_coords
        tmp_pos = [i for i in range(Num_Fixed_Temp)]
        for coord in coord_seed[Num_Fixed_Temp:]:
            store = np.subtract(coord_temp, coord)
            store = np.sum(np.abs(store), axis=1)
            idx = np.argsort(store)[0]
            tmp_pos.append(idx)
        seed_pos.append(tmp_pos)
    return seed_pos
        
        
class ListRWTools:
    #Save and import list
    def import_list2d(self, file, dtype, numpy=False, binary=False):
        """
        import 2-Dimensional list
        
        Parameters
        ----------
        file [str, 0d]: file name
        dtype [int float str]: data type
        binary [bool, 0d]: whether write in binary
        
        Returns
        ----------
        list [dtype, 2d]: 2-Dimensional list
        """
        if binary:
            with open(file, 'rb') as f:
                list = pickle.load(f)
            return list
        else:
            with open(file, 'r') as f:
                ct = f.readlines()
            list = self.str_to_list2d(ct, dtype)
            if numpy:
                return np.array(list, dtype=dtype)
            else:
                return list
    
    def str_to_list2d(self, string, dtype):
        """
        convert string list to 2-Dimensional list
    
        Parameters
        ----------
        string [str, 1d]: list in string form
        dtype [int float str]: data type
        
        Returns
        ----------
        list [dtype, 2d]: 2-Dimensional list
        """
        list = [self.str_to_list1d(item.split(), dtype)
                for item in string]
        return list
    
    def str_to_list1d(self, string, dtype):
        """
        convert string list to 1-Dimensional list
    
        Parameters
        ----------
        string [str, 1d]: list in string form
        dtype [int float str]: data type
        
        Returns
        ----------
        list [dtype, 1d]: 1-Dimensional list
        """
        list = [dtype(i) for i in string]
        return list
    
    def write_list2d(self, file, list, style='{0}', binary=False):
        """
        write 2-Dimensional list
        
        Parameters
        ----------
        file [str, 0d]: file name
        list [num, 2d]: 2-Dimensional list
        style [str, 0d]: style of number
        binary [bool, 0d]: whether write in binary
        """
        if binary:
            with open(file, 'wb') as f:
                pickle.dump(list, f)
        else:
            list_str = self.list2d_to_str(list, style)
            list2d_str = '\n'.join(list_str)
            with open(file, 'w') as f:
                f.write(list2d_str)
    
    def list2d_to_str(self, list, style):
        """
        convert 2-Dimensional list to string list
        
        Parameters
        ----------
        list [num, 2d]: 2-Dimensional list
        style [str, 0d]: string style of number
        
        Returns
        ----------
        list_str [str, 1d]: string of list2d
        """
        list_str = [' '.join(self.list1d_to_str(line, style)) 
                    for line in list]
        return list_str
    
    def list1d_to_str(self, list, style):
        """
        convert 1-Dimensional list to string list
        
        Parameters
        ----------
        list [num, 1d]: 1-Dimensional list
        style [str, 0d]: string style of number

        Returns
        ----------
        list [str, 1d]: 1-Dimensional string list
        """
        list = [style.format(i) for i in list]
        return list
    
    def write_dict(self, file, dict):
        """
        export dict as json
        
        Parameters
        ----------
        file [str, 0d]: name of file 
        dict [dict]: dict or list-dict 
        """
        with open(file, 'w') as obj:
            json.dump(dict, obj)
    
    def import_dict(self, file, trans=False):
        """
        import dict from json
        
        Parameters
        ----------
        file [str, 0d]: name of file 
        
        Returns
        ----------
        dict [list, dict, 1d]: list-dict with int keys
        """
        with open(file, 'r') as obj:
            ct = json.load(obj)
        if trans:
            dict = self.transfer_keys(ct)
        else:
            dict = ct
        return dict
    
    def transfer_keys(self, list_dict):
        """
        transfer keys from string to int

        Parameters
        ---------
        list_dict [list, dict, 1d]: list-dict with str keys

        Returns
        ----------
        new [list, dict, 1d]: list-dict with int keys
        """
        new = []
        #list dict transfer
        if isinstance(list_dict, list):
            for item in list_dict:
                store = {}
                for key in item.keys():
                    store[int(key)] = item[key]
                new.append(store)
        #dict transfer
        elif isinstance(list_dict, dict):
            store = {}
            for key in list_dict.keys():
                store[int(key)] = list_dict[key]
            new = store
        return new
    
    def import_data(self, task, grid=0, sg=0):
        """
        import data according to task
        
        Parameters
        ----------
        task [str, 0d]: name of import data
        grid [int, 0d]: name of grid
        sg [int, 0d]: space group number
        """
        head = f'{Grid_Path}/{grid:03.0f}'
        #import element embedding file
        if task == 'elem':
            elem_embed = self.import_list2d(
                Atom_Init_File, float, numpy=True)
            return elem_embed
        #import atom properties
        if task == 'property':
            property_dict = self.import_dict(New_Atom_File)
            return property_dict
        #import rotation angles
        if task == 'angles':
            angles = self.import_list2d(
                Cluster_Angle_File, float, numpy=True)
            return angles
        #import grid index and distance file
        if task == 'grid':
            grid_idx = self.import_list2d(
                f'{head}_nbr_idx_{sg}.bin', int, binary=True)
            grid_dis =  self.import_list2d(
                f'{head}_nbr_dis_{sg}.bin', float, binary=True)
            return grid_idx, grid_dis
        #import mapping relationship
        if task == 'mapping':
            mapping = self.import_list2d(
                f'{head}_mapping_{sg}.bin', int, binary=True)
            return mapping
        #import lattice vector
        if task == 'latt':
            latt_vec = self.import_list2d(
                f'{head}_latt_vec.bin', float, binary=True)
            return latt_vec
        #import fraction coordinates of grid
        if task == 'frac':
            grid_coords = self.import_list2d(
                f'{head}_frac_coords_{sg}.bin', float, binary=True)
            return grid_coords
        

class SSHTools:
    #SSH to work nodes
    def __init__(self):
        if Job_Queue == 'CPU':
            self.work_nodes = CPU_Nodes
            self.child_nodes = [i for i in CPU_Nodes if i != Host_Node]
        elif Job_Queue == 'GPU':
            self.work_nodes = GPU_Nodes
            self.child_nodes = [i for i in GPU_Nodes if i != Host_Node]
        self.work_nodes_num = len(self.work_nodes)
        self.child_nodes_num = len(self.child_nodes)
    
    def ssh_node(self, shell_script, node):
        """
        SSH to target node and execute command

        Parameters
        ----------
        shell_script [str, 0d]: shell script run on work nodes
        node [str, 0d]: name of node
        """
        port = 22
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.MissingHostKeyPolicy())
        ssh.connect(node, port, timeout=1000)
        ssh.exec_command(shell_script)
        ssh.close()
        
    def change_node_assign(self, path):
        """
        change assign of node of jobs

        Parameters
        ----------
        path [str, 0d]: path of save directory 
        """
        poscars = sorted(os.listdir(path))
        poscars_num = len(poscars)
        node_assign = self.assign_node(poscars_num)
        for i, poscar in enumerate(poscars):
            os.rename(f'{path}/{poscar}', 
                      f'{path}/{poscar[:-4]}-{node_assign[i]}')
        
    def assign_node(self, num_jobs, order=True):
        """
        assign jobs to nodes
        
        Parameters
        ----------
        num_jobs [int, 0d]: number of jobs
        
        Returns
        ----------
        node_assign [int, 1d]: vasp job list of nodes
        """
        num_assign, node_assign = 0, []
        while not num_assign == num_jobs:
            left = num_jobs - num_assign
            assign = left//self.work_nodes_num
            if assign == 0:
                node_assign = node_assign + self.work_nodes[:left]
            else:
                node_seq = [i for i in self.work_nodes]
                for n in range(assign):
                    if np.mod(n+1, 2) == 0:
                        node_assign += node_seq[::-1]
                    else:
                        node_assign += node_seq
            num_assign = len(node_assign)
        if order:
            node_assign = sorted(node_assign)
        return node_assign
    
    def assign_job(self, poscar):
        """
        assign jobs to each node by notation of POSCAR file

        Parameters
        ----------
        poscar [str, 1d]: name of POSCAR files
        
        Returns
        ----------
        batches [str, 1d]: string of jobs assigned to different nodes
        nodes [str, 1d]: job assigned nodes
        """
        store, batches, nodes = [], [], []
        last_node = poscar[0][-3:]
        nodes.append(last_node)
        for item in poscar:
            node = item[-3:]
            if node == last_node:
                store.append(item)
            else:
                batches.append(' '.join(store))
                last_node = node
                store = []
                store.append(item)
                nodes.append(last_node)
        batches.append(' '.join(store))
        return batches, nodes
    
    def group_poscars(self, poscars):
        """
        group poscars by node
        
        Parameters
        ----------
        poscars [str, 1d]: name of poscar

        Returns:
        ----------
        jobs [str, 2d]: grouped poscars
        """
        #poscars grouped by nodes
        assign, jobs = [], []
        for node in self.work_nodes:
            for poscar in poscars:
                label = poscar.split('-')[-1]
                if label == str(node):
                    assign.append(poscar)
            if len(assign) > 0:
                jobs.append(assign)
            assign = []
        return jobs
        
    def is_done(self, path, num_file, flag='FINISH'):
        """
        if the vasp calculation is completed, return True
        
        Parameters
        ----------
        path [str, 0d]: path used to store flags
        num_file [int, 0d]: number of file
        
        Returns
        ----------
        flag [bool, 0d]: whether all nodes are done
        """
        command = f'ls -l {path} | grep {flag} | wc -l'
        flag = self.check_num_file(command, num_file)
        return flag
    
    def remove_flag(self, path):
        """
        remove FINISH flags
        
        Parameters
        ----------
        path [str, 0d]: path used to store flags
        """
        os.system(f'rm {path}/FINISH*')
    
    def check_num_file(self, command, file_num):
        """
        if shell is completed, return True
        
        Returns
        ----------
        flag [bool, 0d]: whether work is done
        """
        flag = False
        finish = os.popen(command)
        finish = int(finish.read())
        if finish == file_num:
            flag = True
        return flag
    
    def count_num(self, path):
        """
        get number of optimized POSCAR
        
        Parameters
        ----------
        path [str, 0d]: path used to store flags
        
        Returns
        ----------
        num [int, 0d]: number of optimized POSCAR
        """
        command = f'ls -l {path} | grep FINISH | wc -l'
        ct = os.popen(command)
        num = int(ct.read())
        return num
    
    def assign_cores(self, job_num, core_num):
        """
        get assigment of cores
        
        Parameters
        ----------
        job_num [int, 0d]: number of jobs
        core_num [int, 0d]: number of cpu cores for each job

        Returns
        ----------
        assign_str [str, 1d]: assigment of cores
        """
        cpu_avail = list(range(os.cpu_count()))
        cpu_num = len(cpu_avail)
        total_core = job_num * core_num
        ratio = total_core // cpu_num
        tmp = cpu_avail.copy()
        for _ in range(ratio):
            tmp += cpu_avail
        assign = np.split(np.array(tmp)[:total_core], job_num)
        #convert to string
        assign_str = []
        for cores in assign:
            assign_str.append(','.join(map(str, cores)))
        return assign_str
    
    
if __name__ == '__main__':
    pass