import os, sys
import time
import paramiko
import pickle
import json
import numpy as np

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.path import *


def system_echo(ct):
    """
    write system log
    
    Parameters
    ----------
    ct [str, 0d]: content
    """
    echo_ct = time.strftime("%Y-%m-%d %H:%M:%S",
                            time.localtime()) + ' -- ' + ct
    print(echo_ct)
    with open(log_file, 'a') as obj:
        obj.write(echo_ct + '\n')


class ListRWTools:
    #Save and import list
    def import_list2d(self, file, dtype, numpy=False, binary=False):
        """
        import 2-dimensional list
        
        Parameters
        ----------
        file [str, 0d]: file name
        dtype [int float str]: data type
        binary [bool, 0d]: whether write in binary
        
        Returns
        ----------
        list [dtype, 2d]: 2-dimensional list
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
        convert string list to 2-dimensional list
    
        Parameters
        ----------
        string [str, 1d]: list in string form
        dtype [int float str]: data type
        
        Returns
        ----------
        list [dtype, 2d]: 2-dimensional list
        """
        list = [self.str_to_list1d(item.split(), dtype)
                for item in string]
        return list
    
    def str_to_list1d(self, string, dtype):
        """
        convert string list to 1-dimensional list
    
        Parameters
        ----------
        string [str, 1d]: list in string form
        dtype [int float str]: data type
        
        Returns
        ----------
        list [dtype, 1d]: 1-dimensional list
        """
        list = [dtype(i) for i in string]
        return list
    
    def write_list2d(self, file, list, style='{0}', binary=False):
        """
        write 2-dimensional list
        
        Parameters
        ----------
        file [str, 0d]: file name
        list [num, 2d]: 2-dimensional list
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
    
    def write_list2d_columns(self, file, lists, styles, head=[]):
        """
        write 2-dimensional list
        
        Parameters
        ----------
        file [str, 0d]: file name
        lists [[num1, 2d], [num2, 2d], ...]: 2-dimensional lists
        styles [[str1, 0d], [str2, 0d], ...]: styles of number corresponding to each list
        head [str, 1d]: the head of the output file
        """
        list_strs = [self.list2d_to_str(lists[i], styles[i]) for i in range(len(styles))] 
        list_str = [''.join([list_strs[i][j] for i in range(len(styles))]) for j in range(len(lists[0]))]
        list2d_str = '\n'.join(list_str) if len(head) == 0 else '\n'.join(head + list_str)
        with open(file, 'w', encoding='utf-8') as f:
            f.write(list2d_str)
        
    def list2d_to_str(self, list, style):
        """
        convert 2-dimensional list to string list
        
        Parameters
        ----------
        list [num, 2d]: 2-dimensional list
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
        convert 1-dimensional list to string list
        
        Parameters
        ----------
        list [num, 1d]: 1-dimensional list
        style [str, 0d]: string style of number

        Returns
        ----------
        list [str, 1d]: 1-dimensional string list
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
    
    def import_dict(self, file):
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
        dict = self.transfer_keys(ct)
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
        for dict in list_dict:
            store = {}
            for key in dict.keys():
                store[int(key)] = dict[key]
            new.append(store)
        return new
    
    
class SSHTools:
    #SSH to node
    def __init__(self):
        pass
    
    def ssh_node(self, shell_script, ip):
        """
        SSH to target node and execute command

        Parameters
        ----------
        shell_script [str, 0d]
        ip [str, 0d]
        """
        port = 22
        user = user
        password = password
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, port, user, password, timeout=1000)
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
        num_nodes = len(nodes)
        num_assign, node_assign = 0, []
        while not num_assign == num_jobs:
            left = num_jobs - num_assign
            assign = left//num_nodes
            if assign == 0:
                node_assign = node_assign + nodes[:left]
            else:
                node_seq = [i for i in nodes]
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
    
    def is_done(self, path, num_file):
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
        command = f'ls -l {path} | grep FINISH | wc -l'
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
    

if __name__ == '__main__':
    import pandas as pd
    from jarvis.db.figshare import data
    from jarvis.core.atoms import Atoms
    #
    dft_2d = data(dataset='dft_2d')
    matpd = data(dataset='twod_matpd')
    c2db = data(dataset='c2db')
    
    names, energys = [], []
    for i, obj in enumerate(dft_2d):
        name = f'DFT2D-{i:04.0f}'
        stru = Atoms.from_dict(obj['atoms'])
        energy = obj['optb88vdw_total_energy']
        if energy != 'na':
            stru.write_cif(f'CIF2D/{name}.cif')
            names.append(name)
            energys.append(energy)
    
    for i, obj in enumerate(matpd):
        name = f'MATPD-{i:05.0f}'
        stru = Atoms.from_dict(obj['atoms'])
        energy = obj['energy_per_atom']
        if energy != 'na':
            stru.write_cif(f'CIF2D/{name}.cif')
            names.append(name)
            energys.append(energy)
    
    for i, obj in enumerate(c2db):
        name = f'C2DB-{i:05.0f}'
        stru = Atoms.from_dict(obj['atoms'])
        energy = obj['etot']/stru.num_atoms
        if energy != 'na':
            stru.write_cif(f'CIF2D/{name}.cif')
            names.append(name)
            energys.append(energy)
    #
    df = pd.DataFrame({'cif':names, 'E':energys})
    df.to_csv('id_prop.csv',index =False ,sep = ',')