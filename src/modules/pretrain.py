import os, sys
import time
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from modules.global_var import *
from modules.utils import SSHTools, system_echo
from modules.predict import PPMData, PPModel
from modules.data_transfer import Transfer


class Initial(SSHTools):
    #make each node consistent with main node
    def __init__(self, sleep_time=1):
        self.sleep_time = sleep_time
    
    def update(self):
        """
        update cpu nodes
        """
        num_node = len(nodes)
        for node in nodes:
            self.create_work_dir(node)
        while not self.is_done(num_node):
            time.sleep(self.sleep_time)
        self.remove()
        
        self.copy_file_to_nodes()
        system_echo('Each node consistent with main node')
    
    def create_work_dir(self, node):
        """
        SSH to target node and update ccop
        """
        ip = f'node{node}'
        shell_script = f'''
                        #!/bin/bash
                        cd /local
                        rm -rf ccop/
                        mkdir ccop/
                        cd ccop/
                        mkdir vasp/
                        touch FINISH-{ip}
                        scp FINISH-{ip} {gpu_node}:/local/ccop/data/
                        rm FINISH-{ip}
                        '''
        self.ssh_node(shell_script, ip)
    
    def copy_file_to_nodes(self):
        cpu_nodes = [f'node{i}' for i in nodes]
        cpu_nodes = ' '.join(cpu_nodes)
        shell_script = f'''
                        #!/bin/bash
                        for i in {cpu_nodes}
                        do
                            scp -r data $i:/local/ccop/
                            scp -r libs $i:/local/ccop/
                            scp -r src $i:/local/ccop/
                        done
                        '''
        os.system(shell_script)
    
    def is_done(self, file_num):
        """
        If shell is completed, return True
        
        Returns
        ----------
        flag [bool, 0d]: whether job is done
        """
        command = f'ls -l data/ | grep FINISH | wc -l'
        flag = self.check_num_file(command, file_num)
        return flag
    
    def remove(self): 
        os.system(f'rm data/FINISH*')


def put_into_grid():
    pass


class CIFTransfer(Transfer):
    #
    def __init__(self, 
                 cutoff=8, nbr=12, dmin=0, dmax=8, step=0.2, var=0.2):
        self.cutoff = cutoff
        self.nbr, self.var = nbr, var
        self.filter = np.arange(dmin, dmax+step, step)
        self.elem_embed = self.import_list2d(
            atom_init_file, int, numpy=True)
    
    def single(self, cif):
        """
        index and distance of near grid points
        
        Parameters
        ----------
        cif [str, 0d]: string of cif 
        
        Returns
        ----------
        atom_fea [int, 2d, np]: feature of atoms
        nbr_fea [float, 2d, np]: distance of near neighbor 
        nbr_idx [int, 2d, np]: index of near neighbor 
        """
        crystal = Structure.from_str(cif, fmt='cif')
        atom_type = np.array(crystal.atomic_numbers) - 1
        all_nbrs = crystal.get_all_neighbors(self.cutoff)
        all_nbrs = [sorted(nbrs, key = lambda x: x[1]) for nbrs in all_nbrs]
        num_near = min(map(lambda x: len(x), all_nbrs))
        nbr_idx, nbr_dis = [], []
        for nbr in all_nbrs:
            idx = list(map(lambda x: x[2], nbr[:num_near]))[:self.nbr]
            dis = list(map(lambda x: x[1], nbr[:num_near]))[:self.nbr]
            nbr_idx.append(idx)
            nbr_dis.append(dis)
        nbr_idx, nbr_dis = np.array(nbr_idx), np.array(nbr_dis)
        atom_fea = self.atom_initilizer(atom_type)
        nbr_fea = self.expand(nbr_dis)
        return atom_fea, nbr_fea, nbr_idx
    
    def batch(self, cifs):
        """
        transfer cifs to input of predict model
        
        Parameters
        ----------
        cifs [str, 1d]: string of structure in cif

        Returns
        ----------
        batch_atom_fea [int, 3d]: batch atom feature
        batch_nbr_fea [float, 4d]: batch neighbor feature
        batch_nbr_fea_idx [float, 3d]: batch neighbor index
        """
        batch_atom_fea, batch_nbr_fea, \
            batch_nbr_fea_idx = [], [], []
        for cif in cifs:
            atom_fea, nbr_fea, nbr_fea_idx = \
                self.single(cif)
            batch_atom_fea.append(atom_fea)
            batch_nbr_fea.append(nbr_fea)
            batch_nbr_fea_idx.append(nbr_fea_idx)
        return batch_atom_fea, batch_nbr_fea, \
                batch_nbr_fea_idx
    

if __name__ == '__main__':
    train_df = pd.read_csv('database/mp_20/train.csv')
    valid_df = pd.read_csv('database/mp_20/val.csv')
    test_df = pd.read_csv('database/mp_20/test.csv')

    train_store = [train_df.iloc[idx] for idx in range(len(train_df))]
    train_cifs = [i['cif'] for i in train_store]
    train_energys = [i['formation_energy_per_atom'] for i in train_store]

    valid_store = [valid_df.iloc[idx] for idx in range(len(valid_df))]
    valid_cifs = [i['cif'] for i in valid_store]
    valid_energys = [i['formation_energy_per_atom'] for i in valid_store]
    
    test_store = [test_df.iloc[idx] for idx in range(len(test_df))]
    test_cifs = [i['cif'] for i in test_store]
    test_energys = [i['formation_energy_per_atom'] for i in test_store]
    
    transfer = CIFTransfer()
    train_atom_fea, train_nbr_fea, train_nbr_fea_idx = transfer.batch(train_cifs)
    valid_atom_fea, valid_nbr_fea, valid_nbr_fea_idx = transfer.batch(valid_cifs)
    test_atom_fea, test_nbr_fea, test_nbr_fea_idx = transfer.batch(test_cifs)
    
    train_data = PPMData(train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energys)
    valid_data = PPMData(valid_atom_fea, valid_nbr_fea, valid_nbr_fea_idx, valid_energys)
    test_data = PPMData(test_atom_fea, test_nbr_fea, test_nbr_fea_idx, test_energys)
    
    ppm = PPModel(100, train_data, valid_data, test_data)
    ppm.train_epochs()
    
    
    
    
    
    
    '''
    checkpoint = torch.load('test/model_best.pth.tar', map_location='cpu')
    model = FineTuneNet(orig_atom_fea_len, nbr_bond_fea_len)
    model.load_state_dict(checkpoint['state_dict'])
    '''