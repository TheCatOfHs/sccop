import os, sys
import time
from monty.dev import requires
import pandas as pd
from pymatgen.core.structure import Structure

import torch
import torch.nn as nn

sys.path.append(f'{os.getcwd()}/src')
from modules.global_var import *
from modules.utils import SSHTools, system_echo
from modules.predict import CrystalGraphConvNet, ConvLayer


class Initial(SSHTools):
    #make each node consistent with main node
    def __init__(self, sleep_time=1):
        self.sleep_time = sleep_time
    
    def update(self):
        num_node = len(nodes)
        for node in nodes:
            self.update_with_ssh(node)
        while not self.is_done(num_node):
            time.sleep(self.sleep_time)
        self.remove()
        system_echo('Each node consistent with main node')
        
    def update_with_ssh(self, node):
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
                        cp -r ~/ccop/data .
                        cp -r ~/ccop/libs .
                        cp -r ~/ccop/src .
                        mkdir vasp/
                        touch FINISH-{ip}
                        mv FINISH-{ip} ~/ccop/data/
                        '''
        self.ssh_node(shell_script, ip)
    
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


class FineTuneNet(CrystalGraphConvNet):
    #Calculate crys_fea
    def __init__(self, orig_atom_fea_len, nbr_fea_len, 
                 h_fea_len=128):
        super(FineTuneNet, self).__init__(orig_atom_fea_len, nbr_fea_len)
        for p in self.parameters():
            p.requires_grad = False
        self.fc_out = nn.Linear(h_fea_len, 1)
    

if __name__ == '__main__':
    df = pd.read_csv('test/test_data.csv')
    store = [df.iloc[idx] for idx in range(len(df))]
    cif = [i['cif'] for i in store]
    formation = [i['formation_energy_per_atom'] for i in store]
    
    def near_property(poscar, cutoff):
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
        crystal = Structure.from_str(poscar, fmt='cif')
        all_nbrs = crystal.get_all_neighbors(cutoff)
        all_nbrs = [sorted(nbrs, key = lambda x: x[1]) for nbrs in all_nbrs]
        num_near = min(map(lambda x: len(x), all_nbrs))
        nbr_idx, nbr_dis = [], []
        for nbr in all_nbrs:
            nbr_idx.append(list(map(lambda x: x[2], nbr[:num_near])))
            nbr_dis.append(list(map(lambda x: x[1], nbr[:num_near])))
        nbr_idx, nbr_dis = list(nbr_idx), list(nbr_dis)
        return nbr_idx, nbr_dis
    
    checkpoint = torch.load('test/model_best.pth.tar', map_location='cpu')
    model = FineTuneNet(orig_atom_fea_len, nbr_bond_fea_len)
    model.load_state_dict(checkpoint['state_dict'])
    for name, module in model._modules.items():
        for p in module.parameters():
            print(p.requires_grad)