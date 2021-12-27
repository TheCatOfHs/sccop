import os, sys
import time

sys.path.append(f'{os.getcwd()}/src')
from modules.global_var import *
from modules.utils import SSHTools, system_echo


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


if __name__ == '__main__':
    from pymatgen.core.structure import Structure
    import pandas as pd
    df = pd.read_csv('test/test_data.csv')
    a = [df.iloc[idx] for idx in range(len(df))]
    cif = [i['cif'] for i in a]
    formation = [i['formation_energy_per_atom'] for i in a]
    
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
    
    print(near_property(cif[0], 8))