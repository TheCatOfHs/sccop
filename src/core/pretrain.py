import os, sys
import numpy as np
import pandas as pd

from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.path import *
from core.utils import *
from core.predict import PPMData, PPModel
from core.data_transfer import Transfer


class Pretrain(Transfer, ListRWTools):
    #Pretrain property predict model
    def __init__(self, cutoff=12, nbr=12, dmin=0, dmax=8, step=0.2, var=0.2):
        self.cutoff = cutoff
        self.nbr, self.var = nbr, var
        self.filter = np.arange(dmin, dmax+step, step)
    
    def pretrain(self):
        path = 'database/CIF2D'
        data = pd.read_csv(f'{path}/id_prop.csv', header=None).values
        #divide into train, valid, test
        data_num = len(data)
        index = np.arange(data_num)
        train_index = index[:int(data_num*0.6)]
        valid_index = index[int(data_num*0.6):int(data_num*0.8)]
        test_index = index[int(data_num*0.8):]
        #get name of cifs
        train_cifs, train_energys = np.transpose(data[train_index])
        valid_cifs, valid_energys = np.transpose(data[valid_index])
        test_cifs, test_energys = np.transpose(data[test_index])
        train_cifs = [f'{path}/{i}.cif' for i in train_cifs]
        valid_cifs = [f'{path}/{i}.cif' for i in valid_cifs]
        test_cifs = [f'{path}/{i}.cif' for i in test_cifs]
        #transfer data to input of model
        train_atom_fea, train_nbr_fea, train_nbr_fea_idx = self.batch(train_cifs)
        valid_atom_fea, valid_nbr_fea, valid_nbr_fea_idx = self.batch(valid_cifs)
        test_atom_fea, test_nbr_fea, test_nbr_fea_idx = self.batch(test_cifs)
        #training prediction model
        train_data = PPMData(train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energys)
        valid_data = PPMData(valid_atom_fea, valid_nbr_fea, valid_nbr_fea_idx, valid_energys)
        test_data = PPMData(test_atom_fea, test_nbr_fea, test_nbr_fea_idx, test_energys)
        ppm = PPModel(100, train_data, valid_data, test_data)
        ppm.train_epochs()
    
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
        crystal = Structure.from_file(cif)
        atom_type = np.array(crystal.atomic_numbers)
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
        elem_embed = self.import_list2d(atom_init_file, int, numpy=True)
        atom_fea = self.get_atom_fea(atom_type, elem_embed)
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
    '''
    ssh = SSHTools()
    cifs = os.listdir('database/CIF2D')
    node_assign = ssh.assign_node(len(cifs), order=False)
    for i, cif in enumerate(cifs):
        stru = Structure.from_file(f'database/CIF2D/{cif}')
        stru.to(filename=f'data/poscar/000/POSCAR-{cif[:-4]}-{node_assign[i]}', fmt='poscar')
    '''
    #cpu_nodes = UpdateNodes()
    #cpu_nodes.update()
    #from core.sub_vasp import ParallelSubVASP
    #vasp = ParallelSubVASP()
    #vasp.sub_job(0, vdW=True)
    #vasp.get_energy(0, 0)
    '''
    rwtools = ListRWTools()
    ct = rwtools.import_list2d('data/vasp_out/Energy-0.dat', str)
    names, energys = [], []
    for i in ct:
        if i[1] == 'True':
            name = i[0].split('-')
            label = name[1] + '-' + name[2]
            names.append(label)
            energys.append(i[2])
    
    
    import shutil
    for i in names:
        file_1 = f'database/CIF2D/{i}.cif'
        file_2 = f'database/CIF2D_vdW/{i}.cif'
        shutil.copy(file_1, file_2)
    
    df = pd.DataFrame({'cif':names, 'E':energys})
    df.to_csv('id_prop.csv',index =False ,sep = ',')
    '''
    model = Pretrain()
    model.pretrain()
    #print(model.import_list2d('test.txt', int))
    