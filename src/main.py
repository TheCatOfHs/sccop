import os, time
import numpy as np

from core.global_var import *
from core.dir_path import *
from core.initialize import InitSampling, UpdateNodes
from core.sample_select import Select, BayesianOpt
from core.sub_vasp import ParallelSubVASP, VASPoptimize
from core.search import ParallelWorkers
from core.predict import PPMData, PPModel
from core.utils import ListRWTools, system_echo
from core.data_transfer import MultiGridTransfer


class CrystalOptimization(ListRWTools):
    #crystal combinational optimization program
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        self.init = InitSampling()
        self.cpu_nodes = UpdateNodes()
        self.transfer = MultiGridTransfer()
        self.bayes = BayesianOpt()
        self.workers = ParallelWorkers()
        self.vasp = ParallelSubVASP()
        
    def main(self):
        #Update cpu nodes
        self.cpu_nodes.update()
        #Initialize storage
        train_pos, train_type, train_symm, train_grid, train_ratio, train_sg = [], [], [], [], [], []
        train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energy = [], [], [], []
        
        start = 0
        for recycle in range(num_recycle):
            system_echo(f'Begin Crystal Combinatorial Optimization Program --- Recycle: {recycle}')
            #Generate structures
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = self.init.generate(recycle)
            system_echo('New initial samples generated')
            
            #Write POSCARs
            select = Select(start)
            select.write_POSCARs(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group)
            #Energy calculation
            self.vasp.sub_job(start, vdW=add_vdW)
            
            #CCOP optimize
            num_round = num_ml_list[recycle]
            for round in range(start, start+num_round):
                #Data import
                atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, energy = \
                    self.import_sampling_data(round)
                #Check number of Training data
                add_num, crys_num = self.check_num_train_data(energy, train_energy)
                system_echo(f'Training set: {crys_num}  New add to training set: {add_num}')
                #Train predict model
                train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energy = \
                    self.train_predict_model(round, add_num, atom_pos, atom_type, atom_symm,
                                             grid_name, grid_ratio, space_group, energy, train_atom_fea, 
                                             train_nbr_fea, train_nbr_fea_idx, train_energy)
                #Update training set
                train_pos += atom_pos
                train_type += atom_type
                train_symm += atom_symm
                train_grid += grid_name
                train_ratio += grid_ratio
                train_sg += space_group
                
                #Initialize start points
                init_pos, init_type, init_symm, init_grid, init_ratio, init_sg = \
                    self.bayes.select(recycle, train_pos, train_type, train_symm,
                                      train_grid, train_ratio, train_sg, train_energy)
                #Search on grid
                self.workers.search(round+1, init_pos, init_type, init_symm, init_grid, init_ratio, init_sg)
                
                #Select samples
                file_head = f'{search_path}/{round+1:03.0f}'
                atom_pos = self.import_list2d(f'{file_head}/atom_pos.dat', int)
                atom_type = self.import_list2d(f'{file_head}/atom_type.dat', int)
                atom_symm = self.import_list2d(f'{file_head}/atom_symm.dat', int)
                grid_name = self.import_list2d(f'{file_head}/grid_name.dat', int)
                grid_ratio = self.import_list2d(f'{file_head}/grid_ratio.dat', float)
                space_group = self.import_list2d(f'{file_head}/space_group.dat', int)
                select = Select(round+1)
                select.samples(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group,
                               train_pos, train_type, train_grid, train_ratio, train_sg)
                #Energy calculation
                self.vasp.sub_job(round+1, vdW=add_vdW)
            
            #Update training set
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, energy = \
                self.import_sampling_data(start+num_round)
            atom_fea, nbr_fea, nbr_fea_idx = \
            self.transfer.get_ppm_input_bh(atom_pos, atom_type, grid_name, grid_ratio, space_group)
            
            train_atom_fea += atom_fea
            train_nbr_fea += nbr_fea
            train_nbr_fea_idx += nbr_fea_idx
            train_energy += energy
            train_pos += atom_pos
            train_type += atom_type
            train_symm += atom_symm
            train_grid += grid_name
            train_ratio += grid_ratio
            train_sg += space_group
            
            #Select searched POSCARs
            select = Select(start+num_round)
            select.export_recycle(recycle, train_pos, train_type, train_symm, train_grid, train_ratio, train_sg, train_energy)
            system_echo(f'End Crystal Combinatorial Optimization Program --- Recycle: {recycle}')
            #VASP optimize
            vasp = VASPoptimize(recycle)
            vasp.run_optimization_low(vdW=add_vdW)
            start += num_round + 1  
        #Select optimized structures
        select.optim_strus()
        #Optimize
        vasp.run_optimization_high(vdW=add_vdW)
    
    def import_sampling_data(self, round):
        """
        import sampling data from ccop round
        
        Parameters
        ----------
        round [int, 0d]: ccop round

        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        energy [float, 1d]: energy of samples
        """
        #select reasonable energy
        file = f'{vasp_out_path}/Energy-{round:03.0f}.dat'
        energy_file = self.import_list2d(file, str, numpy=True)
        mask = np.array(energy_file)[:, 1]
        mask = [True if i=='True' else False for i in mask]
        energys = np.array(energy_file[:, 2], dtype=float)
        #use std to filter high energy structures
        std = np.std(energys[mask])
        mean = np.mean(energys[mask])
        for i, energy in enumerate(energys):
            if energy - mean > std:
                mask[i] = False
        energys = energys[mask].tolist()
        #import sampling data
        head = f'{search_path}/{round:03.0f}'
        atom_pos = self.import_list2d(f'{head}/atom_pos_select.dat', int)
        atom_type = self.import_list2d(f'{head}/atom_type_select.dat', int)
        atom_symm = self.import_list2d(f'{head}/atom_symm_select.dat', int)
        grid_name = self.import_list2d(f'{head}/grid_name_select.dat', int)
        grid_ratio = self.import_list2d(f'{head}/grid_ratio_select.dat', float)
        space_group = self.import_list2d(f'{head}/space_group_select.dat', int)
        grid_name = np.ravel(grid_name)
        grid_ratio = np.ravel(grid_ratio)
        space_group = np.ravel(space_group)
        #filter samples
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.init.filter_samples(mask, atom_pos, atom_type, atom_symm,
                                     grid_name, grid_ratio, space_group)
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, energys

    def check_num_train_data(self, energy, train_energy):
        """
        check number of train data assigned to gpus    

        Parameters
        ----------
        energy [float, 1d]: select energy last round
        train_energy [float, 1d]: energy in train data

        Returns
        ----------
        add_num [int, 0d]: number added to train data
        crys_num [int, 0d]: number of added train data
        """
        poscar_num = len(energy)
        add_num = int(poscar_num*0.8)
        crys_num = add_num + len(train_energy)
        last_batch_num = np.mod(crys_num, train_batchsize)
        if 0 < last_batch_num < num_gpus:
            add_num -= last_batch_num
        return add_num, crys_num

    def train_predict_model(self, round, add_num, atom_pos, atom_type, atom_symm,
                            grid_name, grid_ratio, space_group, energy, train_atom_fea, 
                            train_nbr_fea, train_nbr_fea_idx, train_energy):
        """
        train CGCNN as predict model
        
        Parameters
        ----------
        round [int, 0d]: seaching round
        add_num [int, 0d]: number added to training set
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        energy_select [float, 1d]: select energy last round
        train_atom_fea [float, 3d]: atom feature
        train_nbr_fea [float, 4d]: bond feature 
        train_nbr_fea_idx [int, 3d]: neighbor index
        train_energy [float, 1d]: train energy
        
        Returns
        ----------
        train_atom_fea [float, 3d]: atom feature
        train_nbr_fea [float, 4d]: bond feature
        train_nbr_fea_idx [int, 3d]: neighbor index
        train_energy [float, 1d]: train energy
        """
        #samples from different space groups
        idx = self.init.balance_sampling(len(space_group), space_group)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.init.filter_samples(idx, atom_pos, atom_type, atom_symm,
                                     grid_name, grid_ratio, space_group)
        #training data
        atom_fea, nbr_fea, nbr_fea_idx = \
            self.transfer.get_ppm_input_bh(atom_pos, atom_type,
                                           grid_name, grid_ratio, space_group)
        train_atom_fea += atom_fea[:add_num]
        train_nbr_fea += nbr_fea[:add_num]
        train_nbr_fea_idx += nbr_fea_idx[:add_num]
        train_energy += energy[:add_num]
        #validation data
        valid_atom_fea = atom_fea[add_num:]
        valid_nbr_fea = nbr_fea[add_num:]
        valid_nbr_fea_idx = nbr_fea_idx[add_num:]
        valid_energy = energy[add_num:]
        #training model
        train_data = PPMData(train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energy)
        valid_data = PPMData(valid_atom_fea, valid_nbr_fea, valid_nbr_fea_idx, valid_energy)
        ppm = PPModel(round+1, train_data, valid_data, valid_data)
        ppm.train_epochs()
        #train data added
        train_atom_fea += atom_fea[add_num:]
        train_nbr_fea += nbr_fea[add_num:]
        train_nbr_fea_idx += nbr_fea_idx[add_num:]
        train_energy += energy[add_num:]
        return train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energy
        

if __name__ == '__main__':
    start_time = time.time()
    ccop = CrystalOptimization()
    ccop.main()
    end_time = time.time()
    system_echo(f'Time: {end_time - start_time}')