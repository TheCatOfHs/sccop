import os
import numpy as np

from core.global_var import *
from core.dir_path import *
from core.initialize import InitSampling, UpdateNodes
from core.grid_divide import ParallelDivide
from core.sample_select import Select
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
        self.divide = ParallelDivide()
        self.workers = ParallelWorkers()
        self.vasp = ParallelSubVASP()
    
    def main(self):
        #Update cpu nodes
        self.cpu_nodes.update()
        #Initialize storage
        train_pos, train_type, train_symm, train_grid, train_sg = [], [], [], [], []
        train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energy = [], [], [], []
        
        start = 0
        for recycle in range(num_recycle):
            system_echo(f'Begin Crystal Combinatorial Optimization Program --- Recycle: {recycle}')
            #Generate structures
            atom_pos, atom_type, atom_symm, grid_name, space_group = self.init.generate(recycle)
            system_echo('New initial samples generated')
            
            #Write POSCARs
            select = Select(start)
            select.write_POSCARs(atom_pos, atom_type, atom_symm, grid_name, space_group)
            #Energy calculation
            self.vasp.sub_job(start, vdW=add_vdW)
            
            #CCOP optimize
            num_round = num_ml_list[recycle]
            for round in range(start, start+num_round):
                #Data import
                atom_pos, atom_type, atom_symm, grid_name, space_group, energy = \
                    self.import_sampling_data(round)
                #Check number of Training data
                add_num, crys_num = self.check_num_train_data(energy, train_energy)
                system_echo(f'Training set: {crys_num}  New add to training set: {add_num}')
                #Train predict model
                train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energy = \
                    self.train_predict_model(round, add_num, atom_pos, atom_type,
                                             grid_name, space_group, energy, train_atom_fea, 
                                             train_nbr_fea, train_nbr_fea_idx, train_energy)
                #Update training set
                train_pos += atom_pos
                train_type += atom_type
                train_symm += atom_symm
                train_grid += grid_name
                train_sg += space_group
                
                #Initialize start points
                init_pos, init_type, init_symm, init_grid, init_sg = \
                    self.generate_search_point(train_pos, train_type, train_symm,
                                               train_grid, train_sg, train_energy)
                #Search on grid
                self.workers.search(round+1, init_pos, init_type, init_symm, init_grid, init_sg)
                
                #Select samples
                file_head = f'{search_path}/{round+1:03.0f}'
                atom_pos = self.import_list2d(f'{file_head}/atom_pos.dat', int)
                atom_type = self.import_list2d(f'{file_head}/atom_type.dat', int)
                atom_symm = self.import_list2d(f'{file_head}/atom_symm.dat', int)
                grid_name = self.import_list2d(f'{file_head}/grid_name.dat', int)
                space_group = self.import_list2d(f'{file_head}/space_group.dat', int)
                select = Select(round+1)
                select.samples(atom_pos, atom_type, atom_symm, grid_name, space_group,
                               train_pos, train_type, train_symm, train_grid, train_sg)
                #Energy calculation
                self.vasp.sub_job(round+1, vdW=add_vdW)
            
            #Update training set
            atom_pos, atom_type, atom_symm, grid_name, space_group, energy = \
                self.import_sampling_data(start+num_round)
            atom_fea, nbr_fea, nbr_fea_idx = \
            self.transfer.get_ppm_input_bh(atom_pos, atom_type, grid_name, space_group)
            
            train_atom_fea += atom_fea
            train_nbr_fea += nbr_fea
            train_nbr_fea_idx += nbr_fea_idx
            train_energy += energy
            train_pos += atom_pos
            train_type += atom_type
            train_symm += atom_symm
            train_grid += grid_name
            train_sg += space_group
            
            #Select searched POSCARs
            select = Select(start+num_round)
            select.export_recycle(recycle, train_pos, train_type, train_symm, train_grid, train_sg, train_energy)
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
        space_group [int, 1d]: space group number
        energy [float, 1d]: energy of samples
        """
        #select reasonable energy
        file = f'{vasp_out_path}/Energy-{round:03.0f}.dat'
        energy_file = self.import_list2d(file, str, numpy=True)
        mask = np.array(energy_file)[:, 1]
        mask = [True if i=='True' else False for i in mask]
        energy = np.array(energy_file[:, 2], dtype=float)[mask].tolist()
        #import sampling data
        head = f'{search_path}/{round:03.0f}'
        atom_pos = self.import_list2d(f'{head}/atom_pos_select.dat', int)
        atom_type = self.import_list2d(f'{head}/atom_type_select.dat', int)
        atom_symm = self.import_list2d(f'{head}/atom_symm_select.dat', int)
        grid_name = self.import_list2d(f'{head}/grid_name_select.dat', int)
        space_group = self.import_list2d(f'{head}/space_group_select.dat', int)
        grid_name = np.ravel(grid_name)
        space_group = np.ravel(space_group)
        #filter samples
        atom_pos, atom_type, atom_symm, grid_name, space_group = \
            self.init.filter_samples(mask, atom_pos, atom_type, atom_symm,
                                     grid_name, space_group)
        return atom_pos, atom_type, atom_symm, grid_name, space_group, energy

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

    def train_predict_model(self, round, add_num, atom_pos, atom_type,
                            grid_name, space_group, energy, train_atom_fea, 
                            train_nbr_fea, train_nbr_fea_idx, train_energy):
        """
        train CGCNN as predict model
        
        Parameters
        ----------
        round [int, 0d]: seaching round
        add_num [int, 0d]: number added to training set
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
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
        #training data
        atom_fea, nbr_fea, nbr_fea_idx = \
            self.transfer.get_ppm_input_bh(atom_pos, atom_type,
                                           grid_name, space_group)
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
    
    def generate_search_point(self, train_pos, train_type, train_symm,
                              train_grid, train_sg, train_energy):
        """
        generate initial searching points
        
        Parameters
        ----------
        train_pos [int, 2d]: position of training set
        train_type [int, 2d]: type of training set
        train_symm [int, 2d]: symmetry of training set
        train_grid [int, 1d]: grid of training set
        train_sg [int, 1d]: space group of training set
        train_energy [float, 1d]: energy of training set
        
        Returns
        ----------
        init_pos [int, 2d]: position of initial points
        init_type [int, 2d]: type of initial points
        init_symm [int, 2d]: symmetry of initial points
        init_grid [int, 1d]: grid of initial points
        init_sg [int, 1d]: space group of initial points
        """
        #initialize
        init_pos, init_type, init_symm, init_grid, init_sg = [], [], [], [], []
        #greedy path
        min_idx = np.argsort(train_energy)[:num_seed]
        greed_pos, greed_type, greed_symm, greed_grid, greed_sg =\
            self.point_sampling(num_path_min, min_idx, train_pos, train_type,
                               train_symm, train_grid, train_sg)
        #random path
        all_idx = np.arange(len(train_energy))
        rand_pos, rand_type, rand_symm, rand_grid, rand_sg =\
            self.point_sampling(num_path_rand, all_idx, train_pos, train_type,
                               train_symm, train_grid, train_sg)
        init_pos += greed_pos + rand_pos
        init_type += greed_type + rand_type
        init_symm += greed_symm + rand_symm
        init_grid += greed_grid + rand_grid
        init_sg += greed_sg + rand_sg
        return init_pos, init_type, init_symm, init_grid, init_sg

    def point_sampling(self, num, idx, train_pos, train_type, train_symm,
                      train_grid, train_sg):
        """
        sampling initial points
        """
        init_pos, init_type, init_symm, init_grid, init_sg = [], [], [], [], []
        for _ in range(num):
            seed = np.random.choice(idx)
            init_pos.append(train_pos[seed])
            init_type.append(train_type[seed])
            init_symm.append(train_symm[seed])
            init_grid.append(train_grid[seed])
            init_sg.append(train_sg[seed])
        return init_pos, init_type, init_symm, init_grid, init_sg
    

if __name__ == '__main__':
    ccop = CrystalOptimization()
    ccop.main()