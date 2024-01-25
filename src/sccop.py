import os, shutil
import argparse
import numpy as np

from core.log_print import *
from core.init_sampling import InitSampling, UpdateNodes
from core.sample_select import Select
from core.sub_vasp import ParallelSubVasp, VaspOpt
from core.sub_lammps import ParallelSubLammps, LammpsOpt
from core.multi_SA import ParallelWorkers
from core.GNN_model import GNNData, GNNTrain, batch_balance, get_loader
from core.GNN_tool import ParallelPESUpdate, GNNSlice
from core.utils import ListRWTools, system_echo
from core.data_transfer import MultiGridTransfer
import torch


class CrystalOptimization(ListRWTools, UpdateNodes):
    #crystal combinational optimization program
    def __init__(self):
        UpdateNodes.__init__(self)
        self.init = InitSampling()
        self.transfer = MultiGridTransfer()
        self.PES = ParallelPESUpdate()
        self.slice = GNNSlice()
        self.workers = ParallelWorkers()
        if Energy_Method == 'VASP':
            self.solver = ParallelSubVasp()
        elif Energy_Method == 'LAMMPS':
            self.solver = ParallelSubLammps()
        
    def main(self, recycle):
        #Import data
        if recycle == 0:
            #Pretrain model
            if os.path.exists(Pretrain_Save):
                system_echo('Pretrain GNN start')
                self.pretrain_gnn_model()
                system_echo('Pretrain GNN finish')
            #Initialize list
            start = 0
            convergence, stop = False, False
            train_pos, train_type, train_symm, train_grid, train_ratio, train_sg, train_angles, train_thicks, train_energy = [], [], [], [], [], [], [], [], []
        else:
            if os.path.exists(f'{Optim_Strus_Path}'):
                stop = True
                convergence = True
            else:
                stop = False
                start, convergence, train_pos, train_type, train_symm, train_grid, train_ratio, train_sg, train_angles, train_thicks, train_energy = self.import_recyc_data(recycle)
        
        #End criterion
        if recycle == Num_Recycle or convergence or stop:
            if stop:
                pass
            else:
                select = Select(start)
                if Energy_Method == 'VASP':
                    solver = VaspOpt(recycle)
                elif Energy_Method == 'LAMMPS':
                    solver = LammpsOpt(recycle)
                #Select optimized structures
                select.optim_strus()
                #Optimize
                solver.run_optimization_high()
        else:
            system_echo(f'Begin Symmetry Combinatorial Crystal Optimization Program --- Recycle: {recycle}')
            #Generate structures
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energy = self.init.generate(recycle)
            system_echo('Initial samples generated')
            
            #Write POSCARs
            select = Select(start)
            select.write_POSCARs(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energy)
            #Energy calculation
            if Use_VASP_Scf or Use_LAMMPS_Scf:
                self.solver.sub_job(start)
            else:
                self.write_ml_energy(start, energy)
            
            #SCCOP optimize
            num_iteration = Num_ML_Iter[recycle]
            for iteration in range(start, start+num_iteration):
                #Data import
                atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energy = \
                    self.import_sampling_data(iteration)
                #Train gnn model
                if Use_VASP_Scf or Use_LAMMPS_Scf:
                    train_pos, train_type, train_symm, train_grid, train_ratio, train_sg, train_angles, train_thicks, train_energy = \
                        self.train_gnn_model(iteration, atom_pos, atom_type, atom_symm,
                                             grid_name, grid_ratio, space_group, angles, thicks, energy, 
                                             train_pos, train_type, train_symm, train_grid,
                                             train_ratio, train_sg, train_angles, train_thicks, train_energy)
                else:
                    self.copy_ml_model(iteration+1)
                    self.update_dataset(atom_pos, atom_type, atom_symm,
                                        grid_name, grid_ratio, space_group, angles, thicks, energy,
                                        train_pos, train_type, train_symm, train_grid,
                                        train_ratio, train_sg, train_angles, train_thicks, train_energy)
                #Update PES
                atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energy, crys_vec = self.PES.sub_update_jobs(iteration+1)
                init_pos, init_type, init_symm, init_grid, init_ratio, init_sg, init_angles, init_thicks = \
                    self.slice.candidate_select(atom_pos, atom_type, atom_symm, grid_name, grid_ratio,
                                                space_group, angles, thicks, energy, crys_vec,
                                                train_pos, train_type, train_symm, train_grid, train_ratio,
                                                train_sg, train_angles, train_thicks, train_energy,
                                                self.work_nodes_num)
                #Multi-start ML-SA
                self.workers.search(iteration+1, init_pos, init_type, init_symm, init_grid, init_ratio, init_sg, init_angles, init_thicks)
                
                #Select samples
                select.export_train_set(train_pos, train_type, train_symm, train_grid, train_ratio,
                                        train_sg, train_angles, train_thicks, train_energy)
                os.system(f'python src/core/sample_select.py --iteration {iteration+1}')
                #Energy calculation
                if Use_VASP_Scf or Use_LAMMPS_Scf:
                    self.solver.sub_job(iteration+1)
                else:
                    energy = self.import_ml_energy(iteration+1)
                    self.write_ml_energy(iteration+1, energy)
            
            #Update training set
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energy = \
                self.import_sampling_data(start+num_iteration)
            self.update_dataset(atom_pos, atom_type, atom_symm,
                                grid_name, grid_ratio, space_group, angles, thicks, energy,
                                train_pos, train_type, train_symm, train_grid,
                                train_ratio, train_sg, train_angles, train_thicks, train_energy)
            #Select searched POSCARs
            select = Select(start+num_iteration)
            select.export_recycle(recycle, train_pos, train_type, train_symm,
                                  train_grid, train_ratio, train_sg, train_angles, train_thicks, train_energy)
            system_echo(f'End Symmetry Combinatorial Crystal Optimization Program --- Recycle: {recycle}')
            #structure optimization
            if Energy_Method == 'VASP':
                solver = VaspOpt(recycle)
            elif Energy_Method == 'LAMMPS':
                solver = LammpsOpt(recycle)
            solver.run_optimization_low()
            start += num_iteration + 1  
            #Energy convergence
            convergence = 0
            if recycle >= 1:
                convergence = select.judge_convergence(recycle)
            
            #Save data
            self.export_recyc_data(recycle, start, convergence, train_pos, train_type, train_symm, train_grid, train_ratio, train_sg, 
                                   train_angles, train_thicks, train_energy)
            
    def import_sampling_data(self, iteration):
        """
        import sampling data from sccop iteration
        
        Parameters
        ----------
        iteration [int, 0d]: sccop iteration

        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energy [float, 1d]: energy of samples
        """
        #select reasonable energy
        if Energy_Method == 'VASP':
            file = f'{VASP_Out_Path}/Energy-{iteration:02.0f}.dat'
        elif Energy_Method == 'LAMMPS':
            file = f'{LAMMPS_Out_Path}/Energy-{iteration:02.0f}.dat'
        if os.path.exists(file):
            energy_file = self.import_list2d(file, str, numpy=True)
            mask = np.array(energy_file)[:, 1]
            mask = [True if i=='True' else False for i in mask]
            energys = np.array(energy_file[:, 2], dtype=float)
            #use std to filter high energy structures
            std = np.std(energys[mask])
            mean = np.mean(energys[mask])
            for i, energy in enumerate(energys):
                if energy - mean > 3*std:
                    mask[i] = False
            energys = energys[mask].tolist()
            #import sampling data
            head = f'{Search_Path}/ml_{iteration:02.0f}'
            atom_pos = self.import_list2d(f'{head}/atom_pos_select.dat', int)
            atom_type = self.import_list2d(f'{head}/atom_type_select.dat', int)
            atom_symm = self.import_list2d(f'{head}/atom_symm_select.dat', int)
            grid_name = self.import_list2d(f'{head}/grid_name_select.dat', int)
            grid_ratio = self.import_list2d(f'{head}/grid_ratio_select.dat', float)
            space_group = self.import_list2d(f'{head}/space_group_select.dat', int)
            angles = self.import_list2d(f'{head}/angles_select.dat', int)
            thicks = self.import_list2d(f'{head}/thicks_select.dat', int)
            grid_name = np.concatenate(grid_name)
            grid_ratio = np.concatenate(grid_ratio)
            space_group = np.concatenate(space_group)
            #filter samples
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
                self.init.filter_samples(mask, atom_pos, atom_type, atom_symm,
                                        grid_name, grid_ratio, space_group, angles, thicks)
        else:
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys = \
                [], [], [], [], [], [], [], [], []
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys
    
    def import_recyc_data(self, recyc):
        """
        import recycling save data
        
        Parameters
        ----------
        recyc [int, 0d]: sccop recycle number

        Returns
        ----------
        start [int, 0d]: start search number
        convergence [bool, 0d]: energy convergence
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        energys [float, 1d]: energy of samples
        """
        head = f'{SCCOP_Path}/{Save_Path}/{recyc-1:02.0f}'
        start, convergence = self.import_list2d(f'{head}/record.dat', int)[0]
        convergence = [True if convergence == 1 else False][0]
        #import sampling data
        train_pos = self.import_list2d(f'{head}/train_pos.dat', int)
        train_type = self.import_list2d(f'{head}/train_type.dat', int)
        train_symm = self.import_list2d(f'{head}/train_symm.dat', int)
        train_grid = self.import_list2d(f'{head}/train_grid.dat', int)[0]
        train_ratio = self.import_list2d(f'{head}/train_ratio.dat', float)[0]
        train_sg = self.import_list2d(f'{head}/train_sg.dat', int)[0]
        train_angles = self.import_list2d(f'{head}/train_angles.dat', int)
        train_thicks = self.import_list2d(f'{head}/train_thicks.dat', int)
        train_energy = self.import_list2d(f'{head}/train_energy.dat', float)[0]
        return start, convergence, train_pos, train_type, train_symm, train_grid, train_ratio, train_sg, train_angles, train_thicks, train_energy
    
    def export_recyc_data(self, recyc, start, convergence,
                          train_pos, train_type, train_symm, train_grid,
                          train_ratio, train_sg, train_angles, train_thicks, train_energy):
        """
        export recycling save data
        
        Parameters
        ----------
        recyc [int, 0d]: sccop recycle number
        start [int, 0d]: start search number
        convergence [int, 0d]: energy convergence
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: energy of samples
        """
        #export sampling data
        head = f'{SCCOP_Path}/{Save_Path}/{recyc:02.0f}'
        os.mkdir(head)
        self.write_list2d(f'{head}/record.dat', [[start, convergence]])
        self.write_list2d(f'{head}/train_pos.dat', train_pos)
        self.write_list2d(f'{head}/train_type.dat', train_type)
        self.write_list2d(f'{head}/train_symm.dat', train_symm)
        self.write_list2d(f'{head}/train_grid.dat', [train_grid])
        self.write_list2d(f'{head}/train_ratio.dat', [train_ratio])
        self.write_list2d(f'{head}/train_sg.dat', [train_sg])
        self.write_list2d(f'{head}/train_angles.dat', train_angles)
        self.write_list2d(f'{head}/train_thicks.dat', train_thicks)
        self.write_list2d(f'{head}/train_energy.dat', [train_energy])
    
    def get_drop_num_samples(self, sample_num, batchsize):
        """
        drop number of samples
        
        Parameters
        ----------
        sample_num [int, 0d]: number of samples
        batchsize [int, 0d]: batchsize of training

        Returns
        ----------
        drop_num [int, 0d]: drop number
        """
        left = np.mod(sample_num, batchsize)
        drop_num = 0
        if Job_Queue == 'GPU':
            if 0 < left < Num_GPUs:
                drop_num = left
            else:
                drop_num = 0
        return drop_num
    
    def pretrain_gnn_model(self, batchsize=16, epochs=60, lr_factor=1):
        """
        pretrain gnn model
        """
        #unzip file
        shell = f'''
                #!/bin/bash --login
                cd {SCCOP_Path}/{Pretrain_Save}
                tar -zmxf *
                '''
        os.system(shell)
        #import file
        ct = self.import_list2d(f'{Pretrain_Save}/Energy.dat', str)
        poscars, energys = np.transpose(ct)
        energys = np.array(energys, dtype=float)
        sample_num = len(energys)
        #get gnn input from data
        poscars_train = []
        for poscar in poscars:
            poscars_train.append(f'{Pretrain_Save}/{poscar}')
        atom_fea, nbr_fea, nbr_fea_idx = self.transfer.get_gnn_input_from_stru_batch_parallel(poscars_train, type='POSCAR')
        atom_symm = [[1 for _ in range(len(i))] for i in atom_fea]
        #shuffle data
        train_num = int(.8*sample_num)
        valid_num = sample_num - train_num
        train_drop_num = self.get_drop_num_samples(train_num, batchsize)
        valid_drop_num = self.get_drop_num_samples(valid_num, batchsize)
        train_num = train_num - train_drop_num
        valid_num = valid_num - valid_drop_num
        idx = np.arange(sample_num)
        np.random.shuffle(idx)
        train_idx = idx[:train_num]
        valid_idx = idx[-valid_num:]
        system_echo(f'Training set: {len(train_idx)}  Validation set: {len(valid_idx)}')
        #train data
        train_atom_fea = np.array(atom_fea, dtype=object)[train_idx].tolist()
        train_symm = np.array(atom_symm, dtype=object)[train_idx].tolist()
        train_nbr_fea = np.array(nbr_fea, dtype=object)[train_idx].tolist()
        train_nbr_fea_idx = np.array(nbr_fea_idx, dtype=object)[train_idx].tolist()
        train_energy = np.array(energys, dtype=object)[train_idx].tolist()
        #validation data
        valid_atom_fea = np.array(atom_fea, dtype=object)[valid_idx].tolist()
        valid_symm = np.array(atom_symm, dtype=object)[valid_idx].tolist()
        valid_nbr_fea = np.array(nbr_fea, dtype=object)[valid_idx].tolist()
        valid_nbr_fea_idx = np.array(nbr_fea_idx, dtype=object)[valid_idx].tolist()
        valid_energy = np.array(energys, dtype=object)[valid_idx].tolist()
        #train model
        train_data = GNNData(train_atom_fea, train_symm, train_nbr_fea,
                             train_nbr_fea_idx, train_energy)
        valid_data = GNNData(valid_atom_fea, valid_symm, valid_nbr_fea,
                             valid_nbr_fea_idx, valid_energy)
        gnn = GNNTrain(0, train_data, valid_data, valid_data, 
                            train_batchsize=batchsize, train_epochs=epochs,
                            lr_factor=lr_factor)
        gnn.train_epochs()
        #copy model
        if Dimension == 2:
            shutil.copyfile(f'{Model_Path}/00/model_best.pth.tar', Pretrain_Model_2d)
        elif Dimension == 3:
            shutil.copyfile(f'{Model_Path}/00/model_best.pth.tar', Pretrain_Model_3d)
        shutil.rmtree(f'{Model_Path}/00')
        #export crystal vectors
        atom_fea = train_atom_fea + valid_atom_fea
        atom_symm = train_symm + valid_symm
        nbr_fea = train_nbr_fea + valid_nbr_fea
        nbr_fea_idx = train_nbr_fea_idx + valid_nbr_fea_idx
        energys = train_energy + valid_energy
        dataset = GNNData(atom_fea, atom_symm, nbr_fea, nbr_fea_idx, energys)
        loader = get_loader(dataset, batchsize, 0)
        select = Select(0)
        model = select.get_gnn_model()
        select.load_fea_model(model)
        select.load_out_model(model)
        select.load_normalizer(model)
        crys_vec_np = select.get_crystal_vector_batch(loader).cpu().numpy()
        self.write_list2d(f'{Pretrain_Path}/models/Energy.dat', [energys])
        self.write_list2d(f'{Pretrain_Path}/models/crystal_vectors.bin', crys_vec_np, binary=True)
        #clean memory
        del atom_fea, nbr_fea, nbr_fea_idx
        del train_atom_fea, train_symm, train_nbr_fea, train_nbr_fea_idx, train_energy
        del valid_atom_fea, valid_symm, valid_nbr_fea, valid_nbr_fea_idx, valid_energy
        del train_data, valid_data, gnn
        #release CUDA
        if Job_Queue == 'GPU':
            torch.cuda.empty_cache()
        
    def train_gnn_model(self, iteration, atom_pos, atom_type, atom_symm,
                        grid_name, grid_ratio, space_group, angles, thicks, energy,
                        train_pos, train_type, train_symm, train_grid,
                        train_ratio, train_sg, train_angles, train_thicks, train_energy,
                        batchsize=16, train_valid_ratio=.6):
        """
        train gnn model
        
        Parameters
        ----------
        iteration [int, 0d]: search iteration
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energy [float, 1d]: select energy last iteration
        train_pos [int, 2d]: position of atoms
        train_type [int, 2d]: type of atoms
        train_symm [int, 2d]: symmetry of atoms
        train_grid [int, 1d]: name of grids
        train_ratio [float, 1d]: ratio of grids
        train_sg [int, 1d]: space group number
        train_angles [int, 2d]: cluster rotation angles
        train_thicks [int, 2d]: atom displacement in z-direction
        train_energy [float, 1d]: train energy
        batchsize [int, 0d]: training batchsize
        train_valid_ratio [float, 0d]: ratio of trainset and validation
        """
        if len(grid_name) > 0:
            #sorted by grid and space group for new data
            idx = self.transfer.sort_by_grid_sg(grid_name, space_group)
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
                self.init.filter_samples(idx, atom_pos, atom_type, atom_symm,
                                        grid_name, grid_ratio, space_group, angles, thicks)
            energy = np.array(energy)[idx].tolist()
            #new data
            atom_fea, nbr_fea, nbr_fea_idx = \
                self.transfer.get_gnn_input_batch_general(atom_pos, atom_type, grid_name, grid_ratio, space_group)
            #train set
            if len(train_pos) == 0:
                train_atom_fea, train_symm_tmp, train_nbr_fea, train_nbr_fea_idx, train_energy_tmp = [], [], [], [], []
                valid_atom_fea, valid_symm, valid_nbr_fea, valid_nbr_fea_idx, valid_energy = [], [], [], [], []
            else:
                train_atom_fea, train_nbr_fea, train_nbr_fea_idx = \
                    self.transfer.get_gnn_input_batch_general(train_pos, train_type, train_grid, train_ratio, train_sg)
            #get gnn input from optimzied samples
            optim_file = [i for i in os.listdir(POSCAR_Path) if i.startswith('optim')]
            if len(optim_file) > 0:
                select = Select(0)
                strus, opt_energys = select.collect_recycle(0, len(optim_file))
                opt_atom_fea_bh, opt_nbr_fea_bh, opt_nbr_fea_idx_bh = self.transfer.get_gnn_input_from_stru_batch_parallel(strus)
                opt_symm = [[1 for _ in range(len(i))] for i in opt_atom_fea_bh]
            else:
                opt_energys, opt_symm = [], []
                opt_atom_fea_bh, opt_nbr_fea_bh, opt_nbr_fea_idx_bh = [], [], []
            #divide train set
            if len(train_pos) > 0:
                num = len(train_energy)
                idx = np.arange(num)
                np.random.shuffle(idx)
                add_num = int(train_valid_ratio*num)
                train_idx = idx[:add_num]
                valid_idx = idx[add_num:]
                #validation set
                valid_atom_fea = np.array(train_atom_fea, dtype=object)[valid_idx].tolist()
                valid_symm = np.array(train_symm, dtype=object)[valid_idx].tolist()
                valid_nbr_fea = np.array(train_nbr_fea, dtype=object)[valid_idx].tolist()
                valid_nbr_fea_idx = np.array(train_nbr_fea_idx, dtype=object)[valid_idx].tolist()
                valid_energy = np.array(train_energy, dtype=object)[valid_idx].tolist()
                #train set
                train_atom_fea = np.array(train_atom_fea, dtype=object)[train_idx].tolist()
                train_symm_tmp = np.array(train_symm, dtype=object)[train_idx].tolist()
                train_nbr_fea = np.array(train_nbr_fea, dtype=object)[train_idx].tolist()
                train_nbr_fea_idx = np.array(train_nbr_fea_idx, dtype=object)[train_idx].tolist()
                train_energy_tmp = np.array(train_energy, dtype=object)[train_idx].tolist()
            #divide searched samples
            num = len(energy)
            idx = np.arange(num)
            np.random.shuffle(idx)
            add_num = int(train_valid_ratio*num)
            train_idx = idx[:add_num]
            valid_idx = idx[add_num:]
            #validation set
            valid_atom_fea += np.array(atom_fea, dtype=object)[valid_idx].tolist()
            valid_symm += np.array(atom_symm, dtype=object)[valid_idx].tolist()
            valid_nbr_fea += np.array(nbr_fea, dtype=object)[valid_idx].tolist()
            valid_nbr_fea_idx += np.array(nbr_fea_idx, dtype=object)[valid_idx].tolist()
            valid_energy += np.array(energy, dtype=object)[valid_idx].tolist()
            tuple = valid_atom_fea, valid_symm, valid_nbr_fea, valid_nbr_fea_idx, valid_energy
            num = len(valid_energy)
            if Job_Queue == 'GPU':
                batch_balance(num, batchsize, tuple)
            valid_num = len(valid_energy)
            #train set
            train_atom_fea += np.array(atom_fea, dtype=object)[train_idx].tolist()
            train_symm_tmp += np.array(atom_symm, dtype=object)[train_idx].tolist()
            train_nbr_fea += np.array(nbr_fea, dtype=object)[train_idx].tolist()
            train_nbr_fea_idx += np.array(nbr_fea_idx, dtype=object)[train_idx].tolist()
            train_energy_tmp += np.array(energy, dtype=object)[train_idx].tolist()
            #merge train set
            all_train_atom_fea = opt_atom_fea_bh + train_atom_fea
            all_train_symm = opt_symm + train_symm_tmp
            all_train_nbr_fea = opt_nbr_fea_bh + train_nbr_fea
            all_train_nbr_fea_idx = opt_nbr_fea_idx_bh + train_nbr_fea_idx
            all_train_energy = opt_energys + train_energy_tmp
            tuple = all_train_atom_fea, all_train_symm, all_train_nbr_fea, all_train_nbr_fea_idx, all_train_energy
            num = len(all_train_energy)
            if Job_Queue == 'GPU':
                batch_balance(num, batchsize, tuple)
            train_num = len(all_train_energy)
            system_echo(f'Training set: {train_num}  Validation set: {valid_num}')
            #train model
            train_data = GNNData(all_train_atom_fea, all_train_symm, all_train_nbr_fea,
                                all_train_nbr_fea_idx, all_train_energy)
            valid_data = GNNData(valid_atom_fea, valid_symm, valid_nbr_fea,
                                valid_nbr_fea_idx, valid_energy)
            gnn = GNNTrain(iteration+1, train_data, valid_data, valid_data, train_batchsize=batchsize)
            gnn.train_epochs()
            #update train set
            self.update_dataset(atom_pos, atom_type, atom_symm,
                                grid_name, grid_ratio, space_group, angles, thicks, energy,
                                train_pos, train_type, train_symm, train_grid,
                                train_ratio, train_sg, train_angles, train_thicks, train_energy)
            #use std to filter high energy structures
            idx = []
            std = np.std(train_energy)
            mean = np.mean(train_energy)
            for i, e in enumerate(train_energy):
                if e - mean < 3*std:
                    idx.append(i)
            train_pos, train_type, train_symm, train_grid, train_ratio, train_sg, train_angles, train_thicks = \
                self.init.filter_samples(idx, train_pos, train_type, train_symm,
                                        train_grid, train_ratio, train_sg, train_angles, train_thicks)
            train_energy = np.array(train_energy)[idx].tolist()
            #sorted by grid and space group for train set
            idx = self.transfer.sort_by_grid_sg(train_grid, train_sg)
            train_pos, train_type, train_symm, train_grid, train_ratio, train_sg, train_angles, train_thicks = \
                self.init.filter_samples(idx, train_pos, train_type, train_symm,
                                        train_grid, train_ratio, train_sg, train_angles, train_thicks)
            train_energy = np.array(train_energy)[idx].tolist()
            #clean memory
            del atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group
            del atom_fea, nbr_fea, nbr_fea_idx, train_atom_fea, train_nbr_fea, train_nbr_fea_idx
            del opt_atom_fea_bh, opt_nbr_fea_bh, opt_nbr_fea_idx_bh
            del all_train_atom_fea, all_train_symm, all_train_nbr_fea, all_train_nbr_fea_idx, all_train_energy
            del valid_atom_fea, valid_symm, valid_nbr_fea, valid_nbr_fea_idx, valid_energy
            del train_data, valid_data, gnn
            #release CUDA
            if Job_Queue == 'GPU':
                torch.cuda.empty_cache()
        else:
            old_model_save_path = f'{Model_Path}/{iteration:02.0f}'
            new_model_save_path = f'{Model_Path}/{iteration+1:02.0f}'
            shutil.copytree(old_model_save_path, new_model_save_path)
        return train_pos, train_type, train_symm, train_grid, train_ratio, train_sg, train_angles, train_thicks, train_energy
        
    def update_dataset(self, atom_pos, atom_type, atom_symm,
                       grid_name, grid_ratio, space_group, angles, thicks, energy,
                       train_pos, train_type, train_symm, train_grid,
                       train_ratio, train_sg, train_angles, train_thicks, train_energy):
        #update training set
        train_pos += atom_pos
        train_type += atom_type
        train_symm += atom_symm
        train_grid += grid_name
        train_ratio += grid_ratio
        train_sg += space_group
        train_angles += angles
        train_thicks += thicks
        train_energy += energy
    
    def copy_ml_model(self, iteration):
        """
        copy pretrain ml model
        
        Parameters
        ----------
        iteration [int, 0d]: search iteration
        """
        model_Save_Path = f'{Model_Path}/{iteration:02.0f}'
        if not os.path.exists(model_Save_Path):
            os.mkdir(model_Save_Path)
        if Dimension == 2:
            shutil.copyfile(f'{Pretrain_Model_2d}', f'{model_Save_Path}/model_best.pth.tar')
        elif Dimension == 3:
            shutil.copyfile(f'{Pretrain_Model_3d}', f'{model_Save_Path}/model_best.pth.tar')
    
    def import_ml_energy(self, iteration):
        """
        import ml energy
        
        Parameters
        ----------
        iteration [int, 0d]: search iteration

        Returns
        ----------
        energys [float, 1d]: prediction energy 
        """
        head = f'{Search_Path}/ml_{iteration:02.0f}'
        energys = self.import_list2d(f'{head}/energy_select.dat', float, numpy=True)
        return energys.flatten()
    
    def write_ml_energy(self, iteration, energys):
        """
        export ml energy
        
        Parameters
        ----------
        iteration [int, 0d]: search iteration
        energys [float, 1d]: prediction energy
        """
        num_poscar = len(energys)
        system_echo(f'Start Energy Prediction---itersions: '
                    f'{iteration}, number: {num_poscar}')
        #export energy file
        ct = []
        poscars = os.listdir(f'{POSCAR_Path}/ml_{iteration:02.0f}')
        poscars = sorted(poscars, key=lambda x: x.split('-')[1])
        for poscar, energy in zip(poscars, energys):
            ct.append([f'{poscar}.out', True, energy])
            system_echo(f'{poscar}.out, True, {energy}')
        if Energy_Method == 'VASP':
            self.write_list2d(f'{VASP_Out_Path}/Energy-{iteration:02.0f}.dat', ct)
        elif Energy_Method == 'LAMMPS':
            self.write_list2d(f'{LAMMPS_Out_Path}/Energy-{iteration:02.0f}.dat', ct)
        system_echo(f'Energy file generated successfully!')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recyc', type=int)
    args = parser.parse_args()
    
    recycle = args.recyc
    
    sccop = CrystalOptimization()
    sccop.main(recycle)
    os._exit(0)