import os, sys
import numpy as np
import time

from core.global_var import *
from core.dir_path import *
from core.initial import Initial, UpdateNodes
from core.grid_divide import MultiDivide, GridDivide
from core.data_transfer import MultiGridTransfer
from core.sample_select import Select, OptimSelect
from core.sub_vasp import SubVASP
from core.workers import MultiWorkers, Search
from core.predict import PPMData, PPModel
from core.utils import ListRWTools, system_echo
from core.post_process import PostProcess, VASPoptimize


def convert():
    #TODO pos, type, grid store in string
    #e.g. 1 1 1 - 2 2 2 - 1
    pass

def delete_duplicates():
    #TODO delete duplicates that are already in training set
    pass

def main():
    #TODO ccop program
    pass


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    
    main()
    
    init = Initial(component, ndensity, mindis)
    cpu_nodes = UpdateNodes()
    grid = GridDivide()
    divide = MultiDivide()
    rwtools = ListRWTools()
    mul_transfer = MultiGridTransfer()
    workers = MultiWorkers()
    sub_vasp = SubVASP()
    
    for recyc in range(num_recycle):
        system_echo(f'Begin Crystal Combinatorial Optimization Program --- Recycle: {recyc}')
        
        if recyc == 0:
            #Update each node
            cpu_nodes.update()
            
            #Initialize storage
            grid_store = []
            pos_buffer, type_buffer, grid_buffer = [], [], []
            train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energys = [], [], [], []
        
        #Generate initial structures
        atom_pos, atom_type, grid_init = init.generate(recyc, grid_store)
        system_echo('Initial sample generate')
        
        #Build grid
        build = True
        grid_origin = grid_init
        grid_mutate = grid_init
        grid_store = np.concatenate((grid_store, grid_init))
        if build:
            divide.assign(grid_origin, grid_mutate)
        grid_name = grid_origin
        system_echo('Initial grid build')
        
        #Initial data
        initial = True
        worker = Search(0, 0)
        if initial:
            num_initial = 1000
            for i in grid_init:
                point_num = len(rwtools.import_list2d(f'{grid_prop_path}/{i:03.0f}_frac_coor.bin', float, binary=True))
                grid_point = [i for i in range(point_num)]
                atom_pos += [list(np.random.choice(grid_point, 8, False)) for _ in range(num_initial)]
                atom_type += [[i for i in [29, 30, 6, 7] for _ in range(2)] for _ in range(num_initial)]
                grid_name = np.concatenate((grid_name, [i for _ in range(num_initial)]))
            order = np.argsort(grid_name)
            atom_pos = [atom_pos[i] for i in order]
            atom_type = [atom_type[i] for i in order]
            grid_name = grid_name[order]
            
            #Geometry check
            batch_nbr_dis = mul_transfer.find_batch_nbr_dis(atom_pos, grid_name)
            check_near = [worker.near_check(i) for i in batch_nbr_dis]
            check_overlay = [worker.overlay_check(i, len(i)) for i in atom_pos]
            check = [i and j for i, j in zip(check_near, check_overlay)]
            atom_pos_right, atom_type_right, grid_name_right = [], [], []
            for i, correct in enumerate(check):
                if correct:
                    atom_pos_right.append(atom_pos[i])
                    atom_type_right.append(atom_type[i])
                    grid_name_right.append(grid_name[i])
            grid_name_right = np.array(grid_name_right)
            num_sample = len(grid_name_right)
            idx = np.arange(num_sample)
            system_echo(f'Sampling number: {num_sample}')

            #Select samples
            start = recyc * (num_round + 1)
            select = Select(start)
            select.write_POSCARs(idx, atom_pos_right, atom_type_right, grid_name_right)

            #VASP calculate
            sub_vasp.sub_VASP_job(start)
        
        #CCOP optimize
        for round in range(start, start+num_round):
            #Data import
            energy_file = rwtools.import_list2d(f'{vasp_out_path}/Energy-{round:03.0f}.dat', str, numpy=True)
            true_E = np.array(energy_file)[:,1]
            true_E = [True if i=='True' else False for i in true_E]
            energys = energy_file[:,2][true_E]
            energys = [float(i) for i in energys]
            
            atom_pos = rwtools.import_list2d(f'{search_path}/{round:03.0f}/atom_pos_select.dat', int)
            atom_type = rwtools.import_list2d(f'{search_path}/{round:03.0f}/atom_type_select.dat', int)
            grid_name = rwtools.import_list2d(f'{search_path}/{round:03.0f}/grid_name_select.dat', int)
            grid_name = np.ravel(grid_name)
            atom_pos_right, atom_type_right, grid_name_right = [], [], []
            for i, correct in enumerate(true_E):
                if correct:
                    atom_pos_right.append(atom_pos[i])
                    atom_type_right.append(atom_type[i])
                    grid_name_right.append(grid_name[i])
            
            #TODO Delete duplicates
            #atom_pos_right, atom_type_right, grid_name_right = \
            #    delete_duplicates(atom_pos_right, atom_type_right, grid_name_right)

            num_poscars = len(energys)
            num_add = int(num_poscars*0.6)
            num_crys = num_add + len(train_energys)
            num_last_batch = np.mod(num_crys, train_batchsize)
            if 0 < num_last_batch < num_gpus:
                num_add -= num_last_batch
            system_echo(f'Training set: {num_crys}')       
            atom_fea, nbr_fea, nbr_fea_idx = mul_transfer.batch(atom_pos_right, atom_type_right, grid_name_right)
            system_echo(f'New add to training set: {len(atom_fea)}')
            #Training data
            pos_buffer += atom_pos_right[0:num_add]
            type_buffer += atom_type_right[0:num_add]
            grid_buffer += grid_name_right[0:num_add]
            train_atom_fea += atom_fea[0:num_add]
            train_nbr_fea += nbr_fea[0:num_add]
            train_nbr_fea_idx += nbr_fea_idx[0:num_add]
            train_energys += energys[0:num_add]
            system_echo(f'Training set: {len(train_energys)}')
            #Validation data
            valid_atom_fea = atom_fea[num_add:]
            valid_nbr_fea = nbr_fea[num_add:]
            valid_nbr_fea_idx = nbr_fea_idx[num_add:]
            valid_energys = energys[num_add:]

            #Training
            train = True
            if train:
                train_data = PPMData(train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energys)
                valid_data = PPMData(valid_atom_fea, valid_nbr_fea, valid_nbr_fea_idx, valid_energys)
                ppm = PPModel(round+1, train_data, valid_data, valid_data)
                ppm.train_epochs()

            #Train data added
            train_atom_fea += atom_fea[num_add:]
            train_nbr_fea += nbr_fea[num_add:]
            train_nbr_fea_idx += nbr_fea_idx[num_add:]
            train_energys += energys[num_add:]
            pos_buffer += atom_pos_right[num_add:]
            type_buffer += atom_type_right[num_add:]
            grid_buffer += grid_name_right[num_add:]
            
            #Search
            search = True
            if search:
                num_seed = 30
                min_idx = np.argsort(train_energys)[:num_seed]
                grid_pool = np.array(grid_buffer)[min_idx] 
                all_idx = np.arange(len(train_energys))
                
                #Generate mutate lattice grid
                if round == 0:
                    mutate = False
                else:
                    if np.mod(round, mut_freq) == 0:
                        mutate = True
                    else:
                        mutate = False
                if mutate:
                    num_grid = len(grid_store)
                    grid_origin = np.random.choice(grid_pool, num_mutate)
                    grid_mutate = np.arange(num_grid, num_grid+num_mutate)
                    grid_store = np.concatenate((grid_store, grid_mutate))
                    divide.assign(grid_origin, grid_mutate)
                    system_echo(f'Grid origin: {grid_origin}')
                
                #Initial search start point
                init_pos, init_type, init_grid = [], [], []
                for _ in range(num_paths//2):
                    seed = np.random.choice(min_idx)
                    init_pos.append(pos_buffer[seed])
                    init_type.append(type_buffer[seed])
                    init_grid.append(grid_buffer[seed])
                
                for _ in range(num_paths//2):
                    seed = np.random.choice(all_idx)
                    init_pos.append(pos_buffer[seed])
                    init_type.append(type_buffer[seed])
                    init_grid.append(grid_buffer[seed])
                
                #Lattice mutate
                #TODO clean up the below code
                mut_counter = 0
                mut_num = int(num_paths*mut_ratio)
                mut_latt = sorted(np.random.choice(grid_mutate, mut_num))
                init_frac = [rwtools.import_list2d(f'{grid_prop_path}/{i:03.0f}_frac_coor.bin', float, binary=True) for i in init_grid[:mut_num]]
                stru_frac = [init_frac[i][init_pos[i]] for i in range(mut_num)]
                mut_frac = [rwtools.import_list2d(f'{grid_prop_path}/{i:03.0f}_frac_coor.bin', float, binary=True) for i in mut_latt]
                mut_latt_vec = [rwtools.import_list2d(f'{grid_prop_path}/{i:03.0f}_latt_vec.bin', float, binary=True) for i in mut_latt]
                
                mut_pos = [mul_transfer.put_into_grid(stru_frac[i], mut_latt_vec[i], mut_frac[i], mut_latt_vec[i]) for i in range(mut_num)]
                batch_nbr_dis = mul_transfer.find_batch_nbr_dis(mut_pos, mut_latt)
                check_near = [worker.near_check(i) for i in batch_nbr_dis]
                check_overlay = [worker.overlay_check(i, len(i)) for i in mut_pos]
                check = [i and j for i, j in zip(check_near, check_overlay)]
                for i, correct in enumerate(check):
                    if correct:
                        init_pos[i] = mut_pos[i]
                        init_grid[i] = mut_latt[i]
                        mut_counter += 1
                system_echo(f'Lattice mutate number: {mut_counter}')
                workers.search(round+1, num_paths, init_pos, init_type, init_grid)
            
            #Sample
            atom_pos = rwtools.import_list2d(f'{search_path}/{round+1:03.0f}/atom_pos.dat', int)
            atom_type = rwtools.import_list2d(f'{search_path}/{round+1:03.0f}/atom_type.dat', int)
            grid_name = rwtools.import_list2d(f'{search_path}/{round+1:03.0f}/grid_name.dat', int)
            select = Select(round+1)
            select.samples(atom_pos, atom_type, grid_name)

            #VASP calculate
            sub_vasp.sub_VASP_job(round+1)
        
        #Export searched POSCARS
        select = Select(start+num_round)
        grid_buffer_2d = [[i] for i in grid_buffer]
        select.export(recyc, pos_buffer, type_buffer, grid_buffer_2d)
        system_echo(f'End Crystal Combinatorial Optimization Program --- Recycle: {recyc}')

        #VASP optimize
        vasp = VASPoptimize(recyc)
        vasp.run_optimization_low()

    #Select optimized structures
    opt_slt = OptimSelect(start+num_round)
    opt_slt.optim_select()
    
    #Optimize
    post = PostProcess()
    post.run_optimization()

    #Energy band
    post.run_pbe_band()
    
    #Phonon spectrum
    post.run_phonon()
    
    #Elastic matrix
    post.run_elastic()
    
    #Dielectric matrix
    post.run_dielectric()