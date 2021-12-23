import os, sys
import numpy as np
import time

from modules.global_var import *
from modules.pretrain import Initial
from modules.grid_divide import MultiDivide, GridDivide
from modules.data_transfer import MultiGridTransfer
from modules.sample_select import Select
from modules.sub_vasp import SubVASP
from modules.workers import MultiWorkers, Search
from modules.predict import PPMData, PPModel
from modules.utils import ListRWTools


def convert():
    #TODO pos, type, grid store in string
    #e.g. 1 1 1 - 2 2 2 - 1
    pass

def delete_duplicates():
    #TODO delete duplicates that are already in training set
    pass


if __name__ == '__main__':
    init = Initial()
    grid = GridDivide()
    divide = MultiDivide()
    rwtools = ListRWTools()
    mul_transfer = MultiGridTransfer()
    workers = MultiWorkers()
    sub_vasp = SubVASP()
    
    #Build grid
    build = True
    if build:
        grid.build_grid(0, latt_vec, grain, cutoff)
    grid_store = [0]
    
    #Update each node
    init.update()
    
    #Lattice mutate
    num_grid = len(grid_store)
    grid_origin = [0 for _ in range(num_mutate)]
    grid_mutate = [i for i in range(num_grid, num_grid+num_mutate)]
    grid_store = grid_store + grid_mutate
    mutate = True
    if mutate:
        divide.assign(grid_origin, grid_mutate)
    
    #Initial data
    initial = True
    if initial:
        round = 0
        num_initial = 9000
        grid_point = [i for i in range(1000)]
        atom_pos = [list(np.random.choice(grid_point, 8, False)) for _ in range(num_initial)]
        atom_type = [[i for i in [30, 6, 7, 29] for _ in range(2)] for _ in range(num_initial)]
        grid_name = np.random.choice(grid_store, num_initial)
        grid_name = sorted(grid_name)
        
        #Geometry check
        batch_nbr_dis = mul_transfer.find_batch_nbr_dis(atom_pos, grid_name)

        worker = Search(round, grid_name[0])
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
        idx = np.arange(len(grid_name_right))
        
        #Select samples
        if not os.path.exists(f'{search_dir}/000'):
            os.mkdir(f'{search_dir}/000')
        select = Select(round)
        select.write_POSCARs(idx, atom_pos_right, atom_type_right, grid_name_right)
        
        #VASP calculate
        sub_vasp.sub_VASP_job(round)
    
    pos_buffer, type_buffer, grid_buffer = [], [], []
    train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energys = [], [], [], []
    for round in range(num_round):
        #Data import
        energy_file = rwtools.import_list2d(f'{vasp_out_dir}/Energy-{round:03.0f}.dat', str, numpy=True)
        true_E = np.array(energy_file)[:,1]
        true_E = [bool(i) for i in true_E]
        energys = energy_file[:,2][true_E]
        energys = [float(i) for i in energys]

        atom_pos = rwtools.import_list2d(f'{search_dir}/{round:03.0f}/atom_pos_select.dat', int)
        atom_type = rwtools.import_list2d(f'{search_dir}/{round:03.0f}/atom_type_select.dat', int)
        grid_name = rwtools.import_list2d(f'{search_dir}/{round:03.0f}/grid_name_select.dat', int)
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
        a = int(num_poscars*0.6)        
        atom_fea, nbr_fea, nbr_fea_idx = mul_transfer.batch(atom_pos_right, atom_type_right, grid_name_right)

        #Training data
        pos_buffer += atom_pos_right[0:a]
        type_buffer += atom_type_right[0:a]
        grid_buffer += grid_name_right[0:a]
        train_atom_fea += atom_fea[0:a]
        train_nbr_fea += nbr_fea[0:a]
        train_nbr_fea_idx += nbr_fea_idx[0:a]
        train_energys += energys[0:a]
        
        #Validation data
        valid_atom_fea = atom_fea[a:]
        valid_nbr_fea = nbr_fea[a:]
        valid_nbr_fea_idx = nbr_fea_idx[a:]
        valid_energys = energys[a:]
        
        #Training
        train = True
        if train:
            train_data = PPMData(train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energys)
            valid_data = PPMData(valid_atom_fea, valid_nbr_fea, valid_nbr_fea_idx, valid_energys)
            ppm = PPModel(round+1, train_data, valid_data, valid_data)
            ppm.train_epochs()

        #Train data added
        train_atom_fea += atom_fea[a:]
        train_nbr_fea += nbr_fea[a:]
        train_nbr_fea_idx += nbr_fea_idx[a:]
        train_energys += energys[a:]
        pos_buffer += atom_pos_right[a:]
        type_buffer += atom_type_right[a:]
        grid_buffer += grid_name_right[a:]
        
        #Search
        search = True
        if search:
            num_seed = 30
            min_idx = np.argsort(train_energys)[:num_seed]
            grid_pool = np.array(grid_buffer)[min_idx] 
            
            #Lattice mutate
            if round > 0:
                mutate = True
            else:
                mutate = False
            if mutate:
                num_grid = len(grid_store)
                grid_origin = np.random.choice(grid_pool, num_mutate)
                grid_mutate = [i for i in range(num_grid, num_grid+num_mutate)]
                grid_store = grid_store + grid_mutate
                divide.assign(grid_origin, grid_mutate)
            
            #Initial search start point
            init_pos, init_type, init_grid = [], [], []
            for _ in range(num_paths):
                seed = np.random.choice(min_idx)
                init_pos.append(pos_buffer[seed])
                init_type.append(type_buffer[seed])
                if round > 0:
                    p = 0.8
                else:
                    p = 1
                if np.random.rand() > p:
                    mut = np.random.choice(grid_mutate)
                    init_grid.append(mut)
                else:
                    init_grid.append(grid_buffer[seed])
            
            workers.assign_job(round+1, num_paths, init_pos, init_type, init_grid)

        #Sample
        atom_pos = rwtools.import_list2d(f'{search_dir}/{round+1:03.0f}/atom_pos.dat', int)
        atom_type = rwtools.import_list2d(f'{search_dir}/{round+1:03.0f}/atom_type.dat', int)
        grid_name = rwtools.import_list2d(f'{search_dir}/{round+1:03.0f}/grid_name.dat', int)
        select = Select(round+1)
        select.samples(atom_pos, atom_type, grid_name)

        #VASP calculate
        sub_vasp.sub_VASP_job(round+1)

    #Export searched POSCARS
    select = Select(num_round)
    grid_buffer = [[i] for i in grid_buffer]
    select.export(pos_buffer, type_buffer, grid_buffer)