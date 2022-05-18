import os
import random
import numpy as np

from core.global_var import *
from core.dir_path import *
from core.initialize import InitSampling, UpdateNodes
from core.grid_divide import ParallelDivide
from core.sample_select import Select, OptimSelect
from core.sub_vasp import ParallelSubVASP
from core.search import ParallelWorkers, GeoCheck
from core.predict import PPMData, PPModel
from core.utils import ListRWTools, system_echo
from core.post_process import PostProcess, VASPoptimize


class CrystalOptimization(ListRWTools):
    #crystal combinational optimization program
    def __init__(self, number, component):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        self.init = InitSampling(number, component)
        self.cpu_nodes = UpdateNodes()
        self.divide = ParallelDivide()
        self.workers = ParallelWorkers()
        self.vasp = ParallelSubVASP()
        self.check = GeoCheck()
        self.post = PostProcess()
    
    def main(self):
        #Update cpu nodes
        self.cpu_nodes.update()
        #Initialize storage
        grid_store = np.array([], dtype=int)
        train_pos, train_type, train_symm, train_grid = [], [], [], []
        train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energy = [], [], [], []
        
        start = 0
        for recycle in range(num_recycle):
            system_echo(f'Begin Crystal Combinatorial Optimization Program --- Recycle: {recycle}')
            #Generate structures
            atom_pos, atom_type, atom_symm, grid_name, space_group = self.init.generate(recycle)
            system_echo('New initial samples generated')
            
            #Write POSCARs
            select = Select(start)
            select.write_POSCARs(atom_pos, atom_type, grid_name, space_group)
            #VASP calculate
            #self.vasp.sub_job(start, vdW=add_vdW)
            break
        '''
            #CCOP optimize
            num_round = num_ml_list[recycle]
            grid_store = np.concatenate((grid_store, grid_init))
            for round in range(start, start+num_round):
                #Data import
                atom_pos, atom_type, grid_name, energy_select = \
                    self.data_import(round)
                #Check number of Training data
                add_num, crys_num = self.check_num_train_data(energy_select, train_energy)
                system_echo(f'Training set: {crys_num}  New add to training set: {add_num}')
                #Train predict model
                train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energy = \
                    self.train_predict_model(round, add_num, atom_pos, atom_type, 
                                             grid_name, energy_select, train_atom_fea, 
                                             train_nbr_fea, train_nbr_fea_idx, train_energy)
                #Update training set
                train_pos += atom_pos
                train_type += atom_type
                train_grid += grid_name
                
                #Generate mutate lattice grid
                all_idx = np.arange(len(train_energy))
                min_idx = np.argsort(train_energy)[:num_seed]
                if round == 0:
                    mutate = False
                    grid_mutate = grid_store
                else:
                    mutate = True
                    grid_origin = np.random.choice(np.array(train_grid)[min_idx], num_mutate)
                    if np.mod(round, mut_freq) == 0:
                        grid_mutate, grid_store = self.lattice_mutate(recycle, grid_origin, grid_store)
                    system_echo(f'Grid pool: {grid_mutate}')
                #Initial search start point
                init_pos, init_type, init_grid = \
                    self.generate_search_point(min_idx, all_idx, mutate, grid_mutate, 
                                               train_pos, train_type, train_grid)
                #Search on grid
                paths_num = len(init_grid)
                self.workers.search(round+1, paths_num, init_pos, init_type, init_grid)
                
                #Select samples
                file_head = f'{search_path}/{round+1:03.0f}'
                atom_pos = self.import_list2d(f'{file_head}/atom_pos.dat', int)
                atom_type = self.import_list2d(f'{file_head}/atom_type.dat', int)
                grid_name = self.import_list2d(f'{file_head}/grid_name.dat', int)
                select = Select(round+1)
                select.samples(atom_pos, atom_type, grid_name, train_pos, train_type, train_grid)
                #Single point ernergy calculate
                self.vasp.sub_job(round+1, vdW=add_vdW)
            
            #Update training set
            atom_pos, atom_type, grid_name, energy_select = \
                    self.data_import(start+num_round)
            add_num = len(atom_pos)
            train_pos += atom_pos
            train_type += atom_type
            train_grid += grid_name
            train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energy = \
                    self.train_predict_model(round, add_num, atom_pos, atom_type, 
                                             grid_name, energy_select, train_atom_fea, 
                                             train_nbr_fea, train_nbr_fea_idx, train_energy,
                                             train_model=False)
            
            #Select searched POSCARs
            select = Select(start+num_round)
            select.export(recycle, train_pos, train_type, train_energy, train_grid)
            system_echo(f'End Crystal Combinatorial Optimization Program --- Recycle: {recycle}')
            #VASP optimize
            vasp = VASPoptimize(recycle)
            vasp.run_optimization_low(vdW=add_vdW)
            start += num_round + 1
        
        #Select optimized structures
        opt_slt = OptimSelect()
        opt_slt.optim_select()
        #Optimize
        self.post.run_optimization(vdW=add_vdW)
    '''
    def data_import(self, round):
        """
        import selected data from last round
        
        Parameters
        ----------
        round [int, 0d]: searching round

        Returns
        ----------
        atom_pos_right [int, 2d]: position of atoms
        atom_type_right [int, 2d]: type of atoms
        grid_name_right [int, 1d]: name of grids
        """
        #select right energy configurations
        file = f'{vasp_out_path}/Energy-{round:03.0f}.dat'
        energy_file = self.import_list2d(file, str, numpy=True)
        true_E = np.array(energy_file)[:,1]
        true_E = [True if i=='True' else False for i in true_E]
        energy_right = energy_file[:,2][true_E]
        energy_right = [float(i) for i in energy_right]
        #import structure information
        head = f'{search_path}/{round:03.0f}'
        atom_pos = self.import_list2d(f'{head}/atom_pos_select.dat', int)
        atom_type = self.import_list2d(f'{head}/atom_type_select.dat', int)
        grid_name = self.import_list2d(f'{head}/grid_name_select.dat', int)
        grid_name = np.ravel(grid_name)
        atom_pos_right, atom_type_right, grid_name_right = [], [], []
        for i, correct in enumerate(true_E):
            if correct:
                atom_pos_right.append(atom_pos[i])
                atom_type_right.append(atom_type[i])
                grid_name_right.append(grid_name[i])
        return atom_pos_right, atom_type_right, grid_name_right, energy_right

    def check_num_train_data(self, energy_select, train_energy):
        """
        check number of train data assigned to gpus    

        Parameters
        ----------
        energy_select [float, 1d]: select energy last round
        train_energy [float, 1d]: energy in train data

        Returns
        ----------
        add_num [int, 0d]: number added to train data
        crys_num [int, 0d]: number of added train data
        """
        poscar_num = len(energy_select)
        add_num = int(poscar_num*0.6)
        crys_num = add_num + len(train_energy)
        last_batch_num = np.mod(crys_num, train_batchsize)
        if 0 < last_batch_num < num_gpus:
            add_num -= last_batch_num
        return add_num, crys_num

    def train_predict_model(self, round, add_num, atom_pos, atom_type, 
                            grid_name, energy_select, train_atom_fea, 
                            train_nbr_fea, train_nbr_fea_idx, train_energy,
                            train_model=True):
        """
        train CGCNN as predict model
        
        Parameters
        ----------
        round [int, 0d]: seaching round
        add_num [int, 0d]: number added to training set
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
        energy_select [float, 1d]: select energy last round
        train_atom_fea [float, 3d]: atom feature
        train_nbr_fea [float, 4d]: bond feature 
        train_nbr_fea_idx [int, 3d]: neighbor index
        train_energy [float, 1d]: train energy
        train_model [bool, 0d]: whether train model
        
        Returns
        ----------
        train_atom_fea [float, 3d]: atom feature
        train_nbr_fea [float, 4d]: bond feature
        train_nbr_fea_idx [int, 3d]: neighbor index
        train_energy [float, 1d]: train energy
        """
        #training data
        atom_fea, nbr_fea, nbr_fea_idx = \
            self.transfer.batch(atom_pos, atom_type, grid_name)
        train_atom_fea += atom_fea[:add_num]
        train_nbr_fea += nbr_fea[:add_num]
        train_nbr_fea_idx += nbr_fea_idx[:add_num]
        train_energy += energy_select[:add_num]
        #validation data
        valid_atom_fea = atom_fea[add_num:]
        valid_nbr_fea = nbr_fea[add_num:]
        valid_nbr_fea_idx = nbr_fea_idx[add_num:]
        valid_energy = energy_select[add_num:]
        #training model
        if train_model:
            train_data = PPMData(train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energy)
            valid_data = PPMData(valid_atom_fea, valid_nbr_fea, valid_nbr_fea_idx, valid_energy)
            ppm = PPModel(round+1, train_data, valid_data, valid_data)
            ppm.train_epochs()
        #train data added
        train_atom_fea += atom_fea[add_num:]
        train_nbr_fea += nbr_fea[add_num:]
        train_nbr_fea_idx += nbr_fea_idx[add_num:]
        train_energy += energy_select[add_num:]
        return train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energy
    
    def lattice_mutate(self, recyc, grid_origin, grid_store):
        """
        generate mutate lattice from training set
        
        Parameters
        ----------
        recyc [int, 0d]: recycle of searching
        grid_origin[int, 1d, np]: origin grid
        grid_store [int, 1d]: store of grid
        
        Returns
        ----------
        grid_mutate [int, 1d]: mutate grid
        grid_store [int, 1d]: store of grid
        """
        #generate mutate lattice grid
        if recyc > 0:
            grain_loc = grain_fine
        else:
            grain_loc = grain_coarse
        grid_num = len(grid_store)
        grid_mutate = np.arange(grid_num, grid_num+num_mutate)
        grid_store = np.concatenate((grid_store, grid_mutate))
        self.divide.assign_to_cpu(grain_loc, grid_origin, grid_mutate)
        system_echo(f'Grid origin: {grid_origin}')
        return grid_mutate, grid_store
    
    def generate_search_point(self, min_idx, all_idx, mutate, grid_mutate, 
                              train_pos, train_type, train_grid):
        """
        generate initial searching points
        
        Parameters
        ----------
        min_idx [int, 1d]: min energy index
        all_idx [int, 1d]: all sample index
        mutate [bool, 0d]: whether do lattice mutate
        grid_mutate [int, 1d, np]: name of mutate grid
        train_pos [int, 2d]: position of training set
        train_type [int, 2d]: type of training set
        train_grid [int, 1d]: grid of training set
        
        Returns
        ----------
        init_pos [int, 2d]: position of initial points
        init_type [int, 2d]: type of initial points
        init_grid [int, 1d]: grid of initial points
        """
        init_pos, init_type, init_grid = [], [], []
        greed_pos, greed_type, greed_grid = [], [], []
        rand_pos, rand_type, rand_grid = [], [], []
        #greedy path
        for _ in range(num_paths_min):
            seed = np.random.choice(min_idx)
            greed_pos.append(train_pos[seed])
            greed_type.append(train_type[seed])
            greed_grid.append(train_grid[seed])
        #random select path
        for _ in range(num_paths_rand):
            seed = np.random.choice(all_idx)
            rand_pos.append(train_pos[seed])
            rand_type.append(train_type[seed])
            rand_grid.append(train_grid[seed])
        init_pos += greed_pos + rand_pos
        init_type += greed_type + rand_type
        init_grid += greed_grid + rand_grid
        #mutate
        if mutate:
            #lattice mutate
            mut_counter = 0
            mut_num = int(len(greed_grid)*mut_ratio)
            mut_latt = sorted(np.random.choice(grid_mutate, mut_num))
            mut_pos = self.structure_mutate(mut_num, mut_latt, greed_pos, greed_grid)
            #geometry check
            batch_nbr_dis = self.transfer.find_nbr_dis(mut_pos, mut_latt)
            check_near = [self.check.near(i) for i in batch_nbr_dis]
            check_overlay = [self.check.overlay(i, len(i)) for i in mut_pos]
            check = [i and j for i, j in zip(check_near, check_overlay)]
            for i, correct in enumerate(check):
                if correct:
                    init_pos.append(mut_pos[i])
                    init_type.append(greed_type[i])
                    init_grid.append(mut_latt[i])
                    mut_counter += 1
            system_echo(f'Lattice mutate: {mut_counter}')
            #atom number mutate
            nber_pos, nber_type, nber_grid = self.atom_number_mutate(train_type, train_grid)
            system_echo(f'Atom number mutate: {len(nber_grid)}')
            #atom order mutate
            order_pos, order_type, order_grid = self.atom_order_mutate(greed_pos, greed_type, greed_grid)
            system_echo(f'Atom order mutate: {len(order_grid)}')
            #initial searching points
            init_pos += nber_pos + order_pos
            init_type += nber_type + order_type
            init_grid += nber_grid + order_grid
        return init_pos, init_type, init_grid
    
    def atom_order_mutate(self, greed_pos, greed_type, greed_grid):
        """
        varying order of atomic position
        
        Parameters
        ----------
        greed_pos [int, 2d]: greedy position 
        greed_type [int, 2d]: greedy type
        greed_grid [int, 1d]: greedy grid
        
        Returns
        ----------
        order_pos [int, 2d]: shuffle position
        order_type [int, 2d]: type of atoms
        order_grid [int, 1d]: grid name
        """
        order_pos, order_type, order_grid = [], [], []
        idx = [i for i in range(len(greed_grid))]
        for _ in range(num_paths_order):
            seed = np.random.choice(idx)
            pos = greed_pos[seed].copy()
            np.random.shuffle(pos)
            order_pos.append(list(pos))
            order_type.append(greed_type[seed])
            order_grid.append(greed_grid[seed])
        return order_pos, order_type, order_grid
    
    def atom_number_mutate(self, train_type, train_grid):
        """
        varying atom number in different grids
        
        Parameters
        ----------
        train_type [int, 2d]: type of training set
        train_grid [int, 1d]: grid of training set

        Returns
        ----------
        atom_pos_right [int, 2d]: position of atoms after constrain
        atom_type_right [int, 2d]: type of atoms after constrain
        grid_name_right [int, 1d]: name of grids after constrain
        """
        type_pool = train_type
        type_num = len(type_pool)
        atom_pos, atom_type, grid_name = [], [], []
        grid_pool = np.unique(train_grid)
        if len(grid_pool) > 20:
            grid_pool = np.random.choice(grid_pool, 20, replace=False)
        for i, grid in enumerate(grid_pool):
            #sampling on different grids
            prefix = f'{grid_prop_path}/{grid:03.0f}'
            frac_coor = self.import_list2d(f'{prefix}_frac_coor.bin', float, binary=True)
            point_num = len(frac_coor)
            points = [i for i in range(point_num)]
            grid_name += [grid for _ in range(num_sampling)]
            for _ in range(num_sampling):
                seed = random.randint(0, type_num-1)
                atom_num = len(type_pool[seed])
                atom_pos += [random.sample(points, atom_num)]
                atom_type += [type_pool[seed]]
        #check geometry of structure
        nbr_dis = self.transfer.find_nbr_dis(atom_pos, grid_name)
        check_near = [self.check.near(i) for i in nbr_dis]
        check_overlay = [self.check.overlay(i, len(i)) for i in atom_pos]
        check = [i and j for i, j in zip(check_near, check_overlay)]
        #get correct sample index
        check_num = len(check)
        sample_idx = np.arange(check_num)[check]
        #sampling correct random samples
        sample_num = len(sample_idx)
        if sample_num > num_paths_atom:
            sample_idx = np.random.choice(sample_idx, num_paths_atom, replace=False)
        #add right samples into buffer
        atom_pos_right, atom_type_right, grid_name_right = [], [], []
        for i in sample_idx:
            atom_pos_right.append(atom_pos[i])
            atom_type_right.append(atom_type[i])
            grid_name_right.append(grid_name[i])
        return atom_pos_right, atom_type_right, grid_name_right
    
    def structure_mutate(self, mut_num, mut_latt, init_pos, init_grid):
        """
        put structure into mutate lattice and keep the relative order
        
        Parameters
        ----------
        mut_num [int, 0d]: number of mutate lattice
        mut_latt [int, 1d]: name of mutate lattice
        init_pos [int, 2d]: position of initial points
        init_grid [int, 1d]: grid of initial points

        Returns
        ----------
        mut_pos [int, 2d]: position of mutate structure
        """
        mut_pos = []
        for i in range(mut_num):
            #file name
            init_frac_file = f'{grid_prop_path}/{init_grid[i]:03.0f}_frac_coor.bin'
            mut_frac_file = f'{grid_prop_path}/{mut_latt[i]:03.0f}_frac_coor.bin'
            mut_latt_vec_file = f'{grid_prop_path}/{mut_latt[i]:03.0f}_latt_vec.bin'
            #import necessary file
            init_frac = self.import_list2d(init_frac_file, float, binary=True)
            stru_frac = init_frac[init_pos[i]]
            mut_frac = self.import_list2d(mut_frac_file, float, binary=True)
            mut_latt_vec = self.import_list2d(mut_latt_vec_file, float, binary=True)
            #put into mutate lattice
            mut_pos += [self.transfer.put_into_grid(stru_frac, mut_latt_vec, mut_frac, mut_latt_vec)]
        return mut_pos
    
    def property_calculate(self):
        """
        calculate property of structures
        """
        #energy band
        self.post.run_pbe_band()
        #phonon spectrum
        self.post.run_phonon()
        #elastic matrix
        #self.post.run_elastic()
        #dielectric matrix
        #self.post.run_dielectric()
        #3 order
        #self.post.run_3RD()
        #thermal conductivity
        #self.post.run_thermal_conductivity()
    
    
if __name__ == '__main__':
    ccop = CrystalOptimization(num_latt, component)
    ccop.main()