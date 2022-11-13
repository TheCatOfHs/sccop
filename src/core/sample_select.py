import sys, os
import torch
import numpy as np

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats.distributions import norm
from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from core.path import *
from core.input import *
from core.utils import *
from core.predict import *
from core.data_transfer import DeleteDuplicates


class Select(SSHTools, DeleteDuplicates):
    #Select training samples by active learning
    def __init__(self, iteration, batch_size=1024, num_workers=0):
        DeleteDuplicates.__init__(self)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.iteration = f'{iteration:03.0f}'
        self.model_save_path = f'{model_path}/{self.iteration}'
        self.poscar_save_path = f'{poscar_path}/{self.iteration}'
        self.sh_save_path = f'{search_path}/{self.iteration}'
        self.device = torch.device('cuda')
        self.normalizer = Normalizer(torch.tensor([]))
    
    def samples(self, atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group,
                train_pos, train_type, train_grid, train_ratio, train_sg):
        """
        choose lowest energy structure in different clusters
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 2d]: name of grids
        grid_ratio [float, 2d]: ratio of grids
        space_group [int, 2d]: space group number
        train_pos [int, 2d]: position in training set
        train_type [int, 2d]: type in training set
        train_grid [int, 1d]: grid in training set
        train_ratio [float, 1d]: ratio in training set
        train_sg [int, 1d]: space group in training set
        """
        #delete same structures of searching
        grid_name = np.ravel(grid_name)
        grid_ratio = np.ravel(grid_ratio)
        space_group = np.ravel(space_group)
        idx = self.delete_duplicates(atom_pos, atom_type,
                                     grid_name, grid_ratio, space_group)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group)
        num_crys = len(atom_pos)
        system_echo(f'Delete duplicates in searching samples: {num_crys}')
        #delete same structures compared with training set
        idx = self.delete_same_selected(atom_pos, atom_type, grid_name, grid_ratio, space_group,
                                        train_pos, train_type, train_grid, train_ratio, train_sg)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group)
        tuple = atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group
        batch_balance(num_crys, self.batch_size, tuple)
        num_crys = len(grid_name)
        system_echo(f'Delete duplicates same as trainset: {num_crys}')
        #sort structure in order of grid and space group
        idx = self.sort_by_grid_sg(grid_name, space_group)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group)
        #predict energy and crystal vector
        loader = self.dataloader(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group)
        crys_vec = self.get_crystal_vector(loader)
        models= self.model_select(self.model_save_path)
        model_names = [f'{self.model_save_path}/{i}' for i in models]
        mean_pred = self.get_energy_bagging(model_names, loader)
        #filter by energy
        idx = self.select_samples(mean_pred)
        crys_vec = crys_vec[idx].cpu().numpy()
        mean_pred = mean_pred[idx].cpu().numpy()
        #reduce dimension and clustering
        crys_embedded = self.reduce(crys_vec)
        clusters = self.cluster(num_clusters, crys_embedded)
        idx = self.min_in_cluster(idx, mean_pred, clusters)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group)
        self.write_POSCARs(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group)
        self.preserve_models(models)
    
    def dataloader(self, atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group):
        """
        transfer data to the input of graph network and
        put them into DataLoader
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atom
        atom_type [int, 2d]: type of atom
        atom_symm [int, 2d]: 
        grid_name [int, 1d]: name of grid
        grid_ratio [float, 1d]: grid ratio
        space_group [int, 1d]: space group number
        
        Returns
        ----------
        loader [obj]: dataloader 
        """
        atom_fea, nbr_fea, nbr_fea_idx = \
            self.get_ppm_input_bh(atom_pos, atom_type, grid_name, grid_ratio, space_group)
        targets = np.zeros(len(grid_name))
        dataset = PPMData(atom_fea, atom_symm, nbr_fea, 
                          nbr_fea_idx, targets)
        loader = get_loader(dataset, self.batch_size, self.num_workers)
        return loader
    
    def get_crystal_vector(self, loader):
        """
        get crystal vector by pretrain model
        
        Parameters
        ----------
        loader [obj]: dataloader
        
        Returns
        ----------
        crys_fea [float, 2d, tensor]: crystal feature
        """
        #load crystal vector model
        if dimension == 2:
            model_name = pretrain_model_2d
        elif dimension == 3:
            model_name = pretrain_model_3d
        self.model_vec = FeatureExtractNet()
        params = torch.load(model_name, map_location=self.device)
        self.model_vec.load_state_dict(params['state_dict'])
        #calculate crystal vector
        self.model_vec = DataParallel(self.model_vec)
        self.model_vec.to(self.device)
        self.model_vec.eval()
        crys_vec = []
        with torch.no_grad():
            for input, _ in loader:
                atom_fea, atom_symm, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
                vecs = self.model_vec(atom_fea=atom_fea, 
                                      atom_symm=atom_symm,
                                      nbr_fea=nbr_fea,
                                      nbr_fea_idx=nbr_fea_idx, 
                                      crystal_atom_idx=crystal_atom_idx)
                crys_vec.append(vecs)
        return torch.cat(crys_vec)
        
    def model_select(self, model_path):
        """
        select models with lowest validation mae
        
        Parameters
        ----------
        model_path [str, 0d]: path of prediction model
        
        Returns
        ----------
        best_models [str, 1d, np]: name of best models
        """
        files = os.listdir(model_path)
        models = [i for i in files if i.startswith('check')]
        models = sorted(models)
        valid_file = f'{model_path}/validation.dat'
        mae = self.import_list2d(valid_file, float)
        order = np.argsort(np.ravel(mae))
        sort_models = np.array(models)[order]
        best_models = sort_models[:num_models]
        return best_models
    
    def get_energy_bagging(self, model_names, loader):
        """
        predict enegy by several models

        Parameters
        ----------
        model_names [str, 1d]: full name of models
        loader [obj]: dataloader
        
        Returns
        ----------
        ave_energy [float, 1d, tensor]: mean of energy prediction
        """
        energy_pred = []
        for name in model_names:
            self.load_model(name)
            pred = self.predict_batch(loader)
            energy_pred.append(pred)
        model_pred = torch.cat(energy_pred, axis=1)
        ave_energy = torch.mean(model_pred, axis=1)
        return ave_energy
    
    def load_model(self, model_name):
        """
        load predict model

        Parameters
        ----------
        model_name [str, 0d]: full name of model
        """
        self.model_val = CrystalGraphConvNet()
        params = torch.load(model_name, map_location=self.device)
        #load parameters of prediction model
        self.model_val.load_state_dict(params['state_dict'])
        self.normalizer.load_state_dict(params['normalizer'])
    
    def preserve_models(self, model_names):
        """
        preserve best prediction models
        
        Parameters
        ----------
        model_names [str, 1d]: name of models
        """
        files = os.listdir(self.model_save_path)
        models = [i for i in files if i.startswith('check')]
        models = np.setdiff1d(models, model_names)
        model_str = ' '.join(models)
        shell_script = f'''
                        #!/bin/bash
                        cd {self.model_save_path}
                        rm {model_str}
                        '''
        os.system(shell_script)
    
    def predict_batch(self, loader):
        """
        predict energy in batch
        
        Parameters
        ----------
        loader [obj]: dataloader
        
        Returns
        ----------
        energy [float, 1d, tensor]: prediction energy
        """
        self.model_val = DataParallel(self.model_val)
        self.model_val.to(self.device)
        self.model_val.eval()
        energy = []
        with torch.no_grad():
            for input, _ in loader:
                atom_fea, atom_symm, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
                pred = self.model_val(atom_fea=atom_fea, 
                                      atom_symm=atom_symm,
                                      nbr_fea=nbr_fea,
                                      nbr_fea_idx=nbr_fea_idx, 
                                      crystal_atom_idx=crystal_atom_idx)
                energy.append(self.normalizer.denorm(pred))
        return torch.cat(energy)
    
    def select_samples(self, mean_pred):
        """
        select lowest energy samples
        
        Parameters
        ----------
        mean_pred [1d, tensor]: mean of prediction
        
        Returns
        ----------
        idx [int, 1d, np]: index of samples
        """
        sample_num = len(mean_pred)
        num_min = int(ratio_min_energy*sample_num)
        #get index of samples
        _, mean_idx = torch.sort(mean_pred)
        idx = mean_idx[:num_min].cpu().numpy()
        system_echo(f'After Filter---sample number: {len(idx)}')
        return idx
    
    def reduce(self, crys_fea):
        """
        reduce dimension of crys_fea from 128 to num_components
        
        Parameters
        ----------
        crys_fea [float, 2d, np]: crystal feature vector
        
        Returns
        ----------
        crys_embedded [float, 2d, np]: redcued crystal vector
        """
        crys_embedded = TSNE(n_components=2, 
                             learning_rate='auto', 
                             init='random',
                             random_state=0).fit_transform(crys_fea)
        return crys_embedded
    
    def cluster(self, num, crys_embedded):
        """
        group reduced crys_fea into n clusters
        
        Parameters
        ----------
        num [int, 0d]: number of clusters
        crys_embedded [float, 2d, np]: redcued crystal vector
        
        Returns
        ----------
        kmeans.labels_ [int, 1d, np]: cluster labels 
        """
        sample_num = len(crys_embedded)
        if num > sample_num:
            num = sample_num
        kmeans = KMeans(n_clusters=num).fit(crys_embedded)
        return kmeans.labels_
    
    def min_in_cluster(self, idx, value, clusters):
        """
        select lowest value sample in each cluster

        Parameters
        ----------
        idx [int, 1d, np]: index of samples in input
        value [float, 1d, np]: average prediction
        clusters [int, 1d, np]: cluster labels of samples
        
        Returns
        ----------
        idx_slt [int, 1d, np]: index of select samples
        """
        #
        order = np.argsort(clusters)
        sort_idx = idx[order]
        sort_value = value[order]
        sort_clusters = clusters[order]
        #
        min_idx, min_value = 0, 1e10
        last_cluster = sort_clusters[0]
        idx_slt = []
        for i, cluster in enumerate(sort_clusters):
            if cluster == last_cluster:
                pred = sort_value[i]
                if min_value > pred:
                    min_idx = sort_idx[i]
                    min_value = pred                     
            else:
                idx_slt.append(min_idx)
                last_cluster = cluster
                min_idx = sort_idx[i]
                min_value = sort_value[i]
        idx_slt.append(min_idx)
        return np.array(idx_slt)
    
    def write_POSCARs(self, atom_pos, atom_type, atom_symm,
                      grid_name, grid_ratio, space_group, add_thickness=False):
        """
        position and type are sorted by grid name and space group
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        add_thickness [bool, 0d]: whether get puckered structure
        """
        #make save directory
        if not os.path.exists(self.poscar_save_path):
            os.mkdir(self.poscar_save_path)
        if not os.path.exists(self.sh_save_path):
            os.mkdir(self.sh_save_path)
        #convert to structure object
        num_jobs = len(grid_name)
        node_assign = self.assign_node(num_jobs)
        strus = self.get_stru_bh(atom_pos, atom_type, grid_name, grid_ratio, space_group, add_thickness)
        for i, stru in enumerate(strus):
            file_name = f'{self.poscar_save_path}/POSCAR-{i:03.0f}-{node_assign[i]}'
            stru.to(filename=file_name, fmt='poscar')
        #export dat file of select structures
        self.write_list2d(f'{self.sh_save_path}/atom_pos_select.dat',
                          atom_pos)
        self.write_list2d(f'{self.sh_save_path}/atom_type_select.dat', 
                          atom_type)
        self.write_list2d(f'{self.sh_save_path}/atom_symm_select.dat', 
                          atom_symm)
        self.write_list2d(f'{self.sh_save_path}/grid_name_select.dat', 
                          np.transpose([grid_name]))
        self.write_list2d(f'{self.sh_save_path}/grid_ratio_select.dat', 
                          np.transpose([grid_ratio]))
        self.write_list2d(f'{self.sh_save_path}/space_group_select.dat', 
                          np.transpose([space_group]))    
    
    def export_recycle(self, recyc, atom_pos, atom_type, atom_symm, 
                       grid_name, grid_ratio, space_group, energy):
        """
        export configurations after sccop

        Parameters
        ----------
        recyc [int, 0d]: recycle times
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number 
        energy [float, 1d]: energy of structure
        """
        self.poscar_save_path = f'{poscar_path}/SCCOP-{recyc}'
        self.sh_save_path = self.poscar_save_path
        if not os.path.exists(self.poscar_save_path):
            os.mkdir(self.poscar_save_path)
        #sort structure in order of grid and space group
        idx = self.sort_by_grid_sg(grid_name, space_group)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group)
        energy = np.array(energy)[idx]
        #delete structures that are selected before
        if recyc > 0:
            #delete same selected structures by pos, type, grid, sg
            recyc_pos, recyc_type, recyc_grid, recyc_ratio, recyc_sg = self.collect_select(recyc)
            idx = self.delete_same_selected(atom_pos, atom_type, grid_name, grid_ratio, space_group,
                                            recyc_pos, recyc_type, recyc_grid, recyc_ratio, recyc_sg)
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, grid_ratio, space_group)
            energy = np.array(energy)[idx]
            system_echo(f'Delete duplicates same as previous recycle: {len(grid_name)}')
            #filter structure by energy
            idx = np.argsort(energy)[:2*num_recycle*num_poscars]
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, grid_ratio, space_group)
            energy = np.array(energy)[idx]
            #delete same selected structures by pymatgen
            strus_1 = self.get_stru_bh(atom_pos, atom_type, grid_name, grid_ratio, space_group)
            strus_2 = self.collect_optim(recyc)
            idx = self.delete_same_selected_pymatgen(strus_1, strus_2)
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, grid_ratio, space_group)
            energy = np.array(energy)[idx]
            system_echo(f'Delete duplicates same as previous recycle: {len(grid_name)}')
        #filter structure by energy
        idx = np.argsort(energy)[:2*num_poscars]
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, grid_ratio, space_group)
        energy = np.array(energy)[idx]
        #delete same structures by pymatgen
        idx = self.delete_duplicates_pymatgen(atom_pos, atom_type,
                                              grid_name, grid_ratio, space_group)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group)
        energy = np.array(energy)[idx]
        system_echo(f'Delete duplicates: {len(idx)}')
        #select Top k lowest energy structures
        idx = np.argsort(energy)[:num_poscars]
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, grid_ratio, space_group)
        #puckered structure
        add_thickness = False
        if dimension == 2 and thickness > 0:
            add_thickness = True
        self.write_POSCARs(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, add_thickness)
        system_echo(f'SCCOP optimize structures: {len(grid_name)}')
    
    def collect_select(self, recyc):
        """
        collect selected samples in each recycle
        
        Parameters
        ----------
        recyc [int, 0d]: recycle times
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        """
        atom_pos, atom_type, grid_name, grid_ratio, space_group = [], [], [], [], []
        for i in range(recyc):
            head = f'{poscar_path}/SCCOP-{i}'
            atom_pos += self.import_list2d(f'{head}/atom_pos_select.dat', int)
            atom_type += self.import_list2d(f'{head}/atom_type_select.dat', int)
            grid_name += self.import_list2d(f'{head}/grid_name_select.dat', int)
            grid_ratio += self.import_list2d(f'{head}/grid_ratio_select.dat', float)
            space_group += self.import_list2d(f'{head}/space_group_select.dat', int)
        grid_name = np.array(grid_name).flatten().tolist()
        grid_ratio = np.array(grid_ratio).flatten().tolist()
        space_group = np.array(space_group).flatten().tolist()
        return atom_pos, atom_type, grid_name, grid_ratio, space_group
    
    def collect_optim(self, recyc):
        """
        collect optimized samples in each recycle

        Parameters
        ----------
        recyc [int, 0d]: recycle times
        
        Returns
        ----------
        strus [obj, 1d]: structure object in pymatgen
        """
        #collect optimized structures in each recycle
        full_poscars = []
        for i in range(1, recyc+1):
            poscars = os.listdir(f'{init_strus_path}_{i}')
            full_poscars += [f'{init_strus_path}_{i}/{j}' for j in poscars
                             if not j.startswith('POSCAR-RCSD')]
        #get structure objects
        strus = []
        for poscar in full_poscars:
            stru = Structure.from_file(poscar)
            strus.append(stru)
        return strus
    
    def optim_strus(self):
        """
        select top k minimal energy structures
        """
        if not os.path.exists(sccop_out_path):
            os.mkdir(sccop_out_path)
        #delete same structures
        strus, energy = self.collect_recycle()
        idx = self.delete_same_strus_energy(strus, energy)
        energy = energy[idx]
        #export top k structures
        order = np.argsort(energy)[:num_optims]
        slt_idx = np.array(idx)[order]
        stru_num = len(order)
        assign = self.assign_node(stru_num)
        for i, idx in enumerate(slt_idx):
            stru = strus[idx]
            file_name = f'{sccop_out_path}/POSCAR-{i:03.0f}-{assign[i]}'
            stru.to(filename=file_name, fmt='poscar')
        system_echo(f'Optimize configurations: {stru_num}')
    
    def collect_recycle(self):
        """
        collect poscars and corresponding energys from each recycle
        
        Returns
        ----------
        strus [obj, 1d]: structure object in pymatgen
        energys [float, 1d]: energy of structures
        """
        poscars, energys = [], []
        #get poscars and corresponding energys
        for i in range(1, num_recycle+1):
            energy_file = f'{vasp_out_path}/initial_strus_{i}/Energy.dat'
            energy_dat = self.import_list2d(energy_file, str, numpy=True)
            poscar, energy = np.transpose(energy_dat)
            poscar = [f'{init_strus_path}_{i}/{j[4:]}' for j in poscar]
            poscars = np.concatenate((poscars, poscar))
            energys = np.concatenate((energys, energy))
        #get structure objects
        strus = []
        for poscar in poscars:
            stru = Structure.from_file(poscar)
            strus.append(stru)
        return strus, np.array(energys, dtype=float)

    def ml_sampling(self, recyc, atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group):
        """
        choose lowest energy structure in different clusters
        
        Parameters
        ----------
        recyc [int, 0d]: recycle of sccop
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        """
        crys_vec, mean_pred = self.get_crys_and_energy(atom_pos, atom_type, atom_symm, 
                                                        grid_name, grid_ratio, space_group)
        self.export_buffer(recyc, crys_vec, atom_pos, atom_type, atom_symm, 
                           grid_name, grid_ratio, space_group)
        #select low energy structures as initial points
        idx = np.argsort(mean_pred)[:int(.5*len(mean_pred))]
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group)
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group
    
    def get_crys_and_energy(self, atom_pos, atom_type, atom_symm,
                            grid_name, grid_ratio, space_group):
        """
        get crystal vector and energy by ML
        """
        #balance samples to GPU
        num_crys = len(atom_pos)
        tuple = atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group
        batch_balance(num_crys, 10*self.batch_size, tuple)
        #sort structure in order of grid and space group
        idx = self.sort_by_grid_sg(grid_name, space_group)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group)
        #predict energy and get crystal vector
        loader = self.dataloader(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group)
        crys_vec = self.get_crystal_vector(loader)
        model_names = self.get_init_models()
        mean_pred = self.get_energy_bagging(model_names, loader)
        mean_pred = mean_pred.cpu().numpy()
        crys_vec = crys_vec.cpu().numpy()
        return crys_vec, mean_pred
    
    def get_init_models(self):
        """
        get initial prediction models
        
        Returns
        ----------
        model_names [str, 1d]: full name of models 
        """
        model_dir = sorted(os.listdir(model_path))
        if len(model_dir) == 0:
            if dimension == 2:
                model_names = [pretrain_model_2d]
            elif dimension == 3:
                model_names = [pretrain_model_3d]
        else:
            path = f'{model_path}/{model_dir[-1]}'
            files = os.listdir(path)
            models = [i for i in files if i.startswith('check')]
            model_names = [f'{path}/{i}' for i in models]
        return model_names

    def export_buffer(self, recyc, crys_vec, atom_pos, atom_type, atom_symm,
                      grid_name, grid_ratio, sgs):
        """
        save random searching results
        
        Parameters
        ----------
        recyc [int, 2d]: position of atoms
        crys_vec [float, 2d]: crystal vector
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: grid name
        grid_ratio [float, 1d]: ratio of grids
        sgs [int, 1d]: space group number
        """
        #export sampling results
        self.write_list2d(f'{buffer_path}/crystal_vector_{recyc}.dat',
                          crys_vec, style='{0:8.6f}')
        self.write_list2d(f'{buffer_path}/atom_pos_{recyc}.dat',
                          atom_pos, style='{0:4.0f}')
        self.write_list2d(f'{buffer_path}/atom_type_{recyc}.dat',
                          atom_type, style='{0:4.0f}')
        self.write_list2d(f'{buffer_path}/atom_symm_{recyc}.dat',
                          atom_symm, style='{0:4.0f}')
        self.write_list2d(f'{buffer_path}/grid_name_{recyc}.dat',
                          np.transpose([grid_name]), style='{0:4.0f}')
        self.write_list2d(f'{buffer_path}/grid_ratio_{recyc}.dat',
                          np.transpose([grid_ratio]), style='{0:8.4f}')
        self.write_list2d(f'{buffer_path}/space_group_{recyc}.dat',
                          np.transpose([sgs]), style='{0:4.0f}')
        
    
class FeatureExtractNet(CrystalGraphConvNet):
    #get crystal vector
    def __init__(self):
        super(FeatureExtractNet, self).__init__()
        
    def forward(self, atom_fea, atom_symm, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, atom_symm, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        return crys_fea
    
    
class BayesianOpt(Select):
    #select initial searching points by bayesian method
    def __init__(self, iteration):
        Select.__init__(self, iteration)
    
    def select(self, recyc, atom_pos, atom_type, atom_symm,
               grid_name, grid_ratio, space_group, energys):
        """
        select initial searching points
        
        Parameters
        ----------
        recyc [int, 0d]: recycle of sccop
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group
        energys [float, 1d]: energy of structures
        
        Returns
        ----------
        init_pos [int, 2d]: initial position of atoms
        init_type [int, 2d]: initial type of atoms
        init_symm [int, 2d]: initial symmetry of atoms
        init_grid [int, 1d]: initial name of grids
        init_ratio [float, 1d]: initial ratio of grids
        init_sgs [int, 1d]: initial space group
        """
        #import random sampling buffer
        crys_vec_all, pos, type, symm, grid, ratio, sgs = self.import_buffer(recyc)
        #train gaussian distribution
        gp = GaussianProcessRegressor()
        crys_vec, _ = self.get_crys_and_energy(atom_pos, atom_type, atom_symm, 
                                               grid_name, grid_ratio, space_group)
        gp.fit(crys_vec, energys)
        e_min = min(energys)
        #select most potential structures
        idx = self.ac_max(num_path, crys_vec_all, gp, e_min)
        init_pos, init_type, init_symm, init_grid, init_ratio, init_sgs = \
            self.filter_samples(idx, pos, type, symm, grid, ratio, sgs)
        return init_pos, init_type, init_symm, init_grid, init_ratio, init_sgs
    
    def ac_max(self, num, states, gp, e_min):
        """
        get index of max PI value samples
        
        Parameters
        ----------
        num [int, 0d]: number of samples
        states [list, 2d]: state vectors
        gp [obj, 0d]: gaussian process object
        e_min [float, 0d]: minimal energy
        
        Returns
        ----------
        idx [int, 1d, np]: index of selected samples
        """
        ac = self.PI(states, gp, e_min)
        weight = ac/np.sum(ac)
        idx_all = np.arange(len(ac))
        idx = np.random.choice(idx_all, num, p=weight)
        return idx
    
    def PI(self, states, gp, e_min):
        """
        acquisition function
        
        Parameters
        ----------
        states [list, 2d]: state vectors
        gp [obj, 0d]: gaussian process object
        e_min [float, 0d]: minimal energy

        Returns
        ----------
        ac [float, 1d, np]: acquisition value
        """
        xi = np.random.uniform(low=-.05, high=.05, size=len(states))
        mean, std = gp.predict(states, return_std=True)
        z = (mean - e_min - xi)/std
        ac = 1 - norm.cdf(z)
        return ac
    
    def import_buffer(self, recyc):
        """
        import samples 
        """
        #import buffer data
        crys_vec = []
        atom_pos, atom_type, atom_symm = [], [], []
        grid_name, grid_ratio, space_group = [], [], []
        for i in range(recyc+1):
            crys_vec += self.import_list2d(f'{buffer_path}/crystal_vector_{i}.dat', float)
            atom_pos += self.import_list2d(f'{buffer_path}/atom_pos_{i}.dat', int)
            atom_type += self.import_list2d(f'{buffer_path}/atom_type_{i}.dat', int)
            atom_symm += self.import_list2d(f'{buffer_path}/atom_symm_{i}.dat', int)
            grid_name_2d = self.import_list2d(f'{buffer_path}/grid_name_{i}.dat', int)
            grid_ratio_2d = self.import_list2d(f'{buffer_path}/grid_ratio_{i}.dat', float)
            space_group_2d = self.import_list2d(f'{buffer_path}/space_group_{i}.dat', int)
            grid_name += np.ravel(grid_name_2d).tolist()
            grid_ratio += np.ravel(grid_ratio_2d).tolist()
            space_group += np.ravel(space_group_2d).tolist()
        return crys_vec, atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group
    

if __name__ == '__main__':
    pass