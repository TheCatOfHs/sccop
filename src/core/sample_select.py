import sys, os
import torch
import numpy as np

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.utils import *
from core.predict import *
from core.data_transfer import DeleteDuplicates


class Select(SSHTools, DeleteDuplicates):
    #Select training samples by active learning
    def __init__(self, round, batch_size=1024, num_workers=0):
        DeleteDuplicates.__init__(self)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.round = f'{round:03.0f}'
        self.model_save_path = f'{model_path}/{self.round}'
        self.poscar_save_path = f'{poscar_path}/{self.round}'
        self.sh_save_path = f'{search_path}/{self.round}'
        self.device = torch.device('cuda')
        self.normalizer = Normalizer(torch.tensor([]))
        if not os.path.exists(self.sh_save_path):
            os.mkdir(self.sh_save_path)
        if not os.path.exists(self.poscar_save_path):
            os.mkdir(self.poscar_save_path)
    
    def samples(self, atom_pos, atom_type, atom_symm, grid_name, space_group,
                train_pos, train_type, train_grid, train_sg):
        """
        choose lowest energy structure in different clusters
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 2d]: name of grids
        space_group [int, 2d]: space group number
        train_pos [int, 2d]: position in training set
        train_type [int, 2d]: type in training set
        train_grid [int, 1d]: grid in training set
        train_sg [int, 1d]: space group in training set
        """
        #delete same structures of searching
        grid_name = np.ravel(grid_name)
        space_group = np.ravel(space_group)
        idx = self.delete_duplicates(atom_pos, atom_type,
                                     grid_name, space_group)
        atom_pos, atom_type, atom_symm, grid_name, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, space_group)
        num_crys = len(atom_pos)
        system_echo(f'Delete duplicates in searching samples: {num_crys}')
        #delete same structures compared with training set
        idx = self.delete_same_selected(atom_pos, atom_type, grid_name, space_group,
                                        train_pos, train_type, train_grid, train_sg)
        atom_pos, atom_type, atom_symm, grid_name, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, space_group)
        tuple = atom_pos, atom_type, atom_symm, grid_name, space_group
        batch_balance(num_crys, self.batch_size, tuple)
        num_crys = len(grid_name)
        system_echo(f'Delete duplicates same as trainset: {num_crys}')
        #sort structure in order of grid and space group
        idx = self.sort_by_grid_sg(grid_name, space_group)
        atom_pos, atom_type, atom_symm, grid_name, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, space_group)
        #predict energy and get crystal vector
        loader = self.dataloader(atom_pos, atom_type, grid_name, space_group)
        model_names = self.model_select()
        mean_pred, std_pred, crys_mean = self.mean(model_names, loader)
        idx = self.select_samples(mean_pred, std_pred)
        mean_pred = mean_pred[idx].cpu().numpy()
        crys_mean = crys_mean[idx].cpu().numpy()
        #reduce dimension and clustering
        crys_embedded = self.reduce(crys_mean)
        clusters = self.cluster(crys_embedded, num_clusters)
        idx = self.min_in_cluster(idx, mean_pred, clusters)
        atom_pos, atom_type, atom_symm, grid_name, space_group = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, space_group)
        self.write_POSCARs(atom_pos, atom_type, atom_symm, grid_name, space_group)
    
    def dataloader(self, atom_pos, atom_type, grid_name, space_group):
        """
        transfer data to the input of graph network and
        put them into DataLoader
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atom
        atom_type [int, 2d]: type of atom
        grid_name [int, 1d]: name of grid
        space_group [int, 1d]: space group number
        
        Returns
        ----------
        loader [obj]: dataloader 
        """
        atom_fea, nbr_fea, nbr_fea_idx = \
            self.get_ppm_input_bh(atom_pos, atom_type, grid_name, space_group)
        targets = np.zeros(len(grid_name))
        dataset = PPMData(atom_fea, nbr_fea, 
                          nbr_fea_idx, targets)
        loader = get_loader(dataset, self.batch_size, self.num_workers)
        return loader
    
    def model_select(self):
        """
        select models with lowest validation mae
        
        Returns
        ----------
        best_models [str, 1d, np]: name of best models
        """
        files = os.listdir(self.model_save_path)
        models = [i for i in files if i.startswith('check')]
        models = sorted(models)
        valid_file = f'{self.model_save_path}/validation.dat'
        mae = self.import_list2d(valid_file, float)
        order = np.argsort(np.ravel(mae))
        sort_models = np.array(models)[order]
        best_models = sort_models[:num_models]
        return best_models
    
    def mean(self, model_names, loader):
        """
        calculate mean_pred, std_pred, crys_mean

        Parameters
        ----------
        model_names [str, 1d]: name of models
        loader [obj]: dataloader
        
        Returns
        ----------
        mean_pred [float, 1d, tensor]: mean of predict
        std_pred [float, 1d, tensor]: std of predict
        crys_mean [float, 2d, tensor]: mean of crystal feature
        """
        model_pred, crys_fea = [], []
        for name in model_names:
            self.load_model(name)
            pred, vec = self.predict_batch(loader)
            model_pred.append(pred)
            crys_fea.append(vec)
        model_pred = torch.cat(model_pred, axis=1)
        mean_pred = torch.mean(model_pred, axis=1)
        std_pred = torch.std(model_pred, axis=1)
        crys_mean = torch.zeros(crys_fea[0].shape).to(self.device)
        for tensor in crys_fea:
            crys_mean += tensor
        crys_mean = crys_mean/num_models
        return mean_pred, std_pred, crys_mean
    
    def load_model(self, model_name):
        """
        load predict model

        Parameters
        ----------
        model_name [str, 0d]: name of model
        """
        self.model_vec = FeatureExtractNet(orig_atom_fea_len, 
                                           nbr_bond_fea_len)
        self.model_val = ReadoutNet(orig_atom_fea_len,
                                    nbr_bond_fea_len)
        paras = torch.load(f'{self.model_save_path}/{model_name}', 
                           map_location=self.device)
        self.model_vec.load_state_dict(paras['state_dict'])
        self.model_val.load_state_dict(paras['state_dict'])
        self.normalizer.load_state_dict(paras['normalizer'])
        
    def predict_batch(self, loader):
        """
        calculate crys_fea and energy by predict model
        
        Parameters
        ----------
        loader [obj]: dataloader
        
        Returns
        ----------
        pred [float, 2d, tensor]: predict value
        crys_fea [float, 2d, tensor]: crystal feature
        """
        self.model_vec = DataParallel(self.model_vec)
        self.model_val = DataParallel(self.model_val)
        self.model_vec.to(self.device)
        self.model_val.to(self.device)
        self.model_vec.eval()
        self.model_val.eval()
        pred, crys_fea = [], []
        with torch.no_grad():
            for _, (input, _) in enumerate(loader):
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
                vec = self.model_vec(atom_fea=atom_fea, 
                                     nbr_fea=nbr_fea,
                                     nbr_fea_idx=nbr_fea_idx, 
                                     crystal_atom_idx=crystal_atom_idx)
                crys_fea.append(vec)
            crys_fea = torch.cat(crys_fea, dim=0)
            crys_fea_assign = torch.chunk(crys_fea, num_gpus)
            pred = self.model_val(crys_fea=crys_fea_assign)
        pred = self.normalizer.denorm(pred)
        return pred, crys_fea
    
    def select_samples(self, mean_pred, std_pred):
        """
        select most uncertainty samples and
        lowest energy samples
        
        Parameters
        ----------
        mean_pred [1d, tensor]: mean of prediction
        std_pred [1d, tensor]: std of prediction
        
        Returns
        ----------
        idx [int, 1d, np]: index of samples
        """
        sample_num = len(std_pred)
        num_min = ratio_min_energy*sample_num
        num_max = ratio_max_std*sample_num
        min_top = int(num_min)
        max_top = sample_num - int(num_max)
        #get index of samples
        _, mean_idx = torch.sort(mean_pred)
        _, std_idx = torch.sort(std_pred)
        min_idx = mean_idx[:min_top].cpu().numpy()
        max_idx = std_idx[max_top:].cpu().numpy()
        idx = np.union1d(min_idx, max_idx)
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
        crys_embedded = TSNE(n_components=num_components, 
                             learning_rate='auto', 
                             init='random',
                             random_state=0).fit_transform(crys_fea)
        return crys_embedded
    
    def cluster(self, crys_embedded, num_clusters):
        """
        group reduced crys_fea into n clusters
        
        Parameters
        ----------
        crys_embedded [float, 2d, np]: redcued crystal vector
        
        Returns
        ----------
        kmeans.labels_ [int, 1d, np]: cluster labels 
        """
        kmeans = KMeans(num_clusters, 
                        random_state=0).fit(crys_embedded)
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
    
    def write_POSCARs(self, atom_pos, atom_type, atom_symm, grid_name, space_group):
        """
        position and type are sorted by grid name and space group
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: 
        grid_name [int, 1d]: name of grids
        space_group [int, 1d]: space group number
        """
        #convert to structure object
        num_jobs = len(grid_name)
        node_assign = self.assign_node(num_jobs)
        strus = self.get_stru_bh(atom_pos, atom_type, grid_name, space_group)
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
        self.write_list2d(f'{self.sh_save_path}/space_group_select.dat', 
                          np.transpose([space_group]))    
    
    def export_recycle(self, recyc, atom_pos, atom_type, atom_symm, 
                       grid_name, space_group, energy):
        """
        export configurations after ccop

        Parameters
        ----------
        recyc [int, 0d]: recycle times
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grid
        space_group [int, 1d]: space group number 
        energy [float, 1d]: energy of structure
        """
        self.poscar_save_path = f'{poscar_path}/CCOP-{recyc}'
        self.sh_save_path = self.poscar_save_path
        if not os.path.exists(self.poscar_save_path):
            os.mkdir(self.poscar_save_path)
        #delete structures that are selected before
        if recyc > 0:
            #delete same selected structures by pos, type, grid, sg
            recyc_pos, recyc_type, recyc_grid, recyc_sg = self.collect_select(recyc)
            idx = self.delete_same_selected(atom_pos, atom_type, grid_name, space_group,
                                            recyc_pos, recyc_type, recyc_grid, recyc_sg)
            atom_pos, atom_type, atom_symm, grid_name, space_group = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, space_group)
            energy = np.array(energy)[idx]
            system_echo(f'Delete duplicates same as previous recycle: {len(grid_name)}')
            #sort structure in order of grid and space group
            idx = self.sort_by_grid_sg(grid_name, space_group)
            atom_pos, atom_type, atom_symm, grid_name, space_group = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, space_group)
            energy = np.array(energy)[idx]
            #delete same selected structures by pymatgen
            strus_1 = self.get_stru_bh(atom_pos, atom_type, grid_name, space_group)
            strus_2 = self.collect_optim(recyc)
            idx = self.delete_same_selected_pymatgen(strus_1, strus_2)
            atom_pos, atom_type, atom_symm, grid_name, space_group = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, space_group)
            energy = np.array(energy)[idx]
            system_echo(f'Delete duplicates same as previous recycle: {len(grid_name)}')
        #select Top k lowest energy structures
        idx = np.argsort(energy)[:num_poscars]
        atom_pos, atom_type, atom_symm, grid_name, space_group = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, space_group)
        self.write_POSCARs(atom_pos, atom_type, atom_symm, grid_name, space_group)
        system_echo(f'CCOP optimize structures: {len(grid_name)}')
    
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
        grid_name [int, 1d]: name of grid
        space_group [int, 1d]: space group number
        """
        atom_pos, atom_type, grid_name, space_group = [], [], [], []
        for i in range(recyc):
            head = f'{poscar_path}/CCOP-{i}'
            atom_pos += self.import_list2d(f'{head}/atom_pos_select.dat', int)
            atom_type += self.import_list2d(f'{head}/atom_type_select.dat', int)
            grid_name += self.import_list2d(f'{head}/grid_name_select.dat', int)
            space_group += self.import_list2d(f'{head}/space_group_select.dat', int)
        grid_name = np.array(grid_name).flatten().tolist()
        space_group = np.array(space_group).flatten().tolist()
        return atom_pos, atom_type, grid_name, space_group
    
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
        if not os.path.exists(ccop_out_path):
            os.mkdir(ccop_out_path)
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
            file_name = f'{ccop_out_path}/POSCAR-{i:03.0f}-{assign[i]}'
            stru.to(filename=file_name, fmt='poscar')
        system_echo(f'Optimize configurations: {stru_num}')
    
    def collect_recycle(self):
        """
        collect poscars and corresponding energys from each recycle
        
        Returns
        ----------
        strus []:
        energys []:
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

    
class FeatureExtractNet(CrystalGraphConvNet):
    #Calculate crys_fea
    def __init__(self, orig_atom_fea_len, nbr_fea_len):
        super(FeatureExtractNet, self).__init__(orig_atom_fea_len, nbr_fea_len)
        
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        return crys_fea


class ReadoutNet(CrystalGraphConvNet):
    #Calculate energy by crys_fea
    def __init__(self, orig_atom_fea_len, nbr_fea_len):
        super(ReadoutNet, self).__init__(orig_atom_fea_len, nbr_fea_len)

    def forward(self, crys_fea):
        out = self.fc_out(crys_fea)
        return out
    

if __name__ == '__main__':
    select = Select(1)
    select.optim_strus()
    #from core.sub_vasp import VASPoptimize
    #vasp = VASPoptimize(1)
    #vasp.run_optimization_high(vdW=add_vdW)