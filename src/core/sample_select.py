import sys, os
import re
import shutil
import torch
import numpy as np

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.utils import *
from core.predict import *
from core.data_transfer import DeleteDuplicates


class Select(SSHTools, DeleteDuplicates):
    #Select training samples by active learning
    def __init__(self, round, batch_size=1024, num_workers=0):
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
        
    def samples(self, atom_pos, atom_type, grid_name,
                train_pos, train_type, train_grid):
        """
        choose lowest energy structure in different clusters
        each structure is unique by pos, type and grid

        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 2d]: name of grids
        train_pos [int, 2d]: position in training set
        train_type [int, 2d]: type in training set
        train_grid [int, 1d]: grid in training set
        """
        #delete same structures
        grid_name = np.ravel(grid_name)
        atom_pos, atom_type, grid_name, _ = \
            self.delete_duplicates(atom_pos, atom_type, grid_name)
        num_crys = len(atom_pos)
        system_echo(f'Delete duplicates in searching samples: {num_crys}')
        #delete same structures according to training set
        atom_pos, atom_type, grid_name, _ = \
            self.delete_same_selected(atom_pos, atom_type, grid_name,
                                      train_pos, train_type, train_grid)
        tuple = atom_pos, atom_type, grid_name
        batch_balance(num_crys, self.batch_size, tuple)
        num_crys = len(grid_name)
        system_echo(f'Delete duplicates same as trainset: {num_crys}')
        #predict energy and get crystal vector
        loader = self.dataloader(atom_pos, atom_type, grid_name)
        model_names = self.model_select()
        mean_pred, std_pred, crys_mean = self.mean(model_names, loader)
        idx_few = self.filter(mean_pred, std_pred)
        mean_pred_few = mean_pred[idx_few].cpu().numpy()
        crys_mean_few = crys_mean[idx_few].cpu().numpy()
        #reduce dimension and clustering
        crys_embedded = self.reduce(crys_mean_few)
        clusters = self.cluster(crys_embedded, num_clusters)
        idx_slt = self.min_in_cluster(idx_few, mean_pred_few, clusters)
        self.write_POSCARs(idx_slt, atom_pos, atom_type, grid_name)
    
    def delete_same_selected(self, atom_pos, atom_type, grid_name,
                             train_pos, train_type, train_grid):
        """
        delete same structures that are already selected
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atom
        atom_type [int, 2d]: type of atom
        grid_name [int, 1d]: name of grid
        train_pos [int, 2d]: position in training set
        train_type [int, 2d]: type in training set
        train_grid [int, 1d]: grid in training set
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atom
        atom_type [int, 2d]: type of atom
        grid_name [int, 1d]: name of grid
        index [int, 1d]: index of sample in origin array
        """
        #different length of atoms convert to string
        grid_name = np.transpose([grid_name])
        train_grid = np.transpose([train_grid])
        atom_pos_str = self.list2d_to_str(atom_pos, '{0}')
        atom_type_str = self.list2d_to_str(atom_type, '{0}')
        grid_name_str = self.list2d_to_str(grid_name, '{0}')
        train_pos_str = self.list2d_to_str(train_pos, '{0}')
        train_type_str = self.list2d_to_str(train_type, '{0}')
        train_grid_str = self.list2d_to_str(train_grid, '{0}')
        #delete same structure accroding to pos, type and grid
        array_1 = [i+'-'+j+'-'+k for i, j, k in 
                   zip(atom_pos_str, atom_type_str, grid_name_str)]
        array_2 = [i+'-'+j+'-'+k for i, j, k in 
                   zip(train_pos_str, train_type_str, train_grid_str)]
        array = np.concatenate((array_1, array_2))
        _, index, counts = np.unique(array, return_index=True, return_counts=True)
        #delete structures same as selected set
        same_index = []
        for i, repeat in enumerate(counts):
            if repeat > 1:
                same_index.append(i)
        sample_num = len(grid_name)
        index = np.delete(index, same_index)
        index = [i for i in index if i < sample_num]
        atom_pos = np.array(atom_pos, dtype=object)[index]
        atom_type = np.array(atom_type, dtype=object)[index]
        grid_name = grid_name.flatten()[index]
        #sorted by grid name
        order  = np.argsort(grid_name)
        atom_pos = atom_pos[order]
        atom_type = atom_type[order]
        grid_name = grid_name[order]
        index = np.array(index)[order]
        #convert to list
        atom_pos = atom_pos.tolist()
        atom_type = atom_type.tolist()
        grid_name = grid_name.tolist()
        index = index.tolist()
        return atom_pos, atom_type, grid_name, index
        
    def dataloader(self, atom_pos, atom_type, grid_name):
        """
        transfer data to the input of graph network and
        put them into DataLoader
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atom
        atom_type [int, 2d]: type of atom
        grid_name [int, 1d]: name of grid
        
        Returns
        ----------
        loader [obj]: dataloader 
        """
        last_grid = grid_name[0]
        transfer = Transfer(last_grid)
        atom_feas, nbr_feas, nbr_fea_idxes = [], [], []
        i = 0
        for j, grid in enumerate(grid_name):
            if not grid == last_grid:
                atom_fea, nbr_fea, nbr_fea_idx = \
                    transfer.batch(atom_pos[i:j], atom_type[i:j])    
                atom_feas += atom_fea
                nbr_feas += nbr_fea
                nbr_fea_idxes += nbr_fea_idx
                transfer = Transfer(grid)
                last_grid = grid
                i = j
        atom_fea, nbr_fea, nbr_fea_idx = \
            transfer.batch(atom_pos[i:], atom_type[i:])
        atom_feas += atom_fea
        nbr_feas += nbr_fea
        nbr_fea_idxes += nbr_fea_idx
        targets = np.zeros(len(grid_name))
        dataset = PPMData(atom_feas, nbr_feas, 
                          nbr_fea_idxes, targets)
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
        models = [i for i in files if re.match(r'check', i)]
        models = sorted(models)
        valid_file = f'{self.model_save_path}/validation.dat'
        mae = self.import_list2d(valid_file, float)
        order = np.argsort(np.ravel(mae))
        sort_models = np.array(models)[order]
        best_models = sort_models[:num_models]
        self.remove_models()
        return best_models
    
    def remove_models(self,):
        #TODO remove bad models
        pass
    
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
    
    def filter(self, mean_pred, std_pred):
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
    
    def export(self, recyc, atom_pos, atom_type, energy, grid_name):
        """
        export configurations after ccop

        Parameters
        ----------
        recyc [int, 0d]: recycle times
        atom_pos [int, 2d]: position of atom
        atom_type [int, 2d]: type of atom
        energy [float, 1d]: energy of structure
        grid_name [int, 1d]: name of grid
        """
        self.round = f'CCOP-{recyc}'
        self.poscar_save_path = f'{poscar_path}/{self.round}'
        self.sh_save_path = self.poscar_save_path
        if not os.path.exists(self.poscar_save_path):
            os.mkdir(self.poscar_save_path)
        #delete same structures
        num_crys = len(grid_name)
        system_echo(f'Number of samples in trainset: {num_crys}')
        atom_pos, atom_type, grid_name, index = \
            self.delete_duplicates(atom_pos, atom_type, grid_name)
        energy = np.array(energy)[index]
        num_crys = len(grid_name)
        system_echo(f'Delete duplicates same in trainset: {num_crys}')
        #delete structures that are selected before
        if recyc > 0:
            recyc_pos, recyc_type, recyc_grid = self.collect(recyc)
            atom_pos, atom_type, grid_name, index = \
                self.delete_same_selected(atom_pos, atom_type, grid_name,
                                          recyc_pos, recyc_type, recyc_grid)
            energy = energy[index]
            num_crys = len(grid_name)
            system_echo(f'Delete duplicates same as previous selected: {num_crys}')
            atom_pos, atom_type, grid_name, energy = \
                self.delete_recycle_poscar(recyc, atom_pos, atom_type, grid_name, energy)
            num_crys = len(grid_name)
            system_echo(f'Delete duplicates same as previous recycle: {num_crys}')
        #select Top k lowest energy structures
        #export training set and select samples
        idx_slt = np.argsort(energy)[:num_poscars]
        self.write_POSCARs(idx_slt, atom_pos, atom_type, grid_name)
        self.write_recycle(recyc, idx_slt, atom_pos, atom_type, grid_name)
        system_echo(f'CCOP optimize structures: {num_poscars}')
    
    def delete_recycle_poscar(self, recyc, atom_pos, atom_type, grid_name, energy):
        """
        delete structures that have been optimized in previous recycle
        
        Parameters
        ----------
        recyc [int, 0d]: recycle times
        atom_pos [int, 2d]: position of atom
        atom_type [int, 2d]: type of atom
        grid_name [int, 1d]: name of grid
        energy [float, 1d]: energy of structure
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atom
        atom_type [int, 2d]: type of atom
        grid_name [int, 1d]: name of grid
        energy [float, 1d]: energy of structure
        """
        #transfer pos, type, grid to poscar
        idx_slt = np.argsort(energy)[:3*num_poscars]
        idx_order = self.write_POSCARs(idx_slt, atom_pos, atom_type, grid_name)
        #compare poscars with previous recycle
        poscars = os.listdir(self.poscar_save_path)
        poscars = sorted([i for i in poscars if re.match('POSCAR', i)])
        poscars = [f'{self.poscar_save_path}/{i}' for i in poscars]
        recyc_poscars = self.collect_poscars(recyc)
        index = self.compare_poscars(poscars, recyc_poscars)
        #select unoptimize structures
        idx_filter = np.delete(idx_order, index)
        atom_pos = np.array(atom_pos, dtype=object)[idx_filter]
        atom_type = np.array(atom_type, dtype=object)[idx_filter]
        grid_name = np.array(grid_name)[idx_filter]
        energy = np.array(energy)[idx_filter]
        #transfer to list
        atom_pos = atom_pos.tolist()
        atom_type = atom_type.tolist()
        grid_name = grid_name.tolist()
        energy = energy.tolist()
        #remove poscars and dat files
        files = os.listdir(self.poscar_save_path)
        for file in files:
            os.remove(f'{self.poscar_save_path}/{file}')
        return atom_pos, atom_type, grid_name, energy
    
    def collect(self, recyc):
        """
        import selected samples in each recycle

        Parameters
        ----------
        recyc [int, 0d]: recycle times
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atom
        atom_type [int, 2d]: type of atom
        grid_name [int, 1d]: name of grid
        """
        atom_pos, atom_type, grid_name = [], [], []
        for i in range(recyc):
            atom_pos += self.import_list2d(f'{record_path}/{i}/atom_pos_select.dat', int)
            atom_type += self.import_list2d(f'{record_path}/{i}/atom_type_select.dat', int)
            grid_name += self.import_list2d(f'{record_path}/{i}/grid_name_select.dat', int)
        grid_name = list(np.array(grid_name).flatten())
        return atom_pos, atom_type, grid_name
    
    def collect_poscars(self, recyc):
        """
        list selected samples in each recycle

        Parameters
        ----------
        recyc [int, 0d]: recycle times
        
        Returns
        ----------
        full_poscars [int, 2d]: full name of poscars
        """
        full_poscars = []
        for i in range(1, recyc+1):
            poscars = os.listdir(f'{init_strus_path}_{i}')
            full_poscars += [f'{init_strus_path}_{i}/{j}' for j in poscars] 
        return full_poscars
    
    def write_recycle(self, recyc, index, atom_pos, atom_type, grid_name):
        """
        export position, type, grid of trainset and select samples
        
        Parameters
        ----------
        recyc [int, 0d]: recycle times
        index [int, 1d]: index of select samples
        atom_pos [int, 2d]: position of atom
        atom_type [int, 2d]: type of atom
        grid_name [int, 1d]: name of grid
        """
        dir = f'{record_path}/{recyc}'
        if not os.path.exists(dir):
            os.mkdir(dir)
        atom_pos_np = np.array(atom_pos)
        atom_type_np = np.array(atom_type)
        grid_name_np = np.transpose([grid_name])
        self.write_list2d(f'{dir}/atom_pos.dat', atom_pos_np)
        self.write_list2d(f'{dir}/atom_type.dat', atom_type_np)
        self.write_list2d(f'{dir}/grid_name.dat', grid_name_np)
        self.write_list2d(f'{dir}/atom_pos_select.dat', atom_pos_np[index])
        self.write_list2d(f'{dir}/atom_type_select.dat', atom_type_np[index])
        self.write_list2d(f'{dir}/grid_name_select.dat', grid_name_np[index])
    
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
        order = np.argsort(clusters)
        sort_idx = idx[order]
        sort_value = value[order]
        sort_clusters = clusters[order]
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
    

class OptimSelect(SSHTools, DeleteDuplicates):
    #Select structures from low level optimization
    def __init__(self):
        if not os.path.exists(ccop_out_path):
            os.mkdir(ccop_out_path)
    
    def optim_select(self):
        """
        select optimized structuers from initial_X
        select top k minimal energy structures
        """
        #import optimized energy from each round
        poscars, poscars_full, energys = self.collect()
        #copy poscars in each round
        for poscar in poscars_full:
            shutil.copy(poscar, ccop_out_path)
        index, drop = self.delete_same_names(ccop_out_path, poscars)
        #delete duplicates
        if len(drop) > 0:
            for poscar in poscars[drop]:
                os.remove(f'{ccop_out_path}/{poscar}')
        #select top k poscars
        energys = energys[index]
        poscars = poscars[index]
        order = np.argsort(energys)
        for poscar in poscars[order[num_optims:]]:
            os.remove(f'{ccop_out_path}/{poscar}')
        #write selected poscars
        self.change_node_assign(ccop_out_path)
        select_energy = energys[order[:num_optims]]
        system_echo(f'Optimize configurations: {select_energy}')
    
    def collect(self):
        """
        collect poscars and corresponding energys from each recycle
        """
        poscars, poscars_full, energys = [], [], []
        for i in range(num_recycle):
            recyc = f'initial_strus_{i+1}'
            stru_path = f'{poscar_path}/{recyc}'
            energy_file = f'{vasp_out_path}/{recyc}/Energy.dat'
            energy_dat = self.import_list2d(energy_file, str, numpy=True)
            poscar, energy = np.transpose(energy_dat)
            poscar = [i[4:] for i in poscar]
            poscars = np.concatenate((poscars, poscar))
            full = [f'{stru_path}/{j}' for j in poscar]
            poscars_full = np.concatenate((poscars_full, full))
            energys = np.concatenate((energys, energy))
        energys = np.array(energys, dtype='float32')
        return poscars, poscars_full, energys
    
    
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
    rw = ListRWTools()
    file_head = f'{search_path}/{3:03.0f}'
    atom_pos = rw.import_list2d(f'{file_head}/atom_pos.dat', int)
    atom_type = rw.import_list2d(f'{file_head}/atom_type.dat', int)
    grid_name = rw.import_list2d(f'{file_head}/grid_name.dat', int)
    select = Select(4)
    sample_num = len(atom_pos)
    energy = [1 for _ in range(sample_num)]
    grid_name = np.ravel(grid_name)
    select.export(1, atom_pos, atom_type, energy, grid_name)