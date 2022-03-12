import sys, os
import re
import shutil
import torch
import numpy as np
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from pymatgen.core.structure import Structure


sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.data_transfer import Transfer
from core.utils import ListRWTools, SSHTools, system_echo
from core.predict import CrystalGraphConvNet, batch_balance
from core.predict import DataParallel, Normalizer
from core.predict import PPMData, get_loader
from core.initialize import InitSampling


class Select(ListRWTools, SSHTools):
    #Select training samples by active learning
    def __init__(self, round,
                 batch_size=1024, num_workers=0):
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
        
    def samples(self, atom_pos, atom_type, grid_name):
        """
        choose lowest energy structure in different clusters
        each structure is unique by pos, type and grid
        atom_pos: atom positions in list
        atom_type: atom types in list
        grid_name: grid in np array

        Parameters
        ----------
        atom_pos [int, 2d]: position of atom
        atom_type [int, 2d]: type of atom
        grid_name [int, 2d]: name of grid
        """
        atom_pos, atom_type, grid_name = \
            self.delete_duplicates(atom_pos, atom_type, grid_name)
        self.num_crys = len(atom_pos)
        system_echo(f'Delete duplicates---sample number: {self.num_crys}')
        loader = self.dataloader(atom_pos, atom_type, grid_name)
        model_names = self.model_select()
        mean_pred, std_pred, crys_mean = self.mean(model_names, loader)
        idx_few = self.filter(mean_pred, std_pred)
        mean_pred_few = mean_pred[idx_few].cpu().numpy()
        crys_mean_few = crys_mean[idx_few].cpu().numpy()
        crys_embedded = self.reduce(crys_mean_few)
        clusters = self.cluster(crys_embedded, num_clusters)
        idx_slt = self.min_in_cluster(idx_few, mean_pred_few, clusters)
        self.write_POSCARs(idx_slt, atom_pos, atom_type, grid_name)
        
    def delete_duplicates(self, atom_pos, atom_type, grid_name):
        """
        delete same configurations
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 2d]: grid of atoms

        Returns
        ----------
        pos [int, 2d]: reduced position of atoms
        type [int, 2d]: reduced type of atoms
        grid [int, 1d, np]: reduced grid of atoms
        """
        pos_str = self.list2d_to_str(atom_pos, '{0}')
        type_str = self.list2d_to_str(atom_type, '{0}')
        grid_str = self.list2d_to_str(grid_name, '{0}')
        pos_type_grid = [i+'-'+j+'-'+k for i, j, k in 
                         zip(pos_str, type_str, grid_str)]
        reduce = np.unique(pos_type_grid)
        reduce = np.array([i.split('-') for i in reduce])
        pos_str, type_str, grid_str = \
            reduce[:,0], reduce[:,1], reduce[:,2]

        grid = self.str_to_list1d(grid_str, int)
        order  = np.argsort(grid)
        grid = np.array(grid)[order]
        pos_str = pos_str[order]
        type_str = type_str[order]
        
        pos = self.str_to_list2d(pos_str, int)
        type = self.str_to_list2d(type_str, int)
        num_crys = len(pos)
        tuple = (pos, type, list(grid))
        batch_balance(num_crys, self.batch_size, tuple)
        return pos, type, grid
    
    def dataloader(self, atom_pos, atom_type, grid_name):
        """
        transfer data to the input of graph network and
        put them into DataLoader
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atom
        atom_type [int, 2d]: type of atom
        grid_name [int, 1d, np]: name of grid
        
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
        targets = np.zeros(self.num_crys)
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
        num_min = ratio_min_energy*self.num_crys
        num_max = ratio_max_std*self.num_crys
        _, mean_idx = torch.sort(mean_pred)
        _, std_idx = torch.sort(std_pred)
        min_idx = mean_idx[:int(num_min)].cpu().numpy()
        max_idx = std_idx[:int(num_max)].cpu().numpy()
        idx = np.union1d(min_idx, max_idx)
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
    
    def min_in_cluster(self, idx, mean_pred, clusters):
        """
        select lowest energy sample in each cluster

        Parameters
        ----------
        idx [int, 1d, np]: index of samples in input
        mean_pred [float, 1d, np]: average prediction
        clusters [int, 1d, np]: cluster labels of samples
        
        Returns
        ----------
        idx_slt [int, 1d, np]: index of select samples
        """
        order = np.argsort(clusters)
        sort_idx = idx[order]
        sort_mean_pred = mean_pred[order]
        sort_clusters = clusters[order]
        min_idx, min_pred = 0, 1e10
        last_cluster = sort_clusters[0]
        idx_slt = []
        for i, cluster in enumerate(sort_clusters):
            if cluster == last_cluster:
                pred = sort_mean_pred[i]
                if min_pred > pred:
                    min_idx = sort_idx[i]
                    min_pred = pred                     
            else:
                idx_slt.append(min_idx)
                last_cluster = cluster
                min_idx = sort_idx[i]
                min_pred = sort_mean_pred[i]
        idx_slt.append(min_idx)
        return np.array(idx_slt)
    
    def write_POSCARs(self, idx, pos, type, grid_name):
        """
        write POSCAR files and corresponding pos, type file
        
        Parameters
        ----------
        idx [int, 1d, np]: index of select samples
        pos [int, 2d]: input position
        type [int, 2d]: input type
        grid_name [int, 1d]: input grid name
        """
        num_jobs = len(idx)
        node_assign = self.assign_node(num_jobs)
        grid_slt = np.array(grid_name)[idx]
        order = np.argsort(grid_slt)
        grid_order = grid_slt[order]
        idx_order = idx[order]
        grid_last = grid_slt[0]
        transfer = Transfer(grid_last)
        pos_order, type_order = [], []
        elements = self.import_list2d(elements_file,
                                      str, numpy=True).ravel()
        for i, grid in enumerate(grid_order):
            idx_slt = idx_order[i]
            pos_order.append(pos[idx_slt])
            type_order.append(type[idx_slt])
            if grid_last == grid:
                self.write_POSCAR(pos[idx_slt], type[idx_slt],
                                  i+1, node_assign[i], 
                                  transfer, elements)
            else:
                grid_last = grid
                transfer = Transfer(grid)
                self.write_POSCAR(pos[idx_slt], type[idx_slt],
                                  i+1, node_assign[i], 
                                  transfer, elements)
        self.write_list2d(f'{self.sh_save_path}/atom_pos_select.dat',
                          pos_order)
        self.write_list2d(f'{self.sh_save_path}/atom_type_select.dat', 
                          type_order)
        self.write_list2d(f'{self.sh_save_path}/grid_name_select.dat', 
                          np.transpose([grid_order]))
    
    def write_POSCAR(self, pos, type, num, node, transfer, elements):
        """
        write POSCAR file of one configuration
        
        Parameters
        ----------
        pos [int, 1d]: atom position
        type [int, 1d]: atom type
        round [int, 0d]: searching round
        num [int, 0d]: configuration number
        node [int, 0d]: calculate node
        """
        head_str = ['E = -1', '1']
        latt_str = self.list2d_to_str(transfer.latt_vec, '{0:4.6f}')
        compn = elements[type]
        compn_dict = dict(Counter(compn))
        compn_str = [' '.join(list(compn_dict.keys())),
                     ' '.join([str(i) for i in compn_dict.values()]),
                     'Direct']
        frac_coor = transfer.frac_coor[pos]
        frac_coor_str = self.list2d_to_str(frac_coor, '{0:4.6f}')
        file = f'{self.poscar_save_path}/POSCAR-{self.round}-{num:04.0f}-{node}'
        POSCAR = head_str + latt_str + compn_str + frac_coor_str
        with open(file, 'w') as f:
            f.write('\n'.join(POSCAR))

    def export(self, recycle, atom_pos, atom_type, grid_name):
        """
        export configurations after ccop
        
        Parameters
        ----------
        recycle [int, 0d]: recycle times
        atom_pos [int, 2d]: position of atom
        atom_type [int, 2d]: type of atom
        grid_name [int, 2d]: name of grid
        """
        #delete duplicates in searching samples
        self.num_crys = len(atom_pos)
        system_echo(f'Training set---sample number: {self.num_crys}')
        atom_pos, atom_type, grid_name = \
            self.delete_duplicates(atom_pos, atom_type, grid_name)
        self.num_crys = len(atom_pos)
        system_echo(f'Delete duplicates---sample number: {self.num_crys}')
        #predict energy and calculate crystal vector
        loader = self.dataloader(atom_pos, atom_type, grid_name)
        model_names = self.model_select()
        mean_pred, _, crys_mean = self.mean(model_names, loader)
        idx_all = np.arange(self.num_crys)
        mean_pred_all = mean_pred.cpu().numpy()
        crys_mean_all = crys_mean.cpu().numpy()
        #filter structure by energy
        energy_order = np.argsort(mean_pred_all)
        filter_num = int(len(energy_order)*ratio_round)
        filter = energy_order[:filter_num]
        idx_all = idx_all[filter]
        mean_pred_all = mean_pred_all[filter]
        crys_mean_all = crys_mean_all[filter]
        system_echo(f'Energy filter---sample number: {len(idx_all)}')
        #export poscars
        crys_embedded = self.reduce(crys_mean_all)
        clusters = self.cluster(crys_embedded, num_poscars)
        idx_slt = self.min_in_cluster(idx_all, mean_pred_all, clusters)
        self.round = f'CCOP-{recycle}'
        self.poscar_save_path = f'{poscar_path}/{self.round}'
        self.sh_save_path = self.poscar_save_path
        if not os.path.exists(self.poscar_save_path):
            os.mkdir(self.poscar_save_path)
        self.write_POSCARs(idx_slt, atom_pos, atom_type, grid_name)
        system_echo(f'CCOP optimize structures: {num_poscars}')


class OptimSelect(Select, InitSampling, Transfer, SSHTools):
    #Select structures from low level optimization
    def __init__(self, round):
        Select.__init__(self, round)
        Transfer.__init__(self, 0)
        self.elem_embed = self.import_list2d(
            atom_init_file, int, numpy=True)
        if not os.path.exists(ccop_out_path):
            os.mkdir(ccop_out_path)
    
    def optim_select(self):
        """
        select optimized structuers from initial_X
        choose different structures in low energy configuration
        """
        #import optimized energy from each round
        poscars, poscars_full, energys = [], [], []
        for i in range(num_recycle):
            round = f'initial_strs_{i+1}'
            stru_path = f'{poscar_path}/{round}'
            energy_file = f'{vasp_out_path}/{round}/Energy.dat'
            energy_dat = self.import_list2d(energy_file, str, numpy=True)
            poscar, energy = np.transpose(energy_dat)
            poscar = [i[4:] for i in poscar]
            poscars = np.concatenate((poscars, poscar))
            full = [f'{stru_path}/{j}' for j in poscar]
            poscars_full = np.concatenate((poscars_full, full))
            energys = np.concatenate((energys, energy))
        energys = np.array(energys, dtype='float32')
        #filter structure by energy
        energy_order = np.argsort(energys)
        filter_num = int(len(energy_order)*ratio_round)
        filter = energy_order[:filter_num]
        poscars = poscars[filter]
        poscars_full = poscars_full[filter]
        energys = energys[filter]
        #copy low energy structures from each round
        for poscar in poscars_full:
            shutil.copy(poscar, ccop_out_path)
        num_crys = len(poscars)
        #transfer poscar into input of PPM
        atom_feas, nbr_feas, nbr_fea_idxs = [], [], []
        for poscar in poscars:
            stru = Structure.from_file(f'{ccop_out_path}/{poscar}', sort=True)
            atom_type = self.get_atom_number(stru)
            atom_fea = self.atom_initializer(atom_type)
            nbr_fea_idx, nbr_dis = self.near_property(stru, cutoff, near=self.nbr)
            nbr_fea = self.expand(nbr_dis)
            atom_feas.append(atom_fea)
            nbr_feas.append(nbr_fea)
            nbr_fea_idxs.append(nbr_fea_idx)
        #load data and get crystal vectors
        data = PPMData(atom_feas, nbr_feas, nbr_fea_idxs, energys)
        loader = get_loader(data, 256, 0)
        model_names = self.model_select()
        _, _, crys_mean = self.mean(model_names, loader)
        #select structures by crystal vectors
        idx_all = np.arange(num_crys)
        crys_mean_all = crys_mean.cpu().numpy()
        crys_embedded = self.reduce(crys_mean_all)
        clusters = self.cluster(crys_embedded, num_optims)
        idx_slt = self.min_in_cluster(idx_all, energys, clusters)
        idx_drop = np.setdiff1d(idx_all, idx_slt)
        for i in idx_drop:
            os.remove(f'{ccop_out_path}/{poscars[i]}')
        #write selected poscars
        poscars = sorted(os.listdir(ccop_out_path))
        node_assign = self.assign_node(num_optims)
        for i, poscar in enumerate(poscars):
            os.rename(f'{ccop_out_path}/{poscar}', 
                      f'{ccop_out_path}/POSCAR-{i+1:02.0f}-{node_assign[i]}')
        system_echo(f'Optimize configurations: {num_optims}')
        
    
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
    '''
    #Data import
    matools = ListRWTools()
    atom_pos = matools.import_list2d(f'{search_path}/001/atom_pos.dat', int)
    atom_type = matools.import_list2d(f'{search_path}/001/atom_type.dat', int)
    grid_name = matools.import_list2d(f'{search_path}/001/grid_name.dat', int)
    
    #Select samples
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    select = Select(1)
    start = time.time()
    select.samples(atom_pos, atom_type, grid_name)
    end = time.time()
    print(end - start)
    '''
    opt_slt = OptimSelect(1)
    opt_slt.optim_select()