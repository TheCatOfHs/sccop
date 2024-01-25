import os, time
import random
import shutil
import functools
import numpy as np
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel as DataParallel_raw

from pymatgen.core.structure import Structure


def system_echo(ct):
    """
    write system log
    
    Parameters
    ----------
    ct [str, 0d]: content
    """
    echo_ct = time.strftime("%Y-%m-%d %H:%M:%S",
                            time.localtime()) + ' -- ' + ct
    print(echo_ct)
    with open('log', 'a') as obj:
        obj.write(echo_ct + '\n')


class DataParallel(DataParallel_raw):
    #Scatter outside of the DataPrallel
    def __init__(self, module):
        super(DataParallel, self).__init__(module)
    
    def forward(self, **kwargs):
        new_inputs = [{} for _ in self.device_ids]
        for key in kwargs:
            if key == 'crys_fea':
                for i, device in enumerate(self.device_ids):
                    new_inputs[i][key] = kwargs[key][i].to(device, non_blocking=True)
                break
            if key == 'crystal_atom_idx':
                for i, device in enumerate(self.device_ids):
                    new_inputs[i][key] = [cry_idx.to(device, non_blocking=True) 
                                            for cry_idx in kwargs[key][i]]
            else:
                for i, device in enumerate(self.device_ids):
                    new_inputs[i][key] = kwargs[key][i].to(device, non_blocking=True)
        nones = [[] for _ in self.device_ids]
        replicas = self.replicate(self.module, self.device_ids)
        outputs = self.parallel_apply(replicas, nones, new_inputs)
        return self.gather(outputs, self.output_device)
        

class PPMData(Dataset):
    #Self-define training set of PPM
    def __init__(self, atom_feas, nbr_feas, nbr_fea_idxes, targets):
        self.atom_feas = atom_feas
        self.nbr_feas = nbr_feas
        self.nbr_fea_idxes = nbr_fea_idxes
        self.targets = targets

    def __len__(self):
        """
        length of dataset
        """
        return len(self.targets)
    
    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        """
        get each item in dataset by idx
        
        Returns
        ----------
        atom_fea [float, 2d, tensor]: feature of atoms
        nbr_fea [float, 3d, tensor]: bond feature
        nbr_fea_idx [int, 2d, tensor]: index of neighbors
        target [int, 1d, tensor]: target value
        """
        atom_fea = self.atom_feas[idx]
        nbr_fea = self.nbr_feas[idx]
        nbr_fea_idx = self.nbr_fea_idxes[idx]
        target = self.targets[idx]
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([target])
        return (atom_fea, nbr_fea, nbr_fea_idx), target


def collate_pool(dataset_list):
    """
    collate data of each batch
    
    Parameters
    ----------
    dataset_list [tuple]: output of PPMData
    
    Returns
    ----------
    store_1 [float, 3d]: atom features assigned to gpus
    store_2 [float, 4d]: bond features assigned to gpus
    store_3 [int, 3d]: index of neighbors assigned to gpus
    store_4 [int, 3d]: index of crystals assigned to gpus
    target [float, 2d, tensor]: target values
    """
    assign_plan = batch_divide(dataset_list)
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    num, base_idx, counter = 0, 0, 0
    store_1, store_2, store_3, store_4 = [], [], [], []
    for ((atom_fea, nbr_fea, nbr_fea_idx), target) in dataset_list:
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        base_idx += n_i
        counter += 1
        if counter == assign_plan[num]:
            store_1.append(torch.cat(batch_atom_fea, dim=0))
            store_2.append(torch.cat(batch_nbr_fea, dim=0))
            store_3.append(torch.cat(batch_nbr_fea_idx, dim=0))
            store_4.append(crystal_atom_idx)
            batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
            crystal_atom_idx = []
            base_idx = 0
            num += 1
    return (store_1, store_2, store_3, store_4), \
            torch.stack(batch_target, dim=0)

def batch_divide(dataset_list):
    """
    divide batch by number of gpus
    
    Parameters
    ----------
    dataset_list [tuple]: output of PPMData

    Returns
    ----------
    assign_plan [int, 1d]: gpu assign plan
    """
    num_gpus = 4
    sample_num = len(dataset_list)
    assign_num = sample_num//num_gpus
    assign_plan = []
    counter = 0
    for i in range(num_gpus):
        counter += assign_num
        if i == num_gpus-1:
            assign_plan.append(sample_num)
        else:
            assign_plan.append(counter)
    return assign_plan

def batch_balance(num, batch_size, tuple):
    """
    delete samples that make the inequal assignment
    
    Parameters
    ----------
    num_crys [int, 0d]: number of training samples
    batch_size [int, 0d]: batch size of training
    tuple [tuple]: tuple of pos, type, grid, energy
    """
    num_gpus = 4
    num_last_batch = np.mod(num, batch_size)
    if 0 < num_last_batch < num_gpus:
        for i in tuple:
            del i[-num_last_batch:]

def get_loader(dataset, batch_size, num_workers, shuffle=False):
    """
    returen data loader
        
    Parameters
    ----------
    dataset [obj]: object generated by PPMData
    
    Returns
    ----------
    loader [obj]: data loader of ppm
    """
    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=collate_pool,
                        shuffle=shuffle, num_workers=num_workers,
                        pin_memory=True)
    return loader


class ConvLayer(nn.Module):
    #Graph convolutional layer
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        embedding crystal into vector
        
        Parameters
        ----------
        atom_in_fea [float, 2d]: atom feature vector
        nbr_fea [float, 3d]: bond feature vector
        nbr_fea_idx [int, 2d]: index of neighbors
        
        Returns
        ----------
        out [float, 2d]: crystal feature vector
        """
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out
        

class CrystalGraphConvNet(nn.Module):
    #CGCNN applied in asymmetric unit
    def __init__(self, n_conv=3, orig_atom_fea_len=92, 
                 atom_fea_len=64, nbr_fea_len=41, h_fea_len=128):
        super(CrystalGraphConvNet, self).__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                              nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        self.fc_out = nn.Linear(h_fea_len, 1)
    
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        predict energy
        
        Parameters
        ----------
        atom_fea [float, 2d, tensor]: atom feature 
        nbr_fea [float, 3d, tensor]: bond feature
        nbr_fea_idx [int, 2d, tensor]: neighbor index
        crystal_atom_idx [int, 2d, tensor]: atom index in batch

        Returns
        ----------
        out [float, 2d, tensor]: predict energy
        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        out = self.fc_out(crys_fea)
        return out
    
    def pooling(self, atom_fea, crystal_atom_idx):
        """
        symmetry weighted average
        mix atom vector into crystal vector
        
        Parameters
        ----------
        atom_fea [float, 2d, tensor]: atom vector
        crystal_atom_idx [int, 2d, tensor]: atom index in batch

        Returns
        ----------
        crys_fea [float, 2d, tensor]: crystal vector
        """
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)

    

class PPModel:
    #Train property predict model
    def __init__(self, train_batchsize, train_data, valid_data, test_data, 
                 train_epochs=120, lr=1e-2, num_workers=0, num_gpus=4, print_feq=10):
        self.device = torch.device('cuda')
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.lr = lr
        self.batch_size = train_batchsize
        self.epochs = train_epochs
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.print_feq = print_feq
        self.model_save_path = f'models'
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
    
    def train_epochs(self):
        """
        train model by epochs
        """
        #load data
        train_loader = get_loader(self.train_data, 
                                  self.batch_size, self.num_workers, shuffle=True)
        valid_loader = get_loader(self.valid_data, 
                                  self.batch_size, self.num_workers)
        test_loader = get_loader(self.test_data, 
                                 self.batch_size, self.num_workers)
        sample_target = self.sample_data_list(self.train_data)
        normalizer = Normalizer(sample_target)
        #build prediction model
        model = CrystalGraphConvNet()
        model = DataParallel(model)
        model.to(self.device)
        #set learning rate
        params = model.parameters()
        #training model
        criterion = nn.MSELoss()
        optimizer = optim.Adam(params, lr=self.lr, weight_decay=0)
        scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.1)
        train_mae_buffer, valid_mae_buffer, best_mae_error = [], [], 1e10
        system_echo('-----------Begin Training Property Predict Model------------')
        for epoch in range(0, self.epochs):
            train_mae = self.train_batch(train_loader, model, criterion, optimizer, epoch, normalizer)
            valid_mae = self.validate(valid_loader, model, criterion, epoch, normalizer)
            scheduler.step()
            is_best = valid_mae < best_mae_error
            best_mae_error = min(valid_mae, best_mae_error)
            self.save_checkpoint(epoch,
                {'state_dict': model.module.state_dict(),
                'normalizer': normalizer.state_dict()}, is_best)
            train_mae_buffer.append([train_mae])
            valid_mae_buffer.append([valid_mae])
        system_echo('-----------------Evaluate Model on Test Set-----------------')
        best_checkpoint = torch.load(f'{self.model_save_path}/model_best.pth.tar')
        model = self.model_initial(best_checkpoint)
        model = DataParallel(model)
        model.to(self.device)
        self.validate(test_loader, model, criterion, epoch, normalizer, best_model_test=True)
        write_list2d(f'{self.model_save_path}/train.dat', train_mae_buffer, style='{0:6.4f}')
        write_list2d(f'{self.model_save_path}/valid.dat', valid_mae_buffer, style='{0:6.4f}')
        
    def train_batch(self, loader, model, criterion, optimizer, epoch, normalizer):
        """
        train model one batch
        
        Parameters
        ----------
        loader [obj]: data loader of training set
        model [obj]: property predict model
        criterion [obj]: loss function
        optimizer [obj]: training optimizer
        epoch [int, 0d]: training epoch
        normalizer [obj]: normalize targets
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = AverageMeter()
        model.train()
        start = time.time()
        for input, target in loader:
            data_time.update(time.time() - start)
            target_normed = normalizer.norm(target)
            target_var = target_normed.to(self.device, non_blocking=True)
            pred = model(atom_fea=input[0], nbr_fea=input[1],
                         nbr_fea_idx=input[2], crystal_atom_idx=input[3])
            loss = criterion(pred, target_var)
            mae_error = self.mae(normalizer.denorm(pred.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time.update(time.time() - start)
            start = time.time()
        if epoch % self.print_feq == 0:
            system_echo(f'Epoch: [{epoch:03.0f}][{epoch}/{self.epochs}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})')
        return mae_errors.avg
        
    def validate(self, loader, model, criterion, epoch, normalizer, best_model_test=False):
        """
        test model on validation set
        
        Parameters
        ----------
        loader [obj]: data loader of validation set
        best_model_test [bool]: best validation performance model
        
        Returns
        ----------
        mae_errors.avg [float, 0d]: average mae
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = AverageMeter()
        model.eval()
        pred_all, vasp_all = torch.tensor([]), torch.tensor([])
        start = time.time()
        for input, target in loader:
            #tensor with no grad
            with torch.no_grad():
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
                target_normed = normalizer.norm(target)
                target_var = target_normed.to(self.device, non_blocking=True)
            #calculate loss
            pred = model(atom_fea=atom_fea, nbr_fea=nbr_fea,
                         nbr_fea_idx=nbr_fea_idx, crystal_atom_idx=crystal_atom_idx)
            loss = criterion(pred, target_var)
            pred = normalizer.denorm(pred.data.cpu())
            mae_error = self.mae(pred, target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            batch_time.update(time.time() - start)
            start = time.time()
            if best_model_test:
                pred_all = torch.cat((pred_all, pred))
                vasp_all = torch.cat((vasp_all, target))
        if epoch % self.print_feq == 0:
            system_echo(f'Test: [{epoch}/{self.epochs}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'
                        )
        if best_model_test:
            system_echo(f'MAE of Best model on Testset: {mae_errors.avg:.3f}')
            write_list2d(f'{self.model_save_path}/pred.dat', pred_all, style='{0:8.4f}')
            write_list2d(f'{self.model_save_path}/vasp.dat', vasp_all, style='{0:8.4f}')
        return mae_errors.avg
    
    def model_initial(self, checkpoint):
        """
        initialize model by input len_atom_fea and nbr_fea_len
        
        Parameters
        ----------
        checkpoint [dict]: save parameters of model
        
        Returns
        ----------
        model [obj]: initialized model
        """
        model = CrystalGraphConvNet()
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    def sample_data_list(self, dataset):
        """
        sample data from training set to calculate mean and std
        to normalize targets
        
        Returns
        ----------
        sample_target [float, 2d, tensor]: sampled target 
        """
        if len(dataset) < 500:
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                random.sample(range(len(dataset)), 500)]
        _, sample_target = collate_pool(sample_data_list)
        return sample_target
    
    def mae(self, prediction, target):
        """
        computes the mean absolute error between prediction and target

        Parameters
        ----------
        prediction [float, 2d, tensor]: prediction vector
        target [float, 2d, tensor]: target vector
        
        Returns
        ----------
        mae [float, 0d]
        """
        return torch.mean(torch.abs(target - prediction))
    
    def save_checkpoint(self, epoch, state, is_best):
        """
        save model
        
        Parameters
        ----------
        state [dict]: save data in the form of dictionary
        is_best [bool]: whether model perform best in validation set
        """
        filename = f'{self.model_save_path}/checkpoint-{epoch:03.0f}.pth.tar'
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, f'{self.model_save_path}/model_best.pth.tar')
    

class Normalizer():
    #Normalize a Tensor and restore it later
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)
    
    def norm(self, tensor):
        """
        normalize target tensor
        
        Parameters
        ----------
        tensor [tensor, 2d]: tensor of targets
        
        Returns
        ----------
        normalized tensor [tensor, 2d] 
        """
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        """
        denormalize target tensor
        
        Parameters
        ----------
        normed_tensor [tensor, 2d]: normalized target tensor
        
        Returns
        ----------
        denormed tensor [tensor, 2d]
        """
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        """
        return dictionary of mean and std of sampled targets
        
        Returns
        ----------
        mean and std [dict]
        """
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        """
        load mean and std to denormalize target tensor
        
        Parameters
        ----------
        state_dict [dict]: mean and std in dictionary
        """
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class AverageMeter():
    #Computes and stores the average and current value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        update average value
        
        Parameters
        ----------
        val [float, 0d]: record value
        """
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count
        
def str_to_list1d(string, dtype):
    """
    convert string list to 1-dimensional list
    """
    list = [dtype(i) for i in string]
    return list

def str_to_list2d(string, dtype):
    """
    convert string list to 2-dimensional list
    """
    list = [str_to_list1d(item.split(), dtype)
            for item in string]
    return list

def import_list2d(file, dtype):
    """
    import 2-dimensional list
    """
    with open(file, 'r') as f:
        ct = f.readlines()
    list = str_to_list2d(ct, dtype)
    return np.array(list, dtype=dtype)

def list2d_to_str(list, style):
    """
    convert 2-dimensional list to string list
    """
    list_str = [' '.join(list1d_to_str(line, style)) 
                for line in list]
    return list_str
    
def list1d_to_str(list, style):
    """
    convert 1-dimensional list to string list
    """
    list = [style.format(i) for i in list]
    return list

def write_list2d(file, list, style='{0}'):
    """
    write 2-dimensional list
    """
    list_str = list2d_to_str(list, style)
    list2d_str = '\n'.join(list_str)
    with open(file, 'w') as f:
        f.write(list2d_str)

def get_atom_fea(atom_type, elem_embed):
    """
    initialize atom feature vectors

    Parameters
    ----------
    atom_type [int, 1d]: type of atoms
    elem_embed [int, 2d, np]: embedding of elements
        
    Returns
    ----------
    atom_fea [int, 2d, np]: atom feature vectors
    """
    atom_fea = elem_embed[atom_type]
    return atom_fea
    
def expand(distances):
    dmin, dmax=0, 8 
    step, var=0.2, 0.2
    filter = np.arange(dmin, dmax+step, step)
    return np.exp(-(distances[:, :, np.newaxis] - filter)**2 /
                    var**2)

def single(poscar):
    """
    index and distance of near grid points
        
    Parameters
    ----------
    poscar [str, 0d]: path of poscar
    
    Returns
    ----------
    atom_fea [int, 2d, np]: feature of atoms
    nbr_fea [float, 2d, np]: distance of near neighbor 
    nbr_idx [int, 2d, np]: index of near neighbor 
    """
    cutoff = 15
    crystal = Structure.from_file(poscar)
    atom_type = np.array(crystal.atomic_numbers)
    all_nbrs = crystal.get_all_neighbors(cutoff)
    all_nbrs = [sorted(nbrs, key = lambda x: x[1]) for nbrs in all_nbrs]
    num_near = min(map(lambda x: len(x), all_nbrs))
    nbr_idx, nbr_dis = [], []
    for nbr in all_nbrs:
        idx = list(map(lambda x: x[2], nbr[:num_near]))[:12]
        dis = list(map(lambda x: x[1], nbr[:num_near]))[:12]
        if len(idx) < 12:
            lack = 12 - len(idx)
            idx += lack*[0]
            dis += lack*[9]
        nbr_idx.append(idx)
        nbr_dis.append(dis)
    nbr_idx, nbr_dis = np.array(nbr_idx), np.array(nbr_dis)
    elem_embed = import_list2d('atom_init.dat', int)
    atom_fea = get_atom_fea(atom_type, elem_embed)
    nbr_fea = expand(nbr_dis)
    return atom_fea, nbr_fea, nbr_idx

def batch(poscars):
    """
    transfer poscars to input of predict model
    """
    batch_atom_fea, batch_nbr_fea, \
        batch_nbr_fea_idx = [], [], []
    pool = mp.Pool(processes=4*96)
    counter, buffer = 0, []
    for idx, poscar in enumerate(poscars):
        buffer.append(poscar)
        counter += 1
        if counter == 4*96:
            fea_job = [pool.apply_async(single, (i, )) for i in buffer]
            fea_pool = [p.get() for p in fea_job]
            for atom_fea, nbr_fea, nbr_fea_idx in fea_pool:
                batch_atom_fea.append(atom_fea)
                batch_nbr_fea.append(nbr_fea)
                batch_nbr_fea_idx.append(nbr_fea_idx)
            counter, buffer = 0, []
            system_echo(f'{idx}')
    if 0 < len(buffer) < 4*96:
        fea_job = [pool.apply_async(single, (i, )) for i in buffer]
        fea_pool = [p.get() for p in fea_job]
        for atom_fea, nbr_fea, nbr_fea_idx in fea_pool:
            batch_atom_fea.append(atom_fea)
            batch_nbr_fea.append(nbr_fea)
            batch_nbr_fea_idx.append(nbr_fea_idx)
    pool.close()
    return batch_atom_fea, batch_nbr_fea, \
            batch_nbr_fea_idx


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    batchsize = 512
    data = np.array(import_list2d('Energy.dat', str))
    #divide into train, valid, test
    data_num = len(data)
    index = [i for i in range(data_num)]
    random.shuffle(index)
    train_index = index[:int(data_num*0.6)]
    valid_index = index[int(data_num*0.6):int(data_num*0.8)]
    test_index = index[int(data_num*0.8):]
    #balance gpus
    batch_balance(len(train_index), batchsize, train_index)
    batch_balance(len(valid_index), batchsize, valid_index)
    batch_balance(len(test_index), batchsize, test_index)
    #get name of poscars
    train_poscars, train_energys = np.transpose(data[train_index])
    valid_poscars, valid_energys = np.transpose(data[valid_index])
    test_poscars, test_energys = np.transpose(data[test_index])
    train_poscars = [f'poscars/{i}' for i in train_poscars]
    valid_poscars = [f'poscars/{i}' for i in valid_poscars]
    test_poscars = [f'poscars/{i}' for i in test_poscars]
    train_energys = np.array(train_energys, dtype=float)
    valid_energys = np.array(valid_energys, dtype=float)
    test_energys = np.array(test_energys, dtype=float)
    #transfer data to input of model
    train_atom_fea, train_nbr_fea, train_nbr_fea_idx = batch(train_poscars)
    valid_atom_fea, valid_nbr_fea, valid_nbr_fea_idx = batch(valid_poscars)
    test_atom_fea, test_nbr_fea, test_nbr_fea_idx = batch(test_poscars)
    #training prediction model
    train_data = PPMData(train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energys)
    valid_data = PPMData(valid_atom_fea, valid_nbr_fea, valid_nbr_fea_idx, valid_energys)
    test_data = PPMData(test_atom_fea, test_nbr_fea, test_nbr_fea_idx, test_energys)
    ppm = PPModel(batchsize, train_data, valid_data, test_data)
    ppm.train_epochs()