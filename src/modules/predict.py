import functools
import numpy as np
import os, sys, time, shutil
from random import sample

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel as DataParallel_raw

sys.path.append(f'{os.getcwd()}/src')
from modules.global_var import *
from modules.sub_vasp import system_echo
from modules.utils import ListRWTools


class DataParallel(DataParallel_raw):
    """
    scatter outside of the DataPrallel
    input: Scattered Inputs without kwargs
    """
    def __init__(self, module):
        # Disable all the other parameters
        super(DataParallel, self).__init__(module)

    def forward(self, *inputs, **kwargs):
        assert len(inputs) == 0, "Only support arguments like [variable_name = xxx]"
        new_inputs = [{} for _ in self.device_ids]
        for key in kwargs:
            if key == 'crys_fea':
                for i, device in enumerate(self.device_ids):
                    if i ==0:
                        new_inputs[i][key] = kwargs[key][i]
                    else:
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
        Length of dataset
        """
        return len(self.targets)
    
    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        """
        Get each item in dataset by idx
        
        Returns
        ----------
        atom_fea [float, 2d, tensor]: 
        nbr_fea [float, 3d, tensor]: 
        nbr_fea_idx [int, 2d, tensor]: 
        target [int, 2d, tensor]: 
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
    Collate data of each batch
    
    Parameters
    ----------
    dataset_list [turple]: training set, input of PPM
    
    Returns
    ----------
    store_1 [float, 3d]:
    store_2 [float, 4d]:
    store_3 [int, 3d]:
    store_4 [int, 3d]:
    target [float, 2d, tensor]:
    """
    #should be split into num_gpus 
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

    Parameters
    ----------
    dataset_list []: 

    Returns
    ----------
    []: 
    """
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

def get_loader(dataset, batch_size, num_workers, shuffle=False):
    """
    Returen data loader
        
    Parameters
    ----------
    dataset [obj]: object generated by PPMData
    
    Returns
    ----------
    loader [obj]: 
    """
    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=collate_pool,
                        shuffle=shuffle, num_workers=num_workers,
                        pin_memory=True)
    return loader


class ConvLayer(nn.Module):
    #
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

        Parameters
        ----------
        atom_in_fea [float, 2d]: 
        nbr_fea [float, 3d]: 
        nbr_fea_idx [int, 2d]: 

        Returns
        ----------
        out [float, 2d]: 
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
    #
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1):
        super(CrystalGraphConvNet, self).__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """

        Parameters
        ----------
        atom_fea [float, , tensor]: 
        nbr_fea [float, , tensor]: 
        nbr_fea_idx [int, , tensor]: 
        crystal_atom_idx [int, , tensor]: 

        Returns
        ----------
        out [float, , tensor]: 
        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """

        Parameters
        ----------
        atom_fea [float, , tensor]: 
        crystal_atom_idx [int, , tensor]: 

        Returns
        ----------
        crys_fea [float, , tensor]: 
        """
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)


class PPModel(ListRWTools):
    #Train property predict model
    def __init__(self, round, train_data, valid_data, test_data, 
                 lr=1e-2, batch_size=128, epochs=120, num_workers=0, 
                 num_gpus=2, print_feq=10):
        self.device = torch.device('cuda')
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.print_feq = print_feq
        self.model_save_dir = f'{model_dir}/{round:03.0f}'
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        
    def train_epochs(self):
        """
        Train model by epochs
        """
        train_loader = get_loader(self.train_data, 
                                  self.batch_size, self.num_workers, shuffle=True)
        valid_loader = get_loader(self.valid_data, 
                                  self.batch_size, self.num_workers)
        test_loader = get_loader(self.test_data, 
                                  self.batch_size, self.num_workers)
        sample_target = self.sample_data_list(self.train_data)
        normalizer = Normalizer(sample_target)
        model = self.model_initial(self.train_data)
        model = DataParallel(model)
        model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=0)
        scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.1)
        mae_buffer, best_mae_error = [], 1e10
        system_echo('-----------Begin Training Property Predict Model------------')
        for epoch in range(0, self.epochs):
            self.train_batch(train_loader, model, criterion, optimizer, epoch, normalizer)
            mae_error = self.validate(valid_loader, model, criterion, epoch, normalizer)
            scheduler.step()
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
            self.save_checkpoint(epoch,
                {'state_dict': model.module.state_dict(),
                'normalizer': normalizer.state_dict()
                }, is_best)
            mae_buffer.append([mae_error])
        system_echo('-----------------Evaluate Model on Test Set-----------------')
        best_checkpoint = torch.load(f'{self.model_save_dir}/model_best.pth.tar')
        model = self.model_initial(self.train_data)
        model.load_state_dict(best_checkpoint['state_dict'])
        model = DataParallel(model)
        model.to(self.device)
        self.validate(test_loader, model, criterion, epoch, normalizer, best_model_test=True)
        self.write_list2d(f'{self.model_save_dir}/validation.dat', mae_buffer, '{0:3.4f}')
    
    def train_batch(self, loader, model, criterion, optimizer, epoch, normalizer):
        """
        Train model one batch
        
        Parameters
        ----------
        loader [obj]: data loader of training set
        model [obj]: property predict model
        criterion [obj]: loss function
        optimizer [obj]: training optimizer
        epoch [int]: training epoch
        normalizer [obj]: normalize targets
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = AverageMeter()
        model.train()
        start = time.time()
        for input, target in loader:
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
    
    def validate(self, loader, model, criterion, epoch, normalizer, best_model_test=False):
        """
        Test model on validation set
        
        Parameters
        ----------
        loader [obj]: data loader of validation set
        best_model_test [bool]: best validation performance model
        
        Returns
        ----------
        mae_errors.avg [float, 0d]: 
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = AverageMeter()
        model.eval()
        start = time.time()
        for _, (input, target) in enumerate(loader):
            with torch.no_grad():
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
            target_normed = normalizer.norm(target)
            with torch.no_grad():
                target_var = target_normed.to(self.device, non_blocking=True)
            pred = model(atom_fea=atom_fea, nbr_fea=nbr_fea,
                         nbr_fea_idx=nbr_fea_idx, crystal_atom_idx=crystal_atom_idx)
            loss = criterion(pred, target_var)
            mae_error = self.mae(normalizer.denorm(pred.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            batch_time.update(time.time() - start)
            start = time.time()
        if epoch % self.print_feq == 0:
            system_echo(f'Test: [{epoch}/{self.epochs}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'
                        )
        if best_model_test:
            system_echo(f'Best model MAE {mae_errors.avg:.3f}')
            self.write_list2d(f'{self.model_save_dir}/pred.dat', 
                              normalizer.denorm(pred.data.cpu()), '{0:4.4f}')
            self.write_list2d(f'{self.model_save_dir}/vasp.dat', 
                              target, '{0:4.4f}')
        return mae_errors.avg
    
    def model_initial(self, dataset):
        """
        Initialize model by input len_atom_fea and nbr_fea_len
        
        Returns
        ----------
        model [obj]: 
        """
        structures, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len)
        return model
    
    def sample_data_list(self, dataset):
        """
        Sample data from training set to calculate mean and std
        to normalize targets
        
        Returns
        ----------
        sample_target []: 
        """
        if len(dataset) < 500:
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
        _, sample_target = collate_pool(sample_data_list)
        return sample_target

    def mae(self, prediction, target):
        """
        Computes the mean absolute error between prediction and target

        Parameters
        ----------
        prediction: torch.Tensor (N, 1)
        target: torch.Tensor (N, 1)
        
        Returns
        ----------
        mae []: 
        """
        return torch.mean(torch.abs(target - prediction))
    
    def save_checkpoint(self, epoch, state, is_best):
        """
        Save model
        
        Parameters
        ----------
        state [dict]: save data in the form of dictionary
        is_best [bool]: whether model perform best in validation set
        """
        filename = f'{self.model_save_dir}/checkpoint-{epoch:03.0f}.pth.tar'
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, f'{self.model_save_dir}/model_best.pth.tar')


class Normalizer():
    #Normalize a Tensor and restore it later
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        """
        Normalize target tensor
        
        Parameters
        ----------
        tensor [tensor, 2d]: tensor of targets
        
        Returns
        ----------
        []: 
        """
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        """
        Denormalize target tensor
        
        Parameters
        ----------
        normed_tensor [tensor]: normalized target tensor
        
        Returns
        ----------
        []: 
        """
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        """
        Return dictionary of mean and std of sampled targets
        
        Returns
        ----------
        []: 
        """
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        """
        Load mean and std to denormalize target tensor
        
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
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count

    
if __name__ == '__main__':
    from modules.data_transfer import Transfer
    round = 0
    rwtools = ListRWTools()
    pos_buffer, type_buffer, energy_buffer = [], [], []
    train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energys = [], [], [], []
    valid_atom_fea, valid_nbr_fea, valid_nbr_fea_idx, valid_energys = [], [], [], []
    test_atom_fea, test_nbr_fea, test_nbr_fea_idx, test_energys = [], [], [], []
    
    #Data import
    energy_file = rwtools.import_list2d(f'{vasp_out_dir}/Energy-{round:03.0f}.dat', str, numpy=True)
    true_E = np.array(energy_file)[:,1]
    true_E = [bool(i) for i in true_E]
    energys = energy_file[:,2][true_E]
    energys = [float(i) for i in energys]

    atom_pos = rwtools.import_list2d(f'{search_dir}/{round:03.0f}/atom_pos_select.dat', int)
    atom_type = rwtools.import_list2d(f'{search_dir}/{round:03.0f}/atom_type_select.dat', int)
    atom_pos_right, atom_type_right = [], []
    for i, correct in enumerate(true_E):
        if correct:
            atom_pos_right.append(atom_pos[i])
            atom_type_right.append(atom_type[i])
    grid_name = 1
    transfer = Transfer(grid_name)
    atom_fea, nbr_fea, nbr_fea_idx = transfer.batch(atom_pos_right, atom_type_right)

    num_poscars = len(energys)
    a = int(num_poscars*0.6)
    b = int(num_poscars*0.8)

    #Train data
    train_atom_fea += atom_fea[0:a]
    train_nbr_fea += nbr_fea[0:a]
    train_nbr_fea_idx += nbr_fea_idx[0:a]
    train_energys += energys[0:a]
    energy_buffer += energys[0:a]
    pos_buffer += atom_pos_right[0:a]
    type_buffer += atom_type_right[0:a]
        
    #Validation data
    valid_atom_fea += atom_fea[a:b]
    valid_nbr_fea += nbr_fea[a:b]
    valid_nbr_fea_idx += nbr_fea_idx[a:b]
    valid_energys += energys[a:b]
        
    #Test data
    test_atom_fea += atom_fea[b:]
    test_nbr_fea += nbr_fea[b:]
    test_nbr_fea_idx += nbr_fea_idx[b:]
    test_energys += energys[b:]
    
    #Training
    train = True
    if train:
        train_data = PPMData(train_atom_fea, train_nbr_fea, train_nbr_fea_idx, train_energys)
        valid_data = PPMData(valid_atom_fea, valid_nbr_fea, valid_nbr_fea_idx, valid_energys)
        test_data = PPMData(valid_atom_fea, valid_nbr_fea, valid_nbr_fea_idx, valid_energys)
        ppm = PPModel(round+1, train_data, valid_data, test_data)
        ppm.train_epochs()