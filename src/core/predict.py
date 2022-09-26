import random
import functools
import os, sys, time, shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel as DataParallel_raw

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.path import *
from core.sub_vasp import system_echo
from core.utils import ListRWTools


class DataParallel(DataParallel_raw):
    #Scatter outside of the DataPrallel
    def __init__(self, module):
        super(DataParallel, self).__init__(module)
    
    def forward(self, **kwargs):
        new_inputs = [{} for _ in self.device_ids]
        for key in kwargs:
            #
            if key == 'crys_fea':
                for i, device in enumerate(self.device_ids):
                    new_inputs[i][key] = kwargs[key][i].to(device, non_blocking=True)
                break
            #
            if key == 'crystal_atom_idx':
                for i, device in enumerate(self.device_ids):
                    new_inputs[i][key] = [cry_idx.to(device, non_blocking=True) 
                                            for cry_idx in kwargs[key][i]]
            #
            else:
                for i, device in enumerate(self.device_ids):
                    new_inputs[i][key] = kwargs[key][i].to(device, non_blocking=True)
        nones = [[] for _ in self.device_ids]
        replicas = self.replicate(self.module, self.device_ids)
        outputs = self.parallel_apply(replicas, nones, new_inputs)
        return self.gather(outputs, self.output_device)
        

class PPMData(Dataset):
    #Self-define training set of PPM
    def __init__(self, atom_feas, atom_symm, nbr_feas, nbr_fea_idxes, targets):
        self.atom_feas = atom_feas
        self.atom_symm = atom_symm
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
        atom_symm [int, 2d]: symmetry of atoms
        nbr_fea [float, 3d, tensor]: bond feature
        nbr_fea_idx [int, 2d, tensor]: index of neighbors
        target [int, 1d, tensor]: target value
        """
        atom_fea = self.atom_feas[idx]
        atom_symm = self.atom_symm[idx]
        nbr_fea = self.nbr_feas[idx]
        nbr_fea_idx = self.nbr_fea_idxes[idx]
        target = self.targets[idx]
        #convert to float tensor
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([target])
        return atom_fea, atom_symm, nbr_fea, nbr_fea_idx, target


def collate_pool(dataset_list):
    """
    collate data of each batch
    
    Parameters
    ----------
    dataset_list [tuple]: PPMData
    
    Returns
    ----------
    store_1 [float, 3d]: atom features assigned to gpus
    store_2 [float, 2d]: symmetry weight assigned to gpus
    store_3 [float, 4d]: bond features assigned to gpus
    store_4 [int, 3d]: index of neighbors assigned to gpus
    store_5 [int, 3d]: index of crystals assigned to gpus
    target [float, 2d, tensor]: target values
    """
    assign_plan = batch_divide(dataset_list)
    batch_atom_fea, batch_symm, batch_nbr_fea  = [], [], []
    batch_nbr_fea_idx, crystal_atom_idx, batch_target = [], [], []
    num, base_idx, counter = 0, 0, 0
    store_1, store_2, store_3, store_4, store_5 = [], [], [], [], []
    #divide data by number of gpus
    for atom_fea, atom_symm, nbr_fea, nbr_fea_idx, target in dataset_list:
        #collate batch data
        n_i = len(atom_fea)
        batch_atom_fea.append(atom_fea)
        batch_symm += atom_symm
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(base_idx+nbr_fea_idx)
        crystal_atom_idx.append(torch.LongTensor(base_idx+np.arange(n_i)))
        batch_target.append(target)
        base_idx += n_i
        counter += 1
        #save data
        if counter == assign_plan[num]:
            store_1.append(torch.cat(batch_atom_fea))
            store_2.append(torch.Tensor(batch_symm))
            store_3.append(torch.cat(batch_nbr_fea))
            store_4.append(torch.cat(batch_nbr_fea_idx))
            store_5.append(crystal_atom_idx)
            batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
            batch_symm, crystal_atom_idx = [], []
            base_idx = 0
            num += 1
    return (store_1, store_2, store_3, store_4, store_5), \
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
    
    def forward(self, atom_fea, atom_symm, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        predict energy
        
        Parameters
        ----------
        atom_fea [float, 2d, tensor]: atom feature 
        atom_symm [float, 1d, tensor]: symmetry of atoms
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
        crys_fea = self.pooling(atom_fea, atom_symm, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        out = self.fc_out(crys_fea)
        return out
    
    def pooling(self, atom_fea, atom_symm, crystal_atom_idx):
        """
        symmetry weighted average
        mix atom vector into crystal vector
        
        Parameters
        ----------
        atom_fea [float, 2d, tensor]: atom vector
        atom_symm [float, 1d, tensor]: symmetry of atoms
        crystal_atom_idx [int, 2d, tensor]: atom index in batch

        Returns
        ----------
        crys_fea [float, 2d, tensor]: crystal vector
        """
        summed_fea = [torch.mean(atom_symm[idx_map].view(-1,1)*atom_fea[idx_map],
                                 dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)


class PPModel(ListRWTools):
    #Train property predict model
    def __init__(self, iteration, train_data, valid_data, test_data, 
                 lr=1e-2, num_workers=0, num_gpus=num_gpus, print_feq=10):
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
        self.model_save_path = f'{model_path}/{iteration:03.0f}'
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
        if use_pretrain_model:
            checkpoint = torch.load(pretrain_model)
            model = self.model_initial(checkpoint)
            normalizer.load_state_dict(checkpoint['normalizer'])
        else:
            model = CrystalGraphConvNet()
        model = DataParallel(model)
        model.to(self.device)
        #set learning rate
        if use_pretrain_model:
            out_layer_id = list(map(id, model.module.fc_out.parameters()))
            crysfea_layer = filter(lambda x: id(x) not in out_layer_id, model.parameters())
            params = [{'params': crysfea_layer, 'lr': self.lr*0},
                      {'params': model.module.fc_out.parameters()}]
        else:
            params = model.parameters()
        #training model
        criterion = nn.MSELoss()
        optimizer = optim.Adam(params, lr=self.lr, weight_decay=0)
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
        best_checkpoint = torch.load(f'{self.model_save_path}/model_best.pth.tar')
        model = self.model_initial(best_checkpoint)
        model = DataParallel(model)
        model.to(self.device)
        self.validate(test_loader, model, criterion, epoch, normalizer, best_model_test=True)
        self.write_list2d(f'{self.model_save_path}/validation.dat', mae_buffer, style='{0:6.4f}')
        
    def train_batch(self, loader, model, criterion, optimizer, epoch, normalizer):
        """
        train model one batch
        
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
            data_time.update(time.time() - start)
            target_normed = normalizer.norm(target)
            target_var = target_normed.to(self.device, non_blocking=True)
            pred = model(atom_fea=input[0], atom_symm=input[1], nbr_fea=input[2],
                         nbr_fea_idx=input[3], crystal_atom_idx=input[4])
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
                atom_fea, atom_symm, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
                target_normed = normalizer.norm(target)
                target_var = target_normed.to(self.device, non_blocking=True)
            #calculate loss
            pred = model(atom_fea=atom_fea, atom_symm=atom_symm, nbr_fea=nbr_fea,
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
            self.write_list2d(f'{self.model_save_path}/pred.dat', 
                              pred_all, style='{0:8.4f}')
            self.write_list2d(f'{self.model_save_path}/vasp.dat', 
                              vasp_all, style='{0:8.4f}')
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
        

if __name__ == '__main__':
    pass