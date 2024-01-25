import os, sys, time
import numpy as np

import torch

sys.path.append(f'{os.getcwd()}/src')
from core.log_print import *
from core.utils import *
from core.GNN_model import *
from core.del_duplicates import DeleteDuplicates
from core.utils import ListRWTools, SSHTools


class GNNPredict(DeleteDuplicates):
    #get energy and crystal vector by gnn
    def __init__(self, batch_size=128, num_workers=0):
        DeleteDuplicates.__init__(self)
        self.batch_size = batch_size
        self.num_workers = num_workers
        if Job_Queue == 'CPU':
            self.device = torch.device('cpu')
        elif Job_Queue == 'GPU':
            self.device = torch.device('cuda')
        self.normalizer = Normalizer(torch.tensor([]))
    
    def get_gnn_model(self):
        """
        get initial prediction models
        
        Returns
        ----------
        model [str, 0d]: absolute path of GNN model
        """
        iterations = os.listdir(Model_Path)
        if len(iterations) > 0:
            model = f'{Model_Path}/{iterations[-1]}/model_best.pth.tar'
        else:
            if Use_Pretrain_Model:
                if Dimension == 2:
                    model = Pretrain_Model_2d
                elif Dimension == 3:
                    model = Pretrain_Model_3d
            else:
                model = 'random'
        return model
    
    def load_normalizer(self, model_name):
        """
        load normalizer

        Parameters
        ----------
        model_name [str, 0d]: full name of model
        """
        if model_name == 'random':
            pass
        else:
            params = torch.load(model_name, map_location=self.device)
            self.normalizer.load_state_dict(params['normalizer'])
    
    def load_gnn_model(self, model_name):
        """
        load prediction model

        Parameters
        ----------
        model_name [str, 0d]: full name of model
        """
        self.gnn_model = CrystalGraphConvNet()
        if model_name == 'random':
            pass
        else:
            params = torch.load(model_name, map_location=self.device)
            self.gnn_model.load_state_dict(params['state_dict'])
    
    def load_fea_model(self, model_name):
        """
        load crystal feature model

        Parameters
        ----------
        model_name [str, 0d]: full name of model
        """
        self.vec_model = FeatureExtractNet()
        if model_name == 'random':
            pass
        else:
            params = torch.load(model_name, map_location=self.device)
            self.vec_model.load_state_dict(params['state_dict'])
    
    def load_out_model(self, model_name):
        """
        load readout model

        Parameters
        ----------
        model_name [str, 0d]: full name of model
        """
        self.out_model = ReadoutNet()
        if model_name == 'random':
            pass
        else:
            params = torch.load(model_name, map_location=self.device)
            self.out_model.load_state_dict(params['state_dict'])
    
    def dataloader_all_atoms(self, atom_pos, atom_type, grid_name, grid_ratio, space_group, angles, thicks):
        """
        transfer data to the input of GNN and put them into DataLoader
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atom
        atom_type [int, 2d]: type of atom
        grid_name [int, 1d]: name of grid
        grid_ratio [float, 1d]: grid ratio
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        
        Returns
        ----------
        loader [obj, 0d]: dataloader
        """
        strus = self.get_stru_batch_parallel(atom_pos, atom_type, grid_name, grid_ratio, space_group, angles, thicks)
        atom_fea, nbr_fea, nbr_idx = self.get_gnn_input_from_stru_batch_parallel(strus)
        targets = np.zeros(len(grid_name))
        atom_symm = [[1 for _ in range(len(item))] for item in atom_fea]
        dataset = GNNData(atom_fea, atom_symm, nbr_fea, nbr_idx, targets)
        loader = get_loader(dataset, self.batch_size, self.num_workers)
        return loader
    
    def dataloader_general(self, atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group):
        """
        transfer data to the input of GNN and put them into DataLoader
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atom
        atom_type [int, 2d]: type of atom
        atom_symm [int, 2d]: symmetry of atom
        grid_name [int, 1d]: name of grid
        grid_ratio [float, 1d]: grid ratio
        space_group [int, 1d]: space group number
        
        Returns
        ----------
        loader [obj, 0d]: dataloader 
        """
        atom_fea, nbr_fea, nbr_idx = \
            self.get_gnn_input_batch_general(atom_pos, atom_type, grid_name, grid_ratio, space_group)
        targets = np.zeros(len(grid_name))
        dataset = GNNData(atom_fea, atom_symm, nbr_fea, nbr_idx, targets)
        loader = get_loader(dataset, self.batch_size, self.num_workers)
        return loader
    
    def dataloader_template(self, atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group):
        """
        transfer data to the input of GNN and put them into DataLoader
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atom
        atom_type [int, 2d]: type of atom
        atom_symm [int, 2d]: symmetry of atom
        grid_name [int, 1d]: name of grid
        grid_ratio [float, 1d]: grid ratio
        space_group [int, 1d]: space group number
        
        Returns
        ----------
        loader [obj, 0d]: dataloader
        """
        atom_fea, nbr_fea, nbr_idx = \
            self.get_gnn_input_batch_template(atom_pos, atom_type, grid_name, grid_ratio, space_group)
        targets = np.zeros(len(grid_name))
        dataset = GNNData(atom_fea, atom_symm, nbr_fea, nbr_idx, targets)
        loader = get_loader(dataset, self.batch_size, self.num_workers)
        return loader
    
    def predict_single(self, symm, atom_fea, nbr_fea, nbr_idx):
        """
        get crystal vector and energy for one structure
        
        Parameters
        ----------
        symm [int, 1d]: symmetry of atoms
        atom_fea [float, 2d]: atom feature 
        nbr_fea [float, 3d]: bond feature
        nbr_idx [int, 2d]: neighbor index

        Returns
        ----------
        energy [float, 0d]: prediction energy
        crys_vec_np [float, 1d, np]: crystal vector
        """
        crystal_atom_idx = np.arange(len(atom_fea))
        input_var = (torch.Tensor(atom_fea),
                     torch.Tensor(symm),
                     torch.Tensor(nbr_fea),
                     torch.LongTensor(nbr_idx),
                     [torch.LongTensor(crystal_atom_idx)])
        #get crystal vector
        self.vec_model.eval()
        crys_vec = self.vec_model(*input_var)
        #predict energy
        self.out_model.eval()
        pred = self.out_model(crys_vec)
        energy = self.normalizer.denorm(pred.data).item()
        crys_vec_np = crys_vec.cpu().detach().numpy().flatten()
        return energy, crys_vec_np
        
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
        if Job_Queue == 'GPU':
            self.gnn_model = DataParallel(self.gnn_model)
        self.gnn_model.to(self.device)
        self.gnn_model.eval()
        energy = []
        with torch.no_grad():
            for input, _ in loader:
                atom_fea, atom_symm, nbr_fea, nbr_idx, crystal_atom_idx = input
                pred = self.gnn_model(atom_fea=atom_fea, 
                                      atom_symm=atom_symm,
                                      nbr_fea=nbr_fea,
                                      nbr_idx=nbr_idx, 
                                      crystal_atom_idx=crystal_atom_idx)
                energy.append(self.normalizer.denorm(pred))
        return torch.cat(energy)
    
    def predict_samples(self, loader):
        """
        predict energy of samples
        
        Parameters
        ----------
        loader [obj, 0d]: dataloader

        Returns
        ----------
        energys [float, 1d]: gnn energys
        crys_vec [float, 2d]: crystal vectors
        """
        #predict energy and calculate crystal vector
        crys_vec = self.get_crystal_vector_batch(loader)
        energys = self.readout_crystal_vector_batch(crys_vec)
        energys = energys.cpu().numpy().flatten().tolist()
        crys_vec = torch.cat(crys_vec).cpu().numpy().tolist()
        return energys, crys_vec
    
    def get_crystal_vector_batch(self, loader):
        """
        get crystal vector by pretrain model
        
        Parameters
        ----------
        loader [obj, 0d]: dataloader
        
        Returns
        ----------
        crys_fea_np [float, 3d, list-Tensor]: crystal vector
        """
        #calculate crystal vector
        if Job_Queue == 'GPU':
            self.vec_model = DataParallel(self.vec_model)
        self.vec_model.to(self.device)
        self.vec_model.eval()
        crys_vec = []
        with torch.no_grad():
            for input, _ in loader:
                atom_fea, atom_symm, nbr_fea, nbr_idx, crystal_atom_idx = input
                vecs = self.vec_model(atom_fea=atom_fea, 
                                      atom_symm=atom_symm,
                                      nbr_fea=nbr_fea,
                                      nbr_idx=nbr_idx, 
                                      crystal_atom_idx=crystal_atom_idx)
                crys_vec.append(vecs)
        return crys_vec
    
    def readout_crystal_vector_batch(self, crys_vec):
        """
        get energy prediction from crystal vector
        
        Parameters
        ----------
        crys_fea_np [float, 3d, list-Tensor]: crystal vector

        Returns
        ----------
        energys [Tensor, 2d]: gnn energys
        """
        if Job_Queue == 'GPU':
            self.out_model = DataParallel(self.out_model)
            crys_vec = torch.chunk(crys_vec, Num_GPUs)
        #predict energy
        energys = []
        with torch.no_grad():
            for vec in crys_vec:
                pred = self.out_model(crys_fea=vec)
                pred = self.normalizer.denorm(pred)
                energys.append(pred)
        return torch.cat(energys)


class ParallelPESUpdate(ListRWTools, SSHTools):
    #update PES in parallel
    def __init__(self, wait_time=0.1):
        SSHTools.__init__(self)
        self.wait_time = wait_time
        self.local_grid_path = f'{SCCOP_Path}/{Grid_Path}'
    
    def sub_update_jobs(self, iteration):
        """
        submit PES updating jobs to work nodes
        
        Parameters
        ----------
        iteration [int, 0d]: sccop iteration
        """
        self.iteration = f'{iteration:02.0f}'
        self.model_save_path = f'{Model_Path}/{self.iteration}'
        #submit updating jobs to each work node
        for node in self.work_nodes:
            self.sub_updating(node)
        while not self.is_done(Grid_Path, self.work_nodes_num):
            time.sleep(self.wait_time)
        self.remove_flag_on_host()
        #collect updated samples
        self.unzip_mcts_samples_on_host()
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec = \
            self.collect_mcts_samples_all()
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec
        
    def sub_updating(self, node, job_limit=20, job_monitor=3, cpu_usage=20, wait_time=5, repeat_time=2):
        """
        updating PES by a node
        
        Parameters
        ----------
        node [str, 0d]: name of node
        job_limit [int, 0d]: number of jobs in parallel
        job_monitor [int, 0d]: job number for monitoring sampling
        cpu_usage [float, 0d]: low cpu usage for monitoring sampling
        wait_time [float, 0d]: waiting time for low cpu usage
        """
        #monitor updating process
        job_file = f'sampling_jobs_{node}.dat'
        monitor_script = f'''
                          repeat=0
                          start=$(date +%s)
                          while true
                          do
                              end=$(date +%s)
                              time=$((end - start))
                              if [ $time -ge {Sampling_Time_Limit} ]; then
                                  rm {Grid_Path}/RUNNING*
                                  ps -ef | grep 'python src/core/MCTS.py' | grep -v grep | awk '{{print $2}}' | sort | uniq | xargs kill -9
                                  break
                              fi
                              
                              counter=`ls {Grid_Path} | grep RUNNING | wc -l`
                              cpu_sum=`ps aux | grep 'python src/core/MCTS.py' | grep -v grep | awk '{{sum += $3}} END {{print sum+0}}'`
                              if [ $counter -eq 0 ]; then
                                  cpu_ave=0
                              else
                                  cpu_ave=`echo "$cpu_sum / $counter" | bc -l`
                              fi
                              if [ $counter -le {job_monitor} ]; then
                                  is_cpu_low=`echo "$cpu_ave < {cpu_usage}" | bc -l`
                                  ((repeat++))
                                  if [ $is_cpu_low -eq 1 ]; then
                                      ((repeat++))
                                      if [ $repeat -ge {wait_time} ]; then
                                          rm {Grid_Path}/RUNNING*
                                          ps -ef | grep 'python src/core/MCTS.py' | grep -v grep | awk '{{print $2}}' | sort | uniq | xargs kill -9
                                          break
                                      fi
                                  fi
                                  sleep 1s
                              fi
                              sleep 1s
                              echo $cpu_ave >> data/grid/cpu_use
                              echo $counter >> data/grid/cpu_num
                          done
                          echo ---------- >> data/grid/cpu_use
                          echo ---------- >> data/grid/cpu_num
                          rm {Grid_Path}/RUNNING*
                          '''
        mcts_script = f'''
                       sub_counter=0
                       cat {job_file} | while read line
                       do
                           flag=`echo $line | awk '{{print $NF}}'`
                           if [ $sub_counter -le {job_limit} ]; then
                               python $line >> log&
                               touch {Grid_Path}/RUNNING_$flag
                               ((sub_counter++))
                               echo $sub_counter >> data/grid/counter_log
                           else
                               counter=`ls {Grid_Path} | grep RUNNING | wc -l`
                               echo $counter >> data/grid/counter_log
                               if [ $counter -eq 0 ]; then
                                   sleep {wait_time}s
                               else
                                   if [ $counter -gt {job_limit} ]; then
                                       while true
                                       do
                                           counter=`ls {Grid_Path} | grep RUNNING | wc -l`
                                           echo $counter >> data/grid/counter_log
                                           if [ $counter -le {job_limit} ]; then
                                               break
                                           fi
                                           sleep 1s
                                       done
                                   fi
                               fi
                               python $line >> log&
                               touch {Grid_Path}/RUNNING_$flag
                               ((sub_counter++))
                               echo $sub_counter >> data/grid/counter_log
                           fi
                       done
                       echo ---------- >> data/grid/counter_log
                       '''
        #shell script of PES updating
        shell_script = f'''
                        #!/bin/bash --login
                        {SCCOP_Env}
                        
                        cd {SCCOP_Path}/
                        if [ ! -d {self.model_save_path} ]; then
                            mkdir {self.model_save_path}
                            scp {Host_Node}:{SCCOP_Path}/{self.model_save_path}/model_best.pth.tar {self.model_save_path}/.
                        fi
                        rm {Buffer_Path}/*
                        
                        ls {Recyc_Store_Path} | grep space | awk -F'_' '{{print $3}}' | sed 's/.dat//' > tmp.dat
                        sed 's/^/src\/core\/MCTS.py --flag 1 --node {node} --grid /' tmp.dat > {job_file}
                        rm tmp.dat
                        
                        for i in `seq 1 {repeat_time}`
                        do
                            {mcts_script}
                            {monitor_script}
                            mv tmp_sampling_jobs.dat {job_file}
                        done
                        rm log {job_file}
                        python src/core/MCTS.py --flag 3 --node {node}
                        
                        cd {Buffer_Path}
                        scp {node}.tar.gz {Host_Node}:{self.local_grid_path}/buffer/.
                        
                        cd ../
                        touch FINISH-{node}
                        scp FINISH-{node} {Host_Node}:{self.local_grid_path}/.
                        '''
        self.ssh_node(shell_script, node)
    
    def unzip_mcts_samples_on_host(self):
        """
        unzip mcts samples on host node
        """
        work_nodes = ' '.join([f'{i}' for i in self.work_nodes])
        shell_script = f'''
                        #!/bin/bash --login
                        cd {SCCOP_Path}/{Buffer_Path}
                        for i in {work_nodes}
                        do
                            tar -zxf $i.tar.gz
                        done
                        rm *.tar.gz
                        '''
        os.system(shell_script)
    
    def remove_flag_on_host(self):
        """
        remove flag file on host node
        """
        shell_script = f'''
                        #!/bin/bash --login
                        cd {SCCOP_Path}/{Grid_Path}
                        rm FINISH*
                        '''
        os.system(shell_script)
    
    def remove_flag_on_work_nodes(self, node):
        """
        remove flag file on work nodes
        """
        shell_script = f'''
                        #!/bin/bash --login
                        cd {self.local_grid_path}
                        rm GRID-SEND-DONE*
                        rm *.tar.gz
                        rm sampling*
                        rm FINISH*
                        '''
        self.ssh_node(shell_script, node)
    
    def collect_mcts_samples_all(self):
        """
        collect samples from all nodes

        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: structure energy
        crys_vec [float, 2d]: crystal vectors
        """
        atom_pos, atom_type, atom_symm = [], [], []
        grid_name, grid_ratio, space_group, angles, thicks =  [], [], [], [], []
        energys, crys_vec = [], []
        for node in self.work_nodes:
            #get name of path record files
            pos_file = f'{Buffer_Path}/atom_pos_{node}.dat'
            type_file = f'{Buffer_Path}/atom_type_{node}.dat'
            symm_file = f'{Buffer_Path}/atom_symm_{node}.dat'
            grid_file = f'{Buffer_Path}/grid_name_{node}.dat'
            ratio_file = f'{Buffer_Path}/grid_ratio_{node}.dat'
            sg_file = f'{Buffer_Path}/space_group_{node}.dat'
            angle_file = f'{Buffer_Path}/angles_{node}.dat'
            thick_file = f'{Buffer_Path}/thicks_{node}.dat'
            energy_file = f'{Buffer_Path}/energy_{node}.dat'
            crys_vec_file = f'{Buffer_Path}/crys_vec_{node}.bin'
            #get searching results
            if os.path.exists(energy_file):
                pos = self.import_list2d(pos_file, int)
                type = self.import_list2d(type_file, int)
                symm = self.import_list2d(symm_file, int)
                grid = self.import_list2d(grid_file, int)
                ratio = self.import_list2d(ratio_file, float)
                sg = self.import_list2d(sg_file, int)
                angle = self.import_list2d(angle_file, int)
                thick = self.import_list2d(thick_file, int)
                energy = self.import_list2d(energy_file, float)
                vec = self.import_list2d(crys_vec_file, float, binary=True)
                atom_pos += pos
                atom_type += type
                atom_symm += symm
                grid_name += grid
                grid_ratio += ratio
                space_group += sg
                angles += angle
                thicks += thick
                energys += energy
                crys_vec += vec.tolist()
        #convert to 1d list
        grid_name = np.array(grid_name).flatten().tolist()
        grid_ratio = np.array(grid_ratio).flatten().tolist()
        space_group = np.array(space_group).flatten().tolist()
        energys = np.array(energys).flatten().tolist()
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec
    

class GNNSlice(GNNPredict):
    #PES silicing by GNN
    def __init__(self):
        GNNPredict.__init__(self)
        
    def gnn_slice_local(self, atom_pos, atom_type, atom_symm, 
                        grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec, sample_limit=100):
        """
        reduce number of samples by crystal vector

        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: prediction energys
        crys_vec [float, 2d]: crystal vectors
        sample_limit [int, 0d]: limit of samples
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms after constrain
        atom_type [int, 2d]: type of atoms after constrain
        atom_symm [int, 2d]: symmetry of atoms after constrain
        grid_name [int, 1d]: name of grids after constrain
        grid_ratio [float, 1d]: ratio of grids after constrain
        space_group [int, 1d]: space group number after constrain
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: prediction energys
        """
        #balance sampling by space group
        if len(space_group) > 10*sample_limit:
            idx = self.balance_sampling(10*sample_limit, space_group, energys)
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, grid_ratio, space_group, angles, thicks)
            energys = np.array(energys)[idx]
            crys_vec = np.array(crys_vec)[idx]
        #sorted by space groups and energy
        idx = self.sort_by_sg_energy(space_group, energys)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = np.array(energys)[idx]
        crys_vec = np.array(crys_vec)[idx]
        #delete duplicates by crystal vectors and space groups
        if Use_ML_Clustering:
            idx = self.delete_duplicates_crys_vec_sg(crys_vec, space_group, energys)
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, grid_ratio, space_group, angles, thicks)
            energys = energys[idx]
            crys_vec = crys_vec[idx]
        #reduce dimension and clustering
        idx = self.reduce_by_gnn(crys_vec, space_group, energys, min_num=len(np.unique(space_group)))
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = energys[idx]
        crys_vec = crys_vec[idx]
        #balance sampling by space group
        if len(space_group) > sample_limit:
            idx = self.balance_sampling(sample_limit, space_group, energys)
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
                self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                    grid_name, grid_ratio, space_group, angles, thicks)
            energys = energys[idx]
            crys_vec = crys_vec[idx]
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec
    
    def gnn_slice_global(self, atom_pos, atom_type, atom_symm, 
                         grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec, sample_limit=100):
        """
        geometry constrain to reduce structures
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: prediction energys
        crys_vec [float, 2d]: crystal vectors
        sample_limit [int, 0d]: limit of samples
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms after constrain
        atom_type [int, 2d]: type of atoms after constrain
        atom_symm [int, 2d]: symmetry of atoms after constrain
        grid_name [int, 1d]: name of grids after constrain
        grid_ratio [float, 1d]: ratio of grids after constrain
        space_group [int, 1d]: space group number after constrain
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: prediction energys
        """
        system_echo(f'Start PES slicing')
        #limit sampling number
        idx = np.argsort(energys)[:10*sample_limit]
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = np.array(energys)[idx]
        crys_vec = np.array(crys_vec)[idx]
        #balance sampling by space group
        idx = self.balance_sampling(4*sample_limit, space_group, energys, print=True)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = energys[idx]
        crys_vec = crys_vec[idx]
        #sample clustering
        idx = self.reduce_by_gnn(crys_vec, space_group, energys, min_num=2*sample_limit)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = energys[idx]
        crys_vec = crys_vec[idx]
        #sort structure in order of space group, energy
        idx = self.sort_by_sg_energy(space_group, energys)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = energys[idx]
        crys_vec = crys_vec[idx]
        #delete same samples under same space group
        idx = self.delete_duplicates_sg_pymatgen(atom_pos, atom_type,
                                                 grid_name, grid_ratio, space_group, angles, thicks, energys)
        system_echo(f'Delete duplicates: {len(idx)}')
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = energys[idx]
        crys_vec = crys_vec[idx]
        #balance sampling
        idx = self.balance_sampling(sample_limit, space_group, energys, print=True)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = energys[idx]
        crys_vec = crys_vec[idx]
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec
    
    def update_PES(self, atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks):
        """
        update PES by GNN
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction

        Returns
        ----------
        energys [float, 1d]: energy of structures
        crys_vec [float, 2d]: crystal vectors
        """
        #get data loader
        if Cluster_Search or (Dimension == 2 and Thickness > 0):
            loader = self.dataloader_all_atoms(atom_pos, atom_type, grid_name, grid_ratio, space_group, angles, thicks)
        else:
            if General_Search:
                loader = self.dataloader_general(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group)
            elif Template_Search:
                loader = self.dataloader_template(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group)
        #load GNN model
        model = self.get_gnn_model()
        self.load_fea_model(model)
        self.load_out_model(model)
        self.load_normalizer(model)
        #predict energy and calculate crystal vector
        crys_vec = self.get_crystal_vector_batch(loader)
        energys = self.readout_crystal_vector_batch(crys_vec)
        energys = energys.cpu().numpy().flatten()
        crys_vec = torch.cat(crys_vec).cpu().numpy()
        return energys, crys_vec
    
    def candidate_select(self, atom_pos, atom_type, atom_symm, grid_name, grid_ratio,
                         space_group, angles, thicks, energys, crys_vec, 
                         train_pos, train_type, train_symm, train_grid, train_ratio,
                         train_sg, train_angles, train_thicks, train_energy, node_num):
        """
        select initial searching points
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: energy of structures
        crys_vec [float, 2d]: crystal vectors
        train_pos [int, 2d]: position of atoms
        train_type [int, 2d]: type of atoms
        train_symm [int, 2d]: symmetry of atoms
        train_grid [int, 1d]: name of grids
        train_ratio [float, 1d]: ratio of grids
        train_sg [int, 1d]: space group number
        train_angles [int, 2d]: cluster rotation angles
        train_thicks [int, 2d]: atom displacement in z-direction
        train_energy [float, 1d]: train energy
        node_num [int, 0d]: number of nodes
        
        Returns
        ----------
        init_pos [int, 2d]: initial position of atoms
        init_type [int, 2d]: initial type of atoms
        init_symm [int, 2d]: initial symmetry of atoms
        init_grid [int, 1d]: initial name of grids
        init_ratio [float, 1d]: initial ratio of grids
        init_sgs [int, 1d]: initial space group
        init_angles [int, 2d]: initial angles
        init_thicks [int, 2d]: initial thickness
        """
        #PES slicing
        init_poscar_num = Init_Strus_per_Node*node_num
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec = \
            self.gnn_slice_global(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, 
                                  space_group, angles, thicks, energys, crys_vec, sample_limit=init_poscar_num)
        #select candidates by GNN
        num_path = SA_Path_per_Node*node_num
        idx = self.topk_select(int(Exploration_Ratio*num_path), energys, ratio=1)
        init_pos_GNN, init_type_GNN, init_symm_GNN, init_grid_GNN, init_ratio_GNN, init_sgs_GNN, init_angles_GNN, init_thicks_GNN = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks)
        #select candidates by DFT
        idx = self.topk_select(int((1-Exploration_Ratio)*num_path), train_energy, limit_num=Exploitation_Num, print=True)
        init_pos_DFT, init_type_DFT, init_symm_DFT, init_grid_DFT, init_ratio_DFT, init_sgs_DFT, init_angles_DFT, init_thicks_DFT = \
            self.filter_samples(idx, train_pos, train_type, train_symm, train_grid, train_ratio, train_sg, train_angles, train_thicks)
        #balance exploration and exploitation
        init_pos = init_pos_GNN + init_pos_DFT
        init_type = init_type_GNN + init_type_DFT
        init_symm = init_symm_GNN + init_symm_DFT
        init_grid = init_grid_GNN + init_grid_DFT
        init_ratio = init_ratio_GNN + init_ratio_DFT
        init_sgs = init_sgs_GNN + init_sgs_DFT
        init_angles = init_angles_GNN + init_angles_DFT
        init_thicks = init_thicks_GNN + init_thicks_DFT
        #shuffle
        idx = np.arange(len(init_sgs))
        np.random.shuffle(idx)
        init_pos, init_type, init_symm, init_grid, init_ratio, init_sgs, init_angles, init_thicks = \
            self.filter_samples(idx, init_pos, init_type, init_symm, init_grid, init_ratio, init_sgs, init_angles, init_thicks)
        system_echo(f'Number of SA path: {num_path}')
        return init_pos, init_type, init_symm, init_grid, init_ratio, init_sgs, init_angles, init_thicks
    
    def topk_select(self, num, energys, ratio=.1, limit_num=500, print=False):
        """
        get index of min energy samples
        
        Parameters
        ----------
        num [int, 0d]: number of SA paths
        energys [list, 1d]: energys in trainset
        ratio [int, 0d]: select topk samples
        limit_num [bool, 0d]:
        print [bool, 0d]: whether print log
        
        Returns
        ----------
        idx [int, 1d, np]: index of selected samples
        """
        sample_num = len(energys)
        idx_all = np.arange(sample_num)
        order = np.argsort(energys)
        idx_order = idx_all[order]
        topk_num = min(max(10, int(ratio*sample_num)), limit_num)
        idx_order = idx_order[:topk_num]
        if print:
            system_echo(f'{np.array(energys)[idx_order]}')
        #sampling
        idx = np.random.choice(idx_order, num)
        return idx

    def import_mcts_samples(self, grid):
        """
        import monte carlo tree search samples 

        Returns
        ----------
        grid [int, 0d]: grid name
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: grid name
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: structure energy
        crys_vec [float, 2d]: crystal vector
        """
        #import mcts data
        atom_pos = self.import_list2d(f'{Buffer_Path}/atom_pos_{grid}.dat', int)
        atom_type = self.import_list2d(f'{Buffer_Path}/atom_type_{grid}.dat', int)
        atom_symm = self.import_list2d(f'{Buffer_Path}/atom_symm_{grid}.dat', int)
        grid_name = self.import_list2d(f'{Buffer_Path}/grid_name_{grid}.dat', int)
        grid_ratio = self.import_list2d(f'{Buffer_Path}/grid_ratio_{grid}.dat', float)
        space_group = self.import_list2d(f'{Buffer_Path}/space_group_{grid}.dat', int)
        angles = self.import_list2d(f'{Buffer_Path}/angles_{grid}.dat', int)
        thicks = self.import_list2d(f'{Buffer_Path}/thicks_{grid}.dat', int)
        energys = self.import_list2d(f'{Buffer_Path}/energy_{grid}.dat', float)
        crys_vec = self.import_list2d(f'{Buffer_Path}/crys_vec_{grid}.bin', float, binary=True).tolist()
        #flatten list
        grid_name = np.concatenate(grid_name).tolist()
        grid_ratio = np.concatenate(grid_ratio).tolist()
        space_group = np.concatenate(space_group).tolist()
        energys = np.concatenate(energys).tolist()
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec
    
    def export_mcts_samples(self, grid, atom_pos, atom_type, atom_symm,
                            grid_name, grid_ratio, sgs, angles, thicks, energys, crys_vec):
        """
        export monte carlo tree search samples
        
        Parameters
        ----------
        grid [int, 0d]: grid name
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: grid name
        grid_ratio [float, 1d]: ratio of grids
        sgs [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: structure energy
        crys_vec [float, 2d]: crystal vector
        """
        #export sampling results
        self.write_list2d(f'{Buffer_Path}/atom_pos_{grid}.dat',
                          atom_pos)
        self.write_list2d(f'{Buffer_Path}/atom_type_{grid}.dat',
                          atom_type)
        self.write_list2d(f'{Buffer_Path}/atom_symm_{grid}.dat',
                          atom_symm)
        self.write_list2d(f'{Buffer_Path}/grid_name_{grid}.dat',
                          np.transpose([grid_name]))
        self.write_list2d(f'{Buffer_Path}/grid_ratio_{grid}.dat',
                          np.transpose([grid_ratio]), style='{0:8.4f}')
        self.write_list2d(f'{Buffer_Path}/space_group_{grid}.dat',
                          np.transpose([sgs]))
        self.write_list2d(f'{Buffer_Path}/angles_{grid}.dat',
                          angles)
        self.write_list2d(f'{Buffer_Path}/thicks_{grid}.dat',
                          thicks)
        self.write_list2d(f'{Buffer_Path}/energy_{grid}.dat',
                          np.transpose([energys]), style='{0:9.6f}')
        self.write_list2d(f'{Buffer_Path}/crys_vec_{grid}.bin',
                          crys_vec, binary=True)


if __name__ == '__main__':
    pass