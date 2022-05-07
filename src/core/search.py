import os, sys, time
import itertools
import numpy as np
import torch
import argparse
from functools import reduce

from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.data_transfer import Transfer
from core.utils import ListRWTools, SSHTools, system_echo
from core.predict import CrystalGraphConvNet, Normalizer


parser = argparse.ArgumentParser()
parser.add_argument('--atom_pos', type=int, nargs='+')
parser.add_argument('--atom_type', type=int, nargs='+')
parser.add_argument('--round', type=int)
parser.add_argument('--path', type=int)
parser.add_argument('--node', type=int)
parser.add_argument('--grid_name', type=int)
parser.add_argument('--model_name', type=str)
args = parser.parse_args()


class ParallelWorkers(ListRWTools, SSHTools):
    #Assign sampling jobs to each node
    def __init__(self, sleep_time=1):
        self.sleep_time = sleep_time
    
    def search(self, round, num_paths, init_pos, init_type, init_grid):
        """
        assign jobs by the following files
        initial_pos_XXX.dat: initial position
        initial_type_XXX.dat: initial type
        worker_job_XXX.dat: node job
        e.g. round path node grid model
             1 1 131 1 model_best.pth.tar
             
        Parameters
        ----------
        round [int, 0d]: searching round
        num_paths [int, 0d]: number of search path
        init_pos [int, 2d]: initial position
        init_type [int, 2d]: initial atom type
        init_grid [int, 1d]: initial grid name
        """
        self.round = f'{round:03.0f}'
        self.sh_save_path = f'{search_path}/{self.round}'
        self.model_save_path = f'{model_path}/{self.round}'
        self.generate_job(round, num_paths, init_pos,
                          init_type, init_grid)
        pos, type, job = self.read_job()
        system_echo('Get the job node!')
        for node in nodes:
            self.update_with_ssh(node)
        while not self.is_done(self.sh_save_path, len(nodes)):
            time.sleep(self.sleep_time)
        self.remove_flag(self.sh_save_path)
        system_echo('Successful update each node!')
        #sub searching job to each node
        work_node_num = self.sub_job_to_workers(pos, type, job)
        system_echo('Successful assign works to workers!')
        while not self.is_done(self.sh_save_path, work_node_num):
            time.sleep(self.sleep_time)
        self.unzip()
        #collect searching path
        atom_pos, atom_type, grid_name = self.collect(job)
        system_echo(f'All workers are finished!---sample number: {len(atom_pos)}')
        #export searching results
        self.write_list2d(f'{self.sh_save_path}/atom_pos.dat', 
                          atom_pos, style='{0:3.0f}')
        self.write_list2d(f'{self.sh_save_path}/atom_type.dat',
                          atom_type, style='{0:3.0f}')
        self.write_list2d(f'{self.sh_save_path}/grid_name.dat',
                          grid_name, style='{0:3.0f}')
        self.remove_all_path()
    
    def generate_job(self, round, num_paths, 
                     init_pos, init_type, init_grid):
        """
        generate initial searching files
        
        Parameters
        ----------
        round [int, 0d]: searching round
        num_paths [int, 0d]: number of search path
        init_pos [int, 2d]: initial position
        init_type [int, 2d]: initial atom type
        init_grid [int, 1d]: initial grid name
        """
        if not os.path.exists(self.sh_save_path):
            os.mkdir(self.sh_save_path)
        worker_job = []
        model = 'model_best.pth.tar'
        node_assign = self.assign_node(num_paths)
        for i, node in enumerate(node_assign):
            job = [round, i, node, init_grid[i], model]
            worker_job.append(job)
        #export searching files
        worker_file = f'{self.sh_save_path}/worker_job_{self.round}.dat'
        pos_file = f'{self.sh_save_path}/initial_pos_{self.round}.dat'
        type_file = f'{self.sh_save_path}/initial_type_{self.round}.dat'
        self.write_list2d(worker_file, worker_job)
        self.write_list2d(pos_file, init_pos)
        self.write_list2d(type_file, init_type)
    
    def update_with_ssh(self, node):
        """
        SSH to target node and update ccop file
        """
        ip = f'node{node}'
        shell_script = f'''
                        #!/bin/bash
                        cd /local/ccop/
                        mkdir {self.sh_save_path}
                        cd data/
                        scp -r {gpu_node}:/local/ccop/{self.model_save_path} ppmodel/
                        
                        touch FINISH-{ip}
                        scp FINISH-{ip} {gpu_node}:/local/ccop/{self.sh_save_path}/
                        rm FINISH-{ip}
                        '''
        self.ssh_node(shell_script, ip)
    
    def sub_job_to_workers(self, pos, type, job):
        """
        sub searching jobs to nodes

        Parameters
        ----------
        pos [str, 1d, np]: initial pos
        type [str, 1d, np]: initial type
        job [str, 2d, np]: jobs assgined to nodes
        """
        #poscars grouped by nodes
        pos_node, type_node, job_node = [], [], []
        pos_assign, type_assign, job_assign = [], [], []
        for node in nodes:
            for i, line in enumerate(job):
                label = line[2]
                if label == str(node):
                    pos_assign.append(pos[i])
                    type_assign.append(type[i])
                    job_assign.append(line)
            pos_node.append(pos_assign)
            type_node.append(type_assign)
            job_node.append(np.transpose(job_assign))
            pos_assign, type_assign, job_assign = [], [], []
        #sub job to target node
        for atom_pos, atom_type, assign in zip(pos_node, type_node, job_node):
            self.sampling_with_ssh(atom_pos, atom_type, *assign)
        work_node_num = len(pos_node)
        return work_node_num
    
    def sampling_with_ssh(self, atom_pos, atom_type, 
                          round, path, node, grid_name, model_name):
        """
        SSH to target node and call workers for sampling

        Parameters
        ----------
        atom_pos [str, 1d]: initial atom position
        atom_type [str, 1d]: initial atom type
        round [str, 1d]: searching round
        path [str, 1d]: searching path
        node [str, 1d]: searching node
        grid_name [str, 1d]: name of grid
        model_name [str, 1d]: name of predict model
        """
        ip = f'node{node[0]}'
        search_jobs = []
        for i in range(len(atom_pos)):
            option = f'--atom_pos {atom_pos[i]} --atom_type {atom_type[i]} ' \
                     f'--round {round[i]} --path {path[i]} --node {node[i]} ' \
                     f'--grid_name {grid_name[i]} --model_name {model_name[i]}'
            search_jobs.append(f'nohup python src/core/search.py {option} >& log&')
        search_jobs = ' '.join(search_jobs)
        #ssh to target node then search from different start
        local_sh_save_path = f'/local/ccop/{self.sh_save_path}'
        shell_script = f'''
                        #!/bin/bash
                        cd /local/ccop/
                        {search_jobs}
                        
                        while true;
                        do
                            num=`ps -ef | grep search.py | grep -v grep | wc -l`
                            if [ $num -eq 0 ]; then
                                rm log
                                break
                            fi
                            sleep 1s
                        done
                        
                        cd {self.sh_save_path}
                        tar -zcf search-{ip}.tar.gz *
                        touch FINISH-{ip}
                        scp search-{ip}.tar.gz FINISH-{ip} {gpu_node}:{local_sh_save_path}
                        rm *
                        '''
        self.ssh_node(shell_script, ip)
    
    def unzip(self):
        """
        unzip files of finish path
        """
        zip_file = os.listdir(self.sh_save_path)
        zip_file = [i for i in zip_file if i.endswith('gz')]
        zip_file = ' '.join(zip_file)
        shell_script = f'''
                        #!/bin/bash
                        cd {self.sh_save_path}
                        for i in {zip_file}
                        do
                            tar -zxf $i
                            rm $i
                        done
                        '''
        os.system(shell_script)
    
    def collect(self, job):
        """
        collect searching results of each worker
        number of atoms, type of atoms and grid should be changed
        read in grid order 
        
        Parameters
        ----------
        job [str, 2d, np]: jobs assgined to nodes
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 2d, np]: name of grid
        """
        job = np.transpose(job)
        round, path, node, grid = job[:4]
        atom_pos, atom_type, grid_name, lack = [], [], [], []
        for i in range(len(path)):
            #get name of path record files
            assign = (round[i], path[i], node[i])
            pos_file = self.get_file_name('pos', *assign)
            type_file = self.get_file_name('type', *assign)
            #get searching results
            if os.path.exists(pos_file):
                pos = self.import_list2d(pos_file, int)
                type = self.import_list2d(type_file, int)
                if len(pos) > 0:
                    atom_pos += pos
                    atom_type += type
                    num_sample = len(type)
                    zeros = np.zeros((num_sample, 1))
                    grid_name.append(zeros + int(grid[i]))
                else:
                    lack.append(pos_file)
            else:
                lack.append(pos_file)
        grid_name = np.vstack(grid_name)
        system_echo(f'Lack files: {lack}')
        return atom_pos, atom_type, grid_name
    
    def get_file_name(self, name, round, path, node):
        """
        read result of each worker
        
        Parameters
        ----------
        name [str, 0d]: name of search result
        round [str, 0d]: round of searching
        path [str, 0d]: path of searching
        node [str, 0d]: node used to search
        
        Returns
        ----------
        file [int, 2d]: name of search file 
        """
        file = f'{self.sh_save_path}/' \
            f'{name}-{round.zfill(3)}-{path.zfill(3)}-{node}.dat'
        return file
    
    def read_job(self):
        """
        import initialize file
        
        Returns
        ----------
        pos [str, 1d, np]: initial position string list
        type [str, 1d, np]: initial type string list
        job [str, 2d, np]: job assign string list
        """
        pos = self.read_dat(f'initial_pos_{self.round}.dat')
        type = self.read_dat(f'initial_type_{self.round}.dat')
        job = self.read_dat(f'worker_job_{self.round}.dat', split=True)
        return pos, type, job
    
    def read_dat(self, dat, split=False):
        """
        read initilize file
        
        Parameters
        ----------
        dat [str, 0d]: name of initilize file
        split [bool, 0d]: whether split item
        
        Returns
        ----------
        list [str, 1d or 2d, np]: return string list
        """
        file = f'{self.sh_save_path}/{dat}'
        with open(file, 'r') as f:
            ct = f.readlines()
        if split:
            list = [item.split() for item in ct]
        else:
            list = [item.replace('\n','') for item in ct]
        return np.array(list)
    
    def remove_all_path(self):
        """
        remove file of each path
        """
        shell_script = f'''
                        cd {self.sh_save_path}
                        rm FINISH*
                        rm pos* type* energy*
                        '''
        os.system(shell_script)

class GeoCheck:
    #check geometry property of structure
    def __init__(self):
        pass
    
    def overlay(self, pos, num_atom):
        """
        advanced geometry constrain
        
        Parameters
        ----------
        pos [int, 1d]: position of atoms
        num_atom [int, 0d]: number of atoms
        
        Returns
        ----------
        flag [bool, 0d]: whether atoms are overlay
        """
        pos_differ = np.unique(pos)
        num_differ = len(pos_differ)
        if num_differ == num_atom:
            return True
        else:
            return False
    
    def near(self, nbr_dis):
        """
        advanced geometry constrain
        
        Parameters
        ----------
        nbr_dis [float, 2d, np]: distance of neighbors 
        
        Returns
        ----------
        flag [bool, 0d]: whether atoms are too closely
        """
        nearest = nbr_dis[:,0]
        error_bond = \
            len(nearest[nearest<min_bond])
        if error_bond == 0:
            return True
        else:
            return False
    
    def delete_same_poscars(self, path):
        """
        delete same structures
        
        Parameters
        -----------
        path [str, 0d]: path of poscars
        """
        poscars = sorted(os.listdir(path))
        poscars_num = len(poscars)
        same_poscars = []
        for i in range(poscars_num):
            stru_1 = Structure.from_file(f'{path}/{poscars[i]}')
            for j in range(i+1, poscars_num):
                stru_2 = Structure.from_file(f'{path}/{poscars[j]}')
                same = stru_1.matches(stru_2, ltol=0.1, stol=0.15, angle_tol=5, 
                                      primitive_cell=True, scale=False, 
                                      attempt_supercell=False, allow_subset=False)
                if same:
                    same_poscars.append(poscars[i])
        same_poscars = np.unique(same_poscars)
        for i in same_poscars:
            os.remove(f'{path}/{i}')
        same_poscars_num = len(same_poscars)
        system_echo(f'Delete same structures: {same_poscars_num}')
        
    
class Search(ListRWTools, GeoCheck):
    #Searching on PES by machine-learned potential
    def __init__(self, round, grid_name):
        self.transfer = Transfer(grid_name)
        self.device = torch.device('cpu')
        self.normalizer = Normalizer(torch.tensor([]))
        self.round = f'{round:03.0f}'
        self.model_save_path = f'{model_path}/{self.round}'
        self.sh_save_path = f'{search_path}/{self.round}'
    
    def explore(self, pos, type, model_name, path, node):
        """
        simulated annealing

        Parameters
        ----------
        pos [int, 1d]: atom position in grid
        type [int ,1d]: atom type
        model_name [str, 0d]: name of predict model
        path [int, 0d]: searching path
        node [int, 0d]: assign node
        """
        self.T = T
        self.load_model(model_name)
        pos_1, type_1 = pos, type
        value_1 = self.predict(pos_1, type_1)
        energy_buffer = []
        pos_buffer, type_buffer = [], []
        pos_buffer.append(pos_1)
        type_buffer.append(type_1)
        energy_buffer.append([value_1])
        for _ in range(steps):
            pos_2, type_2 = self.step(pos_1, type_1)
            value_2 = self.predict(pos_2, type_2)
            if self.metropolis(value_1, value_2, self.T):
                pos_1 = pos_2
                type_1 = type_2
                value_1 = value_2
                pos_buffer.append(pos_1)
                type_buffer.append(type_1)
                energy_buffer.append([value_1])
            self.T *= decay
        self.save(pos_buffer, type_buffer,
                  energy_buffer, path, node)
    
    def step(self, pos, type):
        """
        modify atom position under the geometry constrain
        
        Parameters
        ----------
        pos [int, 1d]: inital position of atom
        type [int, 1d]: initial type of atom
        
        Returns
        ----------
        new_pos [int, 1d]: position of atom after 1 SA step
        type [int, 1d]: type of atom after 1 SA step
        """
        new_pos = pos.copy()
        for _ in range(num_move):
            flag = False
            atom_num = len(new_pos)
            while not flag:
                #generate actions
                nbr_dis = self.transfer.grid_nbr_dis[new_pos]
                nbr_idx = self.transfer.grid_nbr_idx[new_pos]
                idx = np.random.randint(0, atom_num)
                actions_mv = self.action_filter(idx, nbr_dis, nbr_idx)
                actions_ex = self.exchange_action(type)
                replace_ex = [-1 for _ in actions_ex]
                actions = np.concatenate((actions_mv, replace_ex))
                #exchange atoms or move atom
                point = np.random.choice(actions)
                if point == -1:
                    actions_num = len(actions_ex)
                    idx = np.random.randint(0, actions_num)
                    idx_1, idx_2 = actions_ex[idx]
                    new_pos[idx_1], new_pos[idx_2] = \
                        new_pos[idx_2], new_pos[idx_1]
                else:
                    new_pos[idx] = point    
                flag = self.overlay(new_pos, atom_num)
        return new_pos, type.copy()
    
    def exchange_action(self, type):
        """
        actions of exchanging atoms
        
        Parameters
        ----------
        type [int, 1d]: type of atoms

        Returns
        ---------
        allow_action [int, 2d]: effective exchange actions
        """
        element = np.unique(type)
        buffer = np.arange(len(element))
        idx_ele, allow_action = [], []
        for ele in element:
            idx = [i for i, j in enumerate(type) if j==ele]
            idx_ele.append(idx)
        for i, j in itertools.combinations(buffer, 2):
            action = itertools.product(idx_ele[i], idx_ele[j])
            allow_action += [i for i in action]
        return allow_action
    
    def action_filter(self, idx, nbr_dis, nbr_idx):
        """
        distance between atoms should bigger than min_bond
        
        Parameters
        ----------
        nbr_dis [float, 2d, np]: grid neighbor distance of atoms
        nbr_idx [int, 2d, np]: grid neighbor index of atoms
        
        Returns
        ----------
        allow [int, 1d, np]: allowed actions
        """
        forbid_idx = []
        nbr_dis = np.delete(nbr_dis, idx, axis=0)
        nbr_idx = np.delete(nbr_idx, idx, axis=0)
        for item in nbr_dis:
            for i, dis in enumerate(item):
                if dis > min_bond:
                    forbid_idx.append(i) 
                    break
        actions = [nbr_idx[i][:j] for i, j in enumerate(forbid_idx)]
        forbid = reduce(np.union1d, actions)
        allow = np.setdiff1d(self.transfer.grid_point_array, forbid)
        return allow
    
    def metropolis(self, value_1, value_2, T):
        """
        metropolis criterion
        
        Parameters
        ----------
        value_1 [float, 0d]: current value
        value_2 [float, 0d]: next value
        T [float, 0d]: annealing temperature
        
        Returns
        ----------
        flag [bool, 0d]: whether do the action
        """
        delta = value_2 - value_1
        if np.exp(-delta/T) > np.random.rand():
            return True
        else:
            return False
    
    def load_model(self, model_name):
        """
        load predict model

        Parameters
        ----------
        model_name [str, 0d]: name of model
        """
        self.model = CrystalGraphConvNet(orig_atom_fea_len, 
                                         nbr_bond_fea_len)
        paras = torch.load(f'{self.model_save_path}/{model_name}', 
                           map_location=self.device)
        self.model.load_state_dict(paras['state_dict'])
        self.normalizer.load_state_dict(paras['normalizer'])
    
    def predict(self, pos, type):
        """
        predict energy

        Parameters
        ----------
        pos [int, 1d]: atom position in grid
        type [int ,1d]: atom type
        
        Returns
        ----------
        value [float, 0d]: predict value
        """
        atom_fea, nbr_fea, nbr_fea_idx = \
            self.transfer.single(pos, type)
        crystal_atom_idx = np.arange(len(atom_fea))
        input_var = (torch.Tensor(atom_fea),
                     torch.Tensor(nbr_fea),
                     torch.LongTensor(nbr_fea_idx),
                     [torch.LongTensor(crystal_atom_idx)])
        self.model.eval()
        pred = self.model(*input_var)
        value = self.normalizer.denorm(pred.data).item()
        return value
    
    def save(self, pos_buffer, type_buffer, energy_buffer, path, node):
        """
        save searching results
        
        Parameters
        ----------
        pos_buffer [int, 2d]: atom positions
        type_buffer [int, 2d]: atom types
        energy_buffer [float, 2d]: configuration energy
        path [int, 0d]: number of path
        node [int, 0d]: searching node
        """
        self.write_list2d(f'{self.sh_save_path}/'
                          f'pos-{self.round}-{path:03.0f}-{node}.dat', 
                          pos_buffer, style='{0:4.0f}')
        self.write_list2d(f'{self.sh_save_path}/'
                          f'type-{self.round}-{path:03.0f}-{node}.dat', 
                          type_buffer, style='{0:4.0f}')
        self.write_list2d(f'{self.sh_save_path}/'
                          f'energy-{self.round}-{path:03.0f}-{node}.dat', 
                          energy_buffer, style='{0:8.4f}')

if __name__ == '__main__':
    atom_pos = args.atom_pos
    atom_type = args.atom_type
    round = args.round
    path = args.path
    node = args.node
    grid_name = args.grid_name
    model_name = args.model_name
    
    #Searching
    worker = Search(round, grid_name)
    worker.explore(atom_pos, atom_type, model_name, path, node)