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
    def __init__(self, repeat=2, sleep_time=1):
        self.repeat = repeat
        self.wait_time = wait_time
        self.sleep_time = sleep_time
    
    def search(self, round, num_paths, init_pos, init_type, init_grid):
        """
        assign jobs by the following files
        initial_pos_XXX.dat: initial position by line
        initial_type_XXX.dat: initial type by line
        worker_job_XXX.dat: node job by line
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
        
        self.sub_job_to_workers(pos, type, job)
        system_echo('Successful assign works to workers!')
        job_finish = self.worker_monitor(pos, type, job, num_paths)
        system_echo(f'Finished searching jobs: {len(job_finish)}/{num_paths}')
        
        atom_pos, atom_type, grid_name = self.collect(job_finish)
        system_echo(f'All workers are finished!---sample number: {len(atom_pos)}')
        
        self.write_list2d(f'{self.sh_save_path}/atom_pos.dat', 
                          atom_pos, style='{0:3.0f}')
        self.write_list2d(f'{self.sh_save_path}/atom_type.dat',
                          atom_type, style='{0:3.0f}')
        self.write_list2d(f'{self.sh_save_path}/grid_name.dat',
                          grid_name, style='{0:3.0f}')
        self.remove_all_path()
    
    def generate_job(self, round, num_paths, init_pos, 
                     init_type, init_grid):
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
    
    def sampling_with_ssh(self, atom_pos, atom_type,
                          round, path, node, grid_name, model_name):
        """
        SSH to target node and call workers for sampling

        Parameters
        ----------
        atom_pos [str, 0d]: initial atom position
        atom_type [str, 0d]: initial atom type
        round [str, 0d]: searching round
        path [str, 0d]: searching path
        node [str, 0d]: searching node
        grid_name [str, 0d]: name of grid
        model_name [str, 0d]: name of predict model
        """
        ip = f'node{node}'
        options = f'--atom_pos {atom_pos} --atom_type {atom_type} ' \
                  f'--round {round} --path {path} --node {node} ' \
                  f'--grid_name {grid_name} --model_name {model_name}'
        postfix = f'{self.round}-{path.zfill(3)}-{node}.dat'
        pos, type, energy = f'pos-{postfix}', f'type-{postfix}', f'energy-{postfix}'
        local_sh_save_path = f'/local/ccop/{self.sh_save_path}'
        shell_script = f'''
                        #!/bin/bash
                        cd /local/ccop/
                        python src/core/search.py {options}
                        
                        cd {self.sh_save_path}
                        if [ -f {pos} -a -f {type} -a -f {energy} ]; then
                            tar -zcf {path}.tar.gz {pos} {type} {energy}
                            scp {path}.tar.gz {gpu_node}:{local_sh_save_path}
                            touch FINISH-{path}
                            scp FINISH-{path} {gpu_node}:{local_sh_save_path}
                            #rm FINISH-{path} {pos} {type} {energy} {path}.tar.gz
                        fi
                        '''
        self.ssh_node(shell_script, ip)
    
    def worker_monitor(self, pos, type, job, num_paths):
        """
        monitor workers whether have done
        
        Parameters
        ----------
        pos [str, 1d, np]: initial pos
        type [str, 1d, np]: initial type
        job [str, 2d, np]: jobs assigned to nodes
        num_paths [int, 0d]: number of search path
        
        Returns
        ----------
        job_finish [str, 2d, np]: finished jobs
        """
        exist_path = np.arange(num_paths)
        time_counter, repeat_counter = 0, 0
        while not self.is_done(self.sh_save_path, num_paths):
            time.sleep(self.sleep_time)
            time_counter += 1
            if time_counter > self.wait_time:
                fail_path, exist_path = self.find_fail_jobs(job)
                num_fail = len(fail_path)
                pos_fail, type_fail, job_fail = \
                    pos[fail_path], type[fail_path], job[fail_path]
                self.sub_job_to_workers(pos_fail, type_fail, job_fail)
                repeat_counter += 1
                time_counter = self.wait_time/2
                system_echo(f'Failure searching jobs: {num_fail}')
                break
            if repeat_counter == self.repeat:
                break 
        self.unzip(exist_path)
        job_finish = job[exist_path]
        return job_finish
    
    def find_fail_jobs(self, job):
        """
        find ssh failure jobs
        
        Parameters
        ----------
        job [str, 2d, np]: all jobs

        Returns
        ----------
        fail_path [int, 1d, np]: index of failure jobs
        exist_path [int, 1d, np]: index of success jobs
        """
        all_path = [int(i) for i in job[:,1]]
        shell_script = f'ls {self.sh_save_path} | grep tar.gz'
        ct = os.popen(shell_script).read().split()
        exist_path = [int(i.split('.')[0]) for i in ct]
        fail_path = np.setdiff1d(all_path, exist_path)
        return fail_path, sorted(exist_path)
    
    def sub_job_to_workers(self, pos, type, job):
        """
        sub searching jobs to nodes

        Parameters
        ----------
        pos [str, 1d, np]: initial pos
        type [str, 1d, np]: initial type
        job [str, 2d, np]: jobs assgined to nodes
        """
        for atom_pos, atom_type, assign in zip(pos, type, job):
            self.sampling_with_ssh(atom_pos, atom_type, *assign)
    
    def unzip(self, exist_path):
        """
        unzip files of finish path
        
        Parameters
        ----------
        exist_path [int, 1d, np]: successful searching path
        """
        zip_file = [f'{i}.tar.gz' for i in exist_path]
        zip_file = ' '.join(zip_file)
        shell_script = f'''
                        #!/bin/bash
                        cd {self.sh_save_path}
                        for i in {zip_file}
                        do
                            tar -zxf $i
                            #rm $i
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
        job [str, 2d, np]: assignment of node
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 2d, np]: name of grid
        """
        assign, grid = job[:,:3], job[:,3]
        sort = np.argsort(grid)
        assign, grid = assign[sort], grid[sort]
        atom_pos, atom_type, grid_name, lack = [], [], [], []
        for i, item in enumerate(assign):
            pos_file = self.get_file_name('pos', *tuple(item))
            type_file = self.get_file_name('type', *tuple(item))
            if os.path.exists(pos_file):
                pos = self.import_list2d(pos_file, int)
                type = self.import_list2d(type_file, int)
                atom_pos += pos
                atom_type += type
                num_sample = len(type)
                zeros = np.zeros((num_sample, 1))
                grid_name.append(zeros + int(grid[i]))
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
        system_echo(f'Delete same structures: {same_poscars}')
        
    
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
        flag = False
        atom_num = len(pos)
        while not flag:
            #generate actions
            new_pos = pos.copy()
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
    
    #Search
    worker = Search(round, grid_name)
    worker.explore(atom_pos, atom_type, model_name, path, node)