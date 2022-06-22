import os, sys, time
import itertools
import numpy as np
import torch
import argparse
from functools import reduce

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.dir_path import *
from core.data_transfer import Transfer
from core.space_group import PlanarSpaceGroup
from core.utils import ListRWTools, SSHTools, system_echo
from core.predict import CrystalGraphConvNet, Normalizer


parser = argparse.ArgumentParser()
parser.add_argument('--pos', type=int, nargs='+')
parser.add_argument('--type', type=int, nargs='+')
parser.add_argument('--symm', type=int, nargs='+')
parser.add_argument('--grid', type=int)
parser.add_argument('--ratio', type=float)
parser.add_argument('--sg', type=int)
parser.add_argument('--round', type=int)
parser.add_argument('--path', type=int)
parser.add_argument('--node', type=int)
args = parser.parse_args()


class ParallelWorkers(ListRWTools, SSHTools):
    #Assign sampling jobs to each node
    def __init__(self, sleep_time=1):
        self.sleep_time = sleep_time
    
    def search(self, round, init_pos, init_type, init_symm,
               init_grid, init_ratio, init_sg):
        """
        optimize structure by ML on grid
        
        Parameters
        ----------
        round [int, 0d]: searching round
        init_pos [int, 2d]: initial position
        init_type [int, 2d]: initial atom type
        init_symm [int, 2d]: initial symmetry
        init_grid [int, 1d]: initial grid name
        init_ratio [float, 1d]: initial grid ratio
        init_sg [int, 1d]: initial space group
        """
        self.round = f'{round:03.0f}'
        self.sh_save_path = f'{search_path}/{self.round}'
        self.model_save_path = f'{model_path}/{self.round}'
        self.generate_job(round, init_pos, init_type, init_symm,
                          init_grid, init_ratio, init_sg)
        pos, type, symm, job = self.read_job()
        system_echo('Get the job node!')
        for node in nodes:
            self.update_with_ssh(node)
        while not self.is_done(self.sh_save_path, len(nodes)):
            time.sleep(self.sleep_time)
        self.remove_flag(self.sh_save_path)
        system_echo('Successful update each node!')
        #sub searching job to each node
        work_node_num = self.sub_job_to_workers(pos, type, symm, job)
        system_echo('Successful assign works to workers!')
        while not self.is_done(self.sh_save_path, work_node_num):
            time.sleep(self.sleep_time)
        self.unzip()
        #collect searching path
        self.collect(job)
    
    def generate_job(self, round, init_pos, init_type, init_symm,
                     init_grid, init_ratio, init_sg):
        """
        assign jobs by the following files
        initial_pos_XXX.dat: initial position
        initial_type_XXX.dat: initial type
        initial_symm_XXX.dat: initial symmetry
        worker_job_XXX.dat: node job
        e.g. round path node grid ratio sg
             1 1 131 1 1 183
        
        Parameters
        ----------
        round [int, 0d]: searching round
        init_pos [int, 2d]: initial position
        init_type [int, 2d]: initial atom type
        init_symm [int, 2d]: initial symmetry
        init_grid [int, 1d]: initial grid name
        init_ratio [float, 1d]: initial grid ratio
        init_sg [int, 1d]: initial space group
        """
        if not os.path.exists(self.sh_save_path):
            os.mkdir(self.sh_save_path)
        worker_job = []
        path_num = len(init_grid)
        node_assign = self.assign_node(path_num)
        for i, node in enumerate(node_assign):
            job = [round, i, node, init_grid[i], init_ratio[i], init_sg[i]]
            worker_job.append(job)
        #export searching files
        pos_file = f'{self.sh_save_path}/initial_pos_{self.round}.dat'
        type_file = f'{self.sh_save_path}/initial_type_{self.round}.dat'
        symm_file = f'{self.sh_save_path}/initial_symm_{self.round}.dat'
        worker_file = f'{self.sh_save_path}/worker_job_{self.round}.dat'
        self.write_list2d(pos_file, init_pos)
        self.write_list2d(type_file, init_type)
        self.write_list2d(symm_file, init_symm)
        self.write_list2d(worker_file, worker_job)
        
    def update_with_ssh(self, node):
        """
        SSH to target node and update ccop file
        """
        ip = f'node{node}'
        shell_script = f'''
                        #!/bin/bash
                        cd /local/ccop/
                        mkdir {self.sh_save_path}
                        mkdir {self.model_save_path}
                        cd {self.model_save_path}
                        
                        scp {gpu_node}:/local/ccop/{self.model_save_path}/model_best.pth.tar .
                        
                        touch FINISH-{ip}
                        scp FINISH-{ip} {gpu_node}:/local/ccop/{self.sh_save_path}/
                        rm FINISH-{ip}
                        '''
        self.ssh_node(shell_script, ip)
    
    def sub_job_to_workers(self, pos, type, symm, job):
        """
        sub searching jobs to nodes

        Parameters
        ----------
        pos [str, 1d, np]: initial pos
        type [str, 1d, np]: initial type
        symm [str, 1d, np]: initial symmetry
        job [str, 2d, np]: jobs assgined to nodes
        """
        #poscars grouped by nodes
        pos_node, type_node, symm_node, job_node = [], [], [], []
        pos_assign, type_assign, symm_assign, job_assign = [], [], [], []
        for node in nodes:
            for i, line in enumerate(job):
                #get node
                label = line[2]
                if label == str(node):
                    pos_assign.append(pos[i])
                    type_assign.append(type[i])
                    symm_assign.append(symm[i])
                    job_assign.append(line)
            pos_node.append(pos_assign)
            type_node.append(type_assign)
            symm_node.append(symm_assign)
            job_node.append(np.transpose(job_assign))
            pos_assign, type_assign, symm_assign, job_assign = [], [], [], []
        #sub job to target node
        for i, j, k, assign in zip(pos_node, type_node, symm_node, job_node):
            self.sampling_with_ssh(i, j, k, *assign)
        work_node_num = len(pos_node)
        return work_node_num
    
    def sampling_with_ssh(self, atom_pos, atom_type, atom_symm,
                          round, path, node, grid_name, grid_ratio, space_group):
        """
        SSH to target node and call workers for sampling

        Parameters
        ----------
        atom_pos [str, 1d]: initial atom position
        atom_type [str, 1d]: initial atom type
        atom_symm [str, 1d]: initial atom symmetry 
        round [str, 1d]: searching round
        path [str, 1d]: searching path
        node [str, 1d]: searching node
        grid_name [str, 1d]: name of grid
        grid_ratio [str, 1d]L ratio of grid
        space_group [str, 1d]: space group number
        """
        ip = f'node{node[0]}'
        search_jobs = []
        for i in range(len(atom_pos)):
            option = f'--pos {atom_pos[i]} --type {atom_type[i]} --symm {atom_symm[i]} ' \
                     f'--round {round[i]} --path {path[i]} --node {node[i]} ' \
                     f'--grid {grid_name[i]} --ratio {grid_ratio[i]} --sg {space_group[i]}'
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
                            #rm $i
                        done
                        '''
        os.system(shell_script)
    
    def collect(self, job):
        """
        collect searching results of each worker
        
        Parameters
        ----------
        job [str, 2d, np]: jobs assgined to nodes
        """
        job = np.transpose(job)
        round, path, node, _, _, _ = job
        atom_pos, atom_type, atom_symm = [], [], []
        grid_name, grid_ratio, space_group, lack =  [], [], [], []
        for i in range(len(path)):
            #get name of path record files
            assign = (round[i], path[i], node[i])
            pos_file = self.get_file_name('pos', *assign)
            type_file = self.get_file_name('type', *assign)
            symm_file = self.get_file_name('symm', *assign)
            grid_file = self.get_file_name('grid', *assign)
            ratio_file = self.get_file_name('ratio', *assign)
            sg_file = self.get_file_name('sg', *assign)
            #get searching results
            if os.path.exists(pos_file):
                pos = self.import_list2d(pos_file, int)
                type = self.import_list2d(type_file, int)
                symm = self.import_list2d(symm_file, int)
                grid = self.import_list2d(grid_file, int)
                ratio = self.import_list2d(ratio_file, float)
                sg = self.import_list2d(sg_file, int)
                if len(pos) > 0:
                    atom_pos += pos
                    atom_type += type
                    atom_symm += symm
                    grid_name += grid
                    grid_ratio += ratio
                    space_group += sg
                else:
                    lack.append(pos_file)
            else:
                lack.append(pos_file)
        system_echo(f'Lack files: {lack}')
        system_echo(f'Number of samples: {len(grid_name)}')
        self.export_results(atom_pos, atom_type, atom_symm,
                            grid_name, grid_ratio, space_group)
    
    def export_results(self, atom_pos, atom_type, atom_symm,
                       grid_name, grid_ratio, space_group):
        """
        export searching results
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 2d]: name of grids
        grid_ratio [float, 2d]: ratio of grids
        space_group [int, 2d]: space group number
        """
        #export searching results
        self.write_list2d(f'{self.sh_save_path}/atom_pos.dat', 
                          atom_pos, style='{0:3.0f}')
        self.write_list2d(f'{self.sh_save_path}/atom_type.dat',
                          atom_type, style='{0:3.0f}')
        self.write_list2d(f'{self.sh_save_path}/atom_symm.dat',
                          atom_symm, style='{0:3.0f}')
        self.write_list2d(f'{self.sh_save_path}/grid_name.dat',
                          grid_name, style='{0:3.0f}')
        self.write_list2d(f'{self.sh_save_path}/grid_ratio.dat',
                          grid_ratio, style='{0:8.4f}')
        self.write_list2d(f'{self.sh_save_path}/space_group.dat',
                          space_group, style='{0:3.0f}')
        self.remove_all()
        
    def get_file_name(self, name, round, path, node):
        """
        read result of each worker
        
        Parameters
        ----------
        name [str, 0d]: file name
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
        symm [str, 1d, np]: initial symmetry string list
        job [str, 2d, np]: job assign string list
        """
        pos = self.read_dat(f'initial_pos_{self.round}.dat')
        type = self.read_dat(f'initial_type_{self.round}.dat')
        symm = self.read_dat(f'initial_symm_{self.round}.dat')
        job = self.read_dat(f'worker_job_{self.round}.dat', split=True)
        return pos, type, symm, job
    
    def read_dat(self, dat, split=False):
        """
        read initilize file as string list
        
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
    
    def remove_all(self):
        """
        remove file of each path
        """
        shell_script = f'''
                        #!/bin/bash
                        cd {self.sh_save_path}
                        rm FINISH*
                        rm pos-* type-* symm-*
                        rm grid-* ratio-* sg-* energy-*
                        '''
        os.system(shell_script)
    

class ActionSpace:
    #action space of optimizing position of atoms
    def action_filter(self, idx, pos, symm, symm_site,
                      ratio, grid_idx, grid_dis, move=True):
        """
        distance between atoms should bigger than min_bond
        
        Parameters
        ----------
        idx [int, 0d]: index of select atom
        pos [int, 1d]: position of atoms
        symm [int, 1d]: symmetry of atoms\\
        symm_site [dict, int:list]: site position grouped by symmetry
        ratio [float, 0d]: grid ratio
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        move [bool, 0d]: move atom or add atom
        
        Returns
        ----------
        allow [int, 1d]: allowable actions
        """
        if move:
            obstacle = np.delete(pos, idx, axis=0)
        else:
            obstacle = pos
        #get forbidden sites
        symm_slt = symm[idx]
        sites = symm_site[symm_slt]
        if len(obstacle) > 0:
            forbid = self.get_forbid(obstacle, ratio, grid_idx, grid_dis)
            occupy = self.get_occupy(symm_slt, pos, symm)
        else:
            forbid, occupy = [], []
        #get allowable sites
        forbid = np.intersect1d(sites, forbid)
        vacancy = np.setdiff1d(sites, occupy)
        allow = np.setdiff1d(vacancy, forbid).tolist()
        return allow
    
    def get_forbid(self, point, ratio, grid_idx, grid_dis):
        """
        get position of forbid area
        
        Parameters
        ----------
        point [int, 1d]: occupied points
        ratio [float, 0d]: grid ratio
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid

        Returns
        ----------
        forbid [int, 1d]: forbid area
        """
        nbr_idx = grid_idx[point]
        nbr_dis = grid_dis[point]
        nbr_dis *= ratio
        forbid_idx = []  
        for item in nbr_dis:
            for i, dis in enumerate(item):
                if dis > min_bond:
                    forbid_idx.append(i) 
                    break
        actions = [nbr_idx[i][:j] for i, j in enumerate(forbid_idx)]
        forbid = reduce(np.union1d, actions)
        return forbid
    
    def get_occupy(self, symm_slt, atom_pos, atom_symm):
        """
        get occupied position by symmetry
        
        Parameters
        ----------
        symm_slt [int, 0d]: select symmetry
        atom_pos [int, 1d]: position of atoms
        atom_symm [int, 1d]: symmetry of atoms

        Returns
        ----------
        occupy [int, 1d]: position of occupied sites
        """
        occupy = []
        for i, pos in enumerate(atom_pos):
            if symm_slt == atom_symm[i]:
                occupy.append(pos)
        return occupy
    
    def exchange_action(self, idx, type, symm):
        """
        actions of exchanging atoms
        
        Parameters
        ----------
        idx [int, 0d]: index of select atom
        type [int, 1d]: type of atoms
        symm [int, 1d]: symmetry of atoms
        
        Returns
        ----------
        allow [int, 2d]: effective exchange actions
        """
        #get index of different elements of same symmetry
        ele_idx = self.get_ele_idx(idx, type, symm)
        #get allow exchange actions
        allow = []
        ele_num = len(ele_idx)
        buffer = np.arange(ele_num)
        if ele_num > 1:
            for i, j in itertools.combinations(buffer, 2):
                action = itertools.product(ele_idx[i], ele_idx[j])
                allow += [i for i in action]
        return allow
    
    def get_ele_idx(self, idx, atom_type, atom_symm):
        """
        get index of different elements of same symmetry
        
        Parameters
        ----------
        idx [int, 0d]: index of select atom in atom_pos
        atom_type [int, 1d]: type of atoms
        atom_symm [int, 1d]: symmetry of atoms

        Returns
        ----------
        ele_idx [int, 2d]: index of same symmetry elements
        """
        #get index with same symmetry
        symm_idx, type = [], []
        symm_slt = atom_symm[idx]
        for i, symm in enumerate(atom_symm):
            if symm == symm_slt:
                symm_idx.append(i)
                type.append(atom_type[i])    
        #get index of different atoms
        ele_idx = []
        for ele in np.unique(type):
            idx = [symm_idx[i] for i, j in enumerate(type) if j==ele]
            ele_idx.append(idx)
        return ele_idx
    
    def scale_action(self, pos, ratio, scale_bound, grid_idx, grid_dis):
        """
        actions of scaling lattice
        
        Parameters
        ----------
        pos [int, 1d]: position of atoms
        ratio [float, 0d]: grid ratio
        scale_bound [float, 1d]: boundary of scaling lattice
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid

        Returns
        ----------
        allow [float, 1d]: effective scale actions
        """
        #get index and distance of points
        pos = np.array(pos)
        point_idx = grid_idx[pos]
        point_dis = grid_dis[pos]
        point_dis *= ratio
        #get minimal distance
        allow, nbr_dis = [], []
        for i, point in enumerate(point_idx):
            atom_idx = np.where(point==pos[:, None])[-1]
            order = np.argsort(atom_idx)[0]
            nearest_idx = atom_idx[order]
            nbr_dis.append(point_dis[i, nearest_idx])
        min_dis = min(nbr_dis)
        #boundary of scaling
        low, up = scale_bound
        hold = 1.001*min_bond/min_dis
        low = max(low, ratio*hold)
        allow = [i for i in np.arange(low, up, .02)]
        return allow
    
    def get_scale_bound(self, latt_vec):
        """
        get boundary of scaling lattice
        
        Parameters
        ----------
        latt_vec [float, 2d, np]: lattice vector

        Returns
        ----------
        low [float, 0d]: lower boundary
        up [float, 0d]: upper boundary
        """
        #get short and long axis of lattice
        norms = np.linalg.norm(latt_vec, axis=1)
        if add_vacuum:
            norms = norms[:2]
        short, long = min(norms), max(norms)
        #get basic scale boundary
        low = len_lower/short
        up = len_upper/long
        return low, up


class GeoCheck(ActionSpace):
    #check geometry property of structure
    def check_near(self, pos, ratio, grid_idx, grid_dis):
        """
        check near distance of atoms
        
        Parameters
        ----------
        pos [int, 1d]: position of atoms
        ratio [float, 0d]: grid ratio
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid

        Returns
        ----------
        flag [bool, 0d]: whether satisfy bond constrain
        """
        flag = True
        forbid = self.get_forbid(pos, ratio, grid_idx, grid_dis)
        for i in pos:
            if i in forbid:
                flag = False
                break
        return flag
    

class Search(GeoCheck, PlanarSpaceGroup, Transfer):
    #Searching on PES by machine-learning potential
    def __init__(self, round):
        Transfer.__init__(self)
        self.device = torch.device('cpu')
        self.normalizer = Normalizer(torch.tensor([]))
        self.round = f'{round:03.0f}'
        self.model_save_path = f'{model_path}/{self.round}'
        self.sh_save_path = f'{search_path}/{self.round}'
    
    def explore(self, pos, type, symm, grid, ratio, sg, path, node):
        """
        simulated annealing

        Parameters
        ----------
        pos [int, 1d]: position of atoms
        type [int, 1d]: type of atoms
        symm [int, 1d]: symmetry of atoms
        grid [int, 0d]: grid name
        ratio [float, 0d]: grid ratio
        sg [int, 0d]: space group number
        path [int, 0d]: path number
        node [int, 0d]: node number
        """
        #load prediction model
        self.load_model()
        #import embeddings and neighbor in min grid
        self.elem_embed = self.import_data('elem', grid, sg)
        latt_vec = self.import_data('latt', grid, sg)
        grid_idx, grid_dis = self.import_data('grid', grid, sg)
        #get boundary of scaling lattice
        scale_bound = self.get_scale_bound(latt_vec)
        #group sites by symmetry
        mapping = self.import_data('mapping', grid, sg)
        symm_site = self.group_symm_sites(mapping)
        #initialize buffer
        pos_1, ratio_1 = pos, ratio
        energy, atom_pos, grid_ratio = [], [], []
        e_1 = self.predict(pos_1, type, ratio_1, grid_idx, grid_dis)
        energy.append(e_1)
        atom_pos.append(pos_1)
        grid_ratio.append(ratio_1)
        #
        for _ in range(3):
            #simulated annealing
            self.T = T
            for _ in range(sa_steps):
                pos_2 = self.atom_step(pos_1, type, symm, symm_site,
                                       ratio_1, grid_idx, grid_dis)
                e_2 = self.predict(pos_2, type, ratio_1, grid_idx, grid_dis)
                if self.metropolis(e_1, e_2, self.T):
                    e_1, pos_1 = e_2, pos_2
                    energy.append(e_1)
                    atom_pos.append(pos_1)
                    grid_ratio.append(ratio_1)
                self.T *= decay
            #
            ratio_2 = self.lattice_step(pos_1, type, ratio_1, scale_bound, grid_idx, grid_dis)
            ratio_1 = ratio_2
        self.save(atom_pos, type, symm, grid, grid_ratio, sg, energy, path, node)
    
    def atom_step(self, pos, type, symm, symm_site, ratio, grid_idx, grid_dis):
        """
        move atoms under the geometry constrain
        
        Parameters
        ----------
        pos [int, 1d]: inital position of atoms
        type [int, 1d]: initial type of atoms
        symm [int, 1d]: initial symm of atoms
        symm_site [dict, int:list]: site position grouped by symmetry
        ratio [float, 0d]: grid ratio
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        
        Returns
        ----------
        new_pos [int, 1d]: position of atom after 1 SA step
        """
        new_pos = pos.copy()
        atom_num = len(new_pos)
        for _ in range(num_jump):
            #generate actions
            idx = np.random.randint(0, atom_num)
            action_mv = self.action_filter(idx, new_pos, symm, symm_site,
                                           ratio, grid_idx, grid_dis)
            action_ex = self.exchange_action(idx, type, symm)
            mask_ex = [-1 for _ in action_ex]
            actions = action_mv + mask_ex
            if len(actions) == 0:
                actions += [-2]
            #keep or exchange atoms or move atom
            actions = np.array(actions, dtype=int)
            point = np.random.choice(actions)
            if point >= 0:
                check_pos = new_pos.copy()
                check_pos[idx] = point
                #check distance of new generate symmetry atoms
                if self.check_near(check_pos, ratio, grid_idx, grid_dis):
                    new_pos = check_pos
            if point == -1:
                action_num = len(action_ex)
                idx = np.random.randint(0, action_num)
                idx_1, idx_2 = action_ex[idx]
                new_pos[idx_1], new_pos[idx_2] = \
                    new_pos[idx_2], new_pos[idx_1]
            if point == -2:
                new_pos = new_pos
        return new_pos
    
    def lattice_step(self, pos, type, ratio, scale_bound, grid_idx, grid_dis):
        """
        move atoms under the geometry constrain
        
        Parameters
        ----------
        pos [int, 1d]: inital position of atoms
        type [int, 1d]: 
        ratio [float, 0d]: grid ratio
        scale_bound [float, 1d]: boundary of scaling lattice
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        
        Returns
        ----------
        new_pos [int, 1d]: position of atom after 1 SA step
        """
        ratios = self.scale_action(pos, ratio, scale_bound, grid_idx, grid_dis)
        print(ratios)
        if len(ratios) > 0: 
            energys = []
            for i in ratios:
                energy = self.predict(pos, type, i, grid_idx, grid_dis)
                energys.append(energy)
            #
            idx = np.argmin(energys)
            new_ratio = ratios[idx]
        else:
            new_ratio = ratio
        return new_ratio
    
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
            if delta == 0:
                return False
            else:
                return True
        else:
            return False
    
    def load_model(self):
        """
        load predict model
        """
        self.model = CrystalGraphConvNet(orig_atom_fea_len, 
                                         nbr_bond_fea_len)
        paras = torch.load(f'{self.model_save_path}/model_best.pth.tar', 
                           map_location=self.device)
        self.model.load_state_dict(paras['state_dict'])
        self.normalizer.load_state_dict(paras['normalizer'])
    
    def predict(self, pos, type, ratio, grid_idx, grid_dis):
        """
        predict energy of one structure

        Parameters
        ----------
        pos [int, 1d]: position of atoms
        type [int ,1d]: type of atoms
        ratio [float, 0d]: grid ratio
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        
        Returns
        ----------
        energy [float, 0d]: predict energy
        """
        #transfer into input of ppm
        atom_fea, nbr_fea, nbr_fea_idx = \
            self.get_ppm_input(pos, type, self.elem_embed,
                               ratio, grid_idx, grid_dis)
        crystal_atom_idx = np.arange(len(atom_fea))
        input_var = (torch.Tensor(atom_fea),
                     torch.Tensor(nbr_fea),
                     torch.LongTensor(nbr_fea_idx),
                     [torch.LongTensor(crystal_atom_idx)])
        #predict energy
        self.model.eval()
        pred = self.model(*input_var)
        energy = self.normalizer.denorm(pred.data).item()
        return energy
    
    def save(self, atom_pos, type, symm,
             grid, grid_ratio, sg, energy, path, node):
        """
        save searching results
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        type [int, 1d]: type of atoms
        symm [int, 1d]: symmetry of atoms
        grid [int, 0d]: grid name
        grid_ratio [float, 1d]: ratio of grids
        sg [int, 0d]: space group number
        energy [float, 1d]: predict energys
        path [int, 0d]: path number
        node [int, 0d]: node number
        """
        #generate 
        num = len(atom_pos)
        atom_type = [type for _ in range(num)] 
        atom_symm = [symm for _ in range(num)] 
        grid_name = [grid for _ in range(num)] 
        space_group = [sg for _ in range(num)] 
        #export results
        self.write_list2d(f'{self.sh_save_path}/'
                          f'pos-{self.round}-{path:03.0f}-{node}.dat', 
                          atom_pos, style='{0:4.0f}')
        self.write_list2d(f'{self.sh_save_path}/'
                          f'type-{self.round}-{path:03.0f}-{node}.dat', 
                          atom_type, style='{0:4.0f}')
        self.write_list2d(f'{self.sh_save_path}/'
                          f'symm-{self.round}-{path:03.0f}-{node}.dat', 
                          atom_symm, style='{0:4.0f}')
        self.write_list2d(f'{self.sh_save_path}/'
                          f'grid-{self.round}-{path:03.0f}-{node}.dat', 
                          np.transpose([grid_name]), style='{0:4.0f}')
        self.write_list2d(f'{self.sh_save_path}/'
                          f'ratio-{self.round}-{path:03.0f}-{node}.dat', 
                          np.transpose([grid_ratio]), style='{0:8.4f}')
        self.write_list2d(f'{self.sh_save_path}/'
                          f'sg-{self.round}-{path:03.0f}-{node}.dat', 
                          np.transpose([space_group]), style='{0:4.0f}')
        self.write_list2d(f'{self.sh_save_path}/'
                          f'energy-{self.round}-{path:03.0f}-{node}.dat', 
                          np.transpose([energy]), style='{0:8.4f}')
        

if __name__ == '__main__':
    #structure
    pos = [11,2,23,14,45,39]
    type = [5,5,6,6,6,6]
    symm = [1,1,1,1,2,2]
    grid = 33
    ratio = 1.0
    sg = 3
    #path label
    round = 1
    path = 0
    node = 131
    #Searching
    worker = Search(round)
    worker.explore(pos, type, symm, grid, ratio, sg, path, node)