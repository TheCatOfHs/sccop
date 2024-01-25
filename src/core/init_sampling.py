import os, sys
import time
import numpy as np

from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from core.log_print import *
from core.utils import *
from core.GNN_tool import GNNSlice
from core.del_duplicates import DeleteDuplicates
from core.grid_generate import GridGenerate
from core.MCTS import ParallelSampling


class UpdateNodes(SSHTools):
    #make child nodes consistent with host node
    def __init__(self, wait_time=0.1):
        SSHTools.__init__(self)
        self.wait_time = wait_time
        if not os.path.exists(Grid_Path):
            self.update()
    
    def update(self):
        """
        update child nodes
        """
        if Dimension == 3:
            if Energy_Method == 'VASP':
                self.add_pressure()
        for node in self.work_nodes:
            self.update_nodes(node)
        while not self.is_done('', self.work_nodes_num):
            time.sleep(self.wait_time)
        self.remove_flag('.')
        system_echo('Child nodes are consistent with host node')
        os.system(f'''
                  #!/bin/bash --login
                  cd {SCCOP_Path}/data
                  date +%s > time.dat
                  ''')
    
    def update_nodes(self, node):
        """
        SSH to target node and copy necessary files
        """
        shell_script = f'''
                        #!/bin/bash --login
                        cd {SCCOP_Path}
                        
                        mkdir {Save_Path}
                        mkdir {POSCAR_Path}
                        mkdir {Model_Path}
                        mkdir {Search_Path}
                        mkdir {Grid_Path}
                        mkdir {Json_Path}
                        mkdir {Buffer_Path}
                        mkdir {Recyc_Store_Path}
                        if [ {Energy_Method} = 'VASP' ]; then
                            mkdir {VASP_Out_Path}
                        fi
                        if [ {Energy_Method} = 'LAMMPS' ]; then
                            mkdir {LAMMPS_Out_Path}
                        fi
                        
                        cd libs/
                        tar -zxf POTCAR.tar.gz
                        rm POTCAR.tar.gz
                        
                        touch FINISH-{node}
                        scp FINISH-{node} {Host_Node}:{SCCOP_Path}/
                        rm FINISH-{node}
                        '''
        self.ssh_node(shell_script, node)
    
    def add_pressure(self):
        """
        add pressure to INCARs
        """
        shell_script = f'''
                        #!/bin/bash --login
                        cd {SCCOP_Path}/{VASP_Files_Path}
                        cd SinglePointEnergy/3d
                        echo >> INCAR
                        echo PSTRESS={Pressure} >> INCAR
                        cd ../../
                        cd Optimization/3d
                        for i in `ls | grep INCAR`
                        do
                            echo >> $i
                            echo PSTRESS={Pressure} >> $i
                        done
                        '''
        os.system(shell_script)
    
    def copy_data_to_nodes(self, node):
        """
        copy atom types file to each node
        """
        shell_script = f'''
                        #!/bin/bash --login
                        cd {SCCOP_Path}
                        scp -r {Host_Node}:{SCCOP_Path}/data .
                        
                        touch FINISH-{node}
                        scp FINISH-{node} {Host_Node}:{SCCOP_Path}/
                        rm FINISH-{node}
                        '''
        self.ssh_node(shell_script, node)
    
    def copy_poscars_to_nodes(self, recyc, node):
        """
        copy initial poscars to each node
        
        Parameters
        ----------
        recyc [int, 0d]: recycle of sccop
        node [int, 0d]: job node
        """
        local_Save_Path = f'{SCCOP_Path}/{POSCAR_Path}'
        local_POSCAR_Path = f'{SCCOP_Path}/{Init_Strus_Path}_{recyc:02.0f}'
        shell_script = f'''
                        #!/bin/bash --login
                        cd {local_Save_Path}
                        scp -r {Host_Node}:{local_POSCAR_Path} .
                        
                        touch FINISH-{node}
                        scp FINISH-{node} {Host_Node}:{SCCOP_Path}/
                        rm FINISH-{node}
                        '''
        self.ssh_node(shell_script, node)
    
    def copy_seeds_to_nodes(self, node):
        """
        copy initial seeds to each node
        
        Parameters
        ----------
        node [int, 0d]: job node
        """
        local_Seed_Path = f'{SCCOP_Path}/{Seed_Path}'
        shell_script = f'''
                        #!/bin/bash --login
                        cd {SCCOP_Path}/data
                        scp -r {Host_Node}:{local_Seed_Path} .
                        
                        touch FINISH-{node}
                        scp FINISH-{node} {Host_Node}:{SCCOP_Path}/
                        rm FINISH-{node}
                        '''
        self.ssh_node(shell_script, node)
    

class AtomAssignPlan(DeleteDuplicates):
    #atom assignment plan
    def __init__(self):
        DeleteDuplicates.__init__(self)
    
    def control_atom_number(self, atom_type):
        """
        control number of atoms
        
        Parameters
        ----------
        atom_type [str, 1d]: minimum number of atoms
        
        Returns
        ----------
        max_atom_type [str, 1d]: maximum number of atoms
        """
        times = 1
        atom_types = []
        type = atom_type
        min_atom, max_atom = Num_Atom
        while len(type) <= max_atom:
            type = [i for i in atom_type for _ in range(times)]
            atom_types.append(type)
            times += 1
        atom_types = [i for i in atom_types if min_atom <= len(i) <= max_atom]
        #export suitable atom types
        self.write_list2d(f'{Grid_Path}/atom_types.dat', atom_types)
    
    def control_atom_number_cluster(self, atom_type, num=10, leaf_num=10, max_num=100):
        """
        control number of atoms for cluster search
        
        Parameters
        ----------
        atom_type [int, 1d, np]: type of atoms
        num [int, 0d]: number of atom assignments
        leaf_num [int, 0d]: maximum leaf number
        max_num [int, 0d]: maximum sampling number
        """
        property_dict = self.import_data('property')
        clusters = [int(i) for i in property_dict.keys()]
        cluster_type_num = self.get_cluster_type_num(clusters, atom_type, property_dict)
        type_order, allow_type_num = self.get_allow_type_num(atom_type, Num_Atom)
        #Monte Carlo Tree Search
        assign_store, assign_tmp = [], [[]]
        while True:
            store = []
            for assign in assign_tmp:
                #get allowable clusters
                allow = self.allow_cluster(assign, clusters, allow_type_num, cluster_type_num)
                if len(allow) == 0:
                    pass
                else:
                    #random sampling
                    np.random.seed()
                    np.random.shuffle(allow)
                    counter = 0
                    for atom in allow:
                        check_assign = assign.copy()
                        check_assign.append(atom)
                        check_assign = sorted(check_assign)
                        if self.check_max_atom(check_assign, allow_type_num, cluster_type_num):
                            if self.check_same_assignment(check_assign, store):
                                store.append(check_assign)
                                counter += 1
                        if counter > leaf_num:
                            break
                if len(assign_store) > max_num:
                    break
            assign_tmp = store
            if len(assign_tmp) > 0:
                for assign in assign_tmp:
                    type_num = cluster_type_num[assign]
                    cluster_num = np.sum(type_num)
                    if cluster_num >= Cluster_Num_Ratio*np.min(Num_Atom):
                        assign_store.append(assign)
            else:
                break
        #calculate ratio of clusters
        type_num_store = self.get_type_num(assign_store, cluster_type_num)
        if len(assign_store) > 0:
            cluster_ratio = self.get_cluster_ratio(assign_store, clusters)
            ratio = self.get_ratio(cluster_ratio)
            order = np.argsort(ratio)
            assign_store = np.array(assign_store, dtype=object)[order][:num].tolist()
            type_num_store = np.array(type_num_store, dtype=object)[order][:num].tolist()
        #get allowable assignments
        allow_types = self.fill_assignment(assign_store, type_num_store, type_order, allow_type_num)
        #export suitable atom types
        self.write_list2d(f'{Grid_Path}/atom_types.dat', allow_types)
    
    def get_cluster_type_num(self, clusters, atom_type, property_dict):
        """
        get number of different atoms in cluster
        
        Parameters
        ----------
        clusters [int, 1d]: type of clusters
        atom_type [int, 1d]: type of target composition
        property_dict [dict, 1d]: property dictionary for new atom

        Returns
        ----------
        cluster_type_num [int, 2d, np]: number of different atoms in each cluster
        """
        cluster_type_num = []
        for cluster in clusters:
            type_num = property_dict[str(cluster)]['type_num']
            type_num = np.array(type_num)
            cluster_type_tmp = [i[0] for i in type_num]
            lack_type= np.setdiff1d(atom_type, cluster_type_tmp)
            #fill atom types
            if len(lack_type) > 0:
                lack_num = len(lack_type)
                zeros = np.zeros(lack_num)
                lack_type_num = np.stack((lack_type, zeros), axis=1)
                type_num = np.concatenate((type_num, lack_type_num))
            #sorted by atomic number
            order = np.argsort(type_num[:, 0])
            type_num = type_num[order]
            cluster_type_num.append(type_num[:, 1])
        return np.array(cluster_type_num, dtype=int)
    
    def get_allow_type_num(self, atom_type, interval):
        """
        get allowable number of different atoms
        
        Parameters
        ----------
        atom_type [int, 1d]: type of atoms
        interval [int, 1d]: range of atom number

        Returns
        ----------
        type_order [int, 1d]: type of atoms
        allow_type_num [int, 2d]: allowable number of different atoms
        """
        type_num = count_atoms(atom_type)
        order = np.argsort(type_num[:, 0])
        type_order, num_order = np.transpose(type_num[order])
        #get suitable ratio
        atom_num = len(atom_type)
        min_atom, max_atom = interval
        allow_type_num = []
        for i in range(min_atom, max_atom+1):
            if i%atom_num == 0:
                ratio = i//atom_num
                allow_type_num.append(ratio*num_order)
        return type_order.tolist(), allow_type_num
    
    def get_type_num(self, assign_store, cluster_type_num):
        """
        get number of different atoms for assignemnt

        Parameters
        ----------
        assign_store [int, 2d]: atom assignment store
        cluster_type_num [int, 2d, np]: number of different atoms in each cluster

        Returns
        ----------
        allow_type_num [int, 2d]: allowable number of different atoms
        """
        assign_type_num = []
        for assign in assign_store:
            type_num = np.sum(cluster_type_num[assign], axis=0)
            assign_type_num.append(type_num)
        return assign_type_num
    
    def allow_cluster(self, assign, clusters, allow_type_num, cluster_type_num):
        """
        get allowable clusters for specific assignment
        
        Parameters
        ----------
        assign [int, 1d]: assignment of atoms
        clusters [int, 1d]: type of clusters
        allow_type_num [int, 2d]: allowable number of different atoms
        cluster_type_num [int, 2d, np]: number of different atoms in each cluster

        Returns
        ----------
        allow [int, 1d]: allowable clusters
        """
        allow = []
        for cluster in clusters:
            tmp_type = assign.copy()
            tmp_type.append(cluster)
            if self.check_max_atom(tmp_type, allow_type_num, cluster_type_num):
                allow.append(cluster)
        return allow
    
    def check_max_atom(self, assign, allow_type_num, cluster_type_num):
        """
        check whether reach maximum atoms

        Parameters
        ----------
        assign [int, 1d]: assignment of atoms
        allow_type_num [int, 2d]: allowable number of different atoms
        cluster_type_num [int, 2d, np]: number of different atoms in each cluster

        Returns
        ----------
        flag [bool, 0d]: whether reach maximum atoms
        """
        flag = True
        max_type_num = allow_type_num[-1]
        assign_type_num = np.sum(cluster_type_num[assign], axis=0)
        for i, j in zip(assign_type_num, max_type_num):
            if i > j:
                flag = False
                break
        return flag
    
    def check_same_assignment(self, assign, assign_store):
        """
        check whether same assignment
        
        Parameters
        ----------
        assign [int, 1d]: assignment of atom
        assign_store [int, 2d]: atom assignment store

        Returns
        ----------
        flag [bool, 0d]: whether same assignment
        """
        flag = True
        num = len(assign)
        for i in assign_store:
            if len(i) == num:
                tmp = np.subtract(i, assign)
                tmp = np.sum(tmp)
                if tmp == 0:
                    flag = False
                    break
        return flag
    
    def fill_assignment(self, assign_store, type_num_store, type_order, allow_type_num):
        """
        fill assignment of atoms
        
        Parameters
        ----------
        assign_store [int, 2d]: atom assignment store
        type_num_store [int, 2d]: number of different atoms store
        type_order [int, 1d]: type of atoms sorted in order
        allow_type_num [int, 2d]: allowable number of different atoms

        Returns
        ----------
        allow_type [int, 2d]: allowable atom types
        """
        allow_type = []
        if len(assign_store) > 0:
            #exist clusters
            for assign, type_num in zip(assign_store, type_num_store):
                for i in allow_type_num:
                    lack_type_num = np.subtract(i, type_num)
                    if np.any(lack_type_num<0):
                        pass
                    else:
                        lack_type = []
                        for num, type in zip(lack_type_num, type_order):
                            for _ in range(num):
                                lack_type.append(type)
                        allow_type.append(assign+lack_type)
        else:
            #no clusters
            for type_num in allow_type_num:
                tmp_type = []
                for num, type in zip(type_num, type_order):
                    for _ in range(num):
                        tmp_type.append(type)
                allow_type.append(tmp_type)
        return allow_type
    
    def get_cluster_ratio(self, assign_store, clusters):
        """
        get ratio of clusters in assignments
        
        Parameters
        ----------
        assign_store [int, 2d]: atom assignment store
        clusters [int, 1d]: type of clusters

        Returns
        ----------
        cluster_ratio [float, 2d]: ratio of clusters in each assignment
        """
        cluster_ratio = []
        for assign in assign_store:
            tmp = Counter(assign)
            type_num = [[i, j] for i, j in tmp.items()]
            type_num = np.array(type_num, dtype=int)
            type_tmp = [i[0] for i in type_num]
            lack_type = np.setdiff1d(clusters, type_tmp)
            #fill atom types
            if len(lack_type) > 0:
                lack_num = len(lack_type)
                zeros = np.zeros(lack_num)
                lack_type_num = np.stack((lack_type, zeros), axis=1)
                type_num = np.concatenate((type_num, lack_type_num))
            #sorted by atomic number
            order = np.argsort(type_num[:, 0])
            type_num = np.array(type_num, dtype=int)[order]
            #get ratio of clusters
            num = type_num[:, 1]
            total = np.sum(num)
            ratio = np.array(num)/total
            cluster_ratio.append(ratio)
        return cluster_ratio
    
    def get_ratio(self, cluster_ratio):
        """
        get ratio score of assignments

        Parameters
        ----------
        cluster_ratio [float, 2d]: ratio of clusters in each assignment

        Returns
        ----------
        ratio [float, 1d]: ratio score
        """
        diff = np.subtract(cluster_ratio, Cluster_Weight)
        ratio = np.sum(np.abs(diff), axis=1)
        return ratio

    
class InitSampling(AtomAssignPlan, UpdateNodes, GridGenerate, ParallelSampling, GNNSlice):
    #generate initial structures
    def __init__(self):
        AtomAssignPlan.__init__(self)
        UpdateNodes.__init__(self)
        ParallelSampling.__init__(self)
        if General_Search:
            self.init_latt_num = self.work_nodes_num*Latt_per_Node
        elif Cluster_Search:
            self.init_latt_num = self.work_nodes_num*Clus_per_Node
        elif Template_Search:
            self.init_latt_num = self.work_nodes_num*Temp_per_Node
        self.init_poscar_num = self.work_nodes_num*Init_Strus_per_Node
        
    def generate(self, recyc):
        """
        generate initial samples
        
        Parameters
        ----------
        recyc [int, 0d]: times of recycle
        
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
        energys [float, 1d]: prediction energys
        """
        #generate initial lattice
        self.initial_poscars(recyc)
        #generate structures randomly
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, \
            space_group, angles, thicks, energys, crys_vec = self.random_sampling(recyc)
        #clustering samples
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec = \
            self.gnn_slice_global(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, 
                                  space_group, angles, thicks, energys, crys_vec, sample_limit=self.init_poscar_num)
        #add random samples
        system_echo(f'Sampling number: {len(atom_pos)}')    
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys
    
    def initial_poscars(self, recyc):
        """
        generate initial poscars randomly
        
        Parameters
        ----------
        recyc [int, 0d]: times of recycle
        """
        if recyc == 0:
            self.generate_atoms()
        #make directory of initial poscars
        init_path = f'{POSCAR_Path}/initial_strus_{recyc:02.0f}'
        os.mkdir(init_path)
        #succeed from last iteration
        if recyc > 0 and Use_Succeed:
            optim_path = f'{POSCAR_Path}/optim_strus_{recyc-1:02.0f}'
            self.succeed_latt(init_path, optim_path)
            self.succeed_grid(init_path, optim_path)
        #generate lattice according to crystal system
        if General_Search or Cluster_Search:
            crys_system = self.crystal_system_sampling(self.init_latt_num)
            for i in range(self.init_latt_num):
                stru = Structure([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1], [[0, 0, 0]])
                stru.to(filename=f'{init_path}/POSCAR-RCSD-{crys_system[i]}-{i:03.0f}', fmt='poscar')
        #generated from template
        if Template_Search:
            files = os.listdir(Seed_Path)
            template = [i for i in files if i.startswith('POSCAR')][0]
            store = []
            for _ in range(self.init_latt_num):
                store.append(Structure.from_file(f'{Seed_Path}/{template}'))
            for i, stru in enumerate(store):
                stru.to(filename=f'{init_path}/POSCAR-Template-0-{i:03.0f}', fmt='poscar')
            if Disturb_Seeds:
                seeds = [i for i in files if i.startswith('POSCAR-Seed')]
                seeds_pos = get_seeds_pos(seeds, template)
                self.write_list2d(f'{Seed_Path}/seeds_pos.dat', seeds_pos)
        #shuffle the lattice poscars
        poscars = [i for i in os.listdir(init_path) if len(i.split('-'))==4]
        order = np.arange(len(poscars))
        np.random.shuffle(order)
        for i, poscar in zip(order, poscars):
            ct = poscar.split('-')
            job, cs = ct[1], ct[2]
            os.rename(f'{init_path}/{poscar}', f'{init_path}/POSCAR-{job}-{cs}-{i:04.0f}')
        system_echo(f'Generate initial lattice')
    
    def generate_atoms(self):
        """
        generate atom type and bond file
        """
        if Cluster_Search:
            self.build_new_atom()
        self.generate_atom_type()
        self.export_bond_file()
        #copy data directory to job nodes
        if self.child_nodes_num > 0:
            for node in self.child_nodes:
                self.copy_data_to_nodes(node)
            while not self.is_done('', self.child_nodes_num):
                time.sleep(self.wait_time)
            self.remove_flag('.')
    
    def generate_atom_type(self):
        """
        generate allowable combination of atom types
        """
        #transfer composition to atom type list
        atom_type = convert_composition_into_atom_type()
        if Cluster_Search:
            self.control_atom_number_cluster(atom_type)
        else:
            self.control_atom_number(atom_type)
    
    def random_sampling(self, recyc):
        """
        build grid of structure

        Parameters
        ----------
        recyc [int, 0d]: times of recycle

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
        energys [float, 1d]: gnn energys
        crys_vec [float, 2d]: crystal vectors
        """
        #copy initial poscars to job nodes
        if self.child_nodes_num > 0:
            for node in self.child_nodes:
                self.copy_poscars_to_nodes(recyc, node)
            while not self.is_done('', self.child_nodes_num):
                time.sleep(self.wait_time)
            self.remove_flag('.')
        #copy seeds to job nodes
        if Template_Search:
            if self.child_nodes_num > 0:
                for node in self.child_nodes:
                    self.copy_seeds_to_nodes(node)
                while not self.is_done('', self.child_nodes_num):
                    time.sleep(self.wait_time)
                self.remove_flag('.')
        #sampling randomly
        start = self.get_latt_num()
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec = \
            self.sampling_on_grid(recyc, start)
        system_echo(f'Initial sampling: {len(grid_name)}')
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec
    
    def get_latt_num(self):
        """
        count number of grids
        
        Returns
        ----------
        num [int, 0d]: number of grids
        """
        file = os.listdir(Grid_Path)
        grids = [i for i in file if i.endswith('latt_vec.bin')]
        if len(grids) == 0:
            num = 0
        else:
            grid = sorted(grids)[-1]
            num = int(grid.split('_')[0]) + 1
        return num
    
    def build_new_atom(self):
        """
        build new atom
        """
        shell_script = f'''
                        #!/bin/bash --login
                        cd {SCCOP_Path}
                        python src/core/cluster.py
                        '''
        os.system(shell_script)
    
    
if __name__ == '__main__':
    pass