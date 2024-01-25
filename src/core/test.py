import os, sys, time
import re
import json
import numpy as np
from collections import Counter

sys.path.append(f'{os.getcwd()}/src')
from core.input import *
from core.path import *
from core.GNN_tool import GNNSlice
from core.utils import *
from core.neighbors import Neighbors
from core.del_duplicates import DeleteDuplicates
from core.sub_lammps import LammpsDPT

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.util.coord import pbc_diff


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
            score = self.get_ratio_score(cluster_ratio)
            order = np.argsort(score)
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
    
    def get_ratio_score(self, cluster_ratio):
        """
        get ratio score of assignments

        Parameters
        ----------
        cluster_ratio [float, 2d]: ratio of clusters in each assignment

        Returns
        ----------
        ratio_score [float, 1d]: ratio score
        """
        diff = np.subtract(cluster_ratio, Cluster_Weight)
        ratio_score = np.sum(np.abs(diff), axis=1)
        return ratio_score


if __name__ == '__main__':
    idx = []
    for base, uniq_idx in zip([1], [[1,2,3]]):
        idx += np.add(base, uniq_idx).tolist()
    print(idx)
    '''
    stru_1 = Structure.from_file(f'POSCAR-000-m3cn0308')
    stru_2 = Structure.from_file(f'POSCAR-003-m3cn0308')
    print(stru_1.matches(stru_2, ltol=0.2, stol=0.3, angle_tol=5, 
                                          primitive_cell=True, scale=False, 
                                          attempt_supercell=False, allow_subset=False))
    '''
    '''
    import torch
    gnn = GNNSlice()
    path = 'data/search/ml_01'
    node = 'select'
    atom_pos = gnn.import_list2d(f'{path}/atom_pos_{node}.dat', int)
    atom_type = gnn.import_list2d(f'{path}/atom_type_{node}.dat', int)
    atom_symm = gnn.import_list2d(f'{path}/atom_symm_{node}.dat', int)
    grid_name = gnn.import_list2d(f'{path}/grid_name_{node}.dat', int)
    grid_ratio = gnn.import_list2d(f'{path}/grid_ratio_{node}.dat', float)
    space_group = gnn.import_list2d(f'{path}/space_group_{node}.dat', int)
    angles = gnn.import_list2d(f'{path}/angles_{node}.dat', int)
    thicks = gnn.import_list2d(f'{path}/thicks_{node}.dat', int)
    #flatten list
    grid_name = np.concatenate(grid_name).tolist()
    grid_ratio = np.concatenate(grid_ratio).tolist()
    space_group = np.concatenate(space_group).tolist()
    #load GNN model
    loader = gnn.dataloader_all_atoms(atom_pos, atom_type, grid_name, grid_ratio, space_group, angles, thicks)
    model = 'data/gnn_model/01/model_best.pth.tar'
    gnn.load_fea_model(model)
    gnn.load_out_model(model)
    gnn.load_normalizer(model)
    #predict energy and calculate crystal vector
    crys_vec = gnn.get_crystal_vector_batch(loader)
    energys = gnn.readout_crystal_vector_batch(crys_vec)
    energys = energys.cpu().numpy().flatten()
    crys_vec = torch.cat(crys_vec).cpu().numpy()
    gnn.write_list2d(f'Energy_{node}.dat', np.transpose([energys]))
    '''
    # from ase.io import read, write
    
    # poscar = read('POSCAR')  
    # write('lammps.data', poscar, format='lammps-data')

    # lammps = read('lammps.data', format='lammps-data') 
    # write('POSCAR_converted', lammps, format='vasp')
    
    '''def extract_energy_from_log(file_path):
        energy_data = []
        with open(file_path, 'r') as file:
            lines = file.readlines()

        start_reading = False
        
        for line in lines:
            line_str = line.strip()
            if line_str.startswith('Step'):
                start_reading = True
                
            if start_reading:
                print(line_str)
                if line_str == '':
                    break
                items = line_str.split()
                if not items[0].isalpha():
                    energy = items[2]
                    energy_data.append(float(energy))

        return energy_data
    
    log_file_path = 'log.lammps'
    energies = extract_energy_from_log(log_file_path)

    print("Extracted Energies:", energies)'''
    
    
    '''
    import copy
    def assign_by_spacegroup(atom_num, symm_site, max_store=100, max_assign=50, max_try=10):
        symms = list(symm_site.keys())
        site_num = [len(i) for i in symm_site.values()]
        symm_num = dict(zip(symms, site_num))
        atoms = list(atom_num.keys())
        #initialize assignment
        init_assign = {}
        for atom in atoms:
            init_assign[atom] = []
        init_store = [init_assign]
        #find site assignment of different atom_num
        counter = 0
        new_store, assign_plan = [], []
        for _ in range(max_try):
            store = init_store
            while True:
                for assign in store:
                    np.random.shuffle(symms)
                    for site in symms:
                        np.random.shuffle(atoms)
                        for atom in atoms:
                            new_assign = copy.deepcopy(assign)
                            new_assign[atom] += [site]
                            save = check_assign(atom_num, symm_num, new_assign)
                            if save == 0:
                                assign_plan.append(new_assign)
                                counter += 1
                            if save == 1:
                                new_store.append(new_assign)
                            #constrain leafs
                            if len(new_store) > max_store:
                                break
                        if len(new_store) > max_store:
                            break
                    if len(new_store) > max_store:
                        break
                if len(new_store) == 0:
                    break
                if counter > max_assign:
                    break
                store = new_store
                store = delete_same_assign(store)
                np.random.shuffle(store)
                new_store = []
            if len(assign_plan) > 0:
                break
        assign_plan = delete_same_assign(assign_plan)
        return assign_plan
    
    def group_symm_sites(mapping):
        symm = [len(i) for i in mapping]
        last = symm[0]
        store, symm_site = [], {}
        for i, s in enumerate(symm):
            if s == last:
                store.append(i)
            else:
                symm_site[last] = store
                store = []
                last = s
                store.append(i)
        symm_site[s] = store
        return symm_site
    
    def check_assign(atom_num, symm_num_dict, assign):
        assign_num, site_used = {}, []
        #check number of atom_num
        save = 1
        for atom in atom_num.keys():
            site = assign[atom]
            num = sum(site)
            if num <= atom_num[atom]:
                assign_num[atom] = num
                site_used += site
            else:
                save = 2
                break
        if save == 1:
            #check number of used sites
            site_used = Counter(site_used)
            for site in site_used.keys():
                if site_used[site] > symm_num_dict[site]:
                    save = 2
                    break
            #whether find a right assignment
            if assign_num == atom_num and save == 1:
                save = 0
        return save
    
    def delete_same_assign(store):
        idx = []
        num = len(store)
        for i in range(num):
            assign_1 = store[i]
            for j in range(i+1, num):
                same = True
                assign_2 = store[j]
                for k in assign_1.keys():
                    if sorted(assign_1[k]) != sorted(assign_2[k]):
                        same = False
                        break
                if same:
                    idx.append(i)
                    break
        new_store = np.delete(store, idx, axis=0).tolist()
        return new_store
    
    print(np.where(np.array([0,1,2])==1)[-1])
    '''
    # with open('mapping/051_mapping_123.bin', 'rb') as f:
    #     mapping = pickle.load(f)
    # symm_site = group_symm_sites(mapping) 
    # print(mapping)
    # print(symm_site)
    # print(np.random.choice([0,1,2], 3, replace=False))
    
    # atom_type = convert_composition_into_atom_type(Composition)
    # assign = AtomAssignPlan()
    # start = time.time()
    # assign.control_atom_number_cluster(atom_type)
    # end = time.time()
    # print(end - start)
    # pos = export_seeds_pos(['POSCAR-Seed-ref'], 'POSCAR-Template-gamma')
    # stru = Structure.from_file(f'{Seed_Path}/POSCAR-Template-gamma')
    # seed_stru = Structure.from_file(f'{Seed_Path}/POSCAR-Seed-ref')
    # atoms = seed_stru.atomic_numbers
    # latt = stru.lattice
    # coords = stru.frac_coords[pos[0]]
    # new_stru = Structure(latt, atoms, coords)
    # new_stru.to(filename='POSCAR-test', fmt='vasp')
    
    
    '''def divide_neighbors(atom_num, centers, points, dis, ratio=1):
        nbr_idx, nbr_dis = [], []
        tmp_idx, tmp_dis = [], []
        last = centers[0]
        for i, center in enumerate(centers):
            if center == last:
                tmp_dis.append(dis[i])
                tmp_idx.append(points[i])
            else:
                order = np.argsort(tmp_dis)
                tmp_idx = np.array(tmp_idx)[order]
                tmp_dis = ratio*np.array(tmp_dis)[order]
                nbr_idx.append(tmp_idx)
                nbr_dis.append(tmp_dis)
                tmp_idx = [points[i]]
                tmp_dis = [dis[i]]
                last = center
        order = np.argsort(tmp_dis)
        tmp_idx = np.array(tmp_idx)[order]
        tmp_dis = ratio*np.array(tmp_dis)[order]
        nbr_idx.append(tmp_idx)
        nbr_dis.append(tmp_dis)    
        #
        uni_atom = np.unique(centers)
        if atom_num > len(uni_atom):
            nbr_idx, nbr_dis = fill_neighbors(atom_num, uni_atom, nbr_idx, nbr_dis)
        return nbr_idx, nbr_dis
    
    def fill_neighbors(atom_num, uni_atom, nbr_idx, nbr_dis):
        atom_idx = np.arange(atom_num)
        lack_idx = np.setdiff1d(atom_idx, uni_atom)
        lack_idx = lack_idx[::-1]
        for i in lack_idx:
            nbr_idx.insert(i, [i])
            nbr_dis.insert(i, [8])
        return nbr_idx, nbr_dis
            
        
    stru = Structure.from_file(f'POSCAR-005-node131')
    atom_num = len(stru.atomic_numbers)
    centers, points, _, dis = stru.get_neighbor_list(8, sites=stru.sites)
    nbr_idx1, nbr_dis1 = divide_neighbors(atom_num, centers, points, dis)
    idx = 4
    print(nbr_dis1[0])
    # print(nbr_idx1[idx][:12])
    # print(nbr_dis1[idx][:12])
    
    centers, points, _, dis = stru.get_neighbor_list(2, sites=stru.sites)
    nbr_idx2, nbr_dis2 = divide_neighbors(atom_num, centers, points, dis)
    print(centers)
    # print(nbr_idx2[idx][:12])
    # print(nbr_dis2[idx][:12])'''
    
    
    
    # import paramiko
    # def ssh_node(shell_script, node):
    #     port = 22
    #     ssh = paramiko.SSHClient()
    #     ssh.set_missing_host_key_policy(paramiko.MissingHostKeyPolicy())
    #     ssh.connect(node, port, timeout=1000)
    #     ssh.exec_command(shell_script)
    #     ssh.close()
    
    # script = f'''
    #           #!/bin/bash
    #           cd /tmp/sccop
    #           echo ok >> test.log
    #           '''
    # ssh_node(script, 'cnode001')

    '''
    def get_nbr_stru(stru, ratio=1):
        centers, points, _, dis = stru.get_neighbor_list(8)
        nbr_idx, nbr_dis = divide_neighbors(centers, points, dis, ratio)
        #get neighbor index and distance
        nbr_idx, nbr_dis = cut_pad_neighbors(nbr_idx, nbr_dis, 50)
        return nbr_idx, nbr_dis
    
    def pad_neighbors(neighbors, nbr_num, pattern):
        if pattern == 'index':
            neighbors = [np.pad(i, (0, nbr_num-len(i)), constant_values=i[0]) for i in neighbors]
        elif pattern == 'distance':
            neighbors = [np.pad(i, (0, nbr_num-len(i)), constant_values=8+1) for i in neighbors]
        return neighbors
    
    def cut_neighbors(neighbors, nbr_num):
        neighbors = map(lambda x: x[:nbr_num], neighbors)
        return neighbors
    
    def cut_pad_neighbors(nbr_idx, nbr_dis, nbr_num):
        nbr_idx_cut = cut_neighbors(nbr_idx, nbr_num)
        nbr_dis_cut = cut_neighbors(nbr_dis, nbr_num)
        nbr_idx_pad = pad_neighbors(nbr_idx_cut, nbr_num, 'index')
        nbr_dis_pad = pad_neighbors(nbr_dis_cut, nbr_num, 'distance')
        nbr_idx_new, nbr_dis_new = np.array(nbr_idx_pad, dtype=int), np.array(nbr_dis_pad)
        return nbr_idx_new, nbr_dis_new
    
    def get_equal_coords(point, coords):
        equal_coords = coords.copy()
        for i, coord in enumerate(equal_coords):
            vec = pbc_diff(point, coord)
            if np.linalg.norm(vec) < 1e-5:
                del equal_coords[i]
                break
        return equal_coords
    
    def reduce_to_dau(nbr_idx, nbr_dis, mapping):
        nbr_idx = np.array(nbr_idx)
        nbr_dis = np.array(nbr_dis)
        #reduce to min area
        for line in mapping:
            if len(line) > 1:
                dau_atom = line[0]
                for atom in line[1:]:
                    nbr_idx[nbr_idx==atom] = dau_atom
        return nbr_idx, nbr_dis
    
    def divide_neighbors(centers, points, dis, ratio=1):
        nbr_idx, nbr_dis = [], []
        tmp_idx, tmp_dis = [], []
        last = centers[0]
        for i, center in enumerate(centers):
            if center == last:
                tmp_dis.append(dis[i])
                tmp_idx.append(points[i])
            else:
                order = np.argsort(tmp_dis)
                tmp_idx = np.array(tmp_idx)[order]
                tmp_dis = ratio*np.array(tmp_dis)[order]
                nbr_idx.append(tmp_idx)
                nbr_dis.append(tmp_dis)
                tmp_idx = [points[i]]
                tmp_dis =  [dis[i]]
                last = center
        order = np.argsort(tmp_dis)
        tmp_idx = np.array(tmp_idx)[order]
        tmp_dis = ratio*np.array(tmp_dis)[order]
        nbr_idx.append(tmp_idx)
        nbr_dis.append(tmp_dis)
        return nbr_idx, nbr_dis
    
    def delete_neighbors(center_idx, nbr_idx, nbr_dis):
        nbr_idx_del, nbr_dis_del = [], []
        for idx, dis in zip(nbr_idx, nbr_dis):
            bool_filter = idx!=center_idx
            nbr_idx_del.append(idx[bool_filter])
            nbr_dis_del.append(dis[bool_filter])
        return nbr_idx_del, nbr_dis_del
        
    def gather_neighbors(center_idx, nbr_idx, nbr_dis):
        tmp = np.stack((nbr_idx, nbr_dis), axis=1).tolist()
        tmp_order = sorted(tmp, key=lambda x:(x[0], x[1]))
        nbr_idx_sort, nbr_dis_sort = np.transpose(tmp_order)
        #gather same neighbors
        last_idx = nbr_idx_sort[0]
        point_nbr_idx, point_nbr_dis = [], []
        start = 0
        for i, idx in enumerate(nbr_idx_sort):
            if last_idx != idx:
                nbr_dis_tmp = nbr_dis_sort[start:i]
                nbr_idx_tmp = [center_idx for _ in range(len(nbr_dis_tmp))]
                point_nbr_idx.append(nbr_idx_tmp)
                point_nbr_dis.append(nbr_dis_tmp)
                last_idx = idx
                start = i
        nbr_dis_tmp = nbr_dis_sort[start:]
        nbr_idx_tmp = [center_idx for _ in range(len(nbr_dis_tmp))]
        point_nbr_idx.append(nbr_idx_tmp)
        point_nbr_dis.append(nbr_dis_tmp)
        return point_nbr_idx, point_nbr_dis
    
    def adjust_neighbors(center_idx, nbr_idx, nbr_dis, center_nbr_idx, center_nbr_dis,
                         point_nbr_idx, point_nbr_dis, point_num=20):
        nbr_idx_new, nbr_dis_new = [], []
        dau_atom_num = len(nbr_idx)
        for i in range(dau_atom_num):
            if i == center_idx:
                nbr_idx_new.append(center_nbr_idx[:point_num])
                nbr_dis_new.append(center_nbr_dis[:point_num])
            else:
                nbr_idx_tmp = np.concatenate((nbr_idx[i], point_nbr_idx[i][:point_num]))
                nbr_dis_tmp = np.concatenate((nbr_dis[i], point_nbr_dis[i][:point_num]))
                order = np.argsort(nbr_dis_tmp)
                nbr_idx_tmp = nbr_idx_tmp[order]
                nbr_dis_tmp = nbr_dis_tmp[order]
                nbr_idx_new.append(nbr_idx_tmp)
                nbr_dis_new.append(nbr_dis_tmp)
        #cutting and padding neighbors
        nbr_idx_new, nbr_dis_new = cut_pad_neighbors(nbr_idx_new, nbr_dis_new, 12)
        return nbr_idx_new, nbr_dis_new
    
    def exclude_self(center_idx, nbr_idx, nbr_dis, tol=1e-6):
        min_dis = nbr_dis[0]
        min_idx = nbr_idx[0]
        if center_idx == min_idx and min_dis < tol:
            nbr_idx_new = nbr_idx[1:]
            nbr_dis_new = nbr_dis[1:]
        return nbr_idx_new, nbr_dis_new
    
    def update_neighbors(pos_1, pos_2, nbr_idx_1, nbr_dis_1, ratio, sg, latt_vec, grid_coords):
        diff = np.subtract(pos_1, pos_2)
        if np.sum(diff) == 0:
            nbr_idx_new, nbr_dis_new = nbr_idx_1, nbr_dis_1
        else:
            #find different index and delete it from neighbors
            diff_idx = np.where(diff!=0)[-1][0]
            nbr_idx_del, nbr_dis_del = delete_neighbors(diff_idx, nbr_idx_1, nbr_dis_1)
            #initialize list
            mapping = []
            dau_coords = grid_coords[pos_2]
            all_points = dau_coords.tolist()
            spg = SpaceGroup.from_int_number(sg)
            #get all equivalent sites and mapping relationship
            for i, point in enumerate(dau_coords):
                coords = spg.get_orbit(point)
                equal_coords = get_equal_coords(point, coords)
                start = len(all_points)
                end = start + len(equal_coords)
                mapping.append([i] + [j for j in range(start, end)])
                all_points += equal_coords
            #get neighbor index and distance of sites in DAU
            atom_type = [1 for _ in range(len(all_points))]
            stru = Structure(latt_vec, atom_type, all_points)
            site = [stru.sites[diff_idx]]
            centers, points, _, dis = stru.get_neighbor_list(8, sites=site)
            center_nbr_idx, center_nbr_dis = divide_neighbors(centers, points, dis, ratio)
            center_nbr_idx, center_nbr_dis = center_nbr_idx[0], center_nbr_dis[0]
            center_nbr_idx, center_nbr_dis = exclude_self(diff_idx, center_nbr_idx, center_nbr_dis)
            center_nbr_idx, center_nbr_dis = reduce_to_dau(center_nbr_idx, center_nbr_dis, mapping)
            point_nbr_idx, point_nbr_dis = gather_neighbors(diff_idx, center_nbr_idx, center_nbr_dis)
            #adjust neighbors
            nbr_idx_new, nbr_dis_new = adjust_neighbors(diff_idx, nbr_idx_del, nbr_dis_del, 
                                                        center_nbr_idx, center_nbr_dis, point_nbr_idx, point_nbr_dis)
        return nbr_idx_new, nbr_dis_new
    
    pos_1 = [0, 1, 2, 4, 5]
    pos_2 = [0, 1, 3, 4, 5]
    ratio = 1
    sg = 1
    latt_vec = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    grid_coords = np.array([[0, 0, 0], [.1, 0, .1], [0, .1, .1], [.1, .1, .1], [.1, .2, .1], [.1, .1, .2]])
    atoms = [1 for _ in range(len(pos_1))]
    coords_1 = grid_coords[pos_1]
    coords_2 = grid_coords[pos_2]
    stru_1 = Structure.from_spacegroup(sg, latt_vec, atoms, coords_1)
    stru_2 = Structure.from_spacegroup(sg, latt_vec, atoms, coords_2)
    nbr_idx_1, nbr_dis_1 = get_nbr_stru(stru_1)
    start = time.time()
    nbr_idx_2, nbr_dis_2 = get_nbr_stru(stru_2)
    end = time.time()
    time_1 = end - start
    start = time.time()
    nbr_idx_3, nbr_dis_3 = update_neighbors(pos_1, pos_2, nbr_idx_1, nbr_dis_1, ratio, sg, latt_vec, grid_coords)
    end = time.time()
    time_2 = end - start
    print(f'{time_1} {nbr_dis_2[4][:12]}')
    print(f'{time_2} {nbr_dis_3[4]}')
    '''
    
    # site = [stru_2.sites[1]]
    # centers, points, off_set, dis = stru_2.get_neighbor_list(8, sites=site)
    # order = np.argsort(dis)[0]
    # print(centers[order])
    # print(points[order])
    # print(dis[order])
    # print(off_set[order])
    '''
    def exchange_atoms(atom_type, atom_symm, max_exchange=10):
        all_type = [atom_type]
        is_simple = [True if len(np.unique(atom_type))==1 else False][0]
        if is_simple or max_exchange == 1:
            pass
        else:
            counter = 0
            num = len(atom_type)
            for i in range(num): 
                type_1, symm_1 = atom_type[i], atom_symm[i]
                for j in range(i+1, num):
                    type_2, symm_2 = atom_type[j], atom_symm[j]
                    if type_1 != type_2 and symm_1 == symm_2 and symm_1 > 0:
                        type_copy = atom_type.copy()
                        type_copy[i] = type_2
                        type_copy[j] = type_1
                        all_type.append(type_copy)
                        counter += 1
                    if counter > max_exchange:
                        break
                if counter > max_exchange:
                    break
        return all_type
    
    print(exchange_atoms([5,5,6,6,6], [1,1,2,2,1]))
    '''
    
    '''
    def convert_composition_into_atom_type(comp):
        elements= re.findall('[A-Za-z]+', comp)
        ele_num = [int(i) for i in re.findall('[0-9]+', comp)]
        atom_type = []
        for ele, num in zip(elements, ele_num):
            for _ in range(num):
                atom_type.append(ele)
        return atom_type
    
    def get_ave_bond(comp):
        if False:
            ave_bond = 0.1
        else:
            atom_type = convert_composition_into_atom_type(comp)
            radius = 0
            for i in atom_type:
                radius += Element(i).atomic_radius.real
            ave_radiu = radius/len(atom_type)
            ave_bond = 2*ave_radiu
        return ave_bond
    
    def get_bond_list(comp):
        atom_types = convert_composition_into_atom_type(comp)
        atom_types = np.unique(atom_types)
        types, radius = [], []
        for i in atom_types:
            ele = Element(i)
            types.append(ele.number)
            radius.append(ele.atomic_radius.real)
        #calculate bond length between atoms
        bond_list = []
        for r_a in radius:
            tmp = []
            for r_b in radius:
                tmp.append(r_a+r_b)
            bond_list.append(tmp)
        return types, bond_list
    
    def get_nbr_bond_list(center, points, types, bond_list):
        center_idx = types.index(center)
        points_idx = [types.index(i) for i in points]
        nbr_bond_list = []
        for idx in points_idx:
            nbr_bond_list.append(bond_list[center_idx][idx])
        return nbr_bond_list
    
    def compare_nbr_bond(nbr_dis, nbr_bond_list):
        flag = True
        for i, j in zip(nbr_dis, nbr_bond_list):
            if i - j < 0:
                flag = False
                break
        return flag
    
    def write_dict(file, dict):
        with open(file, 'w') as obj:
            json.dump(dict, obj)
    
    def transfer_keys(list_dict):
        new = []
        #list dict transfer
        if isinstance(list_dict, list):
            for item in list_dict:
                store = {}
                for key in item.keys():
                    store[int(key)] = item[key]
                new.append(store)
        #dict transfer
        elif isinstance(list_dict, dict):
            store = {}
            for key in list_dict.keys():
                store[int(key)] = list_dict[key]
            new = store
        return new
    
    def import_dict(file, trans=False):
        with open(file, 'r') as obj:
            ct = json.load(obj)
        if trans:
            dict = transfer_keys(ct)
        else:
            dict = ct
        return dict
    
    def export_bonds(comp):
        ave_bond = get_ave_bond(comp)
        ele_types, bond_list = get_bond_list(comp)
        bond_dict = {'ave_bond': ave_bond, 'types': ele_types, 'bond_list': bond_list}
        write_dict('test/bond.json', bond_dict)
    
    def get_atom_neighbors(center_pos, atom_pos, atom_type, ratio, grid_idx, grid_dis):
        point_idx = grid_idx[center_pos]
        point_dis = grid_dis[center_pos]
        point_dis *= ratio
        #find neighbor atoms
        atom_pos = np.array(atom_pos)[:, None]
        atom_idx = np.where(point_idx==atom_pos)[-1]
        #get type and distance of neighbors
        if len(atom_idx) > 0:
            order = np.argsort(atom_idx)
            atom_idx = atom_idx[order]
            nbr_pos = point_idx[atom_idx]
            nbr_dis = point_dis[atom_idx]
            nbr_type = get_neighbor_type(atom_pos, atom_type, nbr_pos)
        else:
            nbr_pos, nbr_type, nbr_dis = [], [], []
        return nbr_pos, nbr_type, nbr_dis
    
    def get_neighbor_type(atom_pos, atom_type, nbr_pos):
        #avoid same number
        tmp_pos = nbr_pos.copy()
        for type, pos in zip(atom_type, atom_pos):
            tmp_pos[tmp_pos==pos] = -type
        #transfer to atom type
        nbr_type = -1*tmp_pos
        return nbr_type
    
    types, bond_list = get_bond_list('C1B2')
    start =  time.time()
    nbr_bond_list = get_nbr_bond_list(5, [5,5,6,6], types, bond_list)
    end = time.time()
    print(f'{end-start} {nbr_bond_list}')
    print(compare_nbr_bond([2, 2, 2, 1.6], nbr_bond_list))
    #export_bonds('C1B2')
    ratio = 1
    grid_idx = np.array([[2, 3, 1, 2], [1, 1, 2, 2]])
    grid_dis = np.array([[1.5, 3.5, 1.5, 2.5], [2, 2, 2, 2]])
    center_pos = 0
    atom_pos = [3, 2]
    atom_type = [5, 1]
    start = time.time()
    nbr_pos, nbr_type, nbr_dis = get_atom_neighbors(center_pos, atom_pos, atom_type, ratio, grid_idx, grid_dis)
    end = time.time()
    print(f'{end-start} {nbr_pos} {nbr_type} {nbr_dis}')
    '''
    
    '''
    def divide_neighbors(centers, points, dis):
        nbr_idx, nbr_dis = [], []
        tmp_idx, tmp_dis = [], []
        last = centers[0]
        for i, center in enumerate(centers):
            if center == last:
                tmp_dis.append(dis[i])
                tmp_idx.append(points[i])
            else:
                order = np.argsort(tmp_dis)
                tmp_idx = np.array(tmp_idx)[order]
                tmp_dis = np.array(tmp_dis)[order]
                nbr_idx.append(tmp_idx)
                nbr_dis.append(tmp_dis)
                tmp_idx = [points[i]]
                tmp_dis =  [dis[i]]
                last = center
        order = np.argsort(tmp_dis)
        tmp_idx = np.array(tmp_idx)[order]
        tmp_dis = np.array(tmp_dis)[order]
        nbr_idx.append(tmp_idx)
        nbr_dis.append(tmp_dis)
        return nbr_idx, nbr_dis
    
    stru = Structure.from_file('test/POSCAR-C-1')
    start = time.time()
    near_1 = stru.get_all_neighbors(3, sites=stru.sites[:3])
    all_nbrs_1 = [sorted(nbrs, key = lambda x: x[1]) for nbrs in near_1]
    nbr1_idx, nbr1_dis = [], []
    for nbr in all_nbrs_1:
        nbr1_dis.append(list(map(lambda x: x[1], nbr)))
        nbr1_idx.append(list(map(lambda x: x[2], nbr)))
    end = time.time()
    time_1 = end - start
    start = time.time()
    near_2 = stru.get_neighbor_list(3, sites=stru.sites[:3])
    nbr2_idx, nbr2_dis = divide_neighbors(near_2[0], near_2[1], near_2[3])
    end = time.time()
    time_2 = end -start
    print(nbr1_dis)
    print(nbr2_dis)

    print(f'{time_1} {nbr1_idx}')
    print(f'{time_2} {nbr2_idx}')
    '''