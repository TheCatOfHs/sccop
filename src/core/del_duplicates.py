import os, sys
import numpy as np
import multiprocessing as pythonmp

from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans

sys.path.append(f'{os.getcwd()}/src')
from core.log_print import *
from core.utils import *
from core.data_transfer import MultiGridTransfer


class SampleClustering():
    #clustering samples by crystal vectors
    def reduce(self, crys_fea, reduce_dim=8):
        """
        reduce Dimension of crys_fea from 64 to num_components
        
        Parameters
        ----------
        crys_fea [float, 2d, np]: crystal feature vector
        reduce_dim [int, 0d]: reduced Dimension
        
        Returns
        ----------
        crys_embedded [float, 2d, np]: redcued crystal vector
        """
        crys_embedded = KernelPCA(n_components=reduce_dim, kernel='rbf').fit_transform(crys_fea)
        return crys_embedded
    
    def cluster(self, num, crys_embedded):
        """
        group reduced crys_fea into n clusters
        
        Parameters
        ----------
        num [int, 0d]: number of clusters
        crys_embedded [float, 2d, np]: redcued crystal vector
        
        Returns
        ----------
        kmeans.labels_ [int, 1d, np]: cluster labels 
        """
        sample_num = len(crys_embedded)
        if num > sample_num:
            num = sample_num
        kmeans = KMeans(n_init=1, init='k-means++', n_clusters=num).fit(crys_embedded)
        return kmeans.labels_

    def min_in_cluster(self, idx, value, clusters):
        """
        select lowest value sample in each cluster

        Parameters
        ----------
        idx [int, 1d]: index of samples in input
        value [float, 1d]: average prediction
        clusters [int, 1d]: cluster labels of samples
        
        Returns
        ----------
        idx_slt [int, 1d, np]: index of select samples
        """
        order = np.argsort(clusters)
        sort_idx = np.array(idx)[order]
        sort_value = np.array(value)[order]
        sort_clusters = np.array(clusters)[order]
        #index of minimas in each cluster
        min_idx, min_value = 0, 1e10
        last_cluster = sort_clusters[0]
        idx_slt = []
        for i, cluster in enumerate(sort_clusters):
            if cluster == last_cluster:
                pred = sort_value[i]
                if min_value > pred:
                    min_idx = sort_idx[i]
                    min_value = pred                     
            else:
                idx_slt.append(min_idx)
                last_cluster = cluster
                min_idx = sort_idx[i]
                min_value = sort_value[i]
        idx_slt.append(min_idx)
        return np.array(idx_slt)
    
    def cluster_by_labels(self, num, crys_vec, labels, energys):
        """
        clustering samples by labels
        
        Parameters
        ----------
        num [int, 0d]: number of samples
        crys_vec [float, 2d]: crystal vectors
        labels [int, 1d]: clustering using different labels
        energys [float, 1d]: prediction energys

        Returns
        ----------
        sample_idx [int, 1d]: index of samples
        """
        idx = np.arange(len(labels))
        array = np.stack((idx, labels, energys), axis=1)
        array = sorted(array, key=lambda x: x[1])
        idx, labels, energys = np.transpose(array)
        idx, labels = np.array(idx, dtype=int), np.array(labels, dtype=int)
        crys_vec = crys_vec[idx]
        #group index by labels
        store_idx, store_vec, store_energy = [], [], []
        cluster_idx, cluster_vec, cluster_energy = [], [], []
        last_label = labels[0]
        for i, label in enumerate(labels):
            if label == last_label:
                store_idx.append(idx[i])
                store_vec.append(crys_vec[i])
                store_energy.append(energys[i])
            else:
                cluster_idx.append(store_idx)
                cluster_vec.append(store_vec)
                cluster_energy.append(store_energy)
                last_label = label
                store_idx, store_vec, store_energy = [], [], []
                store_idx.append(idx[i])
                store_vec.append(crys_vec[i])
                store_energy.append(energys[i])
        cluster_idx.append(store_idx)
        cluster_vec.append(store_vec)
        cluster_energy.append(store_energy)
        #sampling on different labels
        assign = self.get_assign_plan(num, cluster_idx)
        sample_idx = []
        for i, n in enumerate(assign):
            if n > 0:
                if Use_ML_Clustering:
                    #reduce dimension
                    crys_embedded = self.reduce(cluster_vec[i])
                    #clustering
                    clusters = self.cluster(n, crys_embedded)
                    #select lowest energy in each cluster
                    slt_idx = self.min_in_cluster(cluster_idx[i], cluster_energy[i], clusters)
                else:
                    idx = np.argsort(cluster_energy[i])
                    slt_idx = np.array(cluster_idx[i])[idx][:n]
                sample_idx += slt_idx.tolist()
        return sample_idx

    def get_assign_plan(self, num, cluster_idx):
        """
        get assign plan of clusters

        Parameters
        ----------
        num [int, 0d]: number of samples
        cluster_idx [int, 2d]: index of cluster samples

        Returns
        ----------
        assign [int, 1d]: sampling number of each cluster
        """
        #get sampling number of each label
        cluster_num = len(cluster_idx)
        per_num = [len(i) for i in cluster_idx]
        assign = [0 for _ in range(cluster_num)]
        flag = True
        while flag:
            for i in range(cluster_num):
                cover_all = np.all(np.array(assign) > 0)
                if sum(assign) == num and cover_all:
                    flag = False
                    break
                if sum(assign) > num and cover_all:
                    flag = False
                    break
                if sum(per_num) == 0:
                    flag = False
                    break
                if per_num[i] > 0:
                    per_num[i] -= 1
                    assign[i] += 1
        return assign
    
    def balance_sampling(self, num, labels, preds, print=False):
        """
        select samples from different space groups
        
        Parameters
        ----------
        num [int, 0d]: number of samples
        labels [int, 1d]: sampling under different labels
        preds [float, 1d, np]: prediction energys
        print [bool, 0d]: whether print number for each space group
        
        Returns
        ----------
        sample_idx [int, 1d]: index of samples
        """
        idx = np.arange(len(labels))
        array = np.stack((idx, labels, preds), axis=1)
        array = sorted(array, key=lambda x: x[1])
        idx, labels, energys = np.transpose(array)
        idx, labels = np.array(idx, dtype=int), np.array(labels, dtype=int)
        #group index by labels
        store_idx, store_energy = [], []
        cluster_idx, cluster_energy = [], []
        last_label = labels[0]
        store_labels = []
        for i, label in enumerate(labels):
            if label == last_label:
                store_idx.append(idx[i])
                store_energy.append(energys[i])
            else:
                store_labels.append(last_label)
                cluster_idx.append(store_idx)
                cluster_energy.append(store_energy)
                last_label = label
                store_idx, store_energy = [], []
                store_idx.append(idx[i])
                store_energy.append(energys[i])
        store_labels.append(label)
        if print:
            system_echo(f'{store_labels}')
        cluster_idx.append(store_idx)
        cluster_energy.append(store_energy)
        #sampling on different labels
        assign = self.get_assign_plan(num, cluster_idx)
        if print:
            system_echo(f'{assign}')
        sample_idx = []
        for i, n in enumerate(assign):
            if n > 0:
                order = np.argsort(cluster_energy[i])[:n]
                slt_idx = np.array(cluster_idx[i])[order]
                sample_idx += slt_idx.tolist()
        return sample_idx
    

class DeleteDuplicates(MultiGridTransfer, SampleClustering):
    #delete same structures
    def __init__(self):
        MultiGridTransfer.__init__(self)
    
    def delete_duplicates(self, atom_pos, atom_type, atom_symm,
                          grid_name, grid_ratio, space_group, angles, thicks):
        """
        delete same structures by pos, type, symm, grid
        
        Parameters
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: grid of atoms
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        
        Returns
        ----------
        idx [int, 1d, np]: index of unique samples 
        """
        #convert to string list
        pos_str = self.list2d_to_str(atom_pos, '{0}')
        type_str = self.list2d_to_str(atom_type, '{0}')
        symm_str = self.list2d_to_str(atom_symm, '{0}')
        grid_str = self.list1d_to_str(grid_name, '{0}')
        ratio_str = self.list1d_to_str(grid_ratio, '{0:4.1f}')
        group_str = self.list1d_to_str(space_group, '{0}')
        angle_str = self.list2d_to_str(angles, '{0}')
        thick_str = self.list2d_to_str(thicks, '{0}')
        label = [i+'-'+j+'-'+k+'-'+m+'-'+n+'-'+p+'-'+q+'-'+w for i, j, k, m, n, p, q, w in 
                zip(pos_str, type_str, symm_str, grid_str, ratio_str, group_str, angle_str, thick_str)]
        #delete same structure
        _, idx = np.unique(label, return_index=True)
        return idx
    
    def delete_same_selected(self, pos_1, type_1, symm_1, grid_1, ratio_1, sg_1, angle_1, thick_1,
                             pos_2, type_2, symm_2, grid_2, ratio_2, sg_2, angle_2, thick_2):
        """
        delete common structures of set1 and set2
        return unique index of set1
        
        Parameters
        ----------
        pos_1 [int, 2d]: position of atoms in set1
        type_1 [int, 2d]: type of atoms in set1
        symm_1 [int, 2d]: symmetry of atoms in set1
        grid_1 [int, 1d]: name of grids in set1
        ratio_1 [float, 1d]: ratio of grids in set1
        sg_1 [int, 2d]: space group number in set1
        angle_1 [int, 1d]: angle in set1
        thick_1 [int, 1d]: thick in set1
        pos_2 [int, 2d]: position of atoms in set2
        symm_2 [int, 2d]: symmetry of atoms in set2
        type_2 [int, 2d]: type of atoms in set2
        grid_2 [int, 1d]: name of grids in set2
        ratio_2 [float, 1d]: ratio of grids in set2
        sg_2 [int, 1d]: space group number in set2
        angle_2 [int, 1d]: angle in set2
        thick_2 [int, 1d]: thick in set2
        
        Returns
        ----------
        idx [int, 1d]: index of sample in set 1
        """
        #convert to string list
        pos_str_1 = self.list2d_to_str(pos_1, '{0}')
        type_str_1 = self.list2d_to_str(type_1, '{0}')
        symm_str_1 = self.list2d_to_str(symm_1, '{0}')
        grid_str_1 = self.list1d_to_str(grid_1, '{0}')
        ratio_str_1 = self.list1d_to_str(ratio_1, '{0:4.1f}')
        sg_str_1 = self.list1d_to_str(sg_1, '{0}')
        angle_str_1 = self.list2d_to_str(angle_1, '{0}')
        thick_str_1 = self.list2d_to_str(thick_1, '{0}')
        pos_str_2 = self.list2d_to_str(pos_2, '{0}')
        type_str_2 = self.list2d_to_str(type_2, '{0}')
        symm_str_2 = self.list2d_to_str(symm_2, '{0}')
        grid_str_2 = self.list1d_to_str(grid_2, '{0}')
        ratio_str_2 = self.list1d_to_str(ratio_2, '{0:4.1f}')
        sg_str_2 = self.list1d_to_str(sg_2, '{0}')
        angle_str_2 = self.list2d_to_str(angle_2, '{0}')
        thick_str_2 = self.list2d_to_str(thick_2, '{0}')
        #find unique structures
        array_1 = [i+'-'+j+'-'+k+'-'+m+'-'+n+'-'+p+'-'+q+'-'+w for i, j, k, m, n, p, q, w in 
                   zip(pos_str_1, type_str_1, symm_str_1, grid_str_1, ratio_str_1,
                       sg_str_1, angle_str_1, thick_str_1)]
        array_2 = [i+'-'+j+'-'+k+'-'+m+'-'+n+'-'+p+'-'+q+'-'+w for i, j, k, m, n, p, q, w in 
                   zip(pos_str_2, type_str_2, symm_str_2, grid_str_2, ratio_str_2,
                       sg_str_2, angle_str_2, thick_str_2)]
        array = np.concatenate((array_1, array_2))
        _, idx, counts = np.unique(array, return_index=True, return_counts=True)
        #delete structures same as training set
        same = []
        for i, repeat in enumerate(counts):
            if repeat > 1:
                same.append(i)
        num = len(grid_1)
        idx = np.delete(idx, same)
        idx = [i for i in idx if i < num]
        return idx
    
    def reduce_by_gnn(self, crys_vec, space_group, energys, min_num=1):
        """
        reduce number of samples by gnn
        
        Parameters
        ----------
        crys_vec [float, 2d]: crystal vectors
        space_group [int, 1d]: space groups
        energys [float, 1d]: prediction energys
        min_num [int, 0d]: least clustering number
        
        Returns
        ----------
        idx [int, 1d]: index of reduced samples
        """
        #sample clustering
        num = max(min_num, int(.5*len(energys)))
        if Use_ML_Clustering:
            #reduce dimension
            crys_embedded = self.reduce(crys_vec)
            idx = self.cluster_by_labels(num, crys_embedded, space_group, energys)
        else:
            idx = self.balance_sampling(num, space_group, energys)
        return idx
    
    def delete_duplicates_crys_vec_sg(self, crys_vec, space_group, energys, limit=100):
        """
        delete same structures by crystal vectors and space groups
        
        Parameters
        ----------
        crys_vec [float, 2d]: crystal vectors
        space_group [int, 1d]: space groups
        energys [float, 1d]: prediction energys
        
        Returns
        ----------
        idx [int, 1d]: index of unique samples
        """
        last_sg = space_group[0]
        base, base_store = 0, []
        counter = 0
        cores = pythonmp.cpu_count()
        with pythonmp.get_context('fork').Pool(processes=cores) as pool:
            args_list = []
            for i, sg in enumerate(space_group):
                counter += 1
                if sg != last_sg:
                    args_list.append((crys_vec[base:i], energys[base:i]))
                    base_store.append(base)
                    base = i
                    last_sg = sg
                    counter = 0
                else:
                    if np.mod(counter, limit) == 0:
                        args_list.append((crys_vec[base:i], energys[base:i]))
                        base_store.append(base)
                        base = i
                        counter = 0
            if counter > 0:
                args_list.append((crys_vec[base:], energys[base:]))
                base_store.append(base)
            #start parallel jobs
            jobs = [pool.apply_async(self.delete_duplicates_crys_vec, args) for args in args_list]
            pool.close()
            pool.join()
            #get results
            jobs_pool = [p.get() for p in jobs]
            idx = []
            for base, uniq_idx in zip(base_store, jobs_pool):
                idx += np.add(base, uniq_idx).tolist()
        pool.close()
        del pool
        idx = np.unique(idx)
        return idx
    
    def delete_duplicates_crys_vec_parallel(self, crys_vec, energys, limit=100):
        """
        delete same structures by crystal vectors in parallel
        
        Parameters
        ----------
        crys_vec [float, 2d]: crystal vectors
        energys [float, 1d]: prediction energys
        limit [int, 0d]: limit of job length

        Returns
        ----------
        idx [int, 1d]: index of unique samples
        """
        #multi-cores
        num = len(energys)
        base, base_store = 0, []
        counter = 0
        cores = pythonmp.cpu_count()
        with pythonmp.get_context('fork').Pool(processes=cores) as pool:
            args_list = []
            for i in range(num):
                counter += 1
                if np.mod(counter, limit) == 0:
                    args_list.append((crys_vec[base:i], energys[base:i]))
                    base_store.append(base)
                    base = i
                    counter = 0
            if counter > 0:
                args_list.append((crys_vec[base:], energys[base:]))
                base_store.append(base)
            #start parallel jobs
            jobs = [pool.apply_async(self.delete_duplicates_crys_vec, args) for args in args_list]
            pool.close()
            pool.join()
            #get results
            jobs_pool = [p.get() for p in jobs]
            idx = []
            for base, uniq_idx in zip(base_store, jobs_pool):
                idx += np.add(base, uniq_idx).tolist()
        pool.close()
        del pool
        idx = np.unique(idx)
        return idx
    
    def delete_duplicates_crys_vec(self, crys_vec, energys, energy_tol=.01, num_tol=1e-15, same_tol=.1):
        """
        delete same structures by crystal vectors
        
        Parameters
        ----------
        crys_vec [float, 2d]: crystal vectors
        energys [float, 1d]: prediction energys
        energy_tol [float, 0d]: energy tolerance
        num_tol [float, 0d]: numerical tolerance
        same_tol [float, 0d]: tolerance for canberra distance
        
        Returns
        ----------
        idx [int, 1d, np]: index of unique samples
        """
        num = len(crys_vec)
        #compare crystal vectors
        idx = []
        for i in range(num):
            if i in idx:
                pass
            else:
                vec_1 = np.array(crys_vec[i])
                vec_1_abs = np.abs(vec_1)
                energy_1 = energys[i]
                for j in range(i+1, num):
                    if j in idx:
                        pass
                    else:
                        energy_2 = energys[j]
                        if np.abs(energy_1-energy_2) < energy_tol:
                            vec_2 = np.array(crys_vec[j])
                            vec_2_abs = np.abs(vec_2)
                            #calculate relative distance
                            tmp_1 = np.abs(vec_1 - vec_2)
                            tmp_2 = vec_1_abs + vec_2_abs
                            tmp_2[tmp_2<num_tol] = 1
                            diff = np.mean(tmp_1/tmp_2)
                            #judge difference
                            if diff < same_tol:
                                idx.append(j)
                        else:
                            idx = np.unique(idx).tolist()
        all_idx = np.arange(num)
        idx = np.setdiff1d(all_idx, idx)
        return idx
    
    def delete_same_strus_by_energy(self, strus, energys):
        """
        delete same structures by pymatgen
        
        Parameters
        ----------
        strus [obj, 1d]: structure objects in pymatgen
        energys [float, 1d]: energy of structures
        limit [int, 0d]: limit number
        
        Returns
        ----------
        idx [int, 1d, np]: index of different structures
        """
        #multi-cores
        args_list = []
        strus_num = len(strus)
        args_list = self.divide_duplicates_jobs(args_list, 0, strus_num, strus, energys)
        #delete duplicates in parallel
        cores = pythonmp.cpu_count()
        with pythonmp.get_context('fork').Pool(processes=cores) as pool:
            #start parallel jobs
            jobs = [pool.apply_async(self.delete_duplicates_pymatgen, args) for args in args_list]
            pool.close()
            pool.join()
            #get results
            jobs_pool = [p.get() for p in jobs]
            idx = []
            for i in jobs_pool:
                idx += i
        pool.close()
        del pool
        all_idx = np.arange(strus_num)
        idx = np.unique(idx)
        idx = np.setdiff1d(all_idx, idx)
        return idx
    
    def delete_duplicates_sg_pymatgen(self, atom_pos, atom_type, 
                                      grid_name, grid_ratio, space_group, angles, thicks, energys):
        """
        delete same structures in same space group
        
        Parameters
        -----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        grid_name [int, 1d]: grid of atoms
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d]: structure energy
        
        Returns
        ----------
        idx [int, 1d]: index of different structures
        """
        args_list = []
        i, last_sg = 0, space_group[0]
        strus = self.get_stru_batch_parallel(atom_pos, atom_type, grid_name, grid_ratio, space_group, angles, thicks)
        for j, sg in enumerate(space_group):
            if not last_sg == sg:
                args_list = self.divide_duplicates_jobs(args_list, i, j, strus, energys, energy_tol=1e2)
                last_sg = sg
                i = j
        end = len(space_group)
        args_list = self.divide_duplicates_jobs(args_list, i, end, strus, energys, energy_tol=1e2)
        #delete duplicates in parallel
        cores = pythonmp.cpu_count()
        with pythonmp.get_context('fork').Pool(processes=cores) as pool:
            #start parallel jobs
            jobs = [pool.apply_async(self.delete_duplicates_pymatgen, args) for args in args_list]
            pool.close()
            pool.join()
            #get results
            jobs_pool = [p.get() for p in jobs]
            idx = []
            for i in jobs_pool:
                idx += i
        pool.close()
        del pool
        #unique idx
        all_idx = np.arange(len(grid_name))
        idx = np.unique(idx)
        idx = np.setdiff1d(all_idx, idx)
        return sorted(idx)
    
    def divide_duplicates_jobs(self, args_list, start, end,
                               strus, energys, energy_tol=.1, limit=20):
        """
        divide parallel delete duplicates jobs into smaller size for each core

        Parameters
        ----------
        args_list [list:tuple, 1d]: parameters of parallel jobs
        start [int, 0d]: start index
        end [int, 0d]: end index
        strus [obj, 1d]: structure object
        energys [float, 1d]: structure energy
        energy_tol [float, 0d]: energy tolerance
        limit [int, 0d]: limit number
        
        Returns
        ----------
        args_list [list:tuple, 1d]: parameters of parallel jobs
        """
        a = start
        counter = 0
        energy_1 = energys[a]
        for b in range(start, end):
            counter += 1
            energy_2 = energys[b]
            if np.abs(energy_1-energy_2) < energy_tol:
                if counter > limit:
                    args = (a, b, strus)
                    args_list.append(args)
                    a = b
                    counter = 0
                    energy_1 = energys[a]
            else:
                args = (a, b, strus)
                args_list.append(args)
                a = b
                counter = 0
                energy_1 = energys[a]
        args = (a, end, strus)
        args_list.append(args)
        return args_list
    
    def delete_duplicates_pymatgen(self, start, end, strus):
        """
        delete same structures
        
        Parameters
        -----------
        start [int, 0d]: start index
        end [int, 0d]: end index
        strus [obj, 1d]: structure object
        
        Returns
        ----------
        idx [int, 1d]: index of duplicates
        """
        tmp = strus[start:end]
        strus_num = len(tmp)
        idx = []
        #compare structures by pymatgen
        for i in range(strus_num):
            if i not in idx:
                stru_1 = tmp[i]
                for j in range(i+1, strus_num):
                    stru_2 = tmp[j]
                    same = stru_1.matches(stru_2, ltol=0.2, stol=0.3, angle_tol=5, 
                                          primitive_cell=True, scale=False, 
                                          attempt_supercell=False, allow_subset=False)
                    if same:
                        idx.append(j)
        if len(idx) > 0:
            idx = [start+i for i in idx]
        return idx
    
    def compare_pymatgen(self, stru_1, strus_2):
        """
        compare structure 1 with structures
        
        Parameters
        ----------
        stru_1 [obj, 0d]: pymatgen structure object
        strus_2 [obj, 1d]: pymatgen structure objects
        
        Returns
        ----------
        flag [bool, 0d]: structure whether same
        """
        flag = False
        for stru_2 in strus_2:
            same = stru_1.matches(stru_2, ltol=0.2, stol=0.3, angle_tol=5, 
                                  primitive_cell=True, scale=False, 
                                  attempt_supercell=False, allow_subset=False)
            if same:
                flag = True
                break
        return flag
    
    def delete_same_selected_crys_vec_parallel(self, vecs_1, energys_1, vecs_2, energys_2, energy_tol=.01, limit=100):
        """
        delete same structures by crystal vectors in parallel
        
        Parameters
        ----------
        vecs_1 [float, 2d]: crystal vectors newly added
        energys_1 [float, 1d]: prediction energys newly added
        vecs_2 [float, 2d]: crystal vectors in train set
        energys_2 [float, 1d]: prediction energys in train set
        energy_tol [float, 0d]: energy tolerance
        limit [int, 0d]: limit of job length
        
        Returns
        ----------
        idx [int, 1d]: index of unique samples
        """
        #multi-cores
        base, counter = 0, 0
        cores = pythonmp.cpu_count()
        with pythonmp.get_context('fork').Pool(processes=cores) as pool:
            args_list = []
            for i, vec_1 in enumerate(vecs_1):
                energy_1 = energys_1[i]
                for j in range(len(vecs_2)):
                    energy_2 = energys_2[j]
                    if np.abs(energy_1-energy_2) < energy_tol:
                        counter += 1
                        if np.mod(counter, limit) == 0:
                            args_list.append((i, vec_1, energy_1, vecs_2[base:j], energys_2[base:j]))
                            base = j
                            counter = 0
                if counter > 0:
                    args_list.append((i, vec_1, energy_1, vecs_2[base:], energys_2[base:]))
            #start parallel jobs
            jobs = [pool.apply_async(self.delete_selected_crys_vec, args) for args in args_list]
            pool.close()
            pool.join()
            #get results
            jobs_pool = [p.get() for p in jobs]
            idx = []
            for del_idx in jobs_pool:
                idx += del_idx
        pool.close()
        del pool
        all_idx = np.arange(len(vecs_1))
        idx = np.unique(idx)
        idx = np.setdiff1d(all_idx, idx)
        return idx
    
    def delete_selected_crys_vec(self, idx_1, vec_1, energy_1, vecs_2, energys_2, num_tol=1e-15, same_tol=.1):
        """
        delete same structures by crystal vectors
        
        Parameters
        ----------
        idx []:
        vec_1 [float, 1d]: crystal vectors
        energy_1 []:
        vecs_2 [float, 2d]: crystal vectors
        energys_2 []:
        num_tol [float, 0d]: numerical tolerance
        same_tol [float, 0d]: tolerance for canberra distance
        
        Returns
        ----------
        idx [int, 1d, np]: index of unique samples
        """
        idx = []
        vec_1_abs = np.abs(vec_1)
        #compare crystal vectors
        for i, vec_2 in enumerate(vecs_2):
            energy_2 = energys_2[i]
            vec_2_abs = np.abs(vec_2)
            #calculate relative distance
            tmp_1 = np.abs(vec_1 - vec_2)
            tmp_2 = vec_1_abs + vec_2_abs
            tmp_2[tmp_2<num_tol] = 1
            diff = np.mean(tmp_1/tmp_2)
            #judge difference
            if diff < same_tol:
                if energy_1 > energy_2:
                    idx.append(idx_1)
                    break
        return idx
    
    def delete_same_selected_pymatgen_parallel(self, strus_1, strus_2):
        """
        delete common structures of set1 and set2 by pymatgen
        return unique index of set1
        
        Parameters
        ----------
        strus_1 [obj, 1d]: structure objects in set1
        strus_2 [obj, 1d]: structure objects in set2
        
        Returns
        ----------
        idx [int, 1d]: index of sample in set 1
        """
        #multi-cores
        args_list = []
        cores = pythonmp.cpu_count()
        with pythonmp.get_context('fork').Pool(processes=cores) as pool:
            for stru in strus_1:
                args_list.append((stru, strus_2))
            #put atoms into grid with symmetry constrain
            jobs = [pool.apply_async(self.compare_pymatgen, args) for args in args_list]
            pool.close()
            pool.join()
            #get results
            jobs_pool = [p.get() for p in jobs]
            idx = []
            for i, flag in enumerate(jobs_pool):
                if flag:
                    idx.append(i)
        pool.close()
        del pool
        all_idx = np.arange(len(strus_1))
        idx = np.setdiff1d(all_idx, idx)
        return idx
    
    def filter_samples(self, idx, atom_pos, atom_type, atom_symm, 
                       grid_name, grid_ratio, space_group, angles=[], thicks=[]):
        """
        filter samples by index
        
        Parameters
        ----------
        idx [int, 1d]: index of select samples or binary mask
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        
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
        """
        atom_pos = np.array(atom_pos, dtype=object)[idx].tolist()
        atom_type = np.array(atom_type, dtype=object)[idx].tolist()
        atom_symm = np.array(atom_symm, dtype=object)[idx].tolist()
        grid_name = np.array(grid_name, dtype=object)[idx].tolist()
        grid_ratio = np.array(grid_ratio, dtype=object)[idx].tolist()
        space_group = np.array(space_group, dtype=object)[idx].tolist()
        if len(angles) > 0 and len (thicks) > 0:
            angles = np.array(angles, dtype=object)[idx].tolist()
            thicks = np.array(thicks, dtype=object)[idx].tolist()
        elif len(angles) > 0:
            angles = np.array(angles, dtype=object)[idx].tolist()
        elif len(thicks) > 0:
            thicks = np.array(thicks, dtype=object)[idx].tolist()
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks


if __name__ == '__main__':
    pass