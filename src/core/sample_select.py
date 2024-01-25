import sys, os
import argparse
import numpy as np

from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from core.log_print import *
from core.utils import *
from core.GNN_tool import GNNSlice


class Select(SSHTools, GNNSlice):
    #Select training samples by active learning
    def __init__(self, iteration):
        SSHTools.__init__(self)
        GNNSlice.__init__(self)
        self.poscar_save_path = f'{POSCAR_Path}/ml_{iteration:02.0f}'
        self.search_save_path = f'{Search_Path}/ml_{iteration:02.0f}'
    
    def samples(self, limit=1000):
        """
        choose lowest energy structure in different clusters
        
        Parameters
        ----------
        limit [int, 0d]: limit of sample number
        """
        #import searching data
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec = self.import_search_data()
        train_pos, train_type, train_symm, train_grid, train_ratio, train_sg, train_angles, train_thicks, gnn_energys, train_vec = self.import_train_set()
        #delect selected samples
        idx = self.delete_same_selected(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks,
                                        train_pos, train_type, train_symm, train_grid, train_ratio, train_sg, train_angles, train_thicks)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = energys[idx]
        crys_vec = crys_vec[idx]
        #sorted by energy
        idx = np.argsort(energys)[:limit]
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energys = energys[idx]
        crys_vec = crys_vec[idx]
        #delect selected samples by crystal vectors
        if Use_ML_Clustering:
            idx = np.argsort(gnn_energys)
            gnn_energys = gnn_energys[idx]
            train_vec = train_vec[idx]
            idx = self.delete_same_selected_crys_vec_parallel(crys_vec, energys, train_vec, gnn_energys)
            if len(idx) > 0:
                atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
                    self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                        grid_name, grid_ratio, space_group, angles, thicks)
                energys = energys[idx]
                crys_vec = crys_vec[idx]
        if len(idx) > 0:
            #select samples
            samples_num = SA_Strus_per_Node*self.work_nodes_num
            if Use_ML_Clustering:
                #filter by energy
                idx = np.arange(len(energys))
                num = int(max(samples_num, SA_Energy_Ratio*len(energys)))
                order = np.argsort(energys)[:num]
                energys_filter = energys[order]
                crys_vec_filter = crys_vec[order]
                idx = idx[order]
                if num > samples_num:
                    #reduce dimension and clustering
                    crys_embedded = self.reduce(crys_vec_filter)
                    clusters = self.cluster(samples_num, crys_embedded)
                    idx_slt = self.min_in_cluster(idx, energys_filter, clusters)
                else:
                    idx_slt = idx
            else:
                idx_slt = np.argsort(energys)[:samples_num]
            #export POSCAR
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
                self.filter_samples(idx_slt, atom_pos, atom_type, atom_symm, 
                                    grid_name, grid_ratio, space_group, angles, thicks)
            energys = energys[idx_slt]
            self.write_POSCARs(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys)
    
    def write_POSCARs(self, atom_pos, atom_type, atom_symm,
                      grid_name, grid_ratio, space_group, angles, thicks, energys):
        """
        position and type are sorted by grid name and space group
        
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
        energys [float, 1d]: structure energy
        """
        #make save directory
        if not os.path.exists(self.poscar_save_path):
            os.mkdir(self.poscar_save_path)
        if not os.path.exists(self.search_save_path):
            os.mkdir(self.search_save_path)
        #convert to structure object
        num_jobs = len(grid_name)
        node_assign = self.assign_node(num_jobs)
        strus = self.get_stru_batch_parallel(atom_pos, atom_type, grid_name, grid_ratio, space_group, angles, thicks)
        for i, stru in enumerate(strus):
            file_name = f'{self.poscar_save_path}/POSCAR-{i:03.0f}-{node_assign[i]}'
            stru.to(filename=file_name, fmt='poscar')
        #export dat file of select structures
        self.write_list2d(f'{self.search_save_path}/atom_pos_select.dat',
                          atom_pos)
        self.write_list2d(f'{self.search_save_path}/atom_type_select.dat', 
                          atom_type)
        self.write_list2d(f'{self.search_save_path}/atom_symm_select.dat', 
                          atom_symm)
        self.write_list2d(f'{self.search_save_path}/grid_name_select.dat', 
                          np.transpose([grid_name]))
        self.write_list2d(f'{self.search_save_path}/grid_ratio_select.dat', 
                          np.transpose([grid_ratio]))
        self.write_list2d(f'{self.search_save_path}/space_group_select.dat', 
                          np.transpose([space_group]))    
        self.write_list2d(f'{self.search_save_path}/angles_select.dat',
                          angles)
        self.write_list2d(f'{self.search_save_path}/thicks_select.dat',
                          thicks)
        self.write_list2d(f'{self.search_save_path}/energy_select.dat', 
                          np.transpose([energys]))
    
    def export_recycle(self, recyc, atom_pos, atom_type, atom_symm, 
                       grid_name, grid_ratio, space_group, angles, thicks, energy):
        """
        export configurations after sccop
        
        Parameters
        ----------
        recyc [int, 0d]: recycle times
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: name of grids
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number 
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energy [float, 1d]: energy of structure
        """
        self.poscar_save_path = f'{POSCAR_Path}/SCCOP_{recyc:02.0f}'
        self.search_save_path = self.poscar_save_path
        if not os.path.exists(self.poscar_save_path):
            os.mkdir(self.poscar_save_path)
        #sort structure in order of grid and space group
        idx = self.sort_by_grid_sg(grid_name, space_group)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energy = np.array(energy)[idx]
        #delete structures that are selected before
        if recyc > 0:
            #filter structure by energy
            idx = np.argsort(energy)[:5*Num_Opt_Low_per_Node*self.work_nodes_num]
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
            energy = np.array(energy)[idx]
            #delete same selected structures by pymatgen
            strus_1 = self.get_stru_batch_parallel(atom_pos, atom_type, grid_name, grid_ratio, space_group, angles, thicks)
            strus_2 = self.collect_optim(recyc)
            idx = self.delete_same_selected_pymatgen_parallel(strus_1, strus_2)
            atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
            energy = np.array(energy)[idx]
            system_echo(f'Delete duplicates same as previous recycle: {len(grid_name)}')
        #filter structure by energy
        idx = np.argsort(energy)[:int(5*Num_Opt_Low_per_Node*self.work_nodes_num)]
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energy = np.array(energy)[idx]
        #delete same structures by pymatgen
        strus = self.get_stru_batch_parallel(atom_pos, atom_type, grid_name, grid_ratio, space_group, angles, thicks)
        idx = self.delete_same_strus_by_energy(strus, energy)
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energy = np.array(energy)[idx]
        system_echo(f'Delete duplicates: {len(idx)}')
        #select Top k lowest energy structures
        idx = np.argsort(energy)[:Num_Opt_Low_per_Node*self.work_nodes_num]
        atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks = \
            self.filter_samples(idx, atom_pos, atom_type, atom_symm, 
                                grid_name, grid_ratio, space_group, angles, thicks)
        energy = energy[idx]
        #export structures
        self.write_POSCARs(atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energy)
        system_echo(f'SCCOP optimize structures: {len(grid_name)}')
    
    def collect_optim(self, recyc):
        """
        collect optimized samples in each recycle

        Parameters
        ----------
        recyc [int, 0d]: recycle times
        
        Returns
        ----------
        strus [obj, 1d]: structure object
        """
        #collect optimized structures in each recycle
        full_poscars = []
        for i in range(0, recyc):
            poscars = os.listdir(f'{Optim_Strus_Path}_{i:02.0f}')
            full_poscars += [f'{Optim_Strus_Path}_{i:02.0f}/{j}' for j in poscars
                             if len(j.split('-'))==3]
        #get structure objects
        strus = []
        for poscar in full_poscars:
            stru = Structure.from_file(poscar)
            strus.append(stru)
        return strus
    
    def optim_strus(self):
        """
        select top k minimal energy structures
        """
        if not os.path.exists(SCCOP_Out_Path):
            os.mkdir(SCCOP_Out_Path)
        #delete same structures
        names, strus, energys = self.collect_recycle(0, Num_Recycle)
        idx = self.delete_same_strus_by_energy(strus, energys)
        energys = np.array(energys)[idx]
        #export top k structures
        order = np.argsort(energys)[:Num_Opt_High_per_Node*self.work_nodes_num]
        slt_idx = np.array(idx)[order]
        stru_num = len(order)
        assign = self.assign_node(stru_num)
        system_echo(f'{energys[order]}')
        for i, idx in enumerate(slt_idx):
            name = names[idx]
            stru = strus[idx]
            file_name = f'{SCCOP_Out_Path}/POSCAR-{name}-{assign[i]}'
            stru.to(filename=file_name, fmt='poscar')
        system_echo(f'Optimize configurations: {stru_num}')
        
    def collect_recycle(self, start, end):
        """
        collect poscars and corresponding energys from each recycle
        
        Parameters
        ----------
        start [int, 0d]: start number
        end [int, 0d]: end number
        
        Returns
        ----------
        names [str, 1d]: name of structure
        strus [obj, 1d]: structure object in pymatgen
        energys [float, 1d]: energy of structures
        """
        names, poscars, energys = [], [], []
        #get poscars and corresponding energys
        for i in range(start, end):
            if Energy_Method == 'VASP':
                energy_file = f'{VASP_Out_Path}/optim_strus_{i:02.0f}/Energy.dat'
            elif Energy_Method == 'LAMMPS':
                energy_file = f'{LAMMPS_Out_Path}/optim_strus_{i:02.0f}/Energy.dat'
            if os.path.exists(energy_file):
                energy_dat = self.import_list2d(energy_file, str, numpy=True)
                poscar, energy = np.transpose(energy_dat)
                names += [i.split('-')[1] for i in poscar]
                poscar = [f'{Optim_Strus_Path}_{i:02.0f}/{j}' for j in poscar]
                poscars = np.concatenate((poscars, poscar))
                energys = np.concatenate((energys, energy))
        energys = np.array(energys, dtype=float)
        #sorted by energy
        order = np.argsort(energys)
        energys = energys[order].tolist()
        poscars = np.array(poscars)[order].tolist()
        names = np.array(names)[order].tolist()
        #get structure objects
        strus = []
        for poscar in poscars:
            stru = Structure.from_file(poscar)
            strus.append(stru)
        return names, strus, energys
    
    def judge_convergence(self, recycle):
        """
        judge the convergence by energy
        """
        strus_new, energy_new = self.collect_recycle(recycle, recycle+1)
        strus_old, energy_old = self.collect_recycle(0, recycle)
        idx = self.delete_same_selected_pymatgen_parallel(strus_new, strus_old)
        #compare energy
        energy_new = np.array(energy_new)[idx]
        energy_new = [i for i in energy_new if -12 < i < 0]
        energy_old = [i for i in energy_old if -12 < i < 0]
        energy_new_mean = np.mean(energy_new)
        energy_old_mean = np.mean(energy_old)
        flag = 0
        if energy_old_mean - energy_new_mean < Energy_Convergence:
            flag = 1
            system_echo(f'Satisfy energy convergence condition')
        return flag
    
    def import_search_data(self):
        """
        import multi-SA data
        
        Returns
        ----------
        atom_pos [int, 2d]: position of atoms
        atom_type [int, 2d]: type of atoms
        atom_symm [int, 2d]: symmetry of atoms
        grid_name [int, 1d]: grid name
        grid_ratio [float, 1d]: ratio of grids
        space_group [int, 1d]: space group number
        angles [int, 2d]: cluster rotation angles
        thicks [int, 2d]: atom displacement in z-direction
        energys [float, 1d, np]: structure energy
        crys_vec [float, 2d, np]: crystal vector
        """
        atom_pos = self.import_list2d(f'{self.search_save_path}/atom_pos.dat', int)
        atom_type = self.import_list2d(f'{self.search_save_path}/atom_type.dat', int)
        atom_symm = self.import_list2d(f'{self.search_save_path}/atom_symm.dat', int)
        grid_name = self.import_list2d(f'{self.search_save_path}/grid_name.dat', int)
        grid_ratio = self.import_list2d(f'{self.search_save_path}/grid_ratio.dat', float)
        space_group = self.import_list2d(f'{self.search_save_path}/space_group.dat', int)
        angles = self.import_list2d(f'{self.search_save_path}/angles.dat', int)
        thicks = self.import_list2d(f'{self.search_save_path}/thicks.dat', int)
        energys = self.import_list2d(f'{self.search_save_path}/energys.dat', float)
        crys_vec = self.import_list2d(f'{self.search_save_path}/crys_vec.bin', float, binary=True)
        #flatten 2d list
        grid_name = np.concatenate(grid_name)
        grid_ratio = np.concatenate(grid_ratio)
        space_group = np.concatenate(space_group)
        energys = np.concatenate(energys)
        return atom_pos, atom_type, atom_symm, grid_name, grid_ratio, space_group, angles, thicks, energys, crys_vec
    
    def import_train_set(self):
        """
        import train set
        
        Returns
        ----------
        train_pos [int, 2d]: position of atoms
        train_type [int, 2d]: type of atoms
        train_symm [int, 2d]: symmetry of atoms
        train_grid [int, 1d]: grid name
        train_ratio [float, 1d]: ratio of grids
        train_sg [int, 1d]: space group number
        train_angles [int, 2d]: cluster rotation angles
        train_thicks [int, 2d]: atom displacement in z-direction
        gnn_energy [float, 1d]: prediction energy
        train_vec [float, 2d]: crystal vector
        """
        head = f'{SCCOP_Path}/{Save_Path}/tmp'
        train_pos = self.import_list2d(f'{head}/train_pos.dat', int)
        train_type = self.import_list2d(f'{head}/train_type.dat', int)
        train_symm = self.import_list2d(f'{head}/train_symm.dat', int)
        train_grid = self.import_list2d(f'{head}/train_grid.dat', int)
        train_ratio = self.import_list2d(f'{head}/train_ratio.dat', float)
        train_sg = self.import_list2d(f'{head}/train_sg.dat', int)
        train_angles = self.import_list2d(f'{head}/train_angles.dat', int)
        train_thicks = self.import_list2d(f'{head}/train_thicks.dat', int)
        gnn_energy = self.import_list2d(f'{head}/gnn_energy.dat', float)
        train_vec = self.import_list2d(f'{head}/train_vec.bin', float, binary=True)
        #flatten 2d list
        train_grid = np.concatenate(train_grid)
        train_ratio = np.concatenate(train_ratio)
        train_sg = np.concatenate(train_sg)
        gnn_energy = np.concatenate(gnn_energy)
        return train_pos, train_type, train_symm, train_grid, train_ratio, train_sg, train_angles, train_thicks, gnn_energy, train_vec
    
    def export_train_set(self, train_pos, train_type, train_symm, train_grid,
                         train_ratio, train_sg, train_angles, train_thicks, train_energy):
        """
        export train set
        
        Parameters
        ----------
        train_pos [int, 2d]: position of atoms
        train_type [int, 2d]: type of atoms
        train_symm [int, 2d]: symmetry of atoms
        train_grid [int, 1d]: grid name
        train_ratio [float, 1d]: ratio of grids
        train_sg [int, 1d]: space group number
        train_angles [int, 2d]: cluster rotation angles
        train_thicks [int, 2d]: atom displacement in z-direction
        train_energy [float, 1d]: structure energy
        train_vec [float, 2d]: crystal vector
        """
        gnn_energy, train_vec = self.update_PES(train_pos, train_type, train_symm, train_grid,
                                                train_ratio, train_sg, train_angles, train_thicks)
        #export sampling data
        head = f'{SCCOP_Path}/{Save_Path}/tmp'
        if not os.path.exists(head):
            os.mkdir(head)
        self.write_list2d(f'{head}/train_pos.dat', 
                          train_pos)
        self.write_list2d(f'{head}/train_type.dat', 
                          train_type)
        self.write_list2d(f'{head}/train_symm.dat',
                          train_symm)
        self.write_list2d(f'{head}/train_grid.dat',
                          np.transpose([train_grid]))
        self.write_list2d(f'{head}/train_ratio.dat', 
                          np.transpose([train_ratio]))
        self.write_list2d(f'{head}/train_sg.dat',
                          np.transpose([train_sg]))
        self.write_list2d(f'{head}/train_angles.dat', 
                          train_angles)
        self.write_list2d(f'{head}/train_thicks.dat', 
                          train_thicks)
        self.write_list2d(f'{head}/dft_energy.dat',
                          np.transpose([train_energy]))
        self.write_list2d(f'{head}/gnn_energy.dat',
                          np.transpose([gnn_energy]))
        self.write_list2d(f'{head}/train_vec.bin',
                          train_vec, binary=True)
        
    

if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int)
    args = parser.parse_args()
    
    iteration = args.iteration
    select = Select(iteration)
    select.samples()