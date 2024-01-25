import os, sys
import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.util.coord import pbc_diff

sys.path.append(f'{os.getcwd()}/src')
from core.log_print import *
from core.utils import *


class Neighbors(ListRWTools):
    #Neighbors related functions
    def __init__(self, nbr=12, dmin=0, dmax=8, step=0.2, var=0.2):
        self.nbr = nbr
        self.var = var
        self.dmax = dmax
        self.filter = np.arange(dmin, dmax+step, step)
    
    def get_ave_min_max_bond(self, atom_type, property_dict):
        """
        get average and minimum distance between atoms

        Parameters
        ----------
        atom_type [int, 2d]: type of atoms
        property_dict [dict, 2d, str:list]: property dictionary for new atoms
    
        Returns
        ----------
        ave_bond [float, 0d]: average bond length
        min_bond [float, 0d]: minimum bond length
        max_bond [float, 0d]: maximum bond length
        """
        if Min_Dis_Constraint:
            bonds = []
            for atoms in atom_type:
                for i in atoms:
                    if i > 0:
                        radiu_a = Element.from_Z(i).atomic_radius.real
                    else:
                        radiu_a = property_dict[str(i)]['ave_radiu']
                    for j in atoms:
                        if j > 0:
                            radiu_b = Element.from_Z(j).atomic_radius.real
                        else:
                            radiu_b = property_dict[str(j)]['ave_radiu']
                        bonds.append(radiu_a+radiu_b)
            ave_bond = np.mean(bonds)
            min_bond = np.min(bonds)
            max_bond = np.max(bonds)
        else:
            ave_bond, min_bond, max_bond = 0, 0, 0
        return ave_bond, min_bond, max_bond

    def get_bond_list(self, atom_type, property_dict):
        """
        get target bond length list for all combination of atoms

        Parameters
        ----------
        atom_type [int, 2d]: type of atoms
        property_dict [dict, 2d, str:list]: property dictionary for new atoms

        Returns
        ----------
        types [int, 1d]: type of elements
        bond_list [float, 2d]: target bond length list
        """
        atoms = np.unique(np.concatenate(atom_type))
        types, radius = [], []
        for i in atoms:
            if i > 0:
                ele = Element.from_Z(i)
                types.append(ele.number)
                if Min_Dis_Constraint:
                    radius.append(ele.atomic_radius.real)
                else:
                    radius.append(0)
            else:
                types.append(int(i))
                if Min_Dis_Constraint:
                    radius.append(property_dict[str(i)]['ave_radiu'])
                else:
                    radius.append(0)
        #calculate bond length between atoms
        bond_list = []
        for r_a in radius:
            tmp = []
            for r_b in radius:
                tmp.append(r_a+r_b)
            bond_list.append(tmp)
        return types, bond_list
    
    def export_bond_file(self):
        """
        export average bond length, minimum bond length, type of elements, and bond length list
        """
        property_dict = []
        if os.path.exists(New_Atom_File):
            property_dict = self.import_data('property')
        atom_types = self.import_list2d(f'{Grid_Path}/atom_types.dat', int)
        #get bond information
        ave_bond, min_bond, max_bond = self.get_ave_min_max_bond(atom_types, property_dict)
        ele_types, bond_list = self.get_bond_list(atom_types, property_dict)
        bond_dict = {'ave_bond': ave_bond, 'min_bond': min_bond, 'max_bond': max_bond, 
                     'ele_types': ele_types, 'bond_list': bond_list}
        self.write_dict(Bond_File, bond_dict)
        
    def get_nbr_bond_list(self, center_type, nbr_types, ele_types, bond_list):
        """
        get target bond length for neighbor atoms

        Parameters
        ----------
        center_type [int, 0d]: type of center atom
        nbr_types [int, 1d]: type of neighbor atoms
        ele_types [int, 1d]: type of elements
        bond_list [float, 2d]: target bond length list

        Returns
        ----------
        nbr_bond_list [float, 1d]: target bond length list for neighbors
        """
        center_idx = ele_types.index(center_type)
        points_idx = [ele_types.index(i) for i in nbr_types]
        nbr_bond_list = []
        for idx in points_idx:
            nbr_bond_list.append(bond_list[center_idx][idx])
        return nbr_bond_list
    
    def get_nbr_cutoff(self, center_type, ele_types, bond_list):
        """
        get maximum cutoff distance for center atom

        Parameters
        ----------
        center_type [int, 0d]: type of center atom
        ele_types [int, 1d]: type of elements
        bond_list [float, 2d]: target bond length list

        Returns
        ----------
        nbr_cutoff [float, 0d]: 
        """
        center_idx = ele_types.index(center_type)
        nbr_cutoff = np.max(bond_list[center_idx])
        return nbr_cutoff
    
    def get_atom_neighbors(self, center_pos, atom_pos, atom_type, ratio, grid_idx, grid_dis, cutoff, limit=12):
        """
        get neighbors for the center atom
        
        Parameters
        ----------
        center_pos [int, 0d]: position of center atom
        atom_pos [int, 1d]: position of other atoms
        atom_type [int, 1d]: type of other atoms
        ratio [float, 0d]: lattice ratio
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        cutoff [float, 0d]: cutoff distance for neighbors
        limit [int, 0d]: limit for number of neighbors
        
        Returns
        ----------
        center_nbr_pos [int, 1d]: position of neighbor atoms
        center_nbr_type [int, 1d]: type of neighbor atoms
        center_nbr_dis [float, 1d]: distance of neighbor atoms
        """
        point_idx = grid_idx[center_pos]
        point_dis = grid_dis[center_pos]*ratio
        #get sites within cutoff
        tmp_idx, tmp_dis = [], []
        for i, dis in enumerate(point_dis):
            if dis < cutoff:
                tmp_idx.append(point_idx[i])
                tmp_dis.append(point_dis[i])
        #find neighbor atoms
        pos = np.array(atom_pos)[:, None]
        idx = np.where(tmp_idx==pos)[-1]
        #get type and distance of neighbors
        if len(idx) > 0:
            order = np.argsort(idx)
            idx = idx[order]
            center_nbr_pos = np.array(tmp_idx)[idx][:limit]
            center_nbr_dis = np.array(tmp_dis)[idx][:limit]
            center_nbr_type = self.get_neighbor_type(atom_pos, atom_type, center_nbr_pos)[:limit]
        else:
            center_nbr_pos, center_nbr_type, center_nbr_dis = [], [], []
        return center_nbr_pos, center_nbr_type, center_nbr_dis
    
    def get_neighbor_type(self, atom_pos, atom_type, center_nbr_pos):
        """
        get type of neighbors for the center atom

        Parameters
        ----------
        atom_pos [int, 1d]: position of atoms
        atom_type [int, 1d]: type of atoms
        center_nbr_pos [int, 1d]: position of neighbor atoms

        Returns
        ----------
        nbr_type [int, 1d]: type of neighbor atoms
        """
        #avoid same number
        tmp_pos = center_nbr_pos.copy()
        nbr_type = np.zeros(len(tmp_pos))
        for pos, type in zip(atom_pos, atom_type):
            nbr_type[tmp_pos==pos] = type
        return nbr_type
    
    def check_nbr_dis(self, nbr_dis, nbr_bond_list):
        """
        check the distance for neighbors

        Parameters
        ----------
        nbr_dis [float, 1d]: distance of neighbors
        nbr_bond_list [float, 1d]: target bond length list for neighbors
        
        Returns
        ----------
        flag [bool, 0d]: flag of the bond constraints
        """
        flag = True
        if len(nbr_dis) > 0:
            for dis, bond in zip(nbr_dis, nbr_bond_list):
                if dis < bond:
                    flag = False
                    break
        return flag

    def get_neighbors_DAU(self, stru, cutoff, mapping):
        """
        index and distance of near grid points in DAU
        
        Parameters
        ----------
        stru [obj, 0d]: structure object in pymatgen
        cutoff [float, 0d]: cutoff distance
        
        mapping [int, 2d]: mapping between DAU and all grid
        
        Returns
        ----------
        nbr_idx [int, 2d, np]: index of near neighbor in DAU
        nbr_dis [float, 2d, np]: distance of near neighbor in DAU
        """
        #neighbor index and distance of DAU
        dau_atom_num = len(mapping)
        centers, points, _, dis = stru.get_neighbor_list(cutoff, sites=stru.sites[:dau_atom_num])
        if len(centers) > 0:
            nbr_idx, nbr_dis = self.divide_neighbors(centers, points, dis)
            nbr_idx, nbr_dis = self.fill_neighbors(dau_atom_num, centers, nbr_idx, nbr_dis, self.dmax)
        else:
            nbr_idx = [[i] for i in range(dau_atom_num)]
            nbr_dis = [[self.dmax] for _ in range(dau_atom_num)]
        #padding neighbors
        max_nbr_num = max(map(lambda x: len(x), nbr_idx))
        nbr_idx_pad = self.pad_neighbors(nbr_idx, max_nbr_num, 'index')
        nbr_dis_pad = self.pad_neighbors(nbr_dis, max_nbr_num, 'distance')
        nbr_idx, nbr_dis = self.reduce_to_DAU(nbr_idx_pad, nbr_dis_pad, mapping)
        return nbr_idx, nbr_dis
    
    def divide_neighbors(self, centers, points, dis, ratio=1):
        """
        get neighbors of center atoms
        
        Parameters
        ----------
        centers [int, 1d, np]: index of center atoms
        points [int, 1d, np]: index of neighbor atoms
        dis [float, 1d]: distance of neighbor atoms
        ratio [float, 0d]: grid ratio 
        
        Returns
        ----------
        nbr_idx [int, 2d]: neighbor index of each atom
        nbr_dis [float, 2d]: neighbor distance of each atom
        """
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
        return nbr_idx, nbr_dis
    
    def fill_neighbors(self, atom_num, centers, nbr_idx, nbr_dis, dmax):
        """
        ensure each atom has neighbors
        
        Parameters
        ----------
        atom_num [int, 0d]: number of atoms in DAU
        centers [int, 1d]: index of center atoms
        nbr_idx [int, 2d]: index of near neighbor 
        nbr_dis [float, 2d]: distance of near neighbor 
        dmax [float, 0d]: maximum distance
        
        Returns
        ----------
        nbr_idx [int, 2d]: neighbor index of each atom
        nbr_dis [float, 2d]: neighbor distance of each atom
        """
        uni_atom = np.unique(centers)
        if atom_num > len(uni_atom):
            atom_idx = np.arange(atom_num)
            lack_idx = np.setdiff1d(atom_idx, uni_atom)
            lack_idx = lack_idx[::-1]
            for i in lack_idx:
                nbr_idx.insert(i, [i])
                nbr_dis.insert(i, [dmax])
        return nbr_idx, nbr_dis
    
    def reduce_to_DAU(self, nbr_idx, nbr_dis, mapping):
        """
        reduce neighbors to DAU
        
        Parameters
        ----------
        nbr_idx [int, 1d/2d]: index of near neighbor 
        nbr_dis [float, 1d/2d]: distance of near neighbor 
        mapping [int, 2d]: mapping between DAU and all grid

        Returns
        ----------
        nbr_idx [int, 2d, np]: index of near neighbor in DAU
        nbr_dis [float, 2d, np]: distance of near neighbor in DAU
        """
        nbr_idx = np.array(nbr_idx)
        nbr_dis = np.array(nbr_dis)
        #reduce to min area
        for line in mapping:
            if len(line) > 1:
                dau_atom = line[0]
                for atom in line[1:]:
                    nbr_idx[nbr_idx==atom] = dau_atom
        return nbr_idx, nbr_dis
    
    def get_nbr_stru(self, stru, ratio=1):
        """
        get neighbor bonding feature and index
        
        Parameters
        ----------
        stru [obj, 0d]: structure object 
        ratio [float, 0d]: grid ratio

        Returns
        ----------
        nbr_idx [int, 2d, np]: neighbor index of atoms
        nbr_dis [float, 2d, np]: neighbor distance of atoms
        """
        centers, points, _, dis = stru.get_neighbor_list(self.dmax)
        nbr_idx, nbr_dis = self.divide_neighbors(centers, points, dis, ratio=ratio)
        #get neighbor index and distance
        atom_num = len(stru.atomic_numbers)
        nbr_idx, nbr_dis = self.fill_neighbors(atom_num, centers, nbr_idx, nbr_dis, self.dmax)
        nbr_idx, nbr_dis = self.cut_pad_neighbors(nbr_idx, nbr_dis, self.nbr)
        return nbr_idx, nbr_dis
    
    def get_nbr_general(self, atom_pos, ratio, sg, latt_vec, grid_coords, nbr_num=12):
        """
        get neighbor distance and index
        
        Parameters
        ----------
        atom_pos [int, 1d]: position of atoms
        ratio [float, 0d]: grid ratio
        sg [int, 0d]: space group number
        latt_vec [float, 2d, np]: lattice vector
        grid_coords [float, 2d, np]: fraction coordinates of grid
        nbr_num [int, 0d]: number of neighbors
        
        Returns
        ----------
        nbr_idx [int, 2d, np]: neighbor index of atoms
        nbr_dis [float, 2d, np]: neighbor distance of atoms
        """
        mapping = []
        dau_coords = grid_coords[atom_pos]
        dau_atom_num = len(dau_coords)
        all_points = dau_coords.tolist()
        spg = SpaceGroup.from_int_number(sg)
        #get all equivalent sites and mapping relationship
        for i, point in enumerate(dau_coords):
            coords = spg.get_orbit(point)
            equal_coords = self.get_equal_coords(point, coords)
            start = len(all_points)
            end = start + len(equal_coords)
            mapping.append([i] + [j for j in range(start, end)])
            all_points += equal_coords
        #get neighbor index and distance of sites in DAU
        atom_type = [1 for _ in range(len(all_points))]
        stru = Structure(latt_vec, atom_type, all_points)
        centers, points, _, dis = stru.get_neighbor_list(self.dmax, sites=stru.sites[:dau_atom_num])
        nbr_idx, nbr_dis = self.divide_neighbors(centers, points, dis, ratio=ratio)
        #cutting and padding neighbors
        nbr_idx, nbr_dis = self.fill_neighbors(dau_atom_num, centers, nbr_idx, nbr_dis, self.dmax)
        nbr_idx, nbr_dis = self.cut_pad_neighbors(nbr_idx, nbr_dis, nbr_num)
        nbr_idx, nbr_dis = self.reduce_to_DAU(nbr_idx, nbr_dis, mapping)
        return nbr_idx, nbr_dis
    
    def get_equal_coords(self, point, coords):
        """
        get equivalent coordinates
        
        Parameters
        ----------
        point [float, 1d]: point in minimum area
        coords [float, 2d]: all symmetry coordinates

        Returns
        ----------
        equal_coords [float, 2d]: equivalent coordinates
        """
        equal_coords = coords.copy()
        for i, coord in enumerate(equal_coords):
            vec = pbc_diff(point, coord)
            if np.linalg.norm(vec) < 1e-5:
                del equal_coords[i]
                break
        return equal_coords
    
    def get_nbr_fea_general(self, atom_pos, ratio, sg, latt_vec, grid_coords):
        """
        neighbor bond features and index are cutoff by 12 atoms
        
        Parameters
        ----------
        atom_pos [int, 1d]: position of atoms
        ratio [float, 0d]: grid ratio
        sg [int, 0d]: space group number
        latt_vec [float, 2d, np]: lattice vector
        grid_coords [float, 2d, np]: fraction coordinates of grid
        
        Returns
        ----------
        nbr_fea [float, 3d, np]: neighbor feature of atoms
        nbr_idx [int, 2d, np]: neighbor index of atoms
        """
        nbr_idx, nbr_dis = self.get_nbr_general(atom_pos, ratio, sg, latt_vec, grid_coords)
        #get bond features
        nbr_fea = self.expand(nbr_dis)
        return nbr_fea, nbr_idx
    
    def get_atom_fea(self, atom_type, elem_embed):
        """
        initialize atom feature vectors

        Parameters
        ----------
        atom_type [int, 1d]: type of atoms
        elem_embed [int, 2d, np]: embedding of elements
        
        Returns
        ----------
        atom_fea [int, 2d, np]: atom feature vectors
        """
        atom_fea = elem_embed[atom_type]
        return atom_fea
    
    def expand(self, distances):
        """
        near distance expanded in gaussian feature space
        
        Parameters
        ----------
        distances [float, 2d, np]: distance of near neighbors
        
        Returns
        ----------
        nbr_fea [float, 3d, np]: gaussian feature vector
        """
        return np.exp(-(distances[:, :, np.newaxis] - self.filter)**2 /
                      self.var**2)
    
    def pad_neighbors(self, neighbors, nbr_num, pattern):
        """
        pad index or distance of neighbors
        
        Parameters
        ----------
        neighbors [int/float, 2d]: neighbors of atoms
        nbr_num [int, 0d]: number of neighbors
        pattern [str, 0d]: padding index or distance
        
        Returns
        ----------
        neighbors [int/float, 2d]: neighbors after padding
        """
        if pattern == 'index':
            neighbors = [np.pad(i, (0, nbr_num-len(i)), constant_values=i[-1]) for i in neighbors]
        elif pattern == 'distance':
            neighbors = [np.pad(i, (0, nbr_num-len(i)), constant_values=self.dmax+1) for i in neighbors]
        return neighbors
    
    def cut_neighbors(self, neighbors, nbr_num):
        """
        cut index or distance of neighbors

        Parameters
        ----------
        neighbors [int/float, 2d]: neighbors of atoms
        nbr_num [int, 0d]: number of neighbors

        Returns
        ----------
        neighbors [int/float, 2d]: neighbors after cutting
        """
        neighbors = map(lambda x: x[:nbr_num], neighbors)
        return neighbors
    
    def cut_pad_neighbors(self, nbr_idx, nbr_dis, nbr_num):
        """
        cut and pad neighbors
                
        Parameters
        ----------
        nbr_idx [int, 2d]: neighbor index
        nbr_dis [float, 2d]: neighbor distance
        nbr_num [int, 0d]: neighbor number

        Returns
        ----------
        nbr_idx_new [int, 2d, np]: neighbor index after cutting and padding
        nbr_dis_new [float, 2d, np]: neighbor distance after cutting and padding
        """
        nbr_idx_cut = self.cut_neighbors(nbr_idx, nbr_num)
        nbr_dis_cut = self.cut_neighbors(nbr_dis, nbr_num)
        nbr_idx_pad = self.pad_neighbors(nbr_idx_cut, nbr_num, 'index')
        nbr_dis_pad = self.pad_neighbors(nbr_dis_cut, nbr_num, 'distance')
        nbr_idx_new, nbr_dis_new = np.array(nbr_idx_pad, dtype=int), np.array(nbr_dis_pad)
        return nbr_idx_new, nbr_dis_new
    
    def update_neighbors(self, pos_1, pos_2, nbr_idx_1, nbr_dis_1, ratio, sg, latt_vec, grid_coords):
        """
        update neighbors by last step neighbors
        
        Parameters
        ----------
        pos_1 [int, 1d]: old atom position
        pos_2 [int, 1d]: new atom position 
        nbr_idx_1 [int, 2d]: index of neighbors
        nbr_dis_1 [float, 2d]: distance of neighbors
        ratio [float, 0d]: grid ratio
        sg [int, 0d]: space group number
        latt_vec [float, 2d, np]: lattice vector
        grid_coords [float, 2d, np]: fraction coordinates of grid

        Returns
        ----------
        nbr_idx_new [int, 2d]: updated neighbor index
        nbr_dis_new [float, 2d]: updated neighbor distance
        """
        diff = np.subtract(pos_1, pos_2)
        if np.sum(np.abs(diff)) == 0:
            nbr_idx_new, nbr_dis_new = nbr_idx_1, nbr_dis_1
        else:
            dau_coords = grid_coords[pos_2]
            dau_atom_num = len(pos_2)
            diff_idx = np.where(diff!=0)[-1][0]
            all_points = dau_coords.tolist()
            #get all equivalent sites and mapping relationship
            mapping = []
            spg = SpaceGroup.from_int_number(sg)
            for i, point in enumerate(dau_coords):
                coords = spg.get_orbit(point)
                equal_coords = self.get_equal_coords(point, coords)
                start = len(all_points)
                end = start + len(equal_coords)
                mapping.append([i] + [j for j in range(start, end)])
                all_points += equal_coords
            #get neighbor index and distance for new site in DAU
            atom_type = [1 for _ in range(len(all_points))]
            stru = Structure(latt_vec, atom_type, all_points)
            site_in_DAU = [stru.sites[diff_idx]]
            centers, points, _, dis = stru.get_neighbor_list(self.dmax, sites=site_in_DAU)
            dau_nbr_idx, dau_nbr_dis = self.divide_neighbors(centers, points, dis, ratio=ratio)
            dau_nbr_idx, dau_nbr_dis = dau_nbr_idx[0], dau_nbr_dis[0]
            dau_nbr_idx, dau_nbr_dis = self.exclude_self(diff_idx, dau_nbr_idx, dau_nbr_dis)
            dau_nbr_idx, dau_nbr_dis = self.reduce_to_DAU(dau_nbr_idx, dau_nbr_dis, mapping)
            #get neighbor index and distance for same sites in DAU
            symm_idx_all = mapping[diff_idx]
            symm_coords = np.array(all_points)[symm_idx_all]
            atom_type = [1 for _ in range(len(symm_coords)+1)]
            update_nbr_idx, update_nbr_dis = [], []
            for i in range(dau_atom_num):
                if i == diff_idx:
                    update_nbr_idx.append(dau_nbr_idx)
                    update_nbr_dis.append(dau_nbr_dis)
                else:
                    tmp_points = np.concatenate(([dau_coords[i]], symm_coords))
                    stru = Structure(latt_vec, atom_type, tmp_points)
                    site_in_DAU = [stru.sites[0]]
                    centers, points, _, dis = stru.get_neighbor_list(self.dmax, sites=site_in_DAU)
                    if len(centers) > 0:
                        fix_nbr_idx, fix_nbr_dis = self.divide_neighbors(centers, points, dis, ratio=ratio)
                        fix_nbr_idx, fix_nbr_dis = fix_nbr_idx[0], fix_nbr_dis[0]
                        #exclude self point
                        fix_nbr_idx, fix_nbr_dis = self.exclude_self(0, fix_nbr_idx, fix_nbr_dis)
                        bool_filter = fix_nbr_idx!=0
                        fix_nbr_idx = fix_nbr_idx[bool_filter]
                        fix_nbr_dis = fix_nbr_dis[bool_filter]
                        fix_nbr_idx = 0*fix_nbr_idx+diff_idx
                    else:
                        fix_nbr_idx, fix_nbr_dis = [], []
                    update_nbr_idx.append(fix_nbr_idx)
                    update_nbr_dis.append(fix_nbr_dis)
            #adjust neighbors
            nbr_idx_del, nbr_dis_del = self.delete_neighbors(diff_idx, nbr_idx_1, nbr_dis_1)
            nbr_idx_new, nbr_dis_new = self.adjust_neighbors(diff_idx, nbr_idx_del, nbr_dis_del, update_nbr_idx, update_nbr_dis)
        return nbr_idx_new, nbr_dis_new
    
    def delete_neighbors(self, diff_idx, nbr_idx, nbr_dis):
        """
        delete different site index from neighbors

        Parameters
        ----------
        diff_idx [int, 0d]: different index
        nbr_idx [int, 2d]: neighbor index
        nbr_dis [float, 2d]: neighbor distance

        Returns
        ----------
        nbr_idx_del [int, 2d]: neighbor index after deleting
        nbr_dis_del [float, 2d]: neighbor distance after deleting
        """
        nbr_idx_del, nbr_dis_del = [], []
        for idx, dis in zip(nbr_idx, nbr_dis):
            bool_filter = idx!=diff_idx
            nbr_idx_del.append(idx[bool_filter])
            nbr_dis_del.append(dis[bool_filter])
        return nbr_idx_del, nbr_dis_del
    
    def exclude_self(self, center_idx, nbr_idx, nbr_dis, tol=1e-6):
        """
        exclude self point from neighbors
        
        Parameters
        ----------
        center_idx [int, 0d]: center index
        nbr_idx [int, 1d]: neighbor index of center site
        nbr_dis [float, 1d]: neighbor distance of center site
        tol [float, 0d]: tolerance for same point

        Returns
        ----------
        nbr_idx_new [int, 1d]: neighbor index after excluding itself
        nbr_dis_new [float, 1d]: neighbor distance after excluding itself
        """
        min_dis = nbr_dis[0]
        min_idx = nbr_idx[0]
        if center_idx == min_idx and min_dis < tol:
            nbr_idx_new = nbr_idx[1:]
            nbr_dis_new = nbr_dis[1:]
        else:
            nbr_idx_new = nbr_idx
            nbr_dis_new = nbr_dis
        return nbr_idx_new, nbr_dis_new
    
    def adjust_neighbors(self, diff_idx, nbr_idx, nbr_dis, update_nbr_idx, update_nbr_dis, num=20):
        """
        adjust neighbors according to previous data

        Parameters
        ----------
        diff_idx [int, 0d]: different index
        nbr_idx [int, 2d]: neighbor index 
        nbr_dis [float, 2d]: neighbor distance
        update_nbr_idx [int, 2d]: neighbor index of other sites
        update_nbr_dis [float, 2d]: neighbor distance of other sites
        num [int, 0d]: number of points used for finding neighbors

        Returns
        ----------
        nbr_idx_new [int, 2d]: neighbor index after updating
        nbr_dis_new [float, 2d]: neighbor distance after updating
        """
        nbr_idx_new, nbr_dis_new = [], []
        dau_num = len(nbr_idx)
        for i in range(dau_num):
            if i == diff_idx:
                nbr_idx_new.append(update_nbr_idx[i][:num])
                nbr_dis_new.append(update_nbr_dis[i][:num])
            else:
                nbr_idx_tmp = np.concatenate((nbr_idx[i], update_nbr_idx[i][:num]))
                nbr_dis_tmp = np.concatenate((nbr_dis[i], update_nbr_dis[i][:num]))
                order = np.argsort(nbr_dis_tmp)
                nbr_idx_tmp = nbr_idx_tmp[order]
                nbr_dis_tmp = nbr_dis_tmp[order]
                nbr_idx_new.append(nbr_idx_tmp)
                nbr_dis_new.append(nbr_dis_tmp)
        return nbr_idx_new, nbr_dis_new
    
    def get_nbr_fea_template(self, atom_pos, ratio, grid_idx, grid_dis):
        """
        neighbor bond features and index are cutoff by 12 atoms
        
        Parameters
        ----------
        atom_pos [int, 1d]: position of atoms
        ratio [float, 0d]: grid ratio
        grid_idx [int, 2d, np]: neighbor index of grid
        grid_dis [float, 2d, np]: neighbor distance of grid
        
        Returns
        ----------
        nbr_fea [float, 3d, np]: neighbor feature of atoms
        nbr_idx [int, 2d, np]: neighbor index of atoms
        """
        #get index and distance of points
        point_idx = grid_idx[atom_pos]
        point_dis = grid_dis[atom_pos]
        point_dis *= ratio
        #initialize neighbor index and distance
        dau_atom_num = len(atom_pos)
        nbr_idx = np.zeros((dau_atom_num, self.nbr))
        nbr_dis = np.zeros((dau_atom_num, self.nbr))
        atoms = np.array(atom_pos)[:, None]
        for i, point in enumerate(point_idx):
            #find nearest atoms
            atom_idx = np.where(point==atoms)[-1]
            order = np.argsort(atom_idx)
            #get neighbor index and distance
            near_num = len(order)
            if near_num < self.nbr:
                atom_idx = atom_idx[order]
                lack_num = self.nbr - near_num
                nearest_atom = point_idx[i, atom_idx[0]]
                #fill with nearest atom within cutoff
                nbr_idx[i] = np.pad(point_idx[i, atom_idx], (0, lack_num), constant_values=nearest_atom)
                nbr_dis[i] = np.pad(point_dis[i, atom_idx], (0, lack_num), constant_values=self.dmax+1)
            else:
                order = order[:self.nbr]
                atom_idx = atom_idx[order]
                nbr_idx[i] = point_idx[i, atom_idx]
                nbr_dis[i] = point_dis[i, atom_idx]
        nbr_idx, nbr_dis = np.array(nbr_idx, dtype=int), np.array(nbr_dis)
        #get bond features
        nbr_fea = self.expand(nbr_dis)
        nbr_idx = self.idx_transfer(atom_pos, nbr_idx)
        return nbr_fea, nbr_idx
    
    def idx_transfer(self, atom_pos, nbr_idx):
        """
        make nbr_idx consistent with nbr_dis
    
        Parameters
        ----------
        atom_pos [int, 1d]: position of atoms
        nbr_idx [int, 2d, np]: near index of atoms
        
        Returns
        ----------
        nbr_idx [int, 2d, np]: index start from 0
        """
        #avoid same index
        for i, idx in enumerate(atom_pos):
            nbr_idx[nbr_idx==idx] = -i-1
        #transfer to inner index
        nbr_idx = -nbr_idx-1
        return nbr_idx
    
if __name__ == '__main__':
    pass