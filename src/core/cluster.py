import os, sys
import numpy as np

from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from core.log_print import *
from core.utils import *


class BuildNewAtom(ListRWTools):
    #build new atom for clusters
    def __init__(self):
        pass
    
    def import_clusters(self):
        """
        import clusters in directory data/seeds
        
        Returns
        ----------
        strus [obj, 1d]: structure objects
        """
        strus = []
        poscars = os.listdir(f'{Seed_Path}')
        for poscar in poscars:
            stru = Structure.from_file(f'{Seed_Path}/{poscar}')
            strus.append(stru)
        return strus
    
    def get_atoms(self, strus):
        """
        get atom information for each cluster
        
        Parameters
        ----------
        strus [obj, 1d]: structure objects

        Returns
        ----------
        new_type [int, 1d]: atom type of cluster labeled as negative integer
        atom_num [int, 1d]: number of atoms in each cluster
        cluster_atoms [int, 2d]: type of atoms in each cluster
        """
        atom_num, cluster_atoms = [], []
        for stru in strus:
            atoms = list(stru.atomic_numbers)
            num = len(atoms)
            atom_num.append(num)
            cluster_atoms.append(atoms)
        #label each cluster
        num = len(strus)
        new_type = [-i-1 for i in range(num)][::-1]
        return new_type, atom_num, cluster_atoms 
    
    def get_property(self, strus, cluster_atoms):
        """
        get properties of clusters

        Parameters
        ----------
        strus [obj, 1d]: structure objects
        cluster_atoms [int, 2d]: type of atoms in each cluster

        Returns
        ----------
        embedding [int, 2d]: atom embedding
        coords_zero [float, 3d]: cartesian coordinates of cluster 
        ave_radius [float, 1d]: average radius
        affinity [float, 1d]: average affinity
        negativity [float, 1d]: average negativity
        """
        coords_zero, ave_radius, affinity, negativity = [], [], [], []
        for stru in strus:
            #move coordinates to zero
            coords = stru.cart_coords
            mean = np.mean(coords, axis=0)
            tmp_coords = coords - mean
            coords_zero.append(tmp_coords)
            dis = np.linalg.norm(tmp_coords, axis=1)
            #calculate radiu, affinity and negativity
            tmp_radiu, tmp_aff, tmp_neg = [], [], []
            atoms = stru.species
            for atom in atoms:
                tmp_radiu.append(atom.atomic_radius.real)
                tmp_aff.append(atom.electron_affinity)
                tmp_neg.append(atom.X)
            ave_radius.append(np.max(dis) + np.mean(tmp_radiu))
            affinity.append(np.mean(tmp_aff))
            negativity.append(np.mean(tmp_neg))
        #get element embedding of clusters
        elem_embed = self.import_list2d(Atom_Init_File, int, numpy=True)
        embedding = []
        for atoms in cluster_atoms:
            vecs = elem_embed[atoms]
            cluster_embedding = np.mean(vecs, axis=0)
            cluster_embedding[-1] = 1
            cluster_embedding += .01*np.random.rand(len(cluster_embedding))
            embedding.append(cluster_embedding)
        return embedding, coords_zero, ave_radius, affinity, negativity
    
    def get_rotate_angles(self, grain=[2, 2, 2]):
        """
        write cluster rotation angles

        Parameters
        ----------
        grain [int, 1d]: grain of rotation angles
        """
        angles = []
        for i in range(0, grain[0]):
            for j in range(0, grain[1]):
                for k in range(0, grain[2]):
                    alpha = (i*2*np.pi)/grain[0]
                    beta = (j*np.pi)/grain[1]
                    gamma = (k*2*np.pi)/grain[2]
                    angles.append([alpha, beta, gamma])
        return angles
    
    def export_rotate_angles(self, idx, angles):
        """
        write cluster rotation angles

        Parameters
        ----------
        idx [int, 1d]: index of unique angle
        angles [float, 2d]: rotation angles
        """
        angles = np.array(angles)[idx]
        self.write_list2d(Cluster_Angle_File, angles, style='{0:8.4f}')
    
    def export_embedding(self, embedding):
        """
        write new embedding into atom_init.dat
        
        Parameters
        ----------
        embedding [int, 2d]: atom embedding
        """
        embedding_str = '\n' + '\n'.join([' '.join([str(i) for i in line]) for line in embedding])
        with open(Atom_Init_File, 'a') as obj:
            obj.write(embedding_str)

    def export_property(self, atom_type, coords, cluster_atoms, ave_radius, affinity, negativity):
        """
        export properties of clusters
        
        Parameters
        ----------
        atom_type [int, 1d]: atom type of cluster labeled as negative integer
        coords [float, 3d]: cartesian coordinates of cluster 
        cluster_atoms [int, 2d]: type of atoms in each cluster
        ave_radius [float, 1d]: average radius 
        affinity [float, 1d]: average affinity
        negativity [float, 1d]: average negativity
        """
        property_dict = {}
        for i, atom in enumerate(atom_type):
            types = cluster_atoms[i]
            type_num = count_atoms(types)
            property_dict[atom] = {'coords': coords[i].tolist(),
                                   'types': types, 'type_num': type_num.tolist(),
                                   'ave_radiu': ave_radius[i], 'ave_affinity': affinity[i],
                                   'ave_negativity': negativity[i]}
        self.write_dict(New_Atom_File, property_dict)
    

class AtomManipulate(ListRWTools):
    #rotate and move cluster
    def __init__(self):
        pass 
    
    def rotate_atom(self, alpha, beta, gamma, coords):
        """
        rotate atom randomly
        
        Parameters
        ----------
        alpha [float, 0d]: yaw angle range from 0 to 2pi
        beta [float, 0d]: pitch angle range from 0 to pi
        gamma [float, 0d]: roll angle range from 0 to 2pi
        coords [float, 2d, np]: cartesian coordinates of cluster

        Returns
        ----------
        coords_rotate [float, 2d, np]: coordinates after rotation
        """
        #get rotate matrix
        r_11 = np.cos(alpha)*np.cos(gamma) - np.cos(beta)*np.sin(alpha)*np.sin(gamma)
        r_12 = -np.cos(beta)*np.cos(gamma)*np.sin(alpha) - np.cos(alpha)*np.sin(gamma)
        r_13 = np.sin(alpha)*np.sin(beta)
        r_21 = np.cos(gamma)*np.sin(alpha) + np.cos(alpha)*np.cos(beta)*np.sin(gamma)
        r_22 = np.cos(alpha)*np.cos(beta)*np.cos(gamma) - np.sin(alpha)*np.sin(gamma)
        r_23 = -np.cos(alpha)*np.sin(beta)
        r_31 = np.sin(beta)*np.sin(gamma)
        r_32 = np.cos(gamma)*np.sin(beta)
        r_33 = np.cos(beta)
        rotate_mat = [[r_11, r_12, r_13], [r_21, r_22, r_23], [r_31, r_32, r_33]]
        coords_rotate = np.dot(coords, rotate_mat)
        return coords_rotate     
    
    def move_atom(self, center, coords):
        """
        move cluster to center point
        
        Parameters
        ----------
        center [float, 1d]: center coordinates of cluster
        coords [float, 2d, np]: cartesian coordinates of cluster

        Returns
        ----------
        coords_new [float, 2d]: cartesian coordinates of cluster
        """
        coords_new = center + coords
        return coords_new.tolist()
    
    def delete_duplicates_rotation(self, angles, coords, tol=1e-10):
        """
        delete same rotations
        
        Parameters
        ----------
        angles [float, 2d]: rotation angles
        coords [float, 2d, np]: cartesian coordinates of cluster
        tol [float, 0d]: same tolerance

        Returns
        ----------
        idx [int, 1d, np]: index of unique ratation
        """
        coords_rotates = []
        for angle in angles:
            alpha, beta, gamma = angle
            tmp = self.rotate_atom(alpha, beta, gamma, coords)
            coords_rotates.append(tmp)
        #delete same rotations
        idx = []
        num = len(angles)
        for i in range(num):
            coord_1 = coords_rotates[i]
            if i in idx:
                pass
            else:
                for j in range(i+1, num):
                    if j in idx:
                        pass
                    else:
                        coord_2 = coords_rotates[j]
                        if np.sum(np.abs(coord_1-coord_2)) < tol:
                            idx.append(j)
        all_idx = np.arange(num)
        idx = np.setdiff1d(all_idx, idx)
        return idx
    

if __name__ == '__main__':
    build = BuildNewAtom()
    am = AtomManipulate()
    strus = build.import_clusters()
    new_type, atom_num, cluster_atoms = build.get_atoms(strus)
    embedding, coords_zero, ave_radius, affinity, negativity = build.get_property(strus, cluster_atoms)
    angles = build.get_rotate_angles()
    idx = am.delete_duplicates_rotation(angles, coords_zero)
    build.export_rotate_angles(idx, angles)
    build.export_embedding(embedding)
    build.export_property(new_type, coords_zero, cluster_atoms, ave_radius, affinity, negativity)