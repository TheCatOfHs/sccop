import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(f'{os.getcwd()}/src')
from modules.utils import ListRWTools


class VisualizeGrid():
    #
    def __init__(self, grid_name):
        self.grid = grid_name
    
    #Show neighboring points of target point
    def near_points_plot_sub(self, point, near_idx, grid_coor, fig_num, fig_obj):
        """
        Plot neighboring points of center point
        
        Parameters
        ----------
        point ([int, 0d]): [center point]
        near_idx ([int, 2d, np]): [index of near points]
        grid ([float, 2d, np]): [coordinate of grid]
        fig_num ([int, 0d]): [number of fig in subplot]
        fig_obj ([obj]): [figure object]
        """
        points = np.arange(len(grid_coor))
        grid_points = list(set(points) - set([point]) - set(near_idx[point]))
        point_x, point_y, point_z = grid_coor[point]
        grid_x, grid_y, grid_z = grid_coor[grid_points].transpose()
        near_x, near_y, near_z = grid_coor[near_idx[point]].transpose()
        ax = fig_obj.add_subplot(2, 3, fig_num, projection='3d')
        ax.scatter(point_x, point_y, point_z, c='red', s=80)
        ax.scatter(grid_x, grid_y, grid_z, c='green', alpha=.5, s=10)
        ax.scatter(near_x, near_y, near_z, c='red', alpha=.3, s=60)
        if fig_num in [4, 5, 6]:
            ax.set_xlabel('a', fontsize=14)
            ax.set_ylabel('b', fontsize=14)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        if fig_num in [3, 6]:
            ax.set_zlabel('c', fontsize=14)
        else:
            ax.set_zticklabels([])
        ax.set_title('Point_' + str(point), size=14)
    
    def near_points_plot(self, point_list, num_near, near_idx, latt_vec, frac_coor, grid_name):
        """
        Figure of neighboring points
        
        Parameters
        ----------
        point_list ([int, 1d, np]): [list of center points]
        num_near ([int, 0d]): [number of neighboring points]
        near_idx ([int, 2d, np]): [index of near points]
        latt_vec ([float, 2d]): [lattice vector of grid]
        frac_coor ([float, 2d]): [fraction coordinate of grid]
        grid_name ([str, 0d]): [name of grid]
        """
        grid_coor = np.dot(frac_coor, latt_vec)
        plt.rc('font', family='times new roman')
        fig_obj = plt.figure(figsize=(10, 7))
        for fig_num, i in enumerate(point_list):
            self.near_points_plot_sub(i, near_idx[:, 0:num_near], grid_coor, fig_num+1, fig_obj)
        plt.legend(['center point', grid_name + ' grid', 'near points'], 
                    loc=(-2.5, 2.), shadow=True, fontsize=12)
        plt.subplots_adjust(wspace=0.04, hspace=0.15)
        fig_obj.savefig('data/grid/check/near_points_figs/grid_' + grid_name + '.png', dpi=300)

    def show_neighboring_points(self, num_near, point_list, grid_name):
        """
        Save figure of neighboring points
        
        Parameters
        ----------
        point_list ([int, 1d, np]): [list of center points]
        num_near ([int, 0d]): [number of neighboring points]
        grid_name ([str, 0d]): [name of grid]
        """
        file_prefix = self.prop_dir + grid_name
        latt_vec = self.import_mat(file_prefix + '_latt_vec.dat', float)
        near_idx = self.import_mat(file_prefix + '_nbr_idx.dat', int)
        frac_coor = self.import_mat(file_prefix + '_frac_coor.dat', float)
        self.near_points_plot(point_list, num_near, near_idx, latt_vec, frac_coor, grid_name)

    #Put test configuration into grid
    def import_test_POSCAR(self, dir, file):
        """
        Import test POSCAR saved in test POSCAR
        
        Parameters
        ----------
        dir ([str, 0d]): [directory of test POSCAR]
        file ([str, 0d]): [name of test POSCAR]
        """
        with open(dir + file, 'r') as f:
            file_content = f.readlines()
        POSCAR_str = [item.split() for item in file_content]
        test_vector_str = POSCAR_str[2:5]
        test_comp_str = file_content[5:8]
        test_coor_str = POSCAR_str[8:]
        test_coor = [[float(str) for str in line] for line in test_coor_str]
        test_latt_vec = [[float(str) for str in line] for line in test_vector_str]
        return np.dot(test_coor, test_latt_vec), test_comp_str
    
    def put_into_grid(self, test_coor, grid_coor):
        """
        Approximate test configuration in grid, 
        return corresponding index of grid point
        
        Parameters
        ----------
        test_coor ([float, 2d]): [cartesian coordinate of test configuration]
        grid_coor ([float, 2d]): [cartesian coordinate of grid point]
        """
        distance = np.zeros((len(test_coor), len(grid_coor)))
        for i, atom_coor in enumerate(test_coor):
            for j, point_coor in enumerate(grid_coor):
                distance[i, j] = np.sqrt(np.sum((atom_coor - point_coor)**2))
        test_in_grid_idx = list(map(lambda x: np.argmin(x), distance))
        return test_in_grid_idx
    
    def test_in_grid_plot(self, test_coor, test_in_grid_idx, grid_coor, test_file):
        """
        Figure of test configuration in grid

        Parameters
        ----------
        test_coor ([float, 2d]): [cartesian coordinate of test configuration]
        test_in_grid_idx ([int, 1d]): [index of test configuration in grid]
        grid_coor ([float, 2d]): [cartesian coordinate of grid]
        test_file ([str, 0d]): [file name of test configuration]
        """
        atoms = np.arange(0, len(grid_coor))
        test_in_grid = grid_coor[test_in_grid_idx]
        grid_atoms = list(set(atoms) - set(test_in_grid_idx))
        plt.rc('font', family='times new roman')
        ax = plt.subplots(subplot_kw = dict(projection='3d'))[1]
        coor_x, coor_y, coor_z = test_coor.transpose()
        in_grid_x, in_grid_y, in_grid_z = test_in_grid.transpose()
        grid_x, grid_y, grid_z = grid_coor[grid_atoms].transpose()
        ax.scatter(coor_x, coor_y, coor_z, c='red', s=60, label='test POSCAR')
        ax.scatter(in_grid_x, in_grid_y, in_grid_z, c='blue', s=40, label='POSCAR in grid')
        ax.scatter(grid_x, grid_y, grid_z, c='green', alpha=0.3, s=40, label=self.grid+' grid')
        ax.set_xlabel('a', fontsize=14)
        ax.set_ylabel('b', fontsize=14)
        ax.set_zlabel('c', fontsize=14)
        ax.legend(loc=(-.1, .95))
        rmsd = np.sqrt(np.sum((test_coor - test_in_grid)**2))
        ax.set_title('RMSD = ' + '{0:4.4f}'.format(rmsd) + ' $\AA$')
        plt.savefig('data/grid/check/POSCAR_test_in_grid/' + test_file + '_in_grid.png', dpi=300)
    
    def write_POSCAR_in_grid(self, frac_coor, grid_latt_str, comp_str, test_in_grid_idx, test_file):
        """
        Save POSCAR of test configuration in grid
        
        Parameters
        ----------
        frac_coor ([float, 2d]): [fraction coordinate of grid]
        grid_latt_str ([str, 0d]): [string of grid lattice vector]
        comp_str ([str, 0d]): [string of composition]
        test_in_grid_idx ([int, 1d]): [index of test configuration in grid]
        test_file ([str, 0d]): [file name of test configuration]
        """
        head_str = 'E = -1\n 1\n'
        test_in_grid = frac_coor[test_in_grid_idx]
        test_in_grid_str = self.matrix_to_str(test_in_grid, '{0:4.6f}')
        with open('data/grid/check/POSCAR_test_in_grid/' + test_file, 'w') as f:
            f.write(head_str + ''.join(grid_latt_str) + '\n' + ''.join(comp_str) + test_in_grid_str)
    
    def test_in_grid(self, dir, test_file, num, grid_name):
        """
        Test configuration POSCAR file and show it in grid

        Parameters
        ----------
        dir ([str, 0d]): [directory to save test_file]
        test_file ([str, 0d]): [name of test configuration]
        num ([int, 0d]): [index of configuration]
        grid_name ([str, 0d]): [name of grid]
        """
        file_dir = 'data/grid/Property_grid/'
        latt_vec = self.import_mat(file_dir + grid_name + '_latt_vec.dat', float)
        frac_coor = self.import_mat(file_dir + grid_name + '_frac_coor.dat', float)
        grid_coor = np.dot(frac_coor, latt_vec)
        test_coor, test_comp_str = self.import_test_POSCAR(dir, test_file)
        test_in_grid_idx = self.put_into_grid(test_coor, grid_coor)
        grid_latt_str = self.matrix_to_str(latt_vec, '{0:4.4f}')
        test_name = test_file.split('.')[0]
        self.test_in_grid_plot(test_coor, test_in_grid_idx, grid_coor, test_name)
        self.write_POSCAR_in_grid(frac_coor, grid_latt_str, test_comp_str, test_in_grid_idx, f'POSCAR-{num:03.0f}')
        return test_in_grid_idx
    
if __name__ == '__main__':
    rwtools = ListRWTools()
    if False:
        mae = rwtools.import_list2d('test/GaN_ZnO_3/100/validation.dat', float)
        x = np.arange(len(mae))
        y = [min(mae) for i in range(len(mae))]
        plt.rc('font', family='Times New Roman')
        figure_1 = plt.figure(figsize=(8, 6))
        ax_1 = figure_1.add_subplot(1, 1, 1)
        ax_1.scatter(x, mae)
        ax_1.tick_params(labelsize=16)
        ax_1.set_xlabel('Training Round', fontsize=24)
        ax_1.set_ylabel('Mean Absolute Error / eV', fontsize=24)
        plt.savefig('mp_mae.png', dpi=600)
    energy_buffer = []
    if True:
        for i in range(0, 35):
            dir = f'test/GaN_ZnO_6/vasp_out'
            with open(f'{dir}/Energy-{i:03.0f}.dat', 'r') as f:
                ct = f.readlines()
            energy_file = np.array(rwtools.str_to_list2d(ct, str))
            true_E = energy_file[:,1]
            true_E = [bool(i) for i in true_E]
            energys = energy_file[:,2][true_E]
            energys = [float(i) for i in energys]
            average = np.mean(energys)
            energy_buffer.append(average)
        x = np.arange(0, 35)
        plt.rc('font', family='Times New Roman')
        figure_1 = plt.figure(figsize=(8, 6))
        ax_1 = figure_1.add_subplot(1, 1, 1)
        ax_1.scatter(x, energy_buffer)
        ax_1.tick_params(labelsize=16)
        ax_1.set_xlabel('Recycling Round', fontsize=24)
        ax_1.set_ylabel('Energy / eV', fontsize=24)
        #plt.show()
        plt.savefig('energy.png', dpi=600)