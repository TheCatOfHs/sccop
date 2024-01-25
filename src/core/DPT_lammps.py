import os, argparse

from pymatgen.core.structure import Structure
from pymatgen.io.lammps.data import LammpsData


class LammpsDPT():
    #lammps data process tool
    def __init__(self):
        pass
    
    def poscar2lammps(self, path):
        """
        poscar transfer to lammps structure
        
        Parameters
        ----------
        path [str, 0d]: lammps path
        """
        stru = Structure.from_file(f'{path}/POSCAR')
        lammps_stru = LammpsData.from_structure(stru, atom_style='atomic')
        lammps_stru.write_file(f'{path}/lammps.inp') 
        
    def lammps2poscar(self, path):
        """
        lammps structure transfer to poscar
        
        Parameters
        ----------
        path [str, 0d]: lammps path
        """
        self.get_lammps_opt(path)
        lammps_stru = LammpsData.from_file(f'{path}/lammps_opt.inp', atom_style='atomic')
        stru = lammps_stru.structure
        stru.to(filename='POSCAR', fmt='vasp')
    
    def get_lammps_opt(self, path):
        """
        get optimized structure of lammps
        
        Parameters
        ----------
        path [str, 0d]: lammps path
        """
        atom_type, mass = self.get_atoms_from_inp(path)
        latt = self.get_latt_from_log(path)
        coords = self.get_coords_from_dump(path)
        opt = atom_type + ['\n'] + latt + ['\n'] + mass + ['\n'] + coords 
        opt_str = ' '.join(opt)
        with open(f'{path}/lammps_opt.inp', 'w') as obj:
            obj.writelines(opt_str)
    
    def get_atoms_from_inp(self, path):
        """
        get atom type and mass from inp file
        
        Parameters
        ----------
        path [str, 0d]: lammps path

        Returns
        ----------
        atom_type [str, 1d]: atom type
        mass [str, 1d]: atom mass
        """
        with open(f'{path}/lammps.inp', 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if line.startswith('Masses'):
                a = i
            elif line.startswith('Atoms'):
                b = i + 1
        atom_type = lines[:5]
        mass = lines[a:b]
        return atom_type, mass
    
    def get_ave_energy_from_log(self, log_file):
        """
        get average energy from log file
        
        Parameters
        ----------
        path [str, 0d]: lammps path
        
        Returns
        ----------
        log_file [str, 0d]:
        ave_E [float, 0d]: average energy
        """
        atom_num, store = 0, []
        with open(log_file, 'r') as file:
            lines = file.readlines()
        start_reading = False
        for line in lines:
            line_str = line.strip()
            if line_str.startswith('Step'):
                start_reading = True
            if line_str == '':
                start_reading = False
            if start_reading:
                item = line_str.split()
                if item[0] == 'Loop':
                    atom_num = int(item[-2])
                else:
                    store.append(item[-1])
        #get average energy
        energy = store[-1]
        if atom_num > 0:
            ave_E = float(energy)/atom_num
        else:
            ave_E = 1e6
        return ave_E
    
    def get_latt_from_log(self, path):
        """
        get lattice parameters from log file

        Parameters
        ----------
        path [str, 0d]: lammps path

        Returns
        ----------
        latt [str, 1d]: lattice parameters
        """
        store = []
        with open(f'{path}/log.lammps', 'r') as file:
            lines = file.readlines()
        start_reading = False
        for line in lines:
            line_str = line.strip()
            if line_str.startswith('Step'):
                start_reading = True
            if line_str == '':
                start_reading = False
            if start_reading:
                item = line_str.split()
                if item[0] == 'Loop':
                    pass
                else:
                    store.append(item[1:-1])
        #get lattice parameters
        xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = store[-1]
        latt = [[xlo, xhi, 'xlo', 'xhi', '\n'],
                [ylo, yhi, 'ylo', 'yhi', '\n'],
                [zlo, zhi, 'zlo', 'zhi', '\n'],
                [xy, xz, yz, 'xy', 'xz', 'yz', '\n']]
        latt_str = [' '.join(i) for i in latt]
        return latt_str
    
    def get_coords_from_dump(self, path):
        """
        get coordinates from dump file
        
        Parameters
        ----------
        path [str, 0d]: lammps path
        
        Returns
        ----------
        coords [str, 2d]: atom coordinates
        """
        with open(f'{path}/dump.out', 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if line.startswith('ITEM: ATOM'):
                a = i + 1
                break
        #sorted atoms by ID
        coords = sorted(lines[a:], key=lambda x: x[0])
        return coords
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--flag', type=int)
    args = parser.parse_args()
    flag = args.flag
    
    lammps = LammpsDPT()
    paths = [i for i in os.listdir() if i.startswith('calc')]
    if flag == 0:
        for path in paths:
            lammps.poscar2lammps(path)
    elif flag == 1:
        for path in paths:
            lammps.lammps2poscar(path)