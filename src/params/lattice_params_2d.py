import os, sys
import pandas as pd
import numpy as np


from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure, IStructure
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from pymatgen.core.periodic_table import Element, ElementBase
from pymatgen.util.coord import pbc_diff


def judge_crystal_system(a, b, gamma):
    if np.abs(a-b) < 1 and np.abs(gamma-120) < 5:
        crystal_system = 6
    elif np.abs(a-b) < 1 and np.abs(gamma-60) < 5:
        crystal_system = 6
    elif np.abs(a-b) < 1 and np.abs(gamma-90) < 5:
        crystal_system = 4
    elif np.abs(gamma-90) < 5:
        crystal_system = 3
    else:
        crystal_system = 1
    return crystal_system

def find_delta(coords, matrix):
    thicks = []
    for i, coord_1 in enumerate(coords):
        for coord_2 in coords[i:]:
            dis = pbc_diff(coord_1, coord_2)
            dis = np.linalg.norm(np.dot(dis, matrix)[-1])
            thicks.append(dis)
    return max(thicks)
    
def write_dat(file, ct):
    ct_string = '\n'.join([' '.join([str(i) for i in item]) for item in ct])
    with open(file, 'w') as obj:
        obj.writelines(ct_string)
        
if __name__ == '__main__':
    data = os.listdir('database/2d_material/CIF2D_no_vdW')
    
    collect = []
    for idx, cif in enumerate(data):
        radius, negativity, affinity = [], [], []
        write = True
        stru = Structure.from_file(f'database/2d_material/CIF2D_no_vdW/{cif}')
        
        z_dis = find_delta(stru.frac_coords, stru.lattice.matrix)
        if z_dis < 4:
            a, b, c, _, _, gamma = stru.lattice.parameters
            for i in stru.species:
                if i.atomic_radius == None:
                    write = False
                else:
                    radius += [i.atomic_radius.real]
                    negativity += [i.X]
                    affinity += [i.electron_affinity]
            if write:
                vec_a, vec_b = stru.lattice.matrix[:2]
                area = np.linalg.norm(np.cross(vec_a, vec_b))
                collect.append(radius + negativity + affinity + [area] + list(stru.lattice.parameters) + [judge_crystal_system(a, b, gamma)])
            print(idx)
    
    write_dat('lattice_params/lattice_train_2d.dat', collect)