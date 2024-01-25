import os, sys
import pandas as pd
import numpy as np

from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from core.input import *
from core.utils import *


def judge_crystal_system(latt_paras):
    a, b, c, alpha, beta, gamma = latt_paras
    if np.abs(a-b) < 1 and np.abs(a-c) < 1 and np.abs(b-c) < 1 and np.abs(alpha-90) < 5 and np.abs(beta-90) < 5 and np.abs(gamma-90) < 5:
        crystal_system = 7
    elif np.abs(a-b) < 1 and np.abs(alpha-90) < 5 and np.abs(beta-90) < 5 and np.abs(gamma-120) < 5:
        crystal_system = 6
    elif np.abs(a-b) < 1 and np.abs(a-c) < 1 and np.abs(b-c) < 1 and np.abs(alpha-beta) < 5 and np.abs(alpha-gamma) < 5 and np.abs(beta-gamma) < 5:
        crystal_system = 6
    elif np.abs(a-b) < 1 and np.abs(alpha-90) < 5 and np.abs(beta-90) < 5 and np.abs(gamma-90) < 5:
        crystal_system = 4
    elif np.abs(alpha-90) < 5 and np.abs(beta-90) < 5 and np.abs(gamma-90) < 5:
        crystal_system = 3
    elif np.abs(alpha-90) < 5 and np.abs(gamma-90) < 5:
        crystal_system = 2
    else:
        crystal_system = 1
    return crystal_system

def judge_crystal_system(sg):
    if sg in range(1, 3):
        crystal_system = 1
    elif sg in range(3, 16):
        crystal_system = 2
    elif sg in range(16, 75):
        crystal_system = 3
    elif sg in range(75, 143):
        crystal_system = 4
    elif sg in range(143, 168):
        crystal_system = 6
    elif sg in range(168, 195):
        crystal_system = 6
    elif sg in range(195, 231):
        crystal_system = 7
    return crystal_system


def write_dat(file, ct):
    ct_string = '\n'.join([' '.join([str(i) for i in item]) for item in ct])
    with open(file, 'w') as obj:
        obj.writelines(ct_string)
    
if __name__ == '__main__':
    data = pd.read_csv(f'database/mp_20/val.csv').values
    
    collect = []
    for idx, cif in enumerate(data):
        radius, negativity, affinity = [], [], []
        write = True
        stru = Structure.from_str(''.join(cif[7:-1]), fmt='cif')
        for i in stru.species:
            if i.atomic_radius == None:
                write = False
            else:
                radius += [i.atomic_radius.real]
                negativity += [i.X]
                affinity += [i.electron_affinity]
        if write:
            collect.append(radius + negativity + affinity + [stru.volume] + list(stru.lattice.parameters) + [judge_crystal_system(int(cif[-1]))])
        print(idx)
    write_dat('lattice_params/lattice_val_3d.dat', collect)