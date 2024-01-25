import os, sys
import pandas as pd
import numpy as np

from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from core.input import *
from core.utils import *
        

if __name__ == '__main__':
    data = pd.read_csv(f'database/mp_20/val.csv').values
    
    for idx, cif in enumerate(data):
        stru = Structure.from_str(''.join(cif[7:-1]), fmt='cif')
        stru.to(filename=f'database/mp_20/val/POSCAR-{idx:05g}', fmt='poscar')
        print(idx)