import os, sys
import numpy as np

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

sys.path.append(f'{os.getcwd()}/src')
from core.global_var import *
from core.utils import *
from core.grid_sampling import PlanarSpaceGroup


if __name__ == '__main__':
    crystal_system = 2
    
    #space group
    if crystal_system == 0:
        groups = [1, 2]
    if crystal_system == 2:
        groups = [3, 4, 5, 25, 28, 32, 35]
    if crystal_system == 3:
        groups = [75, 99, 100]
    if crystal_system == 5:
        groups = [143, 156, 157, 168, 183]
    
    a, b, c = 5, 5, 10
    alpha, beta, gamma = 90, 90, 90
    latt = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    
    plane = PlanarSpaceGroup()
    sg = groups[1]
    all_grid, mapping = plane.get_grid_points(sg, grain, latt)
    print(len(all_grid))
    atoms = [1 for _ in range(len(all_grid))]
    stru = Structure(latt, atoms, all_grid)
    stru.to(filename='test/POSCAR-p1g1', fmt='poscar')