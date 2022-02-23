import numpy as np
from pymatgen.core.structure import Structure

def single(poscar):
    crystal = Structure.from_file(poscar)
    all_nbrs = crystal.get_all_neighbors(6)
    all_nbrs = [sorted(nbrs, key = lambda x: x[1]) for nbrs in all_nbrs]
    num_near = min(map(lambda x: len(x), all_nbrs))
    nbr_idx, nbr_dis = [], []
    for nbr in all_nbrs:
        idx = list(map(lambda x: x[2], nbr[:num_near]))[:12]
        dis = list(map(lambda x: x[1], nbr[:num_near]))[:12]
        nbr_idx.append(idx)
        nbr_dis.append(dis)
    nbr_idx, nbr_dis = np.array(nbr_idx), np.array(nbr_dis)
    return nbr_dis

if __name__ == '__main__':
    print(single('data/poscar/000/POSCAR-000-0001-131'))