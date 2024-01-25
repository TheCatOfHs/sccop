from re import L
import time
import numpy as np

from pymatgen.core.structure import Structure


def system_echo(ct):
    """
    write system log
    
    Parameters
    ----------
    ct [str, 0d]: content
    """
    echo_ct = time.strftime("%Y-%m-%d %H:%M:%S",
                            time.localtime()) + ' -- ' + ct
    print(echo_ct)
    with open('log', 'a') as obj:
        obj.write(echo_ct + '\n')

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

def import_dat(file):
    with open(file, 'r') as obj:
        ct = obj.readlines()
    poscars = [i.split()[0] for i in ct]
    return poscars
    
def write_dat(file, ct):
    ct_string = '\n'.join([' '.join([str(i) for i in item]) for item in ct])
    with open(file, 'w') as obj:
        obj.writelines(ct_string)
    
    
if __name__ == '__main__':
    poscars = import_dat('Energy.dat')
    
    counter, wrong, collect = 0, 0, []
    for idx, poscar in enumerate(poscars):
        try:
            radius, negativity, affinity = [], [], []
            write = True
            stru = Structure.from_file(f'poscars/{poscar}')
            for i in stru.species:
                if i.atomic_radius == None:
                    write = False
                    break
                elif np.isnan(i.X):
                    write = False
                    break
                elif not isinstance(i.electron_affinity, float):
                    write = False
                    break
                radius += [i.atomic_radius.real]
                negativity += [i.X]
                affinity += [i.electron_affinity]
            if write:
                params = list(stru.lattice.parameters)
                collect.append(radius + negativity + affinity + [stru.volume] + params + [judge_crystal_system(params)])
            counter += 1
            if counter == 1000:
                system_echo(f'Current: {idx}')
                counter = 0
        except:
            system_echo(f'Wrong: {poscar}')
            wrong += 1
        continue
    
    system_echo(f'Wrong number: {wrong}')
    write_dat('lattice_val_3d_OQMD.dat', collect)