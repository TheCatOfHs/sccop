'''
@author: Liang Hanpu
@create date: 2019/08/31
@description: Processing the thermelectronic data and create file CONTROL.
	      Your need to have POSCAR, born_charges.dat, dielectric.dat, FORCE_CONSTANTS_2RD, FORCE_CONSTANTS_3ND.
'''


import numpy as np 
from os import path

def read_born(file_name):
    if path.exists(file_name) == False:
        print('File '+file_name+' not found!')
        exit(0)
    with open(file_name, 'r') as obj:
        born = obj.readlines()
    born_new = []
    for line in born:
        if line.split()[0] in ['1', '2', '3']:
            born_new.append(line)
    out_born = ['  '.join(line.split()[1:]) for line in born_new]
    return out_born

def read_POSCAR():
    if path.exists('POSCAR') == False:
        print('File POSCAR not found!')
        exit(0)
    with open('POSCAR', 'r') as obj:
        poscar = obj.readlines()
    # lattice contants
    lattice = ['\t'.join(line.split()) for line in poscar[2:5]]
    # the name of all atoms
    atom_name = poscar[5].split()
    # the number of all atoms
    atom_num = np.array([int(item) for item in poscar[6].split()])
    # the position of each atom
    position = ['\t'.join(line.split()[0:3]) for line in poscar[8:8+np.sum(atom_num)]]
    return (lattice, atom_name, atom_num, position)

def read_diele(file_name):
    if path.exists(file_name) == False:
        print('File '+file_name+' not found!')
        exit(0)
    with open(file_name, 'r') as obj:
        diele = obj.readlines()
    diele = ['\t'.join(line.split()) for line in diele[0:3]]
    return diele

def generate_CONTROL(born, lattice, atom_name, atom_num, diele, position):
    elements = ''
    types = ''
    for i, item in enumerate(atom_name):
        elements += "\""+item+"\" "
        for j in range(atom_num[i]):
            types += str(i+1)+' '
    pos = ''
    for i in range(np.sum(atom_num)):
        pos += '\n    positions(:,%d)=%s,'%(i+1,position[i])
    born_ct = ''
    count = 0
    for i in range(np.sum(atom_num)):
        for j in range(3):
            born_ct += '\n    born(:,%d,%d)=%s,'%(j+1,i+1,born[count])
            count += 1
    control = """&allocations
    nelements=%d,
    natoms=%d,
    ngrid(:)=10 10 5
&end
&crystal
    lfactor=0.1,
    lattvec(:,1)=%s,
    lattvec(:,2)=%s,
    lattvec(:,3)=%s,
    elements=%s
    types=%s,%s
    epsilon(:,1)=%s,
    epsilon(:,2)=%s,
    epsilon(:,3)=%s,%s
    scell(:)=4 4 1
&end
&parameters
    T=T-place.
    scalebroad=1.0
&end
&flags
    nonanalytic=.TRUE.
    nanowires=.FLASE.
    convergence=.FALSE.
&end"""%(len(atom_name),np.sum(atom_num),lattice[0],lattice[1],lattice[2],elements,types,pos,diele[0],diele[1],diele[2],born_ct)
    with open('CONTROL', 'w') as obj:
        obj.write(control)

if __name__ == '__main__':
    born = read_born('born_charges.dat')
    lattice, atom_name, atom_num, position = read_POSCAR()
    diele = read_diele('dielectric.dat')
    generate_CONTROL(born, lattice, atom_name, atom_num, diele, position)
