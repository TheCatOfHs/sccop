import re
import argparse
import numpy as np
from cspd import atomic_structure_generator


parser = argparse.ArgumentParser()
parser.add_argument('--component', type=str)
parser.add_argument('--ndensity', type=float)
parser.add_argument('--mindis', type=float)
args = parser.parse_args()


def min_dis_mat(component, mindis):
    str_list = re.findall(r'[A-Za-z]+', component)
    num_atom = len(str_list)
    dis_mat = np.zeros((num_atom, num_atom)) + mindis
    return list(dis_mat)
    
    
if __name__ == '__main__':
    component = args.component
    ndensity = args.ndensity
    mindis = args.mindis
    
    dis_mat = min_dis_mat(component, mindis)
    atomic_structure_generator(
        symbols=component,
        lw=True,
        format='vasp',
        #fu=[1, 4],
        cspd_file='CSPD.db',
        #sgn=[1,225],
        ndensity=ndensity,
        #volume=30.0,
        mindis=dis_mat,
        #nstr=300,
        maxatomn=20,
    )