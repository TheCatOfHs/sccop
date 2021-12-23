import os, sys
import argparse

sys.path.append(f'{os.getcwd()}/src')
from modules.workers import Search


parser = argparse.ArgumentParser()
parser.add_argument('--atom_pos', type=int, nargs='+')
parser.add_argument('--atom_type', type=int, nargs='+')
parser.add_argument('--round', type=int)
parser.add_argument('--path', type=int)
parser.add_argument('--node', type=int)
parser.add_argument('--grid_name', type=int)
parser.add_argument('--model_name', type=str)
args = parser.parse_args()
            

if __name__ == "__main__":
    atom_pos = args.atom_pos
    atom_type = args.atom_type
    round = args.round
    path = args.path
    node = args.node
    grid_name = args.grid_name
    model_name = args.model_name
    
    #Search
    worker = Search(round, grid_name)
    worker.explore(atom_pos, atom_type, model_name, path, node)