import os

#Grid
cutoff = 5
latt_vec = [[3.24, 0, 0], [0, 5.54, 0], [0, 0, 5.61]]
grain = [.1, .1, .1]

#Mutate
num_mutate = 6


#Recycling
num_round = 5
num_poscars = 10

#Training


orig_atom_fea_len = 92
nbr_bond_fea_len = 41

#Searching
T = .1
decay = .99
steps = 150
num_paths = 180
min_bond = 1.4
wait_time = 240

#Sample Select
num_models = 5
n_components = 2
n_clusters = 60
ratio_min_energy = 0.2
ratio_max_std = 0.2

#Server
num_gpus = 2
nodes = [131, 132, 133, 134, 135, 136]
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

#Save directory
poscar_dir = 'data/poscar' 
model_dir = 'data/ppmodel'
search_dir = 'data/search'
vasp_out_dir = 'data/vasp_out'
grid_prop_dir = 'data/grid/property'
grid_poscar_dir = 'data/grid/poscar'

#DFT directory
sing_point_energy_dir = 'libs/VASP_inputs/SinglePointEnergy'

#File
log_file = 'data/system.log'
elements_file = 'data/elements.dat'
atom_init_file = 'data/atom_init.dat'