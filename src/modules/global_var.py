import os

#Grid
cutoff = 6
latt_vec = [[5., 0, 0], [0, 5., 0], [0, 0, 5.]]
grain = [.1, .1, .1]

#Mutate
num_mutate = 6
mut_ratio = 0.9
latt_mu = 0.
latt_sigma = 0.2

#Recycling
num_round = 30
num_poscars = 30

#Training
orig_atom_fea_len = 92
nbr_bond_fea_len = 41

#Searching
T = .1
decay = .99
steps = 150
num_paths = 180
min_bond = 1.8
wait_time = 200

#Sample Select
num_models = 5
n_components = 2
n_clusters = 60
ratio_min_energy = 0.2
ratio_max_std = 0.2

#Server
num_gpus = 2
gpu_node = 'node151'
nodes = [131, 132, 133, 134, 135, 136]
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

#Save directory
poscar_dir = 'data/poscar' 
model_dir = 'data/ppmodel'
search_dir = 'data/search'
vasp_out_dir = 'data/vasp_out'
grid_prop_dir = 'data/grid/property'
grid_poscar_dir = 'data/grid/poscar'
ccop_out_dir = 'data/poscar/CCOP'
optim_strs_path = 'data/poscar/optim_strs'

#DFT directory
vasp_files_path = 'libs/VASP_inputs'
sing_point_energy_dir = 'libs/VASP_inputs/SinglePointEnergy'

#File
log_file = 'data/system.log'
elements_file = 'data/elements.dat'
atom_init_file = 'data/atom_init.dat'
KPOINTS_file = 'vasp/KPOINTS'
bandconf_file = 'vasp/bandconf'
pretrain_model = 'database/mp_20/pretrain_model/model_best.pth.tar'

#Property file
optim_vasp_path = f'{vasp_out_dir}/optim_strs'
dielectric_path = f'{optim_vasp_path}/dielectric'
elastic_path = f'{optim_vasp_path}/elastic'
energy_path = f'{optim_vasp_path}/energy'
pbe_band_path = f'{optim_vasp_path}/pbe_band'
phonon_path = f'{optim_vasp_path}/phonon'