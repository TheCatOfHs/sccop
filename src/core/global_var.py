#Grid
cutoff = 8
num_min_atom = 5
num_max_atom = 10
grain_coarse = [.5, .5, 1.2]
grain_fine = [.25, .25, 1.2]
plane_upper = [100, 100, 1]

#Dimension
num_dim = 2
add_vacuum = True
vacuum_space = 15
add_vdW = False

#Mutate
num_mutate = 6
mut_ratio = 0.9
mut_freq = 1
latt_mu = 0.
latt_sigma = 0.2
free_aix = [1, 1, 0]

#Recycling
num_recycle = 2
num_ml_list = [1, 2]
num_seed = 40
num_poscars = 12
num_optims = 6

#Initial Samples
component = 'B1C3'
num_latt = 36
num_Rand = 120
num_sampling = 5
len_mu = 5
len_sigma = 1
len_lower = 4
len_upper = 6
ang_mu = 90
ang_sigma = 10
system_weight = [1/4, 0, 1/4, 1/4, 0, 1/4, 0]

#Training
train_batchsize = 128
train_epochs = 120
orig_atom_fea_len = 92
nbr_bond_fea_len = 41
use_pretrain_model = True

#Searching
T = .1
decay = .99
steps = 300
num_move = 3
num_paths_min = 40
num_paths_rand = 80
num_paths_atom = 30
num_paths_order = 30
min_bond = 1.2

#Sample Select
num_models = 5
num_components = 3
num_clusters = 60
ratio_min_energy = 0.5
ratio_max_std = 0.5

#Server
num_gpus = 2
gpu_node = 'node151'
nodes = [131, 132, 133, 134, 135, 136]