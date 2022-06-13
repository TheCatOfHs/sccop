#Grid
cutoff = 8
num_min_atom = 10
num_max_atom = 20
grain = [.5, .5, 1.2]
plane_upper = [100, 100, 1]

#Dimension
num_dim = 2
add_vacuum = True
vacuum_space = 15
add_vdW = False

#Recycling
num_recycle = 2
num_ml_list = [2, 2]
num_seed = 40
num_poscars = 12
num_optims = 6

#Initial Samples
component = 'XXX'
num_latt = 72
num_Rand = 120
num_ave_sg = 1
num_per_sg = 2
len_mu = 7
len_sigma = 1
len_lower = 6
len_upper = 8
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
decay = .95
latt_steps = 3
sa_steps = 100
num_jump = 2
num_path_min = 60
num_path_rand = 40
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