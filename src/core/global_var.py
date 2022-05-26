#Grid
cutoff = 10
num_min_atom = 5
num_max_atom = 10
grain = [.4, .4, 1.2]
plane_upper = [100, 100, 1]

#Dimension
num_dim = 2
add_vacuum = True
vacuum_space = 15
add_vdW = False

#Recycling
num_recycle = 2
num_ml_list = [1, 1]
num_seed = 40
num_poscars = 12
num_optims = 6

#Initial Samples
component = 'XXX'
num_latt = 72
num_Rand = 120
num_ave_sg = 1
num_per_sg = 10
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
num_jump = 2
num_path_min = 80
num_path_rand = 80
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