#Grid
cutoff = 10
num_min_atom = 2
num_max_atom = 5
grain_coarse = [.5, .5, 100.]
grain_fine = [.25, .25, 100.]
add_vacuum = True

#Mutate
num_mutate = 6
mut_ratio = 0.9
mut_freq = 1
latt_mu = 0.
latt_sigma = 0.2

#Recycling
num_recycle = 2
num_ml_list = [1, 2]
num_seed = 40
num_poscars = 36
num_optims = 6

#Initial Samples
component = 'XXX'
num_RCSD = 36
num_Rand = 120
num_sampling = 4000
len_mu = 5
len_sigma = 1
ang_mu = 90
ang_sigma = 15
system_weight = [0., 0., 0., 0., 1., 0., 0.]

#Training
train_batchsize = 128
train_epochs = 120
orig_atom_fea_len = 92
nbr_bond_fea_len = 41
use_pretrain_model = False

#Searching
T = .1
decay = .99
steps = 300
num_paths_min = 40
num_paths_rand = 80
num_paths_atom = 30
num_paths_order = 30
min_bond = 1.2
wait_time = 300

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