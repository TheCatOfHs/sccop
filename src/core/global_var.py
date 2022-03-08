#Grid
cutoff = 7
grain_coarse = [.5, .5, 5.]
grain_fine = [.5, .5, 5.]

#Mutate
num_mutate = 6
mut_ratio = 0.9
mut_freq = 2
latt_mu = 0.
latt_sigma = 0.2

#Recycling
num_recycle = 2
num_round = 12
num_seed = 30
num_poscars = 30
num_optims = 12
ratio_round = 0.8

#Initial Samples
component = 'B1C3'
ndensity = 0.1 
min_dis_CSPD = 1.4
num_CSPD = 48
num_rand = 10000
num_initial = 100

#Training
train_batchsize = 128
train_epochs = 120
orig_atom_fea_len = 92
nbr_bond_fea_len = 41

#Searching
T = .1
decay = .99
steps = 300
num_paths_min = 90
num_paths_rand = 90
min_bond = 1.4
threshold = -7.5
wait_time = 300

#Sample Select
num_models = 5
num_components = 2
num_clusters = 60
ratio_min_energy = 0.2
ratio_max_std = 0.2

#Server
num_gpus = 2
gpu_node = 'node151'
nodes = [131, 132, 133, 134, 135, 136]