#Base
dimension = int('DIM')
composition = 'XXX'
num_atom = [5, 10]
space_group = [[int('SG1'), int('SG2')]]

#2-Dimension settings
vacuum_space = 15
thickness = 0.1

#Sampling
num_latt = 72
num_Rand = 120
sg_per_latt = 10

#Recycling
num_recycle = 1
num_ml_list = [1]
num_poscars = 12
num_optims = 6
vasp_time_limit = 480

#Searching
latt_steps = 5
sa_steps = 80
num_jump = 1
num_path = 360

#Sample Select
num_models = 5
num_clusters = 60
ratio_min_energy = 0.5