#Server
num_gpus = 2
gpu_node = 'nodeXXX'
nodes = [XXX, XXX]

#Absolute path
SCCOP_path = '/local/sccop'
CPU_local_path = '/local'
MPI_2d_path = 'path_mpi_2d'
MPI_3d_path = 'path_mpi_3d'
VASP_2d = f'{MPI_2d_path} -np 48 vasp_relax_ab'
VASP_3d = f'{MPI_3d_path} -np 48 vasp'

#Save directory
poscar_path = 'data/poscar' 
model_path = 'data/ppmodel'
search_path = 'data/search'
vasp_out_path = 'data/vasp_out'
grid_path = 'data/grid'
json_path = 'data/grid/json'
buffer_path = 'data/grid/buffer'
sccop_out_path = 'data/poscar/SCCOP'
optim_strus_path = 'data/poscar/optim_strus'
init_strus_path = 'data/poscar/initial_strus'

#DFT directory
vasp_files_path = 'libs/VASP_inputs'
sing_point_energy_path = 'libs/VASP_inputs/SinglePointEnergy'

#File
log_file = 'data/system.log'
elements_file = 'data/elements.dat'
atom_init_file = 'data/atom_init.dat'
pretrain_model_2d = 'data/pretrain/model_2d.pth.tar'
pretrain_model_3d = 'data/pretrain/model_3d.pth.tar'

#Property file
optim_vasp_path = f'{vasp_out_path}/optim_strus'
energy_path = f'{optim_vasp_path}/energy'