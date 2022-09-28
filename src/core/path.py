#Server
num_gpus = 2
gpu_node = 'node151'
nodes = [131, 132, 133, 134, 135, 136]

#Absolute path
SCCOP_path = '/local/sccop'
CPU_local_path = '/local'
MPI_2d_path = '/opt/openmpi-1.6.3/bin/mpirun'
MPI_3d_path = '/opt/intel/impi/4.0.3.008/intel64/bin/mpirun'
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
pretrain_model = 'data/pretrain/model_2d.pth.tar'

#Property file
optim_vasp_path = f'{vasp_out_path}/optim_strus'
energy_path = f'{optim_vasp_path}/energy'