#Save directory
poscar_path = 'data/poscar' 
model_path = 'data/ppmodel'
search_path = 'data/search'
vasp_out_path = 'data/vasp_out'
record_path = 'data/record'
grid_path = 'data/grid'
ccop_out_path = 'data/poscar/CCOP'
optim_strus_path = 'data/poscar/optim_strus'
init_strus_path = 'data/poscar/initial_strus'

#DFT directory
vasp_files_path = 'libs/VASP_inputs'
sing_point_energy_path = 'libs/VASP_inputs/SinglePointEnergy'

#File
log_file = 'data/system.log'
elements_file = 'data/elements.dat'
atom_init_file = 'data/atom_init.dat'
KPOINTS_file = 'data/post/KPOINTS'
bandconf_file = 'data/post/bandconf'
pretrain_model = 'data/pretrain/model_2d.pth.tar'

#Property file
optim_vasp_path = f'{vasp_out_path}/optim_strs'
dielectric_path = f'{optim_vasp_path}/dielectric'
elastic_path = f'{optim_vasp_path}/elastic'
energy_path = f'{optim_vasp_path}/energy'
pbe_band_path = f'{optim_vasp_path}/pbe_band'
phonon_path = f'{optim_vasp_path}/phonon'
thermalconductivity_path = f'{optim_vasp_path}/thermalconductivity'

#adsorb file
adsorb_path = 'data/adsorb'
anode_strus_path = f'{adsorb_path}/anode_strus'
anode_energy_path = f'{adsorb_path}/anode_energy'
adsorb_strus_path = f'{adsorb_path}/adsorb_strus'
adsorb_energy_path = f'{adsorb_path}/adsorb_energy'
adsorb_analysis_path = f'{adsorb_path}/adsorb_analysis'

neb_path = f'{adsorb_path}/neb_path'
neb_analysis_path = f'{adsorb_path}/neb_analysis'