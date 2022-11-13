# Symmetry Crystal Combinatorial Optimization Program

This software package implements the Symmetry Crystal Combinatorial Optimization Program (SCCOP) that predicts crystal structure of specific composition. 

SCCOP combines graph neural network and DFT calculation to accelerate the search of crystal structure.
The following paper describes the details of the SCCOP framework:

[Crystal structure prediction and property related feature extraction by graph deep learning](XXX)
![](images/SCCOP.png)

## Table of Contents

- [How to cite](#how-to-cite)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Confiugration](#server-and-absolute-path-configuration)
  - [Customize initial search file](#define-a-customized-search-file)
  - [Submit SCCOP job on cluster](#submit-sccop-job)
  - [Successful example](#successful-example)
- [Data](#data)
- [Authors](#authors)
- [License](#license)

## How to cite

Please cite the following work if you want to use SCCOP.

```
@article{XXX,
  title = {Crystal structure prediction and property related feature extraction by graph deep learning},
  author = {Li, Chuannan and Liang, Hanpu and Zhang, Xie and Lin, Zijing and Wei, Su-Huai},
  journal = {XXX},
  volume = {XXX},
  issue = {XXX},
  pages = {XXX},
  numpages = {XXX},
  year = {XXX},
  month = {XXX},
  publisher = {XXX},
  doi = {XXX},
  url = {XXX}
}
```

##  Prerequisites

Package requirements:

- [PyTorch (1.8.1)](http://pytorch.org/)
- [scikit-learn (1.0.1)](http://scikit-learn.org/stable/)
- [pymatgen (2022.5.26)](http://pymatgen.org/)
- [VASP (5.4.4)](https://www.vaspweb.org/)
- [DPT (0.8.3)](https://github.com/HanpuLiang/Data-Processing-Toolkit)
- [paramiko (2.7.2)](https://www.paramiko.org/)

Hardware requirements:

- [GPU node](https://en.wikipedia.org/wiki/GPU_cluster)
- [CPU node](https://en.wikipedia.org/wiki/Server_(computing))

## Usage
### Server and Absolute Path Configuration
In SCCOP, we use shell command `ssh` and `scp` to transfer files between nodes, you should make sure the `ssh` between nodes without password, here we provide one way as follows:

```bash
cd ~/.ssh
ssh-keygen -t rsa -b 4096
ssh-copy-id -i ~/.ssh/id_rsa.pub user@nodeXXX
```

Then, you need to specify server info and absolute path in `src/core/path.py`, thus GPU node can send jobs to cpu nodes via [paramiko](https://www.paramiko.org/).

```diff
[Server]
# GPU number and name of gpu node
num_gpus = 2 
gpu_node = 'node151'
# List of cpu nodes, thus the cpu name is, e.g., 'nodeXXX' 
nodes = [131, 132, 133, 134, 135, 136] 

[Absolute path]
# Path of SCCOP on GPU node
SCCOP_path = '/local/sccop' 
# Directory of SCCOP on CPU nodes
CPU_local_path = '/local' 
# Path of openmpi
MPI_2d_path = '/opt/openmpi-1.6.3/bin/mpirun' 
MPI_3d_path = '/opt/intel/impi/4.0.3.008/intel64/bin/mpirun' 
# Call VASP for DFT calculation
VASP_2d = f'{MPI_2d_path} -np 48 vasp_relax_ab' 
VASP_3d = f'{MPI_3d_path} -np 48 vasp' 
```

**Note:** the SCCOP should under the `/local` directory of GPU node, e.g., `/local/sccop` which includes `sccop/src`, `sccop/data` and `sccop/libs`. For researchers who want to change the submission of VASP jobs, see the code in `src/core/sub_vasp.py`.

### Define a Customized Search File

To run SCCOP for desired composition, you need to define a customized initial search file, i.e., the `src/core/global_var.py` should be:

```diff
[Base]
# Dimension of target composition
dimension = 2
# The chemical formula of the compound, e.g., 'B1C3'
composition = XXX
# Number of atoms in unit cell
num_atom = [5, 10]
# Search space group
space_group = [[2, 17]]

[2-Dimension settings]
# Vacuum space layer
vacuum_space = 15
# Puckered structure
thickness = 0.1

[Sampling]
# Number of initial lattice
num_latt = 72
# Number of initial structures sent to VASP
num_Rand = 120
# Average space group per lattice
sg_per_latt = 10

[Recycling]
# Number search recycle
num_recycle = 1
# List of ML search and optimize in each recycle
num_ml_list = [1]
# Number of structures that sent to VASP optimize
num_poscars = 12
# High accuracy optimized by VASP
num_optims = 6
# VASP time limit
vasp_time_limit = 480

[Searching]
# Total SA steps = latt_steps*sa_steps
latt_steps = 5
sa_steps = 80
# Metropolis judge interval
num_jump = 1
# Number of SA path
num_path = 360

[Sample Select]
# Number of models to predict energy
num_models = 5
# Number of clusters
num_clusters = 60
ratio_min_energy = 0.5
```

### Submit SCCOP Job

If you install packages in [prerequisites](#prerequisites), and finish the [server and path configuration](#server-and-absolute-path-configuration) and [initial search file](#define-a-customized-search-file), then you need to make sure the `sccop` is under `/local` of GPU node, and you can `cd /local/sccop` to submit sccop job by:

```bash
nohup python src/main.py >& log&
```

After searching, you will get three important files.

- `data/system.dat`: stores the searching process of SCCOP.
- `data/poscars/optim_strus`: stores the POSCAR of searched structures.
- `data/vasp_out/optim_strus/energy/Energy.dat`: stores the energy of searched structures.

### Successful Example

Here we give one successful example of SCCOP, you can find the log file `system.log` and `POSCAR` of searched structures in `examples/`.

Initial sampling structures by symmetry in parallel.
![](images/BC3_log_1.png)

Update prediction model and optimize structures by ML-SA in parallel.
![](images/BC3_log_2.png)

Optimize structures by VASP in parallel.
![](images/BC3_log_3.png)

## Data

We have applied SCCOP to systematic search 82 compositions of B-C-N system, and newly discovered 28 stable low energy configurations, and you can download the data from [B-C-N_POSCAR](BCN_stable_poscars.zip).

## Authors

This software was primarily written by Chuannan Li and Hanpu Liang. 

## License

SCCOP is released under the MIT License.