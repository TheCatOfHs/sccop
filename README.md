# Symmetry Crystal Combinatorial Optimization Program

This software package implements the Symmetry Crystal Combinatorial Optimization Program (SCCOP) that predicts crystal structure of specific composition. 

SCCOP combines graph neural network and DFT calculation to accelerate the search of crystal structure.
The following paper describes the details of the SCCOP framework:

[Crystal structure prediction and property related feature extraction by graph deep learning](XXX)
![](images/SCCOP.png)

SCCOP now only supports the search for 2D materials, and the supporting for 3D is under construction.

## Table of Contents

- [How to cite](#how-to-cite)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Confiugration](#server-and-absolute-path-configuration)
  - [Customize initial search file](#define-a-customized-search-file)
  - [Submit sccop job on cluster](#submit-sccop-job)
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

GPU node send VASP jobs to cpu nodes by python package `paramiko`.

```diff
[Server]
# GPU number and name of gpu node
num_gpus = 2 
gpu_node = 'nodeXXX'
# List of cpu nodes, thus the cpu name is, e.g., 'nodeXXX' 
nodes = [XXX, XXX, XXX] 
```

Last, you need to set up the absolute path of SCCOP on GPU and CPU nodes, as well as the path of VASP software.

```diff
[Absolute path]
# Path of SCCOP on GPU node
SCCOP_path = '/local/sccop' 
# Directory of SCCOP on CPU nodes
CPU_local_path = '/local' 
# Path of VASP 2d and VASP 3d
VASP_2d_path = '/opt/openmpi-1.6.3/bin/mpirun' 
VASP_3d_path = '/opt/intel/impi/4.0.3.008/intel64/bin/mpirun' 
# VASP parallelization
VASP_2d_exe = f'{VASP_2d_path} -np 48 vasp_relax_ab' 
VASP_3d_exe = f'{VASP_3d_path} -np 48 vasp' 
```

**Note:** we recommend that you put SCCOP under the `/local` directory to accelerate the speed of data processing. For researchers who want to change the submission of VASP jobs, see the code in `src/core/sub_vasp.py`.


### Define a Customized Search File

To run SCCOP, you will need to define a customized initial search file, i.e., the `global_var.py` should be:

```diff
[Grid]
# Cut off distance to find neighbor atoms
cutoff = 8 
# Number of atoms in unit cell
num_min_atom = 5 
num_max_atom = 10 
# Grain of grid
grain = [.5, .5, 1.2] 
# Number of gridlines in a, b, c direction
plane_upper = [100, 100, 1]

[Dimension]
# Dimension of target composition
num_dim = 2 
# Whether add vacuum layer
add_vacuum = True 
vacuum_space = 15 
# Whether search puckered structure
puckered = True 
thickness = 0.1 

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

[Initial Samples]
# The chemical formula of the compound, e.g., 'B1C3'
component = 'XXX' 
# Number of initial lattice
num_latt = 72 
# Number of initial structures sent to VASP
num_Rand = 120 
# Average space group per crystal system
num_ave_sg = 10 
# Number of sampling structure = num_cores*num_per_sg
num_cores = 4  
num_per_sg = 5 
# Lattice parameters
len_mu = 5 
len_lower = 4 
len_upper = 6 
len_sigma = 1 
ang_mu = 90 
ang_sigma = 20 
# Sampling weight of crystal system
# [triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic]
system_weight = [1/4, 0, 1/4, 1/4, 0, 1/4, 0] 

[Training]
# Training prediction model parameters
train_batchsize = 64 
train_epochs = 120 
use_pretrain_model = True 

[Searching]
# SA parameters
T = .1 
decay = .95 
# Total SA steps = latt_steps*sa_steps
latt_steps = 3 
sa_steps = 100 
# Metropolis judge interval
num_jump = 2 
# Number of SA path
num_path = 360
# SA cores
sa_cores = 2 
# Distance constraint
min_bond = 1.2 

[Sample Select]
# Number of models to predict energy
num_models = 5 
# TSNE components
num_components = 2 
# Number of clusters
num_clusters = 60 
ratio_min_energy = 0.5 
```


### Submit SCCOP Job

Before training a new CGCNN model, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` to store the structure-property relations of interest.

Then, in directory `cgcnn`, you can train a CGCNN model for your customized dataset by:

```bash
python main.py root_dir
```

You can set the number of training, validation, and test data with labels `--train-size`, `--val-size`, and `--test-size`. Alternatively, you may use the flags `--train-ratio`, `--val-ratio`, `--test-ratio` instead. Note that the ratio flags cannot be used with the size flags simultaneously. For instance, `data/sample-regression` has 10 data points in total. You can train a model by:

```bash
python main.py --train-size 6 --val-size 2 --test-size 2 data/sample-regression
```
or alternatively
```bash
python main.py --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 data/sample-regression
```

You can also train a classification model with label `--task classification`. For instance, you can use `data/sample-classification` by:

```bash
python main.py --task classification --train-size 5 --val-size 2 --test-size 3 data/sample-classification
```

After training, you will get three files in `cgcnn` directory.

- `model_best.pth.tar`: stores the CGCNN model with the best validation accuracy.
- `checkpoint.pth.tar`: stores the CGCNN model at the last epoch.
- `test_results.csv`: stores the `ID`, target value, and predicted value for each crystal in test set.

### Successful Example

![](images/BC3_log_1.png)



![](images/BC3_log_2.png)



![](images/BC3_log_3.png)

Before predicting the material properties, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` for all the crystal structures that you want to predict.
- Obtain a [pre-trained CGCNN model](pre-trained) named `pre-trained.pth.tar`.

Then, in directory `cgcnn`, you can predict the properties of the crystals in `root_dir`:

```bash
python predict.py pre-trained.pth.tar root_dir
```

For instace, you can predict the formation energies of the crystals in `data/sample-regression`:

```bash
python predict.py pre-trained/formation-energy-per-atom.pth.tar data/sample-regression
```

And you can also predict if the crystals in `data/sample-classification` are metal (1) or semiconductors (0):

```bash
python predict.py pre-trained/semi-metal-classification.pth.tar data/sample-classification
```

Note that for classification, the predicted values in `test_results.csv` is a probability between 0 and 1 that the crystal can be classified as 1 (metal in the above example).

After predicting, you will get one file in `sccop` directory:

- `test_results.csv`: stores the `ID`, target value, and predicted value for each crystal in test set. Here the target value is just any number that you set while defining the dataset in `id_prop.csv`, which is not important.

## Data

To reproduce our paper, you can download the corresponding datasets following the [instruction](data/material-data).

## Authors

This software was primarily written by Chuannan Li and Hanpu Liang. 

## License

SCCOP is released under the MIT License.