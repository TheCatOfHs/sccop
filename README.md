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
# *GPU number*
num_gpus = 2 
# Name of gpu node
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
# Path of VASP 2d
VASP_2d_path = '/opt/openmpi-1.6.3/bin/mpirun' 
# Path of VASP 3d
VASP_3d_path = '/opt/intel/impi/4.0.3.008/intel64/bin/mpirun' 
# VASP 2d parallelization
VASP_2d_exe = f'{VASP_2d_path} -np 48 vasp_relax_ab' 
# VASP 3d parallelization
VASP_3d_exe = f'{VASP_3d_path} -np 48 vasp' 
```

Note: we recommend that put SCCOP under the /local directory to accelerate the speed of data processing. For researchers who want to change the submission of VASP jobs, see the code in `src/core/sub_vasp.py`, e.g., use Protable Batch System (PBS) to submit VASP jobs.


### Define a Customized Search File

To input crystal structures to CGCNN, you will need to define a customized dataset. Note that this is required for both training and predicting. 

Before defining a customized dataset, you will need:

- [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) files recording the structure of the crystals that you are interested in
- The target properties for each crystal (not needed for predicting, but you need to put some random numbers in `id_prop.csv`)

You can create a customized dataset by creating a directory `root_dir` with the following files: 

1. `id_prop.csv`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with two columns. The first column recodes a unique `ID` for each crystal, and the second column recodes the value of target property. If you want to predict material properties with `predict.py`, you can put any number in the second column. (The second column is still needed.)

2. `atom_init.json`: a [JSON](https://en.wikipedia.org/wiki/JSON) file that stores the initialization vector for each element. An example of `atom_init.json` is `data/sample-regression/atom_init.json`, which should be good for most applications.

3. `ID.cif`: a [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) file that recodes the crystal structure, where `ID` is the unique `ID` for the crystal.

The structure of the `root_dir` should be:

```diff
[Grid]
cutoff = 8 #
num_min_atom = 5 #
num_max_atom = 10 #
grain = [.5, .5, 1.2] #
plane_upper = [100, 100, 1] #

[Dimension]
num_dim = 2 #
add_vacuum = True #
vacuum_space = 15 #
puckered = True #
thickness = 0.1 #

[Recycling]
num_recycle = 1 #
num_ml_list = [1] #
num_poscars = 12 #
num_optims = 6 #
vasp_time_limit = 480 #

[Initial Samples]
component = 'XXX' #
num_latt = 72 #
num_Rand = 120 #
num_ave_sg = 10 #
num_cores = 4 #
num_per_sg = 5 #
len_mu = 5 #
len_lower = 4 #
len_upper = 6 #
len_sigma = 1 #
ang_mu = 90 #
ang_sigma = 20 #
system_weight = [1/4, 0, 1/4, 1/4, 0, 1/4, 0] #

[Training]
train_batchsize = 64 #
train_epochs = 120 #
use_pretrain_model = True #

[Searching]
T = .1 #
decay = .95 #
latt_steps = 3 #
sa_steps = 100 #
num_jump = 2 #
num_path = 360 #
sa_cores = 2 #
min_bond = 1.2 #

[Sample Select]
num_models = 5 #
num_components = 2 #
num_clusters = 60 #
ratio_min_energy = 0.5 #
```

There are two examples of customized datasets in the repository: `data/sample-regression` for regression and `data/sample-classification` for classification. 

**For advanced PyTorch users**

The above method of creating a customized dataset uses the `CIFData` class in `cgcnn.data`. If you want a more flexible way to input crystal structures, PyTorch has a great [Tutorial](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#sphx-glr-beginner-data-loading-tutorial-py) for writing your own dataset class.

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