# Crystal-Combinatiorial-Optimization-Program

晶体组合优化程序(Crystal-Combinatiorial-Optimization-Program, CCOP)是基于Python3与PyTorch编写的机器学习加速结构搜索程序。目前正在开发中，如有需要可联系邮箱lcn1996@mail.ustc.edu.cn进行交流。

## 环境

shell版本: bash

python版本: python3.5及以后。需要有ase, pytorch, DPT, phonopy库

## 测试样本

test/Optim：POSCAR-015-002-131 
            POSCAR-022-001-131
            POSCAR-024-002-131

需求：

1. Optim文件夹中的POSCAR做结构优化
2. 计算声子谱，能带等性质
3. 通过ssh连接到指定节点调用程序计算，ssh工具在utils.py，节点信息在global_var.py

## VASP计算

目前可计算

1. 结构优化(libs/VASP_inputs/Optimization)
2. 声子谱(libs/VASP_inputs/Phonon), 需安装Phonopy, 运行完毕后会直接输出声子谱图(利用libs/scripts/plot-phonon-band.py)
3. PBE能带(libs/VASP_inputs/ElectronicStructure), 运行完毕后会直接输出能带图(利用libs/scripts/plot-energy-band.py)

注：所有计算的布里渊区路径为

Z(0.0, 0.5, 0.0) $\to$ $\Gamma$(0.0, 0.0, 0.0) $\to$ Y(0.5, 0.0, 0.0) $\to$ A(0.5, 0.0, 0.5) $\to$ B(0.0, 0.0, 0.5) $\to$ D(0.0, 0.5, 0.5) $\to$ E(0.5, 0.5, 0.5) $\to$ C(0.5, 0.5, 0.0)
