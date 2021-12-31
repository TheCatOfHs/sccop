# Crystal-Combinatiorial-Optimization-Program

晶体组合优化程序(Crystal-Combinatiorial-Optimization-Program, CCOP)是基于Python3与PyTorch编写的机器学习加速结构搜索程序。目前正在开发中，如有需要可联系邮箱lcn1996@mail.ustc.edu.cn进行交流。

## 环境

shell版本: bash

python版本: python3.5及以后。需要有ase, pytorch, DPT, phonopy, pymatgen库

## 程序启动

进入ccop，运行python src/main.py

## VASP计算

目前可计算

1. 结构优化(libs/VASP_inputs/Optimization)
2. 声子谱(libs/VASP_inputs/Phonon), 需安装Phonopy, 运行完毕后会直接输出声子谱图(利用libs/scripts/plot-phonon-band.py)
3. PBE能带(libs/VASP_inputs/ElectronicStructure), 运行完毕后会直接输出能带图(利用libs/scripts/plot-energy-band.py)

注：所有计算的布里渊区路径均由pymatgen生成
依赖包：pymatgen seekpath
pip install seekpath


## 节点通信
节点之间需要ssh免密连接
配置方法：
主节点：cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
更新密钥：
scp ~/.ssh/authorized_keys root@ip1:~/.ssh/
scp ~/.ssh/authorized_keys root@ip2:~/.ssh/
...