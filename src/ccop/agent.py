import time
import numpy as np
import threading as td
import multiprocessing as mp
from collections import Counter

import torch

import grid_divide as g3d
import data_transfer as dtr
from environment import CrystalGraphConvNet, Normalizer
from utils import ListRWTools


class Agent:
    
    def __init__(self):
        pass


if __name__ == "__main__":
    print('ok')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #Simulated Annealing
    '''
    matols = MatrixRWTools()
    inizer = dtr.Initializer(grid_name)
    atom_pos = matols.import_mat('data/rounds/003/idx.dat', int)
    atom_type = [[val for val in [30, 6, 7, 29] for _ in range(18)] for _ in range(600)]
    
    n = 0
    round = '001'
    worker = Search(round, inizer)
    start = time.time()
    worker.explore(atom_pos[n], atom_type[n], 'model_best.pth.tar', '0')
    end = time.time()
    print('single', end - start)
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''
    #Data Transfer
    matols = MatrixRWTools()
    inizer = dtr.Initializer(grid_name)
    atom_pos = matols.import_mat('data/rounds/003/idx.dat', int)
    atom_type = [[val for val in [31, 7, 8, 30] for _ in range(18)] for _ in range(600)]
    flags, energys = inizer.import_energy('data/rounds/003/Energy-003-0.dat')
    start = time.time()
    atom_fea, nbr_fea, nbr_fea_idx = inizer.single_init(atom_pos[0], atom_type[0])
    end = time.time()
    crystal_atom_idx = np.arange(len(atom_fea))
    print(end - start)
    '''
    
    
    '''
    matools = MatrixRWTools()
    vasp = matools.import_mat('data/vasp.dat', float).ravel()
    pred = matools.import_mat('data/pred.dat', float).ravel()
    x = np.linspace(vasp.min(), vasp.max(), 10)
    plt.rc('font', family='Times New Roman')
    figure_1 = plt.figure(figsize=(7, 7))
    ax_1 = figure_1.add_subplot(1, 1, 1)
    ax_1.scatter(pred, vasp, c='g', label='Test Set')
    ax_1.plot(x, x, '--', c='r', alpha=0.5, label='y = x')
    ax_1.tick_params(labelsize=12)
    ax_1.set_xlabel('Prediction / eV', fontsize=16)
    ax_1.set_ylabel('DFT Calculation / eV', fontsize=16)
    plt.legend(shadow=True, fontsize=16)
    plt.savefig('data/image.png', dpi=600)
    plt.show()
    
    tx, ty = 119, -300
    with open('data/Energy-003-0.dat', 'r') as f:
        ct = f.readlines()
    energy = np.array([float(item.split()[-1]) for item in ct])
    paths = np.split(energy, 3)
    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(8, 6))
    for path in paths:
        plt.plot(path)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('DFT calculation / eV', fontsize=16)
    plt.xlabel('Simulated annealing steps', fontsize=16)
    plt.legend(('path_1', 'path_2', 'path_3'), shadow=True, fontsize=18)
    plt.text(tx, ty, 'Unknown', c='r', fontsize=15, verticalalignment="bottom", horizontalalignment="center")
    plt.annotate('', xy=(119,-301.), xytext=(tx,ty), arrowprops=dict(arrowstyle="->", color='r', connectionstyle="arc3"))
    plt.savefig('data/sa_image.png', dpi=600)
    #plt.show()
    '''