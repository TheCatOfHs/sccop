import os
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')


def read_file(file_name):
    with open(file_name, 'r') as obj:
        file_content = obj.readlines()
    if file_name == 'kappa.dat':
        file_content = file_content[1:]
    data = [[float(i) for i in item.split()] 
            for item in file_content]
    return data

def plot_kappa(kappa_file):
    T, kappa_x, kappa_y, kappa_z = np.transpose(kappa_file)
    plt.figure(figsize=(8,5))
    plt.plot(T, kappa_x, label='$\kappa$-$x$', marker='s', lw='1', c='r')
    plt.plot(T, kappa_y, label='$\kappa$-$y$', marker='o', lw='1', c='k')
    plt.plot(T, kappa_z, label='$\kappa$-$z$', marker='+', ms=8, mew=2, lw='1', c='b')
    plt.xlabel('Temperature (K)',fontsize=15)
    plt.ylabel('$\kappa$ (W/mK)', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig('Thermal.png', dpi=500)
    plt.close()

def plot_cum_kappa(kappa_file):
    T, kappa_x, kappa_y, kappa_z = np.transpose(kappa_file)
    sum_kappa_x = np.sum(kappa_x)
    sum_kappa_y = np.sum(kappa_y)
    sum_kappa_z = np.sum(kappa_z)
    cum_kappa_x = [np.sum(kappa_x[:i])/sum_kappa_x for i in range(len(kappa_x))]
    cum_kappa_y = [np.sum(kappa_y[:i])/sum_kappa_y for i in range(len(kappa_y))]
    cum_kappa_z = [np.sum(kappa_z[:i])/sum_kappa_z for i in range(len(kappa_z))]
    plt.figure(figsize=(8,5))
    plt.plot(T, cum_kappa_x, label='$\kappa$-$x$', ls='-', lw='1', c='r')
    plt.plot(T, cum_kappa_y, label='$\kappa$-$y$', ls='--', lw='1', c='k')
    plt.plot(T, cum_kappa_z, label='$\kappa$-$z$', ls=':', lw='1', c='b')
    plt.xlabel('Temperature (K)',fontsize=15)
    plt.ylabel('Cumulative $\kappa$ (W/mK)', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig('Cum_Thermal.png', dpi=500)
    plt.close()
    
def plot_scatter(w_file):
    freq, scat = np.transpose(w_file)
    plt.figure(figsize=(8,5))
    plt.scatter(freq, scat, s=20, c='gray', marker='o', edgecolors='k', alpha=0.6)
    plt.xlabel('Frequency (THz)',fontsize=15)
    plt.ylabel('Scattering rate ($\mathregular{ps^{-1}}$)', fontsize=15)
    plt.savefig('Scatter.png', dpi=500)
    plt.close()

def plot_groupv(w_file, v_file):
    freq, _ = np.transpose(w_file)
    v_x, v_y, v_z = np.abs(np.transpose(v_file))
    plt.figure(figsize=(8,5))
    plt.scatter(freq, v_x, s=20, c='gray', label='$\mathregular{v_x}$', marker='o', edgecolors='k', alpha=0.6)
    plt.scatter(freq, v_y, s=20, c='none', label='$\mathregular{v_y}$', marker='s', edgecolors='salmon', alpha=0.8)
    plt.scatter(freq, v_z, s=20, c='blue', label='$\mathregular{v_z}$', marker='^', edgecolors='none', alpha=0.8)
    plt.xlabel('Frequency (THz)',fontsize=15)
    plt.ylabel('v (km/s)', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig('Velocity.png', dpi=500)
    plt.close()

def systemError(content):
    ''' exit program when somewhere errors'''
    content = '**ERROR** : ' + content
    print(content)
    print('Program Exits.')
    exit(0)


if __name__ == '__main__':
    kappa_dat = 'kappa.dat'
    if not os.path.exists(kappa_dat):
        systemError('File kappa.dat not found!')
    else:
        kappa_file = read_file(kappa_dat)
        plot_kappa(kappa_file)
        plot_cum_kappa(kappa_file)
        
    w_dat = 'BTE.w.dat'
    if not os.path.exists(w_dat):
        systemError('File BET.w.dat not found!')
    else:
        w_file = read_file(w_dat)
        plot_scatter(w_file)
        
    v_dat = 'BTE.v.dat'
    if not os.path.exists(v_dat):
        systemError('File BET.v.dat not found!')
    else:
        v_file = read_file(v_dat)
        plot_groupv(w_file, v_file)