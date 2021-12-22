#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys, os, datetime
plt.rc('font', family='Times New Roman')

def read_band(file_name):
    with open(file_name, 'r') as obj:
        file_content = obj.readlines()

    data = np.array([[float(item) for item in line.split()] for line in file_content])
    print('Number of band is {:5.0f}'.format(data.shape[1]-1))
    return data

def plot_band(band):
    font_set = {'family':'Times New Roman', 'weight':'normal', 'size':20}
    k_sym = np.array([0, 19, 39, 59, 79, 99, 119, 139])
    sym_points = ['Z', '$\Gamma$', 'Y', 'A', 'B', 'D', 'E', 'C']
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(band.shape[1]-1):
        ax.plot(band[:,0], band[:,i+1], color='black', linewidth=0.8)
    k = band[:,0]
    for i in k_sym:
        plt.plot([k[i], k[i]], [-100, 100], '--k', lw=0.5)
    plt.plot([-10, 10], [0, 0], '--r', lw=0.5)
    ax.set_xlim([np.min(k), np.max(k)])
    if len(sys.argv) == 3:
        y_min = float(sys.argv[1])
        y_max = float(sys.argv[2])
        ax.set_ylim([y_min, y_max])
    else:
        ax.set_ylim([-6, 8])    # set y range
    
    # ax.set_xlabel('Wave Vector', font_set, fontsize=15)
    ax.set_ylabel('Energy (eV)', font_set, fontsize=15)
    ax.set_xticks(k[k_sym])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_xticklabels(sym_points)

def Open_screen():
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('''
       *-----------------*-----*
       |  Data           |  D  |        Version : Plot series
       |     Processing  |  P  |  Latest Update : 2021-01-10 
       |         Toolkit |  T  |         Author : Liang HP
       *-----------------*-----*
              Current Time: {0}
'''.format(now_time))

def systemEcho(content):
    ''' system information print to screen'''
    print(content)
    systemLog(content)

def systemError(content):
    ''' exit program when somewhere errors'''
    content = '**ERROR** : ' + content
    print(content)
    print('Program Exits.')
    exit(0)

def systemLog(content):
    ''' output information into log file'''
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    out_str = now_time + '---' + content + '\n'
    with open('./system.log', 'a') as obj:
        obj.write(out_str)

if __name__ == '__main__':
    Open_screen()
    if not os.path.exists('DPT.BAND.dat'):
        systemError('File DPT.BAND.dat not found!')
    band = read_band('DPT.BAND.dat')
    plot_band(band)
    plt.savefig('DPT.band.png', dpi=500)
    plt.show()
    

