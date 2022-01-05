#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys, os, datetime
plt.rc('font', family='Times New Roman')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# 使用tex字符
#plt.rc('text', usetex=True)

def read_phonopy(file_name):
    with open(file_name, 'r') as obj:
        file_content = obj.readlines()
    
    k_sym = np.array([float(item) for item in file_content[1].split()[1:]])
    data = [[float(item) for item in line.split()] for line in file_content[2:]]
    data_new = []
    for line in data:
        if len(line) != 0:
            data_new.append(line)
    data = np.array(data_new)

    num_k = 0
    for i in range(1000):
        if data[i+1,0] < data[i,0]:
            num_k = i + 1
            break
    num_phonopy = int(data.shape[0] / num_k)
    print('Number of phonon branch is', num_phonopy)
    phonon = np.zeros((num_k, num_phonopy+1))
    phonon[:,0] = data[0:num_k,0]
    for i in range(num_phonopy):
        phonon[:,i+1] = data[i*num_k:(i+1)*num_k,1]

    phonon_str = '\n'.join([' '.join(['{0:12.6f}'.format(item) for item in line]) for line in phonon])
    with open('phonon_matrix.dat', 'w') as obj:
        obj.write(phonon_str)
    return (phonon, k_sym)

def plot_phonon(phonon, k_sym, k_labels):
    font_set = {'family':'Times New Roman', 'weight':'normal', 'size':20}
    # sym_points = ['Z', '$\Gamma$', 'Y', 'A', 'B', 'D', 'E', 'C']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(phonon.shape[1]-1):
        ax.plot(phonon[:,0], phonon[:,i+1], color='black', linewidth=0.8)
    k = phonon[:,0]
    for i in k_sym:
        plt.plot([i, i], [-100, 100], '-', color='#c2ccd0', lw=0.9)
    plt.plot([np.min(k), np.max(k)], [0, 0], '-', color='#c2ccd0', lw=0.9)
    ax.set_xlim([np.min(k), np.max(k)])
    ax.set_ylim([np.min(phonon[:,1:]), np.max(phonon[:,1:])+1])
    ax.set_ylabel('Frequency (THz)', font_set, fontsize=15)
    ax.set_xticks(k_sym)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_xticklabels(k_labels, fontsize=15)
    fig.subplots_adjust(left=0.18, right=0.95, top=0.95)

def get_k_labels(band_conf):
    with open(band_conf, 'r') as obj:
        ct = obj.readlines()
    all_labels = ct[3].split()[2:]
    all_k = [path.split(' ') for path in ct[2][7:].split(',')]
    k_num = []
    for path in all_k:
        while '' in path:
            path.remove((''))
        k_num.append(len(path)/3)
    k_num = [int(i) for i in k_num]
    for i in range(0, len(k_num)-1):
        cur_ind = sum(k_num[0:i+1])-1
        be_comb_ind = cur_ind + 1
        all_labels[cur_ind] = f'{all_labels[cur_ind]}|{all_labels[be_comb_ind]}'
        all_labels.pop(be_comb_ind)
        k_num[i] -= 1
    print(all_labels)
    return all_labels

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
    if len(sys.argv) == 1:
        systemError('Please Input the Parameter: file name')
    file_name = sys.argv[1]
    band_conf = sys.argv[2]
    if not os.path.exists(file_name):
        systemError('File {0} not found!'.format(file_name))
    phonon, k_sym = read_phonopy(file_name)
    k_labels = get_k_labels(band_conf)
    print(k_sym)
    plot_phonon(phonon, k_sym, k_labels)
    plt.savefig('PHON.png', dpi=500)
    # plt.show()
