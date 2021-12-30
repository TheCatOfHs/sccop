#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys, os, datetime

plt.rc('font', family='Times New Roman')

def read_elastic(file_name):
    with open(file_name, 'r') as obj:
        ct = obj.readlines()
    elastic = np.array([[float(item) for item in line.split()] for line in ct])
    return elastic

def plot_elastic(elastic):
    font_set = {'family':'Times New Roman', 'weight':'normal', 'size':20}
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='polar')
    phi = np.radians(np.linspace(0, 360, 360))
    c11, c22, c12, c66 = elastic[0,0], elastic[1,1], elastic[0,1], elastic[5,5]
    p1 = (c11*c22-c12*c12)/c66
    vtop = c11 + c22 - p1*np.cos(phi)**2*np.sin(phi)**2 - c12*(np.cos(phi)**4 + np.sin(phi)**4)
    vbot = c11*np.sin(phi)**4 + c22*np.cos(phi)**4 + (p1-2*c12)*np.cos(phi)**2*np.sin(phi)**2
    v = -vtop/vbot
    r = np.ones(phi.shape)
    ax.plot(phi, v, color='black', linewidth=1.2)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9)

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
    if not os.path.exists('DPT.elastic_constant.dat'):
        systemError('File DPT.elastic_constant.dat not found!')
    band = read_elastic('DPT.elastic_constant.dat')
    plot_elastic(band)
    plt.savefig('DPT.poisson.png', dpi=500)
    plt.show()

