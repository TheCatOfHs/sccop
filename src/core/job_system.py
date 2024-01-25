import os, sys

sys.path.append(f'{os.getcwd()}/src')
from core.log_print import *


def write_nodes():
    """
    write name of assigned nodes under PBS system
    """
    with open('data/host.dat', 'r') as ct:
        host = ct.readlines()
    with open('data/nodes.dat', 'r') as ct:
        nodes = ct.readlines()
    host_name = host[0].strip()
    node_name = [i.strip() for i in nodes[0].split()]
    shell_script = f'''
                    #!/bin/bash --login
                    cd /tmp/sccop
                    sed -i s/'HOST_NODE'/\"{host_name}\"/g src/core/path.py
                    sed -i \"s/'CPU_NODES'/{node_name}/g\" src/core/path.py
                    rm data/host.dat 
                    rm data/nodes.dat
                    '''
    os.system(shell_script)

class GPUQueue:
    #submit sccop job to gpu Queue
    def __init__(self):
        pass
    
    def get_nodes_pbs(self):
        pass

    def get_nodes_slurm(self):
        pass


if __name__ == '__main__':
    #CPU Queue
    if Job_Queue == 'CPU':
        write_nodes()
            
    #GPU Queue
    elif Job_Queue == 'GPU':
        gpu = GPUQueue()
        if Job_System == 'PBS':
            gpu.get_nodes_pbs()
        elif Job_System == 'SLURM':
            gpu.get_nodes_slurm()