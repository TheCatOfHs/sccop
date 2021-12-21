import os, sys
import time

sys.path.append(os.getcwd())
from ccop.global_var import *
from ccop.utils import SSHTools, system_echo


class Initial(SSHTools):
    #make each node consistent with main node
    def __init__(self, sleep_time=1):
        self.sleep_time = sleep_time
    
    def update(self):
        num_node = len(nodes)
        for node in nodes:
            self.update_with_ssh(node)
        while not self.is_done(num_node):
            time.sleep(self.sleep_time)
        self.remove()
        system_echo('Each node consistent with main node')
        
    def update_with_ssh(self, node):
        """
        SSH to target node and update ccop file
        """
        ip = f'node{node}'
        shell_script = f'''
                    #!/bin/bash
                    cd /local
                    rm -rf ccop/
                    cp -r ~/ccop .
                    > {ip}
                    mv {ip} ~/ccop/program/data/
                    '''
        self.ssh_node(shell_script, ip)

    def is_done(self, file_num):
        """
        If shell is completed, return True
        
        Returns
        ----------
        file_num [int, 0d]: number of file
        """
        command = f'ls -l data/ | grep node | wc -l'
        flag = self.check_num_file(command, file_num)
        return flag
    
    def remove(self): 
        os.system(f'rm data/node*')


if __name__ == '__main__':
    print('ok')