import os, sys
import time
import paramiko
import numpy as np

sys.path.append(f'{os.getcwd()}/src')
from ccop.global_var import *


def system_echo(ct):
    """
    write system log
    
    Parameters
    ----------
    ct ([str, 0d]): [content]
    """
    echo_ct = time.strftime("%Y-%m-%d %H:%M:%S",
                            time.localtime()) + ' -- ' + ct
    print(echo_ct)
    with open(log_file, 'a') as obj:
        obj.write(echo_ct + '\n')


class ListRWTools:
    #Save and import list
    def import_list2d(self, file, dtype, numpy=False):
        """
        import 2-dimensional list
        
        Parameters
        ----------
        file [str, 0d]: file name
        dtype [int float str]: data type
        
        Returns
        ----------
        list [dtype, 2d]: 2-dimensional list
        """
        with open(file, 'r') as f:
            ct = f.readlines()
        list = self.str_to_list2d(ct, dtype)
        if numpy:
            return np.array(list)
        else:
            return list

    def str_to_list2d(self, string, dtype):
        """
        convert string list to 2-dimensional list
    
        Parameters
        ----------
        string [str, 1d]: list in string form
        dtype [int float str]: data type
        
        Returns
        ----------
        list [dtype, 2d]: 2-dimensional list
        """
        list = [self.str_to_list1d(item.split(), dtype)
                for item in string]
        return list
    
    def str_to_list1d(self, string, dtype):
        """
        convert string list to 1-dimensional list
    
        Parameters
        ----------
        string [str, 1d]: list in string form
        dtype [int float str]: data type
        
        Returns
        ----------
        list [dtype, 1d]: 1-dimensional list
        """
        list = [dtype(i) for i in string]
        return list
    
    def write_list2d(self, file, list, style):
        """
        write 2-dimensional list
        
        Parameters
        ----------
        file [str, 0d]: file name
        list [num, 2d]: 2-dimensional list
        style [str, 0d]: style of number
        """
        list_str = self.list2d_to_str(list, style)
        list2d_str = '\n'.join(list_str)
        with open(file, 'w') as f:
            f.write(list2d_str)
        
    def list2d_to_str(self, list, style):
        """
        convert 2-dimensional list to string list
        
        Parameters
        ----------
        list [num, 2d]: 2-dimensional list
        style [str, 0d]: string style of number
        
        Returns
        ----------
        list_str [str, 0d]: string of list2d
        """
        list_str = [' '.join(self.list1d_to_str(line, style)) 
                    for line in list]
        return list_str
    
    def list1d_to_str(self, list, style):
        """
        convert 1-dimensional list to string list
        
        Parameters
        ----------
        list [num, 1d]: 1-dimensional list
        style [str, 0d]: string style of number

        Returns
        ----------
        list [str, 1d]: 1-dimensional string list
        """
        list = [style.format(i) for i in list]
        return list
    

class SSHTools:
    #SSH to node
    def __init__(self):
        pass
    
    def ssh_node(self, shell_script, ip):
        """
        SSH to target node and execute command

        Parameters
        ----------
        """
        port = 22
        user = 'lcn'
        password = '199612qweasd'
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, port, user, password, timeout=1000)
        ssh.exec_command(shell_script)
        ssh.close()

    def assign_node(self, num_jobs):
        """
        assign jobs to nodes
        
        Parameters
        ----------
        num_jobs [int, 0d]: number of jobs
        
        Returns
        ----------
        node_assign [int, 1d]: vasp job list of nodes
        """
        num_nodes = len(nodes)
        num_assign, node_assign = 0, []
        while not num_assign == num_jobs:
            jobs = num_jobs - num_assign
            assign = jobs//num_nodes
            if assign == 0:
                node_assign = node_assign + nodes[:jobs]
            else:
                node_assign = [i for i in nodes for _ in range(assign)]
            num_assign = len(node_assign)
        return sorted(node_assign)
    
    def check_num_file(self, command, file_num):
        """
        If shell is completed, return True
        
        Returns
        ----------
        flag [bool, 0d]: whether works are done
        """
        flag = False
        finish = os.popen(command)
        finish = int(finish.read())
        if finish == file_num:
            flag = True
        return flag


if __name__ == '__main__':
    a = [1, 2, 3]
    b = [2, 3]
    print(np.setdiff1d(a, b))