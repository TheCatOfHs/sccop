import time
from core.path import *
from core.input import *


def system_echo(ct, header=False):
    """
    write system log
    
    Parameters
    ----------
    ct [str, 0d]: content
    """
    if header:
        echo_ct = ct
    else:
        echo_ct = time.strftime("%Y-%m-%d %H:%M:%S",
                                time.localtime()) + ' -- ' + ct 
    with open(Log_File, 'a') as obj:
        obj.write(echo_ct + '\n')

def header():
    system_echo('''      
     _____  _____  _____  _____ ______        
    /  ___|/  __ \/  __ \|  _  || ___ \\       *------------*-------------*
    \ `--. | /  \/| /  \/| | | || |_/ /       |     Version: 1.4.5       |
     `--. \| |    | |    | | | ||  __/        | Last Update: 2024-01-08  |
    /\__/ /| \__/\| \__/\\\ \_/ /| |           |     Authors: SCCOP Group |
    \____/  \____/ \____/ \___/ \_|           *------------*-------------*                                                                                                
    ''', header=True)


if __name__ == '__main__':
    pass