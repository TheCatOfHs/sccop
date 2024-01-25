import os
from core.log_print import *


if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    if Job_Queue == 'GPU':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    header()
    #launch SCCOP
    shell_script = f'''
                    for i in `seq 0 {Num_Recycle}`
                    do
                        python src/sccop.py --recyc $i
                    done
                    '''
    with open(f'{SCCOP_Path}/sccop.sh', 'w') as obj:
        obj.write(shell_script)
    os.system(f'''
              #!/bin/bash --login
              {SCCOP_Env}
              
              cd {SCCOP_Path}
              
              sh sccop.sh
              
              cd data
              date +%s >> time.dat
              ''')
    with open(f'{SCCOP_Path}/{Time_File}' , 'r') as obj:
        ct = obj.readlines()
    start, end = [int(i) for i in  ct]
    system_echo(f'Time: {end - start}')