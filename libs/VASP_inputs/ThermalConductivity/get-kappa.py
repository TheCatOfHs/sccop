import os

if __name__ == '__main__':
    kappa_tot_file = 'BTE.kappa_tensor'
    kappa_mode_file = 'BTE.kappa'
    T = []
    kappa_x = []
    kappa_y = []
    kappa_z = []
    kappa_mode = []
    for i in range(100, 920, 20):
        file = '{0}/T{1}K/'.format(i,i)+kappa_tot_file
        if os.path.exists(file):
            with open('{0}/T{1}K/'.format(i,i)+kappa_tot_file, 'r') as obj:
                ct1 = obj.readlines()
            kappa_x.append(float((ct1[-1].split())[1]))
            kappa_y.append(float((ct1[-1].split())[5]))
            kappa_z.append(float((ct1[-1].split())[9]))
            with open('{0}/T{1}K/'.format(i,i)+kappa_mode_file, 'r') as obj:
                ct2 = obj.readlines()
            temp = [float(item) for item in (ct2[-1].split())[1:]]
            temp.insert(0, i)
            mode = (len(temp)*'%15.9f') % tuple(temp)
            kappa_mode.append(mode)
            T.append(i)
    out_kappa_tot = ''.join(['%15d%15.9f%15.9f%15.9f\n'%(T[i], kappa_x[i], kappa_y[i], kappa_z[i]) for i in range(len(T))])
    out_kappa_tot = '%15s%15s%15s%15s\n'%('T', 'kappa_x', 'kappa_y', 'kappa_z') + out_kappa_tot
    out_kappa_mode = '\n'.join(kappa_mode)
    with open('./kappa.dat', 'w') as obj:
        obj.write(out_kappa_tot)
