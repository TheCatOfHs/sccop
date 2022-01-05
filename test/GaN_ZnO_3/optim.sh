cd POSCAR-test
cp ../../libs/VASP_inputs/Optimization/* .
                                
cp POSCAR POSCAR_0
DPT -v potcar
for i in 1 2 3
do
    cp INCAR_$i INCAR
    cp KPOINTS_$i KPOINTS
    date > vasp-$i.vasp
    /opt/intel/impi/4.0.3.008/intel64/bin/mpirun -np 48 vasp >> vasp-$i.vasp
    date >> vasp-$i.vasp
    cp CONTCAR POSCAR
    cp CONTCAR POSCAR_$i
    cp OUTCAR OUTCAR_$i
    rm WAVECAR CHGCAR
done