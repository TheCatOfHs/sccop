#!/bin/bash

cp POSCAR POSCAR_0
DPT -v potcar
for i in {1..3}
do
    cp INCAR_$i INCAR
    cp KPOINTS_$i KPOINTS
    date > vasp-$i.vasp
    mpirun -machinefile $PBS_NODEFILE -np $NP vasp_std >> vasp-$i.vasp
    date >> vasp-$i.vasp
    cp CONTCAR POSCAR
    cp CONTCAR POSCAR_$i
    cp OUTCAR OUTCAR_$i
    rm WAVECAR CHGCAR
done
