#!/bin/bash

cp POSCAR POSCAR_0
for i in {1..3}
do
    cp INCAR_$i INCAR
    cp KPOINTS_$i KPOINTS
    mpirun -machinefile $PBS_NODEFILE -np $NP vasp_std > vasp.out
    cp CONTCAR POSCAR
    cp CONTCAR POSCAR_$i
    cp OUTCAR OUTCAR_$i
    rm WAVECAR CHGCAR
done
