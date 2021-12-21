#!/bin/bash

for i in {1..2}
do
    cp INCAR_$i INCAR
    cp KPOINTS_$i KPOINTS
    mpirun -machinefile $PBS_NODEFILE -np $NP vasp_std > vasp.out
    cp OUTCAR OUTCAR_$i
    cp IBZKPT IBZKPT_$i
    cp EIGENVAL EIGENVAL_$i
done


