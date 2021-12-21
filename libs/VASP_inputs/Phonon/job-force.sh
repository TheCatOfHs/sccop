#!/bin/bash
for i in {001..048}
do
    mkdir disp-$i
    cp INCAR KPOINTS POTCAR vdw* disp-$i/

    cp POSCAR-$i disp-$i/POSCAR

    cd disp-$i/
        mpirun -machinefile $PBS_NODEFILE -np $NP vasp_gam >> vasp.out 2>>err.vasp
        rm CHG* WAVECAR
        cp vasprun.xml ../vasprun.xml-$i
    cd ..
done
