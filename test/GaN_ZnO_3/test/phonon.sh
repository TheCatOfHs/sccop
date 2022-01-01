p=POSCAR-CCOP-125-135
mkdir $p
cd $p
cp ../../libs/VASP_inputs/Phonon/* .
                                
phonopy -d --dim="2 2 2"
n=`ls | grep POSCAR- | wc -l`
for i in `seq -f%03g 1 $n`
do
    mkdir disp-$i
    cp INCAR KPOINTS POTCAR vdw* disp-$i/
    cp POSCAR-$i disp-$i/POSCAR
                                
    cd disp-$i/
        DPT -v potcar
        /opt/intel/impi/4.0.3.008/intel64/bin/mpirun -np 48 vasp >> vasp.out 2>>err.vasp
        rm CHG* WAVECAR
        cp vasprun.xml ../vasprun.xml-$i
    cd ..
done
                                
phonopy -f vasprun.xml-*
phonopy band.conf
phonopy-bandplot --gnuplot --legacy band.yaml > phonon-$p.dat
python ../../libs/scripts/plot-phonon-band.py phonon-$p.dat band.conf
scp phonon-$p.dat node151:/local/ccop/data/vasp_out/optim_strs/phonon/phonon-$p.dat
scp PHON.png node151:/local/ccop/data/vasp_out/optim_strs/phonon/phonon-$p.png
