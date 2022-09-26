mkdir search
for i in 'B1N1' 'B1N2' 'B1N3' 'B1N4' 'B1N5' 'B2N1' 'B2N3' 'B2N5' 'B3N1' 'B3N2' 'B3N4' 'B3N5' 'B4N1' 'B4N3' 'B4N5' 'B5N1' 'B5N2' 'B5N3' 'B5N4'
do
    mkdir search/$i
    scp -r sfront:/public/BioPhys/lcn/sccop .
    cd sccop/
        sed -i s/'XXX'/$i/g src/core/global_var.py
        python src/main.py > log
        cp log ../search/$i/.
        cp data/poscar/SCCOP-0/POSCAR* ../search/$i/.
        cp data/vasp_out/initial_strus_1/* ../search/$i/.
        cp -r data/poscar ../search/$i/.
        cp -r data/vasp_out ../search/$i/.
    cd ../
    rm -r sccop/
done

for i in 'B1C1' 'B1C2' 'B1C3' 'B1C4' 'B1C5' 'B2C1' 'B2C3' 'B2C5' 'B3C1' 'B3C2' 'B3C4' 'B3C5' 'B4C1' 'B4C3' 'B4C5' 'B5C1' 'B5C2' 'B5C3' 'B5C4'
do
    mkdir search/$i
    scp -r sfront:/public/BioPhys/lcn/sccop .
    cd sccop/
        sed -i s/'XXX'/$i/g src/core/global_var.py
        python src/main.py > log
        cp log ../search/$i/.
        cp data/poscar/SCCOP-0/POSCAR* ../search/$i/.
        cp data/vasp_out/initial_strus_1/* ../search/$i/.
        cp -r data/poscar ../search/$i/.
        cp -r data/vasp_out ../search/$i/.
    cd ../
    rm -r sccop/
done

for i in 'C1N1' 'C1N2' 'C1N3' 'C1N4' 'C1N5' 'C2N1' 'C2N3' 'C2N5' 'C3N1' 'C3N2' 'C3N4' 'C3N5' 'C4N1' 'C4N3' 'C4N5' 'C5N1' 'C5N2' 'C5N3' 'C5N4'
do
    mkdir search/$i
    scp -r sfront:/public/BioPhys/lcn/sccop .
    cd sccop/
        sed -i s/'XXX'/$i/g src/core/global_var.py
        python src/main.py > log
        cp log ../search/$i/.
        cp data/poscar/SCCOP-0/POSCAR* ../search/$i/.
        cp data/vasp_out/initial_strus_1/* ../search/$i/.
        cp -r data/poscar ../search/$i/.
        cp -r data/vasp_out ../search/$i/.
    cd ../
    rm -r sccop/
done

for i in 'B1C1N1' 'B1C1N2' 'B1C1N3' 'B1C2N1' 'B1C2N2' 'B1C2N3' 'B1C3N1' 'B1C3N2' 'B1C3N3' 'B2C1N1' 'B2C1N2' 'B2C1N3' 'B2C2N1' 'B2C2N3' 'B2C3N1' 'B2C3N2' 'B2C3N3' 'B3C1N1' 'B3C1N2' 'B3C1N3' 'B3C2N1' 'B3C2N2' 'B3C2N3' 'B3C3N1' 'B3C3N2'
do
    mkdir search/$i
    scp -r sfront:/public/BioPhys/lcn/sccop .
    cd sccop/
        sed -i s/'XXX'/$i/g src/core/global_var.py
        python src/main.py > log
        cp log ../search/$i/.
        cp data/poscar/SCCOP-0/POSCAR* ../search/$i/.
        cp data/vasp_out/initial_strus_1/* ../search/$i/.
        cp -r data/poscar ../search/$i/.
        cp -r data/vasp_out ../search/$i/.
    cd ../
    rm -r sccop/
done