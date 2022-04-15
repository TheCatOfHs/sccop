mkdir search
for i in 'B1C1' 'B1C2' 'B1C3' 'B2C1' 'B1N1' 'B2N1' 'B3N1' 'B4N1' 'B5N1' 'C1N1' 'C2N1' 'C3N1' 'C4N1' 'C5N1' 'C3N4' 'B1C1N1' 'B1C2N1'
do
    mkdir search/$i
    scp -r sfront:/public/BioPhys/lcn/ccop .
    cd ccop/
        sed -i s/'XXX'/$i/g src/core/global_var.py
        python src/main.py > log
        cp log ../search/$i/.
        cp data/poscar/optim_strs_sym/* ../search/$i/.
        cp data/vasp_out/optim_strs/energy/* ../search/$i/.
    cd ../
    rm -r ccop/
done
