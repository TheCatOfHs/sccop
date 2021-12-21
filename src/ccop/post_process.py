

class PostProcess:
    ''' process the crystals by VASP to relax the structures and calculate the properties'''
    def __init__(self, str_file, run_dir):
        self.str_file = str_file
        self.run_dir = run_dir

    def run_optimization(self):
        # prepare the optimization files and submit the job
        pass
    
    def run_phonon(self):
        # prepare the phonon spectrum files and submit the job
        pass
    
    def run_pbe_band(self):
        # prepare the electronic structure files by PBE method and submit the job
        pass


if __name__ == '__main__':
    a = PostProcess('POSCAR-015-002-131', './VASP_calculations')