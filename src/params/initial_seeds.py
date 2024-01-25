import os

from pymatgen.ext.matproj import MPRester


def download_from_database(seeds_num, max_atom, composition, mp_api_key):
    """
    get initial grids from materials project

    Parameters
    ----------
    seeds_num [int, 0d]: number of seeds
    max_atom [int, 0d]: maximum atom number
    composition [str, 0d]: chemical composition
    mp_api_key [str, 0d]: api of materials project
    """
    with MPRester(mp_api_key) as mpr:
        docs = mpr.summary.search(formula=composition,
                                  num_sites=(0, max_atom),
                                  fields=["structure", 'energy_above_hull'],
                                  sort_fields=['energy_above_hull'],
                                  energy_above_hull=(0, .5))
        if not os.path.exists('seeds'):
            os.mkdir('seeds')
        for i, data in enumerate(docs[:seeds_num]):
            data.structure.to(filename=f'seeds/POSCAR-{composition}-{i:03.0f}', fmt='vasp')
    
    
if __name__ == '__main__':
    seeds_num = 20
    max_atom = 300
    composition = 'B1'
    mp_api_key = '02WvvHP3PhKXOrgB9Vep3JznRGL8wkwZ'
    download_from_database(seeds_num, max_atom, composition, mp_api_key)