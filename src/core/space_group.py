import os, sys
import numpy as np
from fractions import Fraction

sys.path.append(f'{os.getcwd()}/src')
from core.log_print import *


class PlaneSpaceGroup:
    #17 plane space groups
    def __init__(self):
        pass
    
    def triclinic_001(frac_grain):
        """
        plane space group P1
        """
        equal_1 = []
        #inner
        for i in np.arange(0, 1, frac_grain[0]):
            for j in np.arange(0, 1, frac_grain[1]):
                equal_1.append([i, j, 0])
        dau_area = equal_1
        return np.array(dau_area, dtype=float)
    
    def triclinic_002(frac_grain):
        """
        plane space group P211
        """
        equal_2 = []
        #point
        equal_1 = [[0, 0, 0], [0, .5, 0], [.5, 0, 0], [.5, .5, 0]]
        #boundary
        for i in np.arange(0, .5, frac_grain[0]):
            if 0 < i:
                equal_2.append([i, 0, 0])
        for j in np.arange(0, .5, frac_grain[1]):
            if 0 < j:
                equal_2.append([0, j, 0])
                equal_2.append([.5, j, 0])
        #inner
        for i in np.arange(0, .5, frac_grain[0]):
            for j in np.arange(0, 1, frac_grain[1]):
                if 0 < i and 0 < j:
                    equal_2.append([i, j, 0])
        dau_area = equal_1 + equal_2
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_003(frac_grain):
        """
        plane space group P1m1
        """
        equal_1, equal_2 = [], []
        #boundary
        for j in np.arange(0, 1, frac_grain[1]):
            equal_1.append([0, j, 0])
            equal_1.append([.5, j, 0])
        #inner
        for i in np.arange(0, .5, frac_grain[0]):
            for j in np.arange(0, 1, frac_grain[1]):
                if 0 < i:
                    equal_2.append([i, j, 0])
        dau_area = equal_1 + equal_2
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_004(frac_grain):
        """
        plane space group P1g1
        """
        equal_2 = []
        #boundary
        for j in np.arange(0, .5, frac_grain[1]):
                equal_2.append([0, j, 0])
                equal_2.append([.5, j, 0])
        #inner
        for i in np.arange(0, .5, frac_grain[0]):
            for j in np.arange(0, 1, frac_grain[1]):
                if 0 < i:
                    equal_2.append([i, j, 0])
        dau_area = equal_2
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_005(frac_grain):
        """
        plane space group C1m1
        """
        equal_2, equal_4 = [], []
        #boundary
        for j in np.arange(0, .5, frac_grain[1]):
            equal_2.append([0, j, 0])
            equal_2.append([.5, j, 0])
        #inner
        for i in np.arange(0, .5, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 0 < i:
                    equal_4.append([i, j, 0])
        dau_area = equal_2 + equal_4
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_025(frac_grain):
        """
        plane space group P2mm
        """
        equal_2, equal_4 = [], []
        #point
        equal_1 = [[0, 0, 0], [0, .5, 0], [.5, 0, 0], [.5, .5, 0]]
        #boundary
        for i in np.arange(0, .5, frac_grain[0]):
            if 0 < i:
                equal_2.append([i, 0, 0])
                equal_2.append([i, .5, 0])
        for j in np.arange(0, .5, frac_grain[1]):
            if 0 < j:
                equal_2.append([0, j, 0])
                equal_2.append([.5, j, 0])
        #inner
        for i in np.arange(0, .5, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 0 < i and 0 < j:
                    equal_4.append([i, j, 0])
        dau_area = equal_1 + equal_2 + equal_4
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_028(frac_grain):
        """
        plane space group P2mg
        """
        equal_4 = []
        #point
        equal_2 = [[0, 0, 0], [0, .5, 0]]
        #boundary
        for j in np.arange(0, 1, frac_grain[1]):
            equal_2.append([.25, j, 0])
        for i in np.arange(0, .25, frac_grain[0]):
            if 0 < i:
                equal_4.append([i, 0, 0])
        for j in np.arange(0, .5, frac_grain[1]):
            if 0 < j:
                equal_4.append([0, j, 0])
        #inner
        for i in np.arange(0, .25, frac_grain[0]):
            for j in np.arange(0, 1, frac_grain[1]):
                if 0 < i and 0 < j:
                    equal_4.append([i, j, 0])
        dau_area = equal_2 + equal_4
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_032(frac_grain):
        """
        plane space group P2gg
        """
        equal_4 = []
        #point
        equal_2 = [[0, 0, 0], [0, .5, 0]]
        #boundary
        for i in np.arange(0, .5, frac_grain[0]):
            if 0 < i:
                equal_4.append([i, 0, 0])
        for j in np.arange(0, .5, frac_grain[1]):
            if 0 < j:
                equal_4.append([0, j, 0])
        #inner
        for i in np.arange(0, .5, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 0 < i and 0 < j:
                    equal_4.append([i, j, 0])
        dau_area = equal_2 + equal_4
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_035(frac_grain):
        """
        plane space group C2mm
        """
        equal_8 = []
        #point
        equal_2 = [[0, 0, 0], [0, .5, 0]]
        equal_4 = [[.25, .25, 0], [.25, 0., 0]]
        #boundary
        for i in np.arange(0, .25, frac_grain[0]):
            if 0 < i:
                equal_4.append([i, 0, 0])
                equal_4.append([i, .5, 0])
        for j in np.arange(0, .5, frac_grain[1]):
            if 0 < j:
                equal_4.append([0, j, 0])
            if 0 < j < .25:
                equal_8.append([.25, j, 0])
        #inner
        for i in np.arange(0, .25, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 0 < i and 0 < j:
                    equal_8.append([i, j, 0])
        dau_area = equal_2 + equal_4 + equal_8
        return np.array(dau_area, dtype=float)
    
    def tetragonal_075(frac_grain):
        """
        plane space group P4
        """
        equal_4 = []
        #point
        equal_1 = [[0, 0, 0], [.5, .5, 0]]
        equal_2 = [[0, .5, 0]]
        #boundary
        for i in np.arange(0, .5, frac_grain[0]):
            if 0 < i:
                equal_4.append([i, .5, 0])
        for j in np.arange(0, .5, frac_grain[1]):
            if 0 < j:
                equal_4.append([0, j, 0])
        #inner
        for i in np.arange(0, .5, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 0 < i and 0 < j:
                    equal_4.append([i, j, 0])
        dau_area = equal_1 + equal_2 + equal_4
        return np.array(dau_area, dtype=float)
    
    def tetragonal_099(frac_grain, delta=1e-10):
        """
        plane space group P4mm
        """
        equal_4, equal_8 = [], []
        #point
        equal_1 = [[0, 0, 0], [.5, .5, 0]]
        equal_2 = [[0, .5, 0]]
        #boundary
        for i in np.arange(0, .5, frac_grain[0]):
            if 0 < i:
                equal_4.append([i, 0, 0])
                equal_4.append([i, i, 0])
        for j in np.arange(0, .5, frac_grain[1]):
            if 0 < j:
                equal_4.append([.5, j, 0])
        #inner
        for i in np.arange(0, .5, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 0 < i and 0 < j < i-delta:
                    equal_4.append([i, j, 0])
        dau_area = equal_1 + equal_2 + equal_4 + equal_8
        return np.array(dau_area, dtype=float)
    
    def tetragonal_100(frac_grain, delta=1e-10):
        """
        plane space group P4gm
        """
        equal_4, equal_8 = [], []
        #point
        equal_2 = [[0, 0, 0], [0, .5, 0]]
        #boundary
        for i in np.arange(0, .5, frac_grain[0]):
            if 0 < i:
                equal_4.append([i, -i+.5, 0])
                equal_8.append([i, 0, 0])
        #inner
        for i in np.arange(0, .5, frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 0 < i and 0 < j < -i+.5-delta:
                    equal_8.append([i, j, 0])
        dau_area = equal_2 + equal_4 + equal_8
        return np.array(dau_area, dtype=float)
    
    def hexagonal_143(frac_grain, delta=1e-10):
        """
        plane space group P3
        """
        equal_3 = []
        #point
        equal_1 = [[0, 0, 0], [Fraction(1, 3), Fraction(2, 3), 0], [Fraction(2, 3), Fraction(1, 3), 0]]
        #boundary
        for i in np.arange(0, Fraction(2, 3), frac_grain[0]):
            if 0 < i:
                equal_3.append([i, .5*i, 0])
        for j in np.arange(0, Fraction(2, 3), frac_grain[1]):
            if 0 < j:
                equal_3.append([.5*j, j, 0])
        #inner
        for i in np.arange(0, 1, frac_grain[0]):
            for j in np.arange(0, 1, frac_grain[1]):
                if .5*i < j-delta and 2*i-1 < j-delta and j < 2*i-delta and j < .5*i+.5-delta:
                    equal_3.append([i, j, 0])
        dau_area = equal_1 + equal_3
        return np.array(dau_area, dtype=float)

    def hexagonal_156(frac_grain, delta=1e-10):
        """
        plane space group P3m1
        """
        equal_3, equal_6 = [], []
        #point
        equal_1 = [[0, 0, 0], [Fraction(1, 3), Fraction(2, 3), 0], [Fraction(2, 3), Fraction(1, 3), 0]]
        #boundary
        for i in np.arange(0, Fraction(2, 3), frac_grain[0]):
            if 0 < i:
                equal_3.append([i, .5*i, 0])
            if Fraction(1, 3) < i:
                equal_3.append([i, -i+1, 0])
        for j in np.arange(0, Fraction(2, 3), frac_grain[1]):
            if 0 < j:
                equal_3.append([.5*j, j, 0])
        #inner
        for i in np.arange(0, Fraction(2, 3), frac_grain[0]):
            for j in np.arange(0, Fraction(2, 3), frac_grain[1]):
                if .5*i < j-delta and 0 < j < 2*i-delta and j < -i+1-delta:
                    equal_6.append([i, j, 0])
        dau_area = equal_1 + equal_3 + equal_6 
        return np.array(dau_area, dtype=float)
    
    def hexagonal_157(frac_grain, delta=1e-10):
        """
        plane space group P31m
        """
        equal_6 = []
        #point
        equal_1 = [[0, 0, 0]]
        equal_2 = [[Fraction(1, 3), Fraction(2, 3), 0]]
        equal_3 = [[.5, .5, 0]]
        #boundary
        for i in np.arange(0, .5, frac_grain[0]):
            if 0 < i:
                equal_3.append([i, i, 0])
                equal_3.append([i, 0, 0])
        for i in np.arange(.5, Fraction(2, 3), frac_grain[0]):
            if .5 < i:
                equal_6.append([i, 2*i-1, 0])
        #inner
        for i in np.arange(0, Fraction(2, 3), frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 2*i-1 < j-delta and 0 < j < i-delta and j < -i+1-delta:
                    equal_6.append([i, j, 0])
        dau_area = equal_1 + equal_2 + equal_3 + equal_6
        return np.array(dau_area, dtype=float)
    
    def hexagonal_168(frac_grain, delta=1e-10):
        """
        plane space group P6
        """
        equal_6 = []
        #point
        equal_1 = [[0, 0, 0]]
        equal_2 = [[Fraction(1, 3), Fraction(2, 3), 0]]
        equal_3 = [[.5, .5, 0]]
        #boundary
        for i in np.arange(0, .5, frac_grain[0]):
            if 0 < i:
                equal_6.append([i, 0, 0])
        for i in np.arange(.5, Fraction(2, 3), frac_grain[0]):
            if .5 < i:
                equal_6.append([i, 2*i-1, 0])
        #inner
        for i in np.arange(0, Fraction(2, 3), frac_grain[0]):
            for j in np.arange(0, .5, frac_grain[1]):
                if 2*i-1 < j-delta and 0 < j < i-delta and j < -i+1-delta:
                    equal_6.append([i, j, 0])
        dau_area = equal_1 + equal_2 + equal_3 + equal_6
        return np.array(dau_area, dtype=float)
    
    def hexagonal_183(frac_grain, delta=1e-10):
        """
        plane space group P6mm
        """
        equal_6, equal_12 = [], []
        #point
        equal_1 = [[0, 0, 0]]
        equal_2 = [[Fraction(1, 3), Fraction(2, 3), 0]]
        equal_3 = [[.5, .5, 0]]
        #boundary
        for i in np.arange(0, Fraction(2, 3), frac_grain[0]):
            if 0 < i:
                equal_6.append([i, .5*i, 0])
            if 0 < i < .5:
                equal_6.append([i, 0, 0])
        for j in np.arange(0, Fraction(1, 3), frac_grain[1]):
            if 0 < j:
                equal_6.append([.5*(j+1), j, 0])
        #inner
        for i in np.arange(0, Fraction(2, 3), frac_grain[0]):
            for j in np.arange(0, Fraction(1, 3), frac_grain[1]):
                if 2*i-1 < j-delta and 0 < j < .5*i-delta:
                    equal_12.append([i, j, 0])
        dau_area = equal_1 + equal_2 + equal_3 + equal_6 + equal_12
        return np.array(dau_area, dtype=float)


class LayerSpaceGroup:
    #80 layer groups
    def __init__(self):
        pass
    

class BulkSpaceGroup:
    #230 space groups
    def __init__(self):
        pass
    
    def triclinic_001(frac_grain=[24, 24, 24]):
        """
        space group group P1
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 <= y < 1 and 0 <= z < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def triclinic_002(frac_grain=[24, 24, 24]):
        """
        space group group P-1
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 < y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def monoclinic_003(frac_grain=[24, 24, 24]):
        """
        space group group P121
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 <= y < 1 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and 0 <= x <= 1/2 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and 0 <= x <= 1/2 and 0 <= y < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def monoclinic_004(frac_grain=[24, 24, 24]):
        """
        space group group P1211
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 <= y < 1 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and 0 < x < 1/2 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and 0 < x < 1/2 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif z == 0 and x == 0 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and x == 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and x == 0 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and x == 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def monoclinic_005(frac_grain=[24, 24, 24]):
        """
        space group group C121
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.5)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z <= 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def monoclinic_006(frac_grain=[24, 24, 24]):
        """
        space group group P1m1
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 <= y <= 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def monoclinic_007(frac_grain=[24, 24, 24]):
        """
        space group group P1c1
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 < y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif y == 0 and 0 <= z < 1/2 and 0 <= x < 1:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and 0 <= z < 1/2 and 0 <= x < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def monoclinic_008(frac_grain=[24, 24, 24]):
        """
        space group group C1m1
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 <= y < 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif y == 1/4 and 0 <= x < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def monoclinic_009(frac_grain=[24, 24, 24]):
        """
        space group group C1c1
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 < y < 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif y == 0 and 0 <= z < 1/2 and 0 <= x < 1:
                        dau_area.append([x, y, z])
                    elif y == 1/4 and 0 <= z < 1/2 and 0 <= x < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def monoclinic_010(frac_grain=[24, 24, 24]):
        """
        space group group P12/m1
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y <= 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/2 and 0 <= y <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z <= 1/2 and 0 <= y <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def monoclinic_011(frac_grain=[24, 24, 24]):
        """
        space group group P121/m1
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 < y <= 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif y == 0 and 0 < z < 1/2 and 0 <= x < 1:
                        dau_area.append([x, y, z])
                    elif y == 0 and z == 0 and 0 <= x <= 1/2:
                        dau_area.append([x, y, z])
                    elif y == 0 and z == 1/2 and 0 <= x <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def monoclinic_012(frac_grain=[24, 24, 24]):
        """
        space group group C12/m1
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/2 and 0 <= y <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z <= 1/2 and 0 <= y < 1/4:
                        dau_area.append([x, y, z])
                    elif y == 1/4 and 0 < x < 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif y == 1/4 and x == 1/4 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def monoclinic_013(frac_grain=[24, 24, 24]):
        """
        space group group P12/c1
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < z <= 1/4 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 < z <= 1/4 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and z == 0 and 0 <= y <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and z == 0 and 0 <= y <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def monoclinic_014(frac_grain=[24, 24, 24]):
        """
        space group group P121/c1
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1 and 0 < y < 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < y < 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif y == 0 and 0 < x < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif y == 1/4 and 0 <= z < 1/2 and 0 < x < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 1/4 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def monoclinic_015(frac_grain=[24, 24, 24]):
        """
        space group group C12/c1
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < z <= 1/4 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 < z <= 1/4 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and 0 <= y < 1/4 and 0 < x < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and 1/4 < y < 1/2 and 0 < x < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and z == 0 and 0 <= y <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and z == 0 and 0 <= y < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 0 and y == 1/4 and 0 < x <= 1/4:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and y == 1/4 and 0 < x <= 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_016(frac_grain=[24, 24, 24]):
        """
        space group group P222
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 < y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/2 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z <= 1/2 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/2 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/2 and y == 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z <= 1/2 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z <= 1/2 and y == 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_017(frac_grain=[24, 24, 24]):
        """
        space group group P2221
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 < y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 1/4 <= z <= 3/4 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 1/4 <= z <= 3/4 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 1/4 <= z <= 1/2 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 1/4 <= z <= 1/2 and y == 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 1/4 <= z <= 1/2 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 1/4 <= z <= 1/2 and y == 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_018(frac_grain=[24, 24, 24]):
        """
        space group group P21212
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/2 and 0 <= y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_019(frac_grain=[24, 24, 24]):
        """
        space group group P212121
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/2 and 0 < y < 1/2 and 0 < z < 1:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/2 and y == 0 and 1/2 <= z < 1:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/2 and y == 1/2 and 0 < z <= 1/2:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/2 and 0 < y < 1/2 and z == 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_020(frac_grain=[24, 24, 24]):
        """
        space group group C2221
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/4 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 1/4 <= z < 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/2 and y == 1/2 and z == 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_021(frac_grain=[24, 24, 24]):
        """
        space group group C222
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/4 and 0 < y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/2 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 < y <= 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and y == 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/2 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/2 and y == 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_022(frac_grain=[24, 24, 24]):
        """
        space group group F222
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/4 and 0 < y < 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/2 and 0 < y < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 1/4 <= z <= 3/4 and 0 < y < 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and y == 1/4 and 1/4 <= z <= 3/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/2 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 1/4 <= z <= 1/2 and y == 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 1/4 <= z <= 1/2 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 1/4 <= z <= 3/4 and y == 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_023(frac_grain=[24, 24, 24]):
        """
        space group group I222
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/2 and 0 <= y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/2 and y == 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and x == 0 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_024(frac_grain=[24, 24, 24]):
        """
        space group group I212121
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 < y < 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < y <= 1/4 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 < y <= 1/4 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 0 and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 1/2 and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/4 and 0 < y < 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/4 and 0 < y < 1/2 and z == 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y <= 1/4 and z == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < y <= 1/4 and z == 1/2:
                        dau_area.append([x, y, z])
                    elif y == 0 and 0 < z <= 1/4 and x == 0:
                        dau_area.append([x, y, z])
                    elif y == 0 and 0 < z <= 1/4 and x == 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/4 and y == 0 and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/4 and y == 1/2 and z == 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_025(frac_grain=[24, 24, 24]):
        """
        space group group Pmm2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 <= y <= 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_026(frac_grain=[24, 24, 24]):
        """
        space group group Pmc21
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 < y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/2 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/2 and y == 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_027(frac_grain=[24, 24, 24]):
        """
        space group group Pcc2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 < y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/2 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z < 1/2 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/2 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/2 and y == 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z < 1/2 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z < 1/2 and y == 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_028(frac_grain=[24, 24, 24]):
        """
        space group group Pma2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/4 and 0 <= y < 1 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y <= 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_029(frac_grain=[24, 24, 24]):
        """
        space group group Pca21
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/4 and 0 <= y < 1 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/2 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 <= z < 1/2 and 0 <= y < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_030(frac_grain=[24, 24, 24]):
        """
        space group group Pnc2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y <= 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= y <= 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_031(frac_grain=[24, 24, 24]):
        """
        space group group Pmn21
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 < y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/2 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/2 and y == 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_032(frac_grain=[24, 24, 24]):
        """
        space group group Pba2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/2 and 0 <= y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_033(frac_grain=[24, 24, 24]):
        """
        space group group Pna21
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/2 and 0 < y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/2 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/2 and y == 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_034(frac_grain=[24, 24, 24]):
        """
        space group group Pnn2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/2 and 0 <= y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_035(frac_grain=[24, 24, 24]):
        """
        space group group Cmm2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/4 and 0 <= y <= 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 <= y <= 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_036(frac_grain=[24, 24, 24]):
        """
        space group group Cmcm21
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 <= y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_037(frac_grain=[24, 24, 24]):
        """
        space group group Ccc2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/4 and 0 < y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/2 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 < y <= 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and y == 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/2 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/2 and y == 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_038(frac_grain=[24, 24, 24]):
        """
        space group group Amm2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 <= y <= 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_039(frac_grain=[24, 24, 24]):
        """
        space group group Abm2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 < y <= 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/2 and 0 < y <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z < 1/2 and 0 < y <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/2 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z < 1/2 and y == 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_040(frac_grain=[24, 24, 24]):
        """
        space group group Ama2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/4 and 0 < y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/2 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/4 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/4 and y == 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/2 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/2 and y == 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_041(frac_grain=[24, 24, 24]):
        """
        space group group Aba2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/2 and 0 <= y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_042(frac_grain=[24, 24, 24]):
        """
        space group group Fmm2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/4 and 0 <= y < 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 <= z < 1/2 and 0 <= y < 1/4:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/4 and y == 1/4 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 <= z < 1/2 and y == 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_043(frac_grain=[24, 24, 24]):
        """
        space group group Fdd2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/4 and 0 <= y < 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_044(frac_grain=[24, 24, 24]):
        """
        space group group Imm2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 <= y <= 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_045(frac_grain=[24, 24, 24]):
        """
        space group group Iba2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/2 and 0 <= y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_046(frac_grain=[24, 24, 24]):
        """
        space group group Ima2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/4 and 0 <= y < 1 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y <= 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_047(frac_grain=[24, 24, 24]):
        """
        space group group Pmmm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 <= y <= 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_048(frac_grain=[24, 24, 24]):
        """
        space group group Pnnn
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/4 and 0 < y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y <= 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 < y < 1/2 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 < y < 1/2 and 1/2 < z < 3/4:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 0 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 < y <= 1/4 and z == 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 < y < 1/2 and z == 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 < y <= 1/4 and z == 3/4:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 0 and z == 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 1/2 and z == 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_049(frac_grain=[24, 24, 24]):
        """
        space group group Pccm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 < y < 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/4 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z <= 1/4 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 1/2 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/4 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/4 and y == 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z <= 1/4 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z <= 1/4 and y == 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_050(frac_grain=[24, 24, 24]):
        """
        space group group Pban
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/4 and 0 <= y < 1 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y < 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 <= y < 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and 0 <= y <= 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and 0 <= y <= 1/2 and z == 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y < 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y < 1/2 and z == 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 <= y <= 1/4 and z == 0:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 <= y <= 1/4 and z == 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_051(frac_grain=[24, 24, 24]):
        """
        space group group Pmma
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/4 and 0 <= y <= 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/2 and 0 <= y <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_052(frac_grain=[24, 24, 24]):
        """
        space group group Pnna
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 < y < 1/4 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif 1/4 <= x <= 3/4 and y == 0 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif y == 1/4 and 0 < z <= 1/4 and 0 <= x < 1:
                        dau_area.append([x, y, z])
                    elif 1/2 <= x < 1 and 0 < y < 1/4 and z == 0:
                        dau_area.append([x, y, z])
                    elif 1/2 <= x < 1 and 0 < y < 1/4 and z == 1/2:
                        dau_area.append([x, y, z])
                    elif 1/2 <= x <= 3/4 and y == 0 and z == 0:
                        dau_area.append([x, y, z])
                    elif 1/2 <= x <= 3/4 and y == 0 and z == 1/2:
                        dau_area.append([x, y, z])
                    elif 1/2 <= x < 1 and y == 1/4 and z == 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_053(frac_grain=[24, 24, 24]):
        """
        space group group Pmna
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 <= y < 1 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/2 and 0 <= y <= 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/4 and 0 <= y < 1 and z == 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_054(frac_grain=[24, 24, 24]):
        """
        space group group Pcca
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 < y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < y < 1/2 and 1/4 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 < y < 1/2 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif 1/4 <= x < 1/2 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif 1/4 <= x < 1/2 and y == 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 1/2 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_055(frac_grain=[24, 24, 24]):
        """
        space group group Pbam
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/2 and 0 <= y < 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_056(frac_grain=[24, 24, 24]):
        """
        space group group Pccn
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/4 and 0 <= y < 1 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 1/4 <= y <= 3/4 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([0, .5, 0])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_057(frac_grain=[24, 24, 24]):
        """
        space group group Pbcm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1 and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and 1/2 <= y < 1 and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 1/2 <= y < 1 and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and 1/4 <= y <= 3/4 and z == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 1/2 <= y <= 3/4 and z == 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 1/2 <= y <= 3/4 and z == 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_058(frac_grain=[24, 24, 24]):
        """
        space group group Pnnm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/2 and 0 <= y < 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_059(frac_grain=[24, 24, 24]):
        """
        space group group Pmmn
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 <= y <= 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/4 and 0 <= y <= 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/4 and 0 <= y <= 1/2 and z == 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and x == 1/4 and 0 <= y <= 1/4:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and x == 1/4 and 0 <= y <= 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)

    def orthorhombic_060(frac_grain=[24, 24, 24]):
        """
        space group group Pbcn
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/4 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 1/4 <= z < 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([0, .5, 0])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_061(frac_grain=[24, 24, 24]):
        """
        space group group Pbca
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/2 and 0 <= y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([0, .5, 0])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_062(frac_grain=[24, 24, 24]):
        """
        space group group Pnma
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/2 and 0 < y <= 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/2 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([.5, 0, 0])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_063(frac_grain=[24, 24, 24]):
        """
        space group group Cmcm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 <= y < 1/2 and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/2 and 0 <= y < 1/4 and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/4 and y == 1/4 and z == 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_064(frac_grain=[24, 24, 24]):
        """
        space group group Cmca
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/4 and 0 <= y < 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 <= y < 1/2 and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/4 and 0 <= y <= 1/4 and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/4 and 0 <= y < 1/4 and z == 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 <= y <= 1/4 and z == 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_065(frac_grain=[24, 24, 24]):
        """
        space group group Cmmm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/4 and 0 <= y <= 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 <= y <= 1/4 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_066(frac_grain=[24, 24, 24]):
        """
        space group group Cccm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/4 and 0 < y < 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/4 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 < y <= 1/4 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and y == 1/2 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/4 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/4 and y == 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])   
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_067(frac_grain=[24, 24, 24]):
        """
        space group group Cmma
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 < y <= 1/4 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/4 and y == 0 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/4 and 0 < y <= 1/4 and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/4 and 0 < y <= 1/4 and z == 1/2:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/4 and y == 0 and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/4 and y == 0 and z == 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_068(frac_grain=[24, 24, 24]):
        """
        space group group Ccca
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 < y < 1/4 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < z < 1/4 and 0 < y < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 < z <= 1/4 and 0 < y <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/4 and y == 1/4 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/4 and y == 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/2 and 0 < y < 1/4 and z == 0:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and y == 0 and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/2 and y == 0 and z == 0:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([0, .25, 0])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_069(frac_grain=[24, 24, 24]):
        """
        space group group Fmmm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/4 and 0 <= y < 1/4 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 <= z <= 1/4 and 0 <= y < 1/4:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/4 and y == 1/4 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 <= z <= 1/4 and y == 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_070(frac_grain=[24, 24, 24]):
        """
        space group group Fddd
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/8 and 0 < y < 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 1/8 and 1/8 < z < 5/8 and 0 < y < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z <= 1/2 and 1/4 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/4 and y == 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 1/4 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/8 and 0 < y <= 1/8 and z == 1/8:
                        dau_area.append([x, y, z])
                    elif x == 1/8 and 0 < y <= 1/8 and z == 5/8:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_071(frac_grain=[24, 24, 24]):
        """
        space group group Immm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/4 and 0 <= y <= 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 <= y < 1/4 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and y == 1/4 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_072(frac_grain=[24, 24, 24]):
        """
        space group group Ibam
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/4 and 0 < y < 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < y < 1/2 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 < y < 1/4 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and y == 1/2 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 1/2 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and y == 1/4 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_073(frac_grain=[24, 24, 24]):
        """
        space group group Ibca
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/4 and 0 < y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < y <= 1/4 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 <= z < 1/4 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and y == 1/2 and 1/4 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 0 <= y <= 1/4 and z == 1/4:
                        dau_area.append([x, y, z])
                    elif y == 0 and x == 1/4 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def orthorhombic_074(frac_grain=[24, 24, 24]):
        """
        space group group Imma
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/4 and 0 < y <= 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and 1/4 <= z <= 3/4 and 0 < y <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/4 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and y == 0 and 1/4 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_075(frac_grain=[24, 24, 24]):
        """
        space group group P4
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 <= y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and x == 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_076(frac_grain=[24, 24, 24]):
        """
        space group group P41
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 < y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/4 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z < 1/4 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 0 and 0 <= z < 3/4:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 1/2 and 0 <= z < 3/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/4 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/4 and y == 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z < 1/4 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z < 1/4 and y == 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_077(frac_grain=[24, 24, 24]):
        """
        space group group P42
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 < y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/2 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z < 1/2 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/2 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z < 1/2 and y == 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z < 1/2 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z < 1/2 and y == 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_078(frac_grain=[24, 24, 24]):
        """
        space group group P43
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1.1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 < y < 1/2 and 0 < z <= 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 3/4 < z <= 1 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 3/4 < z <= 1 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 0 and 1/4 < z <= 1:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 1/2 and 1/4 < z <= 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 3/4 < z <= 1 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 3/4 < z <= 1 and y == 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 3/4 < z <= 1 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 3/4 < z <= 1 and y == 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_079(frac_grain=[24, 24, 24]):
        """
        space group group I4
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 <= y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and x == 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_080(frac_grain=[24, 24, 24]):
        """
        space group group I41
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y <= 1/2 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= y <= 1/2 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_081(frac_grain=[24, 24, 24]):
        """
        space group group P-4
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 <= y < 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and x == 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_082(frac_grain=[24, 24, 24]):
        """
        space group group I-4
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < z < 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 0 and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and x == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/2 and y == 0 and z == 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([0, 0, 0])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_083(frac_grain=[24, 24, 24]):
        """
        space group group P4/m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 <= y < 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and x == 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_084(frac_grain=[24, 24, 24]):
        """
        space group group P42/m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 <= y < 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and x == 1/2 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_085(frac_grain=[24, 24, 24]):
        """
        space group group P4/n
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 < y < 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < y <= 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and 0 < y < 1/4 and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and 0 < y < 1/4 and z == 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/4 and y == 1/4 and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/4 and y == 1/4 and z == 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_086(frac_grain=[24, 24, 24]):
        """
        space group group P42/n
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 < y < 1 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y < 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/2 and y == 0 and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and 0 < y < 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and 0 < y < 1/4 and z == 1/4:
                        dau_area.append([x, y, z])
                    elif 1/2 < x < 1 and 0 < y < 1/4 and z == 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 1/2 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/4 and y == 1/4 and z == 1/4:
                        dau_area.append([x, y, z])
                    elif 1/2 < x <= 3/4 and y == 1/4 and z == 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_087(frac_grain=[24, 24, 24]):
        """
        space group group Im/4
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 <= y < 1/2 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and 0 <= y < 1/4 and z == 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and x == 1/2 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/4 and y == 1/4 and z == 1/4:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, .25], [.5, 0, .25]]
        return np.array(dau_area, dtype=float)
    
    def tetragonal_088(frac_grain=[24, 24, 24]):
        """
        space group group I41/a
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.3)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/4 and 0 < y < 1/4 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif y == 0 and x == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif y == 1/4 and x == 0 and 1/8 <= z <= 5/8:
                        dau_area.append([x, y, z])
                    elif y == 1/4 and x == 1/4 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_089(frac_grain=[24, 24, 24]):
        """
        space group group P422
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 <= y < 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and 0 <= y < 1/2 and 0 < x <= 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and x-y >= 0 and 0 <= y < 1/2 and 0 < x <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and x == 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, 0], [0, 0, .5], [.5, .5, 0], [.5, .5, .5]]
        return np.array(dau_area, dtype=float)
    
    def tetragonal_090(frac_grain=[24, 24, 24]):
        """
        space group group P4212
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/2 and 0 <= y < 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and 0 <= x < 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and x-y >= 0 and 0 <= x < 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 0 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and x == 0 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[.5, 0, 0], [.5, 0, .5]]
        return np.array(dau_area, dtype=float)
    
    def tetragonal_091(frac_grain=[24, 24, 24]):
        """
        space group group P4122
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.2)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1 and 0 <= y < 1 and 0 < z < 1/8:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < z < 1/8 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/2 and 0 <= y < 1 and z == 0:
                        dau_area.append([x, y, z])
                    elif z == 1/8 and x+y <= 1 and 0 < x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif z == 0 and x == 0 and 0 <= y < 1:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([0, 0, .125])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_092(frac_grain=[24, 24, 24]):
        """
        space group group P41212
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.2)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 <= y < 1 and 0 < z < 1/8:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and 0 <= x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1 and 1/2 <= y < 1 and z == 1/8:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_093(frac_grain=[24, 24, 24]):
        """
        space group group P4222
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y <= 1/2 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= y <= 1/2 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and 0 <= y <= 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and -x+y >= 0 and x+y <= 1 and 0 < x < 1/2 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y <= 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y <= 1/2 and z == 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= y <= 1/2 and z == 0:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([.5, .5, .25])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_094(frac_grain=[24, 24, 24]):
        """
        space group group P42212
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 < y < 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 < z < 1/2 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 0 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and 0 < x < 1/2 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and x-y >= 0 and 0 < x < 1/2 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 < y < 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 < z < 1/2 and y == 0:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 0 and z == 0:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, .5], [0, .5, 0], [.5, .5, .5]]
        return np.array(dau_area, dtype=float)
    
    def tetragonal_095(frac_grain=[24, 24, 24]):
        """
        space group group P4322
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1.1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.2)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1 and 0 <= y < 1 and 0 < z < 1/8:
                        dau_area.append([x, y, z])
                    elif x == 1 and 0 < z < 1/8 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif 1/2 <= x < 1 and 0 <= y < 1 and z == 0:
                        dau_area.append([x, y, z])
                    elif z == 1/8 and x-y >= 0 and 0 < x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif x == 1 and 0 <= y < 1 and z == 0:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([1, 0, .125])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_096(frac_grain=[24, 24, 24]):
        """
        space group group P43212
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.2)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 <= y < 1 and 0 < z < 1/8:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and 0 <= x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif 1/2 <= x < 1 and 0 <= y < 1 and z == 1/8:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_097(frac_grain=[24, 24, 24]):
        """
        space group group I422
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 <= y < 1/2 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and 0 < x <= 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x+y <= 1/2 and 0 < x <= 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and x == 1/2 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, 0], [0, 0, .25], [.5, .5, 0]]
        return np.array(dau_area, dtype=float)
    
    def tetragonal_098(frac_grain=[24, 24, 24]):
        """
        space group group I4122
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.2)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1 and 0 < z < 1/8:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y <= 1/2 and 0 < z < 1/8:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= y <= 1/2 and 0 < z < 1/8:
                        dau_area.append([x, y, z])
                    elif z == 0 and x+y <= 1 and -x+y >= 0 and 0 < x < 1/2 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and 1/4 <= y <= 3/4 and z == 1/8:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y <= 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 1/4 <= y <= 1/2 and z == 1/8:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 1/4 <= y <= 1/2 and z == 1/8:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([.5, .5, 0])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_099(frac_grain=[24, 24, 24]):
        """
        space group group P4mm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x and y <= 1/2 and -x+y >= 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_100(frac_grain=[24, 24, 24]):
        """
        space group group P4bm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and x+y <= 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_101(frac_grain=[24, 24, 24]):
        """
        space group group P42cm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and y < 1/2 and -x+y >= 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y < 1/2 and -x+y >= 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and 0 < x and -x+y >= 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_102(frac_grain=[24, 24, 24]):
        """
        space group group P42nm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and y < 1/2 and -x+y >= 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y < 1/2 and -x+y >= 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and 0 < x and -x+y >= 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_103(frac_grain=[24, 24, 24]):
        """
        space group group P4cc
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 <= y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and x == 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_104(frac_grain=[24, 24, 24]):
        """
        space group group P4nc
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 <= y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and x == 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_105(frac_grain=[24, 24, 24]):
        """
        space group group P42mc
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 <= y <= 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_106(frac_grain=[24, 24, 24]):
        """
        space group group P42bc
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 <= y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_107(frac_grain=[24, 24, 24]):
        """
        space group group I4mm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x and y <= 1/2 and -x+y >= 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_108(frac_grain=[24, 24, 24]):
        """
        space group group I4cm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and x+y <= 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_109(frac_grain=[24, 24, 24]):
        """
        space group group I41md
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 <= y <= 1/2 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_110(frac_grain=[24, 24, 24]):
        """
        space group group I41cd
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 <= y < 1/2 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_111(frac_grain=[24, 24, 24]):
        """
        space group group P-42m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and y < 1/2 and -x+y >= 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y < 1/2 and -x+y >= 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and 0 < x and -x+y >= 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_112(frac_grain=[24, 24, 24]):
        """
        space group group P-42c
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 < y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < z <= 1/4 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z <= 1/4 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 1/2 and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= z <= 1/4 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < z <= 1/4 and y == 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z <= 1/4 and y == 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= z <= 1/4 and y == 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_113(frac_grain=[24, 24, 24]):
        """
        space group group P-421m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and x+y <= 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_114(frac_grain=[24, 24, 24]):
        """
        space group group P-421c
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 <= y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([.5, .5, 0])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_115(frac_grain=[24, 24, 24]):
        """
        space group group P-4m2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 <= y <= 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and 0 <= x <= 1/2 and 0 <= y <= 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and x-y >= 0 and 0 <= x <= 1/2 and 0 <= y <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_116(frac_grain=[24, 24, 24]):
        """
        space group group P-4c2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y <= 1/2 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= y <= 1/2 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and 0 < y <= 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x+y <= 1 and -x+y >= 0 and 0 < x < 1/2 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < y <= 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y <= 1/2 and z == 1/4:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, 0], [.5, .5, 0], [.5, .5, .25]]
        return np.array(dau_area, dtype=float)
    
    def tetragonal_117(frac_grain=[24, 24, 24]):
        """
        space group group P-4b2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < z < 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and x+y <= 1/2 and 0 < x < 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and x+y <= 1/2 and 0 < x < 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 0 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, 0], [0, 0, .5], [.5, 0, 0], [.5, 0, .5]]
        return np.array(dau_area, dtype=float)
    
    def tetragonal_118(frac_grain=[24, 24, 24]):
        """
        space group group P-4n2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y <= 1/2 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= y <= 1/2 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and 0 <= y < 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and -x+y <= 1/2 and x+y >= 1/2 and 0 < x < 1/2 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= y <= 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= y <= 1/2 and z == 1/4:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, 0], [0, .5, .25]]
        return np.array(dau_area, dtype=float)
    
    def tetragonal_119(frac_grain=[24, 24, 24]):
        """
        space group group I-4m2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 <= y <= 1/2 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and 0 <= x <= 1/2 and 0 <= y <= 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x+y <= 1/2 and 0 <= x <= 1/2 and 0 <= y <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_120(frac_grain=[24, 24, 24]):
        """
        space group group I-4c2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1/2 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < z < 1/4 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and x+y <= 1/2 and 0 < x < 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x-y >= 0 and 0 < x < 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 0 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, 0], [0, 0, .25], [.5, 0, 0], [.5, 0, .25]]
        return np.array(dau_area, dtype=float)
    
    def tetragonal_121(frac_grain=[24, 24, 24]):
        """
        space group group I-42m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x and y < 1/2 and -x+y >= 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and x == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and x == 0 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_122(frac_grain=[24, 24, 24]):
        """
        space group group I-42d
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1 and 0 < z < 1/8:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y <= 1/2 and 0 < z < 1/8:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= y <= 1/2 and 0 < z < 1/8:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and 0 <= y < 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and 1/4 <= y <= 3/4 and z == 1/8:
                        dau_area.append([x, y, z])
                    elif x == 0 and 1/4 <= y <= 1/2 and z == 1/8:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= y < 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 1/4 <= y <= 1/2 and z == 1/8:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, 0], [.5, .5, 0]]
        return np.array(dau_area, dtype=float)
    
    def tetragonal_123(frac_grain=[24, 24, 24]):
        """
        space group group P4/mmm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x and y <= 1/2 and -x+y >= 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_124(frac_grain=[24, 24, 24]):
        """
        space group group P4/mcc
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 <= y < 1/2 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x-y >= 0 and 0 < x <= 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and x == 1/2 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, .25], [.5, .5, .25]]
        return np.array(dau_area, dtype=float)
    
    def tetragonal_125(frac_grain=[24, 24, 24]):
        """
        space group group P4/nbm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/2 and 0 < y <= 1/2 and x+y <= 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and 0 <= x < 1/2 and 0 <= y <= 1/2 and x+y <= 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and x-y >= 0 and 0 <= x < 1/2 and 0 <= y <= 1/2 and x+y <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, .5, .5], [.5, 0, 0]]
        return np.array(dau_area, dtype=float)
    
    def tetragonal_126(frac_grain=[24, 24, 24]):
        """
        space group group P4/nnc
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 <= y < 1/2 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and 0 <= x <= 1/2 and 0 <= y <= 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x-y > 0 and 0 < x < 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x-y == 0 and 0 <= x <= 1/4 and 0 <= y <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 1/2 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 1/2 and 1/4 < z < 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([.5, 0, .25])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_127(frac_grain=[24, 24, 24]):
        """
        space group group P4/mbm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and x+y <= 1/2 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_128(frac_grain=[24, 24, 24]):
        """
        space group group P4/mnc
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 <= y < 1/2 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x+y <= 1/2 and 0 < x <= 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and x == 1/2 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([0, 0, .25])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_129(frac_grain=[24, 24, 24]):
        """
        space group group P4/nmm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 <= y <= 1/2 and x+y <= 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and 0 <= x <= 1/4 and 0 <= y <= 1/4 and x-y <= 0:
                        dau_area.append([x, y, z])
                    elif z == 0 and 1/4 < x <= 1/2 and 0 <= y <= 1/4 and x+y <= 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and 0 <= x <= 1/4 and 0 <= y <= 1/4 and x-y <= 0:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and 1/4 < x <= 1/2 and 0 <= y <= 1/4 and x+y <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_130(frac_grain=[24, 24, 24]):
        """
        space group group P4/ncc
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/2 and 0 <= y < 1/2 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 0 and x+y <= 1/2 and 0 <= x <= 1/4 and 0 < y <= 1/4:
                        dau_area.append([x, y, z])
                    elif 1/4 < x <= 1/2 and 0 < y < 1/4 and z == 0:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x-y <= 0 and 0 <= x < 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([.5, .5, 0])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_131(frac_grain=[24, 24, 24]):
        """
        space group group P42/mmc
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 <= y <= 1/2 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x-y >= 0 and 0 <= x <= 1/2 and 0 <= y <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_132(frac_grain=[24, 24, 24]):
        """
        space group group P42/mcm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and y < 1/2 and -x+y >= 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y < 1/2 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/2 and y == 1/2 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 1/2 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_133(frac_grain=[24, 24, 24]):
        """
        space group group P42/nbc
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1/2 and 0 <= y < 1/2 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 0 and x+y <= 1/2 and 0 <= x <= 1/2 and 0 < y <= 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x+y < 1/2 and 0 <= x <= 1/2 and 0 <= y <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 1/2 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x+y == 1/2 and 0 <= x <= 1/4 and 0 < y <= 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([.5, .5, 0])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_134(frac_grain=[24, 24, 24]):
        """
        space group group P42/nnm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.8)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 <= y < 1 and x-y <= 0 and x+y <= 1 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/2 and y == 0 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y <= 0 and 0 <= x <= 1/2 and 0 <= y <= 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x-y <= 0 and x+y <= 1/2 and 0 <= x <= 1/2 and 0 <= y <= 1/2:
                        dau_area.append([x, y, z])
                    elif z == 3/4 and x-y <= 0 and x+y <= 1/2 and 0 < x < 1/2 and 0 < y < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_135(frac_grain=[24, 24, 24]):
        """
        space group group P42/mbc
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/2 and 0 <= y < 1/2 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y < 1/2 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x+y <= 1/2 and 0 < x < 1/2 and 0 <= y < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 0 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, .25], [.5, 0, .25]]
        return np.array(dau_area, dtype=float)
    
    def tetragonal_136(frac_grain=[24, 24, 24]):
        """
        space group group P42/mnm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and y < 1/2 and -x+y >= 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 <= y < 1/2 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/2 and y == 1/2 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 1/2 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_137(frac_grain=[24, 24, 24]):
        """
        space group group P42/nmc
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 <= y <= 1/2 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y <= 0 and 0 <= x <= 1/2 and 0 <= y <= 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x+y <= 1/2 and 0 <= x <= 1/4 and 0 <= y <= 1/4:
                        dau_area.append([x, y, z])
                    elif 1/4 < x <= 1/2 and 0 <= y < 1/4 and z == 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_138(frac_grain=[24, 24, 24]):
        """
        space group group P42/ncm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x < 1/4 and 0 < y < 1/2 and x-y < 0 and x+y <= 1/2 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x-y == 0 and 0 <= x <= 1/4 and 0 <= y <= 1/4 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x-y == 0 and 0 < x <= 1/4 and 0 < y <= 1/4 and 1/2 < z <= 3/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and 0 < y < 1/2 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_139(frac_grain=[24, 24, 24]):
        """
        space group group I4/mmm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x and y <= 1/2 and -x+y >= 0 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x+y <= 1/2 and 0 <= x and y <= 1/2 and -x+y >= 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_140(frac_grain=[24, 24, 24]):
        """
        space group group I4/mcm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and x+y <= 1/2 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x-y >= 0 and 0 < x and 0 <= y and x+y <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([0, 0, .25])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_141(frac_grain=[24, 24, 24]):
        """
        space group group I41/amd
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.2)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x <= 1/2 and 0 <= y <= 1/2 and 0 < z < 1/8:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y <= 0 and 0 <= x <= 1/2 and 0 <= y <= 1/2:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/2 and 0 <= y <= 1/4 and z == 1/8:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def tetragonal_142(frac_grain=[24, 24, 24]):
        """
        space group group I41/acd
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.2)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and 0 < y <= 1/2 and 0 < z < 1/8:
                        dau_area.append([x, y, z])
                    elif z == 0 and x+y <= 1/2 and 0 <= x < 1/2 and 0 < y <= 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/4 and 0 < y <= 1/2 and z == 1/8:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 1/2 and 0 < z < 1/8:
                        dau_area.append([x, y, z])
                    elif z == 1/8 and x == 1/2 and 0 <= y <= 1/4:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([.5, .5, 0])
        return np.array(dau_area, dtype=float)
    
    def trigonal_143(frac_grain=[24, 24, 24]):
        """
        space group group P3
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_144(frac_grain=[24, 24, 24]):
        """
        space group group P31
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.4)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 <= y < 1 and 0 <= z < Fraction(1, 3):
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_145(frac_grain=[24, 24, 24]):
        """
        space group group P32
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.4)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 <= y < 1 and 0 <= z < Fraction(1, 3):
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_146(frac_grain=[24, 24, 24]):
        """
        space group group R3
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.4)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 <= z < Fraction(1, 3):
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < Fraction(1, 3):
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 <= z < Fraction(1, 3):
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z < Fraction(1, 3):
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_147(frac_grain=[24, 24, 24]):
        """
        space group group P-3
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y > 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and x-y > 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_148(frac_grain=[24, 24, 24]):
        """
        space group group R-3
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 < z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y > 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif z == Fraction(1, 6) and x-2*y > 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif z == Fraction(1, 6) and -2*x+y > 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif 0 < x <= Fraction(1, 3) and y == x/2 and z == Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 < z <= Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z < Fraction(1, 6):
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_149(frac_grain=[24, 24, 24]):
        """
        space group group P312
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-2*y >= 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif z == 0 and -2*x+y >= 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and x-2*y >= 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and -2*x+y >= 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_150(frac_grain=[24, 24, 24]):
        """
        space group group P321
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and x-y >= 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_151(frac_grain=[24, 24, 24]):
        """
        space group group P3112
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 <= y < 1 and 0 < z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif z == 0 and x-2*y >= 0 and 0 <= x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif z == 0 and -x+2*y >= 1 and 0 <= x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif z == Fraction(1, 6) and -2*x+y >= 0 and 0 <= x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif z == Fraction(1, 6) and 2*x-y >= 1 and 0 <= x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_152(frac_grain=[24, 24, 24]):
        """
        space group group P3121
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 <= y < 1 and 0 < z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif z == 0 and -x+y >= 0 and 0 <= x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif z == Fraction(1, 6) and -x+y >= 0 and 0 <= x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_153(frac_grain=[24, 24, 24]):
        """
        space group group P3212
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 <= y < 1 and 0 < z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif z == 0 and x-2*y >= 0 and 0 <= x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif z == 0 and -x+2*y >= 1 and 0 <= x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif z == Fraction(1, 6) and x+y <= 1 and 0 < x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([0, 0, Fraction(1, 6)])
        return np.array(dau_area, dtype=float)
    
    def trigonal_154(frac_grain=[24, 24, 24]):
        """
        space group group P3221
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 <= y < 1 and 0 < z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and 0 <= x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif z == Fraction(1, 6) and x-y >= 0 and 0 <= x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_155(frac_grain=[24, 24, 24]):
        """
        space group group R32
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 < z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif z == Fraction(1, 6) and -x+y <= Fraction(1, 3) and 0 < x <= Fraction(1, 3) and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 < z <= Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z < Fraction(1, 6):
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_156(frac_grain=[24, 24, 24]):
        """
        space group group P3m1
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if -x+2*y >= 0 and x+y <= 1 and 2*x-y >= 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_157(frac_grain=[24, 24, 24]):
        """
        space group group P31m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= y and 2*x-y <= 1 and x+y < 1 and x-y >= 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x+y == 1 and 2*x-y <= 1 and x-y >= 0 and 0 <= y <= Fraction(1, 3) and 0 <= z < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_158(frac_grain=[24, 24, 24]):
        """
        space group group P3c1
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_159(frac_grain=[24, 24, 24]):
        """
        space group group P31c
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_160(frac_grain=[24, 24, 24]):
        """
        space group group R3m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.4)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if -x+2*y >= 0 and x+y <= 1 and 2*x-y >= 0 and 0 <= z < Fraction(1, 3):
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_161(frac_grain=[24, 24, 24]):
        """
        space group group R3c
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.4)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 <= z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 <= z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z < Fraction(1, 6):
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_162(frac_grain=[24, 24, 24]):
        """
        space group group P-31m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= y and 2*x-y <= 1 and x+y < 1 and x-y >= 0 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-2*y >= 0 and 0 <= y and 2*x-y <= 1 and x+y < 1 and x-y >= 0:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and x-2*y >= 0 and 0 <= y and 2*x-y <= 1 and x+y < 1 and x-y >= 0:
                        dau_area.append([x, y, z])
                    elif x+y == 1 and 0 <= y <= Fraction(1, 3) and 2*x-y <= 1 and x-y >= 0 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[Fraction(2, 3), Fraction(1, 3), 0], [Fraction(2, 3), Fraction(1, 3), 1/2]]
        return np.array(dau_area, dtype=float)
    
    def trigonal_163(frac_grain=[24, 24, 24]):
        """
        space group group P-31c
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y > 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x-2*y >= 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and -2*x+y >= 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_164(frac_grain=[24, 24, 24]):
        """
        space group group P-3m1
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.4)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < y and 2*x-y <= 1 and x-2*y >= 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif y == 0 and 2*x-y <= 1 and x-2*y >= 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_165(frac_grain=[24, 24, 24]):
        """
        space group group P-3c1
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y > 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x-y >= 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_166(frac_grain=[24, 24, 24]):
        """
        space group group R-3m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if -x+2*y >= 0 and x+y <= 1 and 2*x-y >= 0 and 0 < z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and -x+2*y >= 0 and x+y <= 1 and 2*x-y >= 0:
                        dau_area.append([x, y, z])
                    elif z == Fraction(1, 6) and x <= Fraction(1, 3) and -x+2*y >= 0 and x+y <= 1 and 2*x-y >= 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def trigonal_167(frac_grain=[24, 24, 24]):
        """
        space group group R-3c
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.2)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 < z < Fraction(1, 12):
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y > 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif z == Fraction(1, 12) and x-y <= Fraction(1, 3) and 0 < x and 0 <= y <= Fraction(1, 3) and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= Fraction(1, 12):
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 < z < Fraction(1, 12):
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z <= Fraction(1, 12):
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_168(frac_grain=[24, 24, 24]):
        """
        space group group P6
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= y and 2*x-y <= 1 and x+y < 1 and x-y > 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x+y == 1 and 0 <= y <= Fraction(1, 3) and 2*x-y <= 1 and x-y > 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_169(frac_grain=[24, 24, 24]):
        """
        space group group P61
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.2)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 <= y < 1 and 0 <= z < Fraction(1, 6):
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)

    def hexagonal_170(frac_grain=[24, 24, 24]):
        """
        space group group P65
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.2)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 <= y < 1 and 0 <= z < Fraction(1, 6):
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_171(frac_grain=[24, 24, 24]):
        """
        space group group P62
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1.1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x < 1 and 0 < y and x-y > 0 and 0 <= z < Fraction(1, 3):
                        dau_area.append([x, y, z])
                    elif x == 1 and 0 < y <= 1/2 and 0 <= z < Fraction(1, 3):
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/2 and y == 0 and 0 <= z < Fraction(1, 3):
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/2 and y == x and 0 <= z < Fraction(1, 3):
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_172(frac_grain=[24, 24, 24]):
        """
        space group group P64
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1.1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1.1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x < 1 and 0 < y and x-y > 0 and 0 <= z < Fraction(1, 3):
                        dau_area.append([x, y, z])
                    elif 1/2 <= x < 1 and y == 0 and 0 <= z < Fraction(1, 3):
                        dau_area.append([x, y, z])
                    elif x == 1 and 1/2 <= y <= 1 and 0 <= z < Fraction(1, 3):
                        dau_area.append([x, y, z])
                    elif 1/2 <= x < 1 and y == x and 0 <= z < Fraction(1, 3):
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_173(frac_grain=[24, 24, 24]):
        """
        space group group P63
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_174(frac_grain=[24, 24, 24]):
        """
        space group group P-6
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_175(frac_grain=[24, 24, 24]):
        """
        space group group P6/m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= y and 2*x-y <= 1 and x+y < 1 and x-y > 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x+y == 1 and 0 <= y <= Fraction(1, 3) and 2*x-y <= 1 and x-y > 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_176(frac_grain=[24, 24, 24]):
        """
        space group group P63/m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y > 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_177(frac_grain=[24, 24, 24]):
        """
        space group group P622
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= y and 2*x-y <= 1 and x+y < 1 and x-y > 0 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-2*y >= 0 and 0 <= y and 2*x-y <= 1 and x+y < 1 and x-y > 0:
                        dau_area.append([x, y, z])
                    elif z == 1/2 and x-2*y >= 0 and 0 <= y and 2*x-y <= 1 and x+y < 1 and x-y > 0:
                        dau_area.append([x, y, z])
                    elif x+y == 1 and 0 <= y <= Fraction(1, 3) and 2*x-y <= 1 and x-y > 0 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[Fraction(2, 3), Fraction(1, 3), 0], [Fraction(2, 3), Fraction(1, 3), 1/2]]
        return np.array(dau_area, dtype=float)
    
    def hexagonal_178(frac_grain=[24, 24, 24]):
        """
        space group group P6122
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.2)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 <= y < 1 and 0 < z < Fraction(1, 12):
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and 0 <= x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif z == Fraction(1, 12) and x-2*y >= 0 and 0 <= x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif z == Fraction(1, 12) and -x+2*y >= 1 and 0 <= x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_179(frac_grain=[24, 24, 24]):
        """
        space group group P6522
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.2)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= x < 1 and 0 <= y < 1 and 0 < z < Fraction(1, 12):
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and 0 <= x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
                    elif z == Fraction(1, 12) and x+y <= 1 and 0 < x < 1 and 0 <= y < 1:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([0, 0, Fraction(1, 12)])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_180(frac_grain=[24, 24, 24]):
        """
        space group group P6222
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1.1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x < 1 and 0 < y and x-y > 0 and 0 < z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif x == 1 and 0 < y <= 1/2 and x-y > 0 and 0 < z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif y == 0 and x <= 1/2 and x-y > 0 and 0 < z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif z == 0 and 2*x-y <= 1 and x < 1 and 0 < y and x-y > 0:
                        dau_area.append([x, y, z])
                    elif z == Fraction(1, 6) and x-2*y >= 0 and x < 1 and 0 < y and x-y > 0:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/2 and y == x and 0 < z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif x == 1 and 0 < y <= 1/2 and z == Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif y == 0 and x == 0 and 0 < z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/2 and y == 0 and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/2 and y == 0 and z == Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/2 and y == x and z == 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_181(frac_grain=[24, 24, 24]):
        """
        space group group P6422
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1.1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1.1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x < 1 and 0 < y and x-y > 0 and 0 < z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/2 and y == 0 and 0 < z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif 1/2 <= x < 1 and y == x and 0 < z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif z == Fraction(1, 6) and x+y >= 1 and x < 1 and 0 < y and x-y > 0:
                        dau_area.append([x, y, z])
                    elif z == 0 and 2*x-y <= 1 and x < 1 and 0 < y and x-y > 0:
                        dau_area.append([x, y, z])
                    elif x == 1 and 1/2 <= y < 1 and 0 < z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/2 and y == 0 and z == 0:
                        dau_area.append([x, y, z])
                    elif y == 0 and x == 0 and 0 < z < Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif 1/2 <= x < 1 and y == x and z == 0:
                        dau_area.append([x, y, z])
                    elif 1/2 <= x < 1 and y == x and z == Fraction(1, 6):
                        dau_area.append([x, y, z])
                    elif z == Fraction(1, 6) and x == 1 and 1/2 <= y <= 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_182(frac_grain=[24, 24, 24]):
        """
        space group group P6322
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.4)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 < z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x-2*y >= 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and -2*x+y >= 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_183(frac_grain=[24, 24, 24]):
        """
        space group group P6mm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.4)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= y and 2*x-y <= 1 and x-2*y >= 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_184(frac_grain=[24, 24, 24]):
        """
        space group group P6cc
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= y and 2*x-y <= 1 and x+y < 1 and x-y > 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x+y == 1 and 0 <= y <= Fraction(1, 3) and 2*x-y <= 1 and x-y > 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_185(frac_grain=[24, 24, 24]):
        """
        space group group P63cm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= y and 2*x-y <= 1 and x+y < 1 and x-y >= 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
                    elif x+y == 1 and 0 <= y <= Fraction(1, 3) and 2*x-y <= 1 and x-y >= 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_186(frac_grain=[24, 24, 24]):
        """
        space group group P63mc
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.4)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*1)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < y and 2*x-y <= 1 and x-2*y >= 0 and 0 <= z < 1:
                        dau_area.append([x, y, z])
                    elif y == 0 and 2*x-y <= 1 and x-2*y >= 0 and 0 <= z < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_187(frac_grain=[24, 24, 24]):
        """
        space group group P-6m2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if -x+2*y >= 0 and x+y <= 1 and 2*x-y >= 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_188(frac_grain=[24, 24, 24]):
        """
        space group group P-6c2
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-2*y >= 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif z == 0 and -2*x+y >= 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_189(frac_grain=[24, 24, 24]):
        """
        space group group P-62m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= y and 2*x-y <= 1 and x+y < 1 and x-y >= 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_190(frac_grain=[24, 24, 24]):
        """
        space group group P-62c
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1 and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and 0 < x and 0 <= y and 2*x-y <= 1 and x+y < 1 and -x+2*y <= 1:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == Fraction(1, 3) and y == Fraction(2, 3) and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif x == Fraction(2, 3) and y == Fraction(1, 3) and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_191(frac_grain=[24, 24, 24]):
        """
        space group group P6/mmm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.4)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= y and 2*x-y <= 1 and x-2*y >= 0 and 0 <= z <= 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_192(frac_grain=[24, 24, 24]):
        """
        space group group P6/mcc
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= y and 2*x-y <= 1 and x+y < 1 and x-y > 0 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and x-2*y >= 0 and 0 <= y and 2*x-y <= 1 and x+y < 1 and x-y > 0:
                        dau_area.append([x, y, z])
                    elif x+y == 1 and 0 <= y <= Fraction(1, 3) and 2*x-y <= 1 and x-y > 0 and 0 <= z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 0 and y == 0 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([Fraction(2, 3), Fraction(1, 3), .25])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_193(frac_grain=[24, 24, 24]):
        """
        space group group P63/mcm
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 <= y and 2*x-y <= 1 and x+y < 1 and x-y >= 0 and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-2*y >= 0 and 0 <= y and 2*x-y <= 1 and x+y < 1 and x-y >= 0:
                        dau_area.append([x, y, z])
                    elif x+y == 1 and 0 <= y <= Fraction(1, 3) and 2*x-y <= 1 and x-y >= 0 and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([Fraction(2, 3), Fraction(1, 3), 0])
        return np.array(dau_area, dtype=float)
    
    def hexagonal_194(frac_grain=[24, 24, 24]):
        """
        space group group P63/mmc
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.8)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if -x+2*y >= 0 and x+y <= 1 and 2*x-y >= 0 and 0 < z <= 1/4:
                        dau_area.append([x, y, z])
                    elif z == 0 and x-y >= 0 and -x+2*y >= 0 and x+y <= 1 and 2*x-y >= 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def cubic_195(frac_grain=[24, 24, 24]):
        """
        space group group P23
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*1)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x+y < 1 and y-z > 0 and x-z >= 0 and z > 0:
                        dau_area.append([x, y, z])
                    elif z == 0 and 0 <= x <= 1/2 and 0 < y <= 1/2 and x+y < 1:
                        dau_area.append([x, y, z])
                    elif x+y == 1 and y >= 1/2 and y-z > 0 and x-z >= 0 and z > 0:
                        dau_area.append([x, y, z])
                    elif y-z == 0 and -x+z >= 0 and x+y < 1 and x-z >= 0 and z > 0:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, 0], [.5, .5, 0], [.5, .5, .5]]
        return np.array(dau_area, dtype=float)
    
    def cubic_196(frac_grain=[24, 24, 24]):
        """
        space group group F23
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(-int(fg_c*.3), int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x-y > 0 and x+z < 1/2 and x-z < 1/2 and y+z >= 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif x-y == 0 and x+y <= 1/2 and x+z < 1/2 and x-z < 1/2 and y+z >= 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif x+z == 1/2 and y-z == 0 and x-y > 0 and x-z < 1/2 and y+z >= 0:
                        dau_area.append([x, y, z])
                    elif x-z == 1/2 and y+z == 0 and x-y > 0 and x+z < 1/2 and y-z >= 0:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[.25, .25, .25], [.25, .25, -.25], [.5, 0, 0]]
        return np.array(dau_area, dtype=float)
    
    def cubic_197(frac_grain=[24, 24, 24]):
        """
        space group group I23
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x-y > 0 and x+y < 1 and y-z >= 0 and z > 0:
                        dau_area.append([x, y, z])
                    elif z == 0 and x <= 1/2 and x-y > 0 and x+y < 1 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif x-y == 0 and y-z == 0 and x+y < 1 and z > 0:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([0, 0, 0])
        return np.array(dau_area, dtype=float)
    
    def cubic_198(frac_grain=[24, 24, 24]):
        """
        space group group P213
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(-int(fg_c*.6), int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x <= 1/2 and y < 1/2 and 0 < x-z < 1/2 and y+z >= 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == 1/2 and 0 < z < x:
                        dau_area.append([x, y, z])
                    elif x-z == 1/2 and x+y <= 1/2 and 0 < x <= 1/2 and y < 1/2 and y+z >= 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == x and z == x:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([0, 0, 0])
        return np.array(dau_area, dtype=float)
    
    def cubic_199(frac_grain=[24, 24, 24]):
        """
        space group group I213
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x < 1/2 and y < 1/2 and z > 0 and x-z > 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 1/4 <= y < 1/2 and 0 < z <= y:
                        dau_area.append([x, y, z])
                    elif 1/4 < x < 1/2 and y == 1/2 and 1/4 <= z < x:
                        dau_area.append([x, y, z])
                    elif 1/4 <= x < 1/2 and 0 <= y < 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and y == x and z == x:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 1/2 and 1/4 <= z < 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([0, 0, 0])
        return np.array(dau_area, dtype=float)
    
    def cubic_200(frac_grain=[24, 24, 24]):
        """
        space group group Pm-3
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x <= 1/2 and y <= 1/2 and z >= 0 and x-z > 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/2 and y == x and z == x:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def cubic_201(frac_grain=[24, 24, 24]):
        """
        space group group Pn-3
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if z > 0 and x-y > 0 and x+y < 1 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/2 and 0 <= y < x and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/4 and y == x and z == y:
                        dau_area.append([x, y, z])
                    elif 1/2 < x <= 3/4 and y == 1 - x and z == y:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([.5, .5, .5])
        return np.array(dau_area, dtype=float)
    
    def cubic_202(frac_grain=[24, 24, 24]):
        """
        space group group Fm-3
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if z >= 0 and x-y > 0 and x+z < 1/2 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif x-y == 0 and x <= 1/4 and z >= 0 and x+z < 1/2 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif x+z == 1/2 and y-z == 0 and z >= 0 and x-y > 0:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([.25, .25, .25])
        return np.array(dau_area, dtype=float)
    
    def cubic_203(frac_grain=[24, 24, 24]):
        """
        space group group Fd-3
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(-int(fg_c*.3), int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x-y > 0 and x+y < 1/2 and y-z >= 0 and y+z >= 0:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/8 and y == x and z == y:
                        dau_area.append([x, y, z])
                    elif x+y == 1/2 and y-z == 0 and x-y > 0 and y+z >= 0:
                        dau_area.append([x, y, z])
                    elif x+y == 1/2 and y+z == 0 and z >= -(1/8) and x-y > 0 and y-z > 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def cubic_204(frac_grain=[24, 24, 24]):
        """
        space group group Im-3
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x <= 1/2 and z >= 0 and x-y > 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif 0 <= x <= 1/4 and y == x and z == y:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def cubic_205(frac_grain=[24, 24, 24]):
        """
        space group group Pa-3
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x < 1/2 and y < 1/2 and z >= 0 and x-z > 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif 0 <= x < 1/2 and y == x and z == x:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([.5, 0, 0])
        return np.array(dau_area, dtype=float)
    
    def cubic_206(frac_grain=[24, 24, 24]):
        """
        space group group Ia-3
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if z > 0 and x-z > 0 and x+z <= 1/2 and y-z >= 0 and y+z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/4 and 0 <= y < 1/2 and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and y == x and z == x:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, 0], [.25, .25, .25]]
        return np.array(dau_area, dtype=float)
    
    def cubic_207(frac_grain=[24, 24, 24]):
        """
        space group group P432
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if z > 0 and x-y >= 0 and x+y < 1 and y-z > 0:
                        dau_area.append([x, y, z])
                    elif z == 0 and x <= 1/2 and x-y >= 0 and x+y < 1 and y-z > 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y == 1/2 and 0 < z < 1/2:
                        dau_area.append([x, y, z])
                    elif y-z == 0 and x <= 1/2 and z > 0 and x-y >= 0 and x+y < 1:
                        dau_area.append([x, y, z])
                    elif z == 0 and y == 0 and 0 <= x <= 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[.5, .5, 0], [.5, .5, .5]]
        return np.array(dau_area, dtype=float)
    
    def cubic_208(frac_grain=[24, 24, 24]):
        """
        space group group P4232
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(-int(fg_c*.3), int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 0 < x-z < 1/2 and 0 < x+z < 1/2 and 0 <= y-z < 1/2 and 0 <= y+z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and y == x and z == x:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and y == x and z == -x:
                        dau_area.append([x, y, z])
                    elif x-z == 1/2 and y <= 1/4 and 0 < x+z < 1/2 and 0 <= y-z < 1/2 and 0 <= y+z < 1/2:
                        dau_area.append([x, y, z])
                    elif x+z == 1/2 and y <= 1/4 and 0 < x-z < 1/2 and 0 <= y-z < 1/2 and 0 <= y+z < 1/2:
                        dau_area.append([x, y, z])
                    elif y-z == 1/2 and x >= 1/4 and 0 < x-z < 1/2 and 0 < x+z < 1/2 and 0 <= y+z < 1/2:
                        dau_area.append([x, y, z])
                    elif y+z == 1/2 and x >= 1/4 and 0 < x-z < 1/2 and 0 < x+z < 1/2 and 0 <= y-z < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and 0 <= y <= 1/4 and z == 0:
                        dau_area.append([x, y, z])
                    elif 1/4 <= x < 1/2 and y == 1/2 and z == 0:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, 0], [.25, .25, .25], [.25, .25, -.25]]
        return np.array(dau_area, dtype=float)
    
    def cubic_209(frac_grain=[24, 24, 24]):
        """
        space group group F432
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(-int(fg_c*.3), int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x-y > 0 and x+y < 1/2 and y-z >= 0 and y+z > 0:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and y == x and 0 <= z <= y:
                        dau_area.append([x, y, z])
                    elif x+y == 1/2 and z >= 0 and x-y > 0 and y-z >= 0 and y+z > 0:
                        dau_area.append([x, y, z])
                    elif y == 0 and z == 0 and 0 < x < 1/2:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and y == 1/4 and 0 <= z <= 1/4:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, 0], [.5, 0, 0]]
        return np.array(dau_area, dtype=float)
    
    def cubic_210(frac_grain=[24, 24, 24]):
        """
        space group group F4132
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(-int(fg_b*.2), int(fg_b*.2)):
                y = Fraction(j, fg_b)
                for k in range(-int(fg_c*.2), int(fg_c*.2)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if y < 1/8 and z < 1/8 and x-y > 0 and x+y < 1/2 and y+z > 0 and x-z >= 0 and x+z <= 1/2:
                        dau_area.append([x, y, z])
                    elif 1/8 < x < 3/8 and y == 1/8 and (1 - 4*x)/4 <= z < 1/8:
                        dau_area.append([x, y, z])
                    elif z == 1/8 and x+y <= 1/4 and y < 1/8 and x-y > 0 and y+z > 0 and x-z >= 0 and x+z <= 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/8 and y == x and z == x:
                        dau_area.append([x, y, z])
                    elif x+y == 1/2 and x+z == 1/2 and y < 1/8 and z < 1/8 and x-y > 0 and y+z > 0 and x-z >= 0:
                        dau_area.append([x, y, z])
                    elif y+z == 0 and z >= 0 and y < 1/8 and z < 1/8 and x-y > 0 and x+y < 1/2 and x-z >= 0 and x+z <= 1/2:
                        dau_area.append([x, y, z])
                    elif z == 1/8 and y == -(1/8) and 1/8 < x <= 3/8:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, 0], [.125, .125, .125], [.375, .125, .125], [.5, 0, 0]]
        return np.array(dau_area, dtype=float)
    
    def cubic_211(frac_grain=[24, 24, 24]):
        """
        space group group I432
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if z > 0 and x-z > 0 and x+z < 1/2 and y-z >= 0 and y+z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/2 and 0 <= y <= x and z == 0:
                        dau_area.append([x, y, z])
                    elif 0 < x < 1/4 and y == x and z == x:
                        dau_area.append([x, y, z])
                    elif 1/4 < x < 1/2 and (1-2*x)/2 <= y <= 1/4 and z == (1-2*x)/2:
                        dau_area.append([x, y, z])
                    elif 1/4 <= x < 1/2 and x < y < 1/2 and z == (1-2*y)/2:
                        dau_area.append([x, y, z])
                    elif z == 0 and x == 1/2 and 0 <= y <= 1/4:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, 0], [.25, .25, .25]]
        return np.array(dau_area, dtype=float)
    
    def cubic_212(frac_grain=[24, 24, 24]):
        """
        space group P4332
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(-int(fg_c*.6), int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x-z <= 1/2 and 0 < y+z < 1/2 and -2*x+y+z < 0 and x-2*y+z < 0 and 2*x-y+z < 1/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 3/8 and y == (1-2*x)/2 and z == -y:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/8 and x < y <= (1+2*x)/2 and z == 2*x-y:
                        dau_area.append([x, y, z])
                    elif x-2*y+z == 0 and y <= 1/8 and x-z <= 1/2 and 0 < y+z < 1/2 and -2*x+y+z < 0 and 2*x-y+z < 1/2:
                        dau_area.append([x, y, z])
                    elif 2*x-y+z == 1/2 and x >= 3/8 and x-z <= 1/2 and 0 < y+z < 1/2 and -2*x+y+z < 0 and x-2*y+z < 0:
                        dau_area.append([x, y, z])
                    elif 0 < x <= 1/8 and y == x and z == 2*x-y:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, .5, -.5], [.375, .125, -.125]]
        return np.array(dau_area, dtype=float)
    
    def cubic_213(frac_grain=[24, 24, 24]):
        """
        space group P4132
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(-int(fg_a*.3), int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.8)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if y-z >= 0 and 0 < -x+y < 1/2 and x-y+2*z > 0 and 2*x+y+z < 3/2 and x+y+2*z < 3/2:
                        dau_area.append([x, y, z])
                    elif 0 < x < 3/8 and y == x and z == y:
                        dau_area.append([x, y, z])
                    elif x-y + 2*z == 0 and z <= 1/8 and y-z >= 0 and 0 < -x+y < 1/2 and 2*x+y+z < 3/2 and x+y+2*z < 3/2:
                        dau_area.append([x, y, z])
                    elif 3/8 <= x < 1/2 and x < y < 1 - x and z == (3-4*x-2*y)/2:
                        dau_area.append([x, y, z])
                    elif x+y+2*z == 3/2 and z >= 3/8 and y-z >= 0 and 0 < -x+y < 1/2 and x-y+2*z > 0 and 2*x+y+z < 3/2:
                        dau_area.append([x, y, z])
                    elif 3/8 <= x < 1/2 and y == (3-3*x)/3 and z == (-x+y)/2:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, 0], [.375, .375, .375]]
        return np.array(dau_area, dtype=float)
    
    def cubic_214(frac_grain=[24, 24, 24]):
        """
        space group I4132
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(-int(fg_a*.4), int(fg_a*.2)):
            x = Fraction(i, fg_a)
            for j in range(-int(fg_b*.2), int(fg_b*.2)):
                y = Fraction(j, fg_b)
                for k in range(-int(fg_c*.2), int(fg_c*.4)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x < 1/8 and -(1/8) < y < 1/8 and -x+z > 0 and 0 <= -y+z < 1/4 and -x+y-z < 1/8:
                        dau_area.append([x, y, z])
                    elif x == 1/8 and y+z <= 1/4 and -(1/8) < y < 1/8 and -x+z > 0 and 0 <= -y+z < 1/4 and -x+y-z < 1/8:
                        dau_area.append([x, y, z])
                    elif y == 1/8 and x+z <= 1/4 and x < 1/8 and -x+z > 0 and 0 <= -y+z < 1/4 and -x+y-z < 1/8:
                        dau_area.append([x, y, z])
                    elif y == -(1/8) and -x+z >= 1/4 and x < 1/8 and 0 <= -y+z < 1/4 and -x+y-z < 1/8:
                        dau_area.append([x, y, z])
                    elif -(1/8) < x < 1/8 and y == x and z == x:
                        dau_area.append([x, y, z])
                    elif -(3/8) < x < 1/8 and -(1/8) < y <= 0 and z == (1+4*y)/4:
                        dau_area.append([x, y, z])
                    elif -x+y-z == 1/8 and -x+y <= 1/4 and x < 1/8 and -(1/8) < y < 1/8 and -x+z > 0 and 0 <= -y+z < 1/4:
                        dau_area.append([x, y, z])
                    elif x == 1/8 and -(1/8) < y <= 0 and z == (1+4*y)/4:
                        dau_area.append([x, y, z])
                    elif y == -(1/8) and z == 1/8 and -(3/8) < x <= -(1/8):
                        dau_area.append([x, y, z])
                    elif -(3/8) < x <= -(1/4) and y == -(1/8) and z == (-1-4*x)/4:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[-.125, -.125, -.125], [.125, -.125, .125], [.125, .125, .125]]
        return np.array(dau_area, dtype=float)
    
    def cubic_215(frac_grain=[24, 24, 24]):
        """
        space group P-43m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*1)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if z > 0 and x-y >= 0 and x+y <= 1 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif z == 0 and x <= 1/2 and x-y >= 0 and x+y <= 1 and y-z >= 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def cubic_216(frac_grain=[24, 24, 24]):
        """
        space group F-43m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(-int(fg_c*.3), int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x-y >= 0 and x+y <= 1/2 and y-z >= 0 and y+z >= 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def cubic_217(frac_grain=[24, 24, 24]):
        """
        space group I-43m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x < 1/2 and z >= 0 and x-y >= 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and z == 0 and y <= 1/4 and x-y >= 0 and y-z >= 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def cubic_218(frac_grain=[24, 24, 24]):
        """
        space group P-43n
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x < 1/2 and y < 1/2 and z >= 0 and x-z > 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif x-z == 0 and y-z == 0 and x < 1/2 and y < 1/2 and z >= 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and z == 0 and y <= 1/4 and x-z > 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and z == 0 and x <= 1/4 and x-z > 0 and y-z >= 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def cubic_219(frac_grain=[24, 24, 24]):
        """
        space group F-43c
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(-int(fg_c*.3), int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x-y >= 0 and x+y < 1/2 and y-z >= 0 and y+z > 0:
                        dau_area.append([x, y, z])
                    elif x+y == 1/2 and x-y == 0 and z >= 0 and y-z >= 0 and y+z > 0:
                        dau_area.append([x, y, z])
                    elif y+z == 0 and x-y == 0 and x+y < 1/2 and y-z > 0:
                        dau_area.append([x, y, z])
                    elif y+z == 0 and y-z == 0 and x <= 1/4 and x-y >= 0 and x+y < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def cubic_220(frac_grain=[24, 24, 24]):
        """
        space group I-43d
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if 1/4 < x <= 1/2 and 1/4 < y < 1/2 and z >= 0 and x-z > 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif x == 1/4 and z == 0 and 3/8 <= y < 1/2 and x-z > 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif y == 1/4 and x == 1/2 and z >= 1/8 and x-z > 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif y == 1/2 and z >= 1/4 and 1/4 < x <= 1/2 and x-z > 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif x-z == 0 and y-z == 0 and 1/4 < x <= 1/2 and 1/4 < y < 1/2 and z >= 0:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([.5, .5, .5])
        return np.array(dau_area, dtype=float)
    
    def cubic_221(frac_grain=[24, 24, 24]):
        """
        space group Pm-3m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.6)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x <= 1/2 and z >= 0 and x-y >= 0 and y-z >= 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def cubic_222(frac_grain=[24, 24, 24]):
        """
        space group Pn-3n
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.5)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.5)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x < 1/2 and z >= 0 and x-y > 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and z == 0 and y <= 1/4 and x-y > 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and y-z == 0 and z > 0 and x-y > 0:
                        dau_area.append([x, y, z])
                    elif x-y == 0 and y-z == 0 and 0 <= z <= 1/4 and x < 1/2:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def cubic_223(frac_grain=[24, 24, 24]):
        """
        space group Pm-3n
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.6)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if z >= 0 and x-z > 0 and x+z < 1/2 and y-z >= 0 and y+z < 1/2:
                        dau_area.append([x, y, z])
                    elif x-z == 0 and y-z == 0 and z >= 0 and x+z < 1/2 and y+z < 1/2:
                        dau_area.append([x, y, z])
                    elif x+z == 1/2 and y <= 1/4 and z >= 0 and x-z > 0 and y-z >= 0 and y+z < 1/2:
                        dau_area.append([x, y, z])
                    elif y+z == 1/2 and x <= 1/4 and z >= 0 and x-z > 0 and x+z < 1/2 and y-z >= 0:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([.25, .25, .25])
        return np.array(dau_area, dtype=float)
    
    def cubic_224(frac_grain=[24, 24, 24]):
        """
        space group Pn-3m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.5)):
                y = Fraction(j, fg_b)
                for k in range(-int(fg_c*.3), int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if x-y >= 0 and x+z < 1/2 and x-z < 1/2 and y+z >= 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif x+z == 1/2 and y <= 1/4 and x-y >= 0 and x-z < 1/2 and y+z >= 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif x-z == 1/2 and y <= 1/4 and x-y >= 0 and x+z < 1/2 and y+z >= 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif x == 1/2 and z == 0 and 0 <= y <= 1/4:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def cubic_225(frac_grain=[24, 24, 24]):
        """
        space group Fm-3m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if z >= 0 and x-y >= 0 and x+y <= 1/2 and y-z >= 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def cubic_226(frac_grain=[24, 24, 24]):
        """
        space group Fm-3c
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.3)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if z >= 0 and x-y >= 0 and x+y < 1/2 and y-z > 0:
                        dau_area.append([x, y, z])
                    elif x+y == 1/2 and x-y == 0 and z >= 0 and y-z > 0:
                        dau_area.append([x, y, z])
                    elif y-z == 0 and x <= 1/4 and z >= 0 and x-y >= 0 and x+y < 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area.append([.25, .25, .25])
        return np.array(dau_area, dtype=float)
    
    def cubic_227(frac_grain=[24, 24, 24]):
        """
        space group Fd-3m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.25)):
                y = Fraction(j, fg_b)
                for k in range(-int(fg_c*.25), int(fg_c*.25)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if y < 1/8 and x-y >= 0 and x+y <= 1/2 and y+z >= 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif y == 1/8 and x+z >= 1/4 and x-y >= 0 and x+y <= 1/2 and y+z >= 0 and y-z >= 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def cubic_228(frac_grain=[24, 24, 24]):
        """
        space group Fd-3c
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.5)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.5)):
                y = Fraction(j, fg_b)
                for k in range(-int(fg_c*.25), int(fg_c*.5)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if y < 1/8 and x-y > 0 and x+y <= 1/2 and y+z > 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif y == 1/8 and x+z <= 1/4 and x-y >= 0 and x+y <= 1/2 and y+z > 0 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif x-y == 0 and y-z == 0 and y < 1/8 and x+y <= 1/2 and y+z > 0:
                        dau_area.append([x, y, z])
                    elif y+z == 0 and y-z == 0 and x <= 1/4 and y < 1/8 and x-y > 0 and x+y <= 1/2:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, 0], [.375, .375, .375]]
        return np.array(dau_area, dtype=float)
    
    def cubic_229(frac_grain=[24, 24, 24]):
        """
        space group Im-3m
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(0, int(fg_a*.6)):
            x = Fraction(i, fg_a)
            for j in range(0, int(fg_b*.5)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if z >= 0 and x-y >= 0 and x+z < 1/2 and y-z >= 0:
                        dau_area.append([x, y, z])
                    elif x+z == 1/2 and y <= 1/4 and z >= 0 and x-y >= 0 and y-z >= 0:
                        dau_area.append([x, y, z])
        return np.array(dau_area, dtype=float)
    
    def cubic_230(frac_grain=[24, 24, 24]):
        """
        space group Ia-3d
        """
        dau_area = []
        fg_a, fg_b, fg_c = frac_grain
        for i in range(-int(fg_a*.2), int(fg_a*.2)):
            x = Fraction(i, fg_a)
            for j in range(-int(fg_b*.2), int(fg_b*.2)):
                y = Fraction(j, fg_b)
                for k in range(0, int(fg_c*.3)):
                    z = Fraction(k, fg_c)
                    #DAU
                    if -(1/8) < x < 1/8 and -(1/8) < y < 1/8 and z < 1/4 and -x+z >= 0 and x+z > 0 and -y+z > 0 and y+z >= 0:
                        dau_area.append([x, y, z])
                    elif x == 1/8 and 0 < -y+z <= 1/4 and 0 <= y+z <= 1/4 and -(1/8) < y < 1/8 and z < 1/4 and -x+z >= 0 and x+z > 0:
                        dau_area.append([x, y, z])
                    elif x == -(1/8) and 0 < y < 1/8 and z < 1/4 and -x+z >= 0 and x+z > 0 and -y+z > 0 and y+z >= 0:
                        dau_area.append([x, y, z])
                    elif y == 1/8 and x+z >= 1/4 and -(1/8) < x < 1/8 and z < 1/4 and -x+z >= 0 and -y+z > 0 and y+z >= 0:
                        dau_area.append([x, y, z])
                    elif y == -(1/8) and 0 <= -x+z <= 1/4 and -(1/8) < x < 1/8 and z < 1/4 and x+z > 0 and -y+z > 0 and y+z >= 0:
                        dau_area.append([x, y, z])
                    elif z == 1/4 and 0 <= y < 1/8 and -(1/8) < x < 1/8 and -x+z >= 0 and x+z > 0 and -y+z > 0 and y+z >= 0:
                        dau_area.append([x, y, z])
                    elif -y+z == 0 and -x+z == 0 and -(1/8) < x < 1/8 and -(1/8) < y < 1/8 and z < 1/4 and x+z > 0 and y+z >= 0:
                        dau_area.append([x, y, z])
                    elif x == -(1/8) and 0 < y < 1/8 and z == 1/4 and -x+z >= 0 and x+z > 0 and -y+z > 0 and y+z >= 0:
                        dau_area.append([x, y, z])
                    elif y == 1/8 and x+z >= 1/4 and -(1/8) < x < 1/8 and z == 1/4 and -x+z >= 0 and -y+z > 0 and y+z >= 0:
                        dau_area.append([x, y, z])
        #point
        dau_area += [[0, 0, 0], [-.125, 0, .25], [.125, .125, .125], [.125, -.125, .125], [.125, 0, .25]]
        return np.array(dau_area, dtype=float)
    
    
    
def write_dat(file, ct):
    ct_string = '\n'.join([' '.join([str(i) for i in item]) for item in ct])
    with open(file, 'w') as obj:
        obj.writelines(ct_string)


if __name__ == '__main__':
    import time
    sg = 230
    sg_name = 'Ia-3d'
    file_1 = f'check_area/{sg:03g}_{sg_name}.dat'
    latt = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    from pymatgen.core.structure import Structure
    start = time.time()
    grid = BulkSpaceGroup.cubic_230()
    end = time.time()
    print(end - start)
    atoms = [1 for _ in range(len(grid))]
    stru = Structure.from_spacegroup(sg, latt, atoms, grid)
    grid_all = stru.frac_coords
    write_dat(file_1, grid_all)
    '''
    file_2 = f'check_symm/{sg:03g}_{sg_name}.dat'
    symm_site = bsg.dau_area_sampling(sg, 5, grid)
    write_dat(file_2, [list(symm_site.keys())])
    
    sparse_grid = np.vstack([v for v in symm_site.values()])
    atoms = [1 for _ in range(len(sparse_grid))]
    stru = Structure.from_spacegroup(sg, latt, atoms, sparse_grid)
    grid_all = stru.frac_coords
    write_dat(file_2, grid_all)
    '''