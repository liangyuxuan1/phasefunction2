import numpy as np
from collections import namedtuple

def readInParam(mcofile):
    simuParam = namedtuple('simuParam', 
                'Num_of_Photons, dz, dr, ndz, ndr, nda, n, mua, mus, g, d, SpecularReflectance, DiffuseReflectance, AbsorbedFraction, Transmittance')

    with open(mcofile,'r') as fid:
        while True:
            line = fid.readline()
            li = line.strip()
            if li.startswith('InParam'):
                break

        line = fid.readline()   # skip the line of mco filename 

        line = fid.readline()   # number of photons
        NoP = line.split()[0]

        line = fid.readline()   # dz, dr
        line_items = line.split()
        dz, dr = line_items[0], line_items[1]

        line = fid.readline()   # ndz, ndr, nda
        line_items = line.split()
        ndz, ndr, nda = line_items[0], line_items[1], line_items[2]

        for i in range(4):
            line = fid.readline()   # skip 4 lines

        line = fid.readline()       # n	mua	mus	g d
        line_items = line.split()
        n, mua, mus, g, d = line_items[0], line_items[1], line_items[2], line_items[3], line_items[4]

        for i in range(3):
            line = fid.readline()   # skip 3 lines

        line = fid.readline()       # Specular reflectance
        Specular_reflectance = line.split()[0]

        line = fid.readline()       # Diffuse reflectance
        Diffuse_reflectance = line.split()[0]

        line = fid.readline()       # Absorbed fraction
        Absorbed_fraction = line.split()[0]

        line = fid.readline()       # Transmittance
        Transmittance = line.split()[0]

    fid.close()

    InParam = simuParam(int(NoP), float(dz), float(dr), int(ndz), int(ndr), int(nda), 
                        float(n), float(mua), float(mus), float(g), float(d), 
                        float(Specular_reflectance), float(Diffuse_reflectance), float(Absorbed_fraction), float(Transmittance))

    return InParam


def readRd_r(mcofile, num):
    
    Rd_r = np.zeros(num)

    with open(mcofile,'r') as fid:
        while True:
            line = fid.readline()
            li = line.strip()
            if li.startswith('Rd_r'):
                break
        
        for i in range(num):
            x = fid.readline()
            Rd_r[i] = float(x)

    fid.close()

    return Rd_r

def readRd_a(mcofile, num):
    
    Rd_a = np.zeros(num)

    with open(mcofile,'r') as fid:
        while True:
            line = fid.readline()
            li = line.strip()
            if li.startswith('Rd_a'):
                break
        
        for i in range(num):
            x = fid.readline()
            Rd_a[i] = float(x)

    fid.close()

    return Rd_a


def readRd_ra(mcofile, nr, na):
    
    Rd_ra = np.zeros(nr*na)

    with open(mcofile,'r') as fid:
        while True:
            line = fid.readline()
            li = line.strip()
            if li.startswith('Rd_ra'):
                break
        
        idx = 0
        while idx < nr*na:
            line = fid.readline().split()
            for i in range(len(line)):
                Rd_ra[idx] = float(line[i])
                idx = idx +1                

    fid.close()

    Rd_ra = Rd_ra.reshape((nr, na))

    return Rd_ra


def readImage(mcofile, nr, key):

    h, w = 2*nr+1, 2*nr+1    
    img = np.zeros(h*w)

    with open(mcofile,'r') as fid:
        while True:
            line = fid.readline()
            li = line.strip()
            if li.startswith(key):
                break
        
        idx = 0
        while idx < h*w:
            line = fid.readline().split()
            for i in range(len(line)):
                img[idx] = float(line[i])
                idx = idx +1                

    fid.close()

    img = img.reshape((h, w))

    return img