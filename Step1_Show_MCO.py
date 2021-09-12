import numpy as np
from collections import namedtuple
import seaborn as sns
import seaborn_image as isns
import matplotlib.pyplot as plt


def readInParam(mcofile):
    simuParam = namedtuple('simuParam', 'Num_of_Photons, dz, dr, ndz, ndr, nda')

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

    fid.close()

    InParam = simuParam(int(NoP), float(dz), float(dr), int(ndz), int(ndr), int(nda))

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


if __name__=='__main__':

    mcofile = './train/a06_s07_g01_01.mco'

    InParam = readInParam(mcofile)
    print('InParam')
    print(InParam)
    print('\n')

    Rd_r = readRd_r(mcofile, InParam.ndr)
    print('Rd_r')
    print(Rd_r)
    print('\n')

    Rd_a = readRd_a(mcofile, InParam.nda)
    print('Rd_a')
    print(Rd_a)
    print('\n')

    Rd_ra = readRd_ra(mcofile, InParam.ndr, InParam.nda)

    fig = plt.figure(figsize=(6,4))

    xaxis = (np.arange(InParam.ndr) + 0.5) * InParam.dr
    plt.plot(xaxis, Rd_r)
    plt.xlabel('Radius [cm]')
    plt.ylabel('Rd_r')
    plt.show()

    xaxis = (np.arange(InParam.nda) + 0.5) * 90.0/InParam.nda
    plt.plot(xaxis, Rd_a)
    plt.xlabel('Alpha [degree]')
    plt.ylabel('Rd_a')
    plt.show()

    img = np.transpose(np.log10(Rd_ra + 1e-10))
    isns.imshow(img, cmap='hot')
    plt.show()

    print('done')




