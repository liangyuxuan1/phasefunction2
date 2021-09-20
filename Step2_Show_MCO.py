import numpy as np
from collections import namedtuple
import seaborn as sns
import seaborn_image as isns
import matplotlib.pyplot as plt

import sys
sys.path.append('./src')

import readMCO

# ========================================================

if __name__=='__main__':

    mcofile = 'RawData_MCML_Train/Adipose_g0_0001.mco'

    InParam = readMCO.readInParam(mcofile)
    Rd_r = readMCO.readRd_r(mcofile, InParam.ndr)
    Rd_a = readMCO.readRd_a(mcofile, InParam.nda)
    Rd_ra = readMCO.readRd_ra(mcofile, InParam.ndr, InParam.nda)
    Rd_xy = readMCO.readImage(mcofile, InParam.ndr, 'Rd_xy')
    
    fig = plt.figure(figsize=(6,4))
    xaxis = (np.arange(InParam.ndr) + 0.5) * InParam.dr
    plt.plot(xaxis, Rd_r)
    plt.xlabel('Radius [cm]')
    plt.ylabel('Rd_r')
    plt.show()

    fig = plt.figure(figsize=(6,4))
    xaxis = (np.arange(InParam.nda) + 0.5) * 90.0/InParam.nda
    plt.plot(xaxis, Rd_a)
    plt.xlabel('Alpha [degree]')
    plt.ylabel('Rd_a')
    plt.show()
    
    img = np.transpose(np.log10(Rd_ra + 1e-10))
    isns.imshow(img, cmap='gist_heat')
    plt.show()

    img = np.log10(Rd_xy + 1e-10)
    isns.imshow(img, cmap='hot')
    plt.show()

    print(InParam)
    print(np.sum(Rd_r))
    print(np.sum(Rd_a))
    print(np.mean(Rd_ra))
    print(np.mean(Rd_xy))

    print('done')




