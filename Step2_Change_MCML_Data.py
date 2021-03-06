import numpy as np
from collections import namedtuple
import seaborn as sns
import seaborn_image as isns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

import sys

from skimage import color
sys.path.append('./src')
import readMCO

import cv2

#matplotlib.use("Agg")

def changeMCML_rawData(data_path, img_path, dataListFile):
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    if not os.path.exists(os.path.join(img_path, 'image')):
        os.mkdir(os.path.join(img_path, 'image'))

    datalist = pd.DataFrame(columns=['Image', 'ua', 'us', 'g', 'Tissue', 'dr', 'ndr',
                                    'SpecularReflectance', 'Absorption', 'Reflectance', 'Transmittance', 'ImgSize', 'Outlier', 'OutlierPercent',
                                    'Run', 'Error', 'estimated_g', 'delta_g'])
    profiles = []
    tissueNames = []
    files = sorted(glob.glob(os.path.join(data_path, '*.mco')))
    for mcofile in files:
        print(mcofile)

        InParam = readMCO.readInParam(mcofile)

        Rd_xy   = readMCO.readImage(mcofile, InParam.ndr, 'Rd_xy')
        total_W = np.sum(Rd_xy)
        Rd_xy = Rd_xy[1:-1,1:-1]
        outlier_W = np.abs(total_W - np.sum(Rd_xy))

        _, filename = mcofile.split(os.path.sep)
        np.save(os.path.join(img_path, filename), Rd_xy)

        tissue = filename.split('_')[0]
        w, h = Rd_xy.shape

        beamWidth = 0.03 # cm
        bw = int(beamWidth/InParam.dr)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bw,bw))
        kernel = kernel.astype(float)
        kernel /= np.sum(kernel)
        #img = cv2.filter2D(Rd_xy, -1, kernel)

        img = Rd_xy
        img = np.power(img, 0.5)

        if filename.find('0001') != -1:
            print('saving reflectance image ...')
            isns.imshow(img, cmap='gist_heat', dx=InParam.dr, units='cm')
            plt.savefig(os.path.join(img_path, 'image', filename[:-3]+'png'), bbox_inches='tight')
            plt.close('all')
            
            if (InParam.g == 0.6) or (InParam.g == 0.65):
                profiles = np.concatenate((profiles, Rd_xy[InParam.ndr, :]))
                tissueNames.append(tissue)
            ax = 10*np.arange(-InParam.ndr+1, InParam.ndr)*InParam.dr

        pdRow = {'Image':filename, 'ua':InParam.mua, 'us':InParam.mus, 'g':InParam.g, 
                'Tissue':tissue, 'dr':InParam.dr, 'ndr':InParam.ndr, 
                'SpecularReflectance':InParam.SpecularReflectance,
                'Absorption':InParam.AbsorbedFraction, 'Reflectance':InParam.DiffuseReflectance, 
                'Transmittance':InParam.Transmittance,
                'ImgSize':w, 'Outlier':outlier_W, 'OutlierPercent':100*outlier_W/total_W, 
                'Run':0, 'Error': 0, 'estimated_g':0, 'delta_g':0}
        datalist = datalist.append(pdRow, ignore_index=True)

    datalist.to_csv(os.path.join(img_path, dataListFile), index=False)

    profiles = profiles.reshape(-1, len(ax))
    np.savez(os.path.join(img_path, 'image', 'profiles.npz'), x=ax, y=profiles)

    plt.subplots(dpi=300)
    for i in range(len(profiles)):
        plt.plot(ax, profiles[i,:], label=tissueNames[i], color=plt.cm.tab20(i))
    #axx.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
    plt.legend()
    plt.xlabel('x [mm]')
    plt.ylabel('Rd [cm'r'$^{-2}$]')
    plt.savefig(os.path.join(img_path, 'image', 'profiles.png'), bbox_inches='tight')

# ==================================================================
if __name__=='__main__':

    imgSize = 401

    src_path        = f"RawData_MCML_Val_{imgSize}"
    dst_path        = f"ImageCW_Val_{imgSize}"
    outputFile      = f"ValDataCW_MCML_{imgSize}.csv"
    changeMCML_rawData(data_path=src_path, img_path=dst_path, dataListFile=outputFile)


    src_path        = f"RawData_MCML_Train_{imgSize}"
    dst_path        = f"ImageCW_Train_{imgSize}"
    outputFile      = f"TrainDataCW_MCML_{imgSize}.csv"
    changeMCML_rawData(data_path=src_path, img_path=dst_path, dataListFile=outputFile)

    '''
    data = np.load(os.path.join(dst_path, 'image', 'profiles.npz'))
    ax = data['x']
    profiles = data['y']
    for i in range(len(profiles)):
        sns.lineplot(x=ax, y=profiles[i,:])
    plt.xlabel('X [cm]')
    plt.ylabel('Rd')
    plt.show()
    '''

    print('done')




