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
                                    'SpecularReflectance', 'Absorption', 'Reflectance', 'Transmittance', 'ImgSize', 'Outlier', 'OutlierPercent', 'Error'])
    profiles = []
    files = sorted(glob.glob(os.path.join(data_path, '*.mco')))
    for mcofile in files:
        InParam = readMCO.readInParam(mcofile)
        Rd_xy   = readMCO.readImage(mcofile, InParam.ndr, 'Rd_xy')

        total_W = np.sum(Rd_xy)
        Rd_xy = Rd_xy[1:-1,1:-1]
        outlier_W = np.abs(total_W - np.sum(Rd_xy))

        w, h = Rd_xy.shape

        print(mcofile)

        _, filename = mcofile.split(os.path.sep)
        tissue = filename.split('_')[0]

        beamWidth = 0.05 # cm
        bw = int(beamWidth/InParam.dr)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bw,bw))
        kernel = kernel.astype(float)
        kernel /= np.sum(kernel)
        img = cv2.filter2D(Rd_xy, -1, kernel)
        # img = Rd_xy

        np.save(os.path.join(img_path, filename), img)

        if filename.find('0001') != -1:
            print('saving reflectance image ...')
            isns.imshow(img, cmap='gist_heat', dx=InParam.dr, units='cm')
            plt.savefig(os.path.join(img_path, 'image', filename[:-3]+'png'), bbox_inches='tight')
            plt.close('all')
            
            profiles = np.concatenate((profiles, Rd_xy[InParam.ndr, :]))
            ax = np.arange(-InParam.ndr+1, InParam.ndr)*InParam.dr

        pdRow = {'Image':filename, 'ua':InParam.mua, 'us':InParam.mus, 'g':InParam.g, 
                'Tissue':tissue, 'dr':InParam.dr, 'ndr':InParam.ndr, 
                'SpecularReflectance':InParam.SpecularReflectance,
                'Absorption':InParam.AbsorbedFraction, 'Reflectance':InParam.DiffuseReflectance, 
                'Transmittance':InParam.Transmittance,
                'ImgSize':w, 'Outlier':outlier_W, 'OutlierPercent':100*outlier_W/total_W, 'Error': 0}
        datalist = datalist.append(pdRow, ignore_index=True)

    datalist.to_csv(os.path.join(img_path, dataListFile), index=False)

    profiles = profiles.reshape(-1, len(ax))
    np.savez(os.path.join(img_path, 'image', 'profiles.npz'), x=ax, y=profiles)

    for i in range(len(profiles)):
        sns.lineplot(x=ax, y=profiles[i,:])
    plt.xlabel('X [cm]')
    plt.ylabel('Rd')
    plt.savefig(os.path.join(img_path, 'image', 'profiles.png'), bbox_inches='tight')

# ==================================================================
if __name__=='__main__':

    imgSize = 101

    src_path        = f"RawData_MCML_Train_{imgSize}"
    dst_path        = f"ImageCW_Train_{imgSize}_2"
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

    src_path        = f"RawData_MCML_Val_{imgSize}"
    dst_path        = f"ImageCW_Val_{imgSize}_2"
    outputFile      = f"ValDataCW_MCML_{imgSize}.csv"
    changeMCML_rawData(data_path=src_path, img_path=dst_path, dataListFile=outputFile)

    print('done')




