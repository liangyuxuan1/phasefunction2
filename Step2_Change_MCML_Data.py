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

matplotlib.use("Agg")

def changeMCML_rawData(data_path, img_path, dataListFile):
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    if not os.path.exists(os.path.join(img_path, 'image')):
        os.mkdir(os.path.join(img_path, 'image'))

    datalist = pd.DataFrame(columns=['Image', 'ua', 'us', 'g', 'Tissue', 'dr', 'ndr',
                                    'SpecularReflectance', 'Absorption', 'Reflectance', 'Transmittance', 'ImgSize', 'Outlier', 'OutlierPercent', 'Error'])

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

        np.save(os.path.join(img_path, filename), Rd_xy)

        if filename.find('0001') != -1:
            print('saving reflectance image ...')
            img = np.log10(Rd_xy + 1e-10)
            isns.imshow(img, cmap='gist_heat', vmin=-10, vmax=2, dx=InParam.dr, units='cm')
            plt.savefig(os.path.join(img_path, 'image', filename[:-3]+'png'), bbox_inches='tight')
            plt.close('all')

        pdRow = {'Image':filename, 'ua':InParam.mua, 'us':InParam.mus, 'g':InParam.g, 
                'Tissue':tissue, 'dr':InParam.dr, 'ndr':InParam.ndr, 
                'SpecularReflectance':InParam.SpecularReflectance,
                'Absorption':InParam.AbsorbedFraction, 'Reflectance':InParam.DiffuseReflectance, 
                'Transmittance':InParam.Transmittance,
                'ImgSize':w, 'Outlier':outlier_W, 'OutlierPercent':100*outlier_W/total_W, 'Error': 0}
        datalist = datalist.append(pdRow, ignore_index=True)

    datalist.to_csv(os.path.join(img_path, dataListFile), index=False)

# ==================================================================
if __name__=='__main__':

    imgSize = 101

    src_path        = f"RawData_MCML_Train_{imgSize}"
    dst_path        = f"ImageCW_Train_{imgSize}"
    outputFile      = f"TrainDataCW_MCML_{imgSize}.csv"
    changeMCML_rawData(data_path=src_path, img_path=dst_path, dataListFile=outputFile)

    src_path        = f"RawData_MCML_Val_{imgSize}"
    dst_path        = f"ImageCW_Val_{imgSize}"
    outputFile      = f"ValDataCW_MCML_{imgSize}.csv"
    changeMCML_rawData(data_path=src_path, img_path=dst_path, dataListFile=outputFile)

    print('done')




