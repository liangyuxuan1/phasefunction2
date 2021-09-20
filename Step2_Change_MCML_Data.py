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

    datalist = pd.DataFrame(columns=['Image', 'ua', 'us', 'g', 'Tissue', 'Events', 
                                    'SpecularReflectance', 'Absorption', 'Reflectance', 'Transmittance'])

    files = sorted(glob.glob(os.path.join(data_path, '*.mco')))
    for mcofile in files:
        InParam = readMCO.readInParam(mcofile)
        Rd_xy   = readMCO.readImage(mcofile, InParam.ndr, 'Rd_xy')

        print(mcofile)

        _, filename = mcofile.split(os.path.sep)
        tissue = filename.split('_')[0]

        np.save(os.path.join(img_path, filename), Rd_xy)

        if filename.find('0001') != -1:
            print('saving reflectance image ...')
            img = np.log10(Rd_xy + 1e-10)
            isns.imshow(img, cmap='gist_heat')
            plt.savefig(os.path.join(img_path, 'image', filename[:-3]+'png'), bbox_inches='tight')
            plt.close('all')

        pdRow = {'Image':filename, 'ua':InParam.mua, 'us':InParam.mus, 'g':InParam.g, 
                'Tissue':tissue, 'Events':'TBD', 
                'SpecularReflectance':InParam.SpecularReflectance,
                'Absorption':InParam.AbsorbedFraction, 'Reflectance':InParam.DiffuseReflectance, 
                'Transmittance':InParam.Transmittance}
        datalist = datalist.append(pdRow, ignore_index=True)

    datalist.to_csv(os.path.join(img_path, dataListFile), index=False)

# ==================================================================
if __name__=='__main__':

    src_path        = "RawData_MCML_Train"
    dst_path        = "ImageCW_Train"
    outputFile      = "TrainDataCW_MCML.csv"
    changeMCML_rawData(data_path=src_path, img_path=dst_path, dataListFile=outputFile)

    src_path        = "RawData_MCML_Val"
    dst_path        = "ImageCW_Val"
    outputFile      = "ValDataCW_MCML.csv"
    changeMCML_rawData(data_path=src_path, img_path=dst_path, dataListFile=outputFile)

    print('done')




