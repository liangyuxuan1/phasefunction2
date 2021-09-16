import numpy as np
from collections import namedtuple
import seaborn as sns
import seaborn_image as isns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

matplotlib.use("Agg")

def read_mose_results(datalist, datapath, imgpath):

    df_MOSE_results = pd.DataFrame(columns=['TotalPhotonNum', 'AbsorpPhotonNum', 'TransmitPhotonNum',
                                            'Runtime', 'SpecularReflectance', '3DCWTransmittance', 
                                            'Reflectance', 'Transmittance'])

    image_dir = os.path.join(imgpath, 'images')
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    for i in range(len(datalist)):
        file_name = os.path.join(datapath, datalist['Image'].loc[i])
        print(file_name)

        with open(file_name,'r') as fid:
            line = fid.readline()   # skip first line: Spectrum
            
            line = fid.readline()   # TotalPhotonNum
            TotalPhotonNum = int(line.split()[1])

            line = fid.readline()   # AbsorpPhotonNum
            AbsorpPhotonNum = int(line.split()[1])

            line = fid.readline()   # TransmitPhotonNum
            TransmitPhotonNum = int(line.split()[1])

            line = fid.readline()   # Runtime
            Runtime = float(line.split()[1])

            line = fid.readline()   # skip Domain

            line = fid.readline()   # SpecularReflectance
            SpecularReflectance = float(line.split()[1])

            line = fid.readline()   # 3DCWTransmittance
            Transmittance = float(line.split()[1])

            line = fid.readline()   # skip blank line

            # =======================================================
            line = fid.readline()   # 3DCWTransmittanceTop
            TransmittanceTop = float(line.split()[1])

            line = fid.readline()   # CountX CountY
            line_items = line.split()
            CountX, CountY = int(line_items[2]), int(line_items[3])

            line = fid.readline()   # 3DCWTransmittanceTopXY
            TransmittanceTopXY = np.zeros((CountX,CountY))
            for row in range(CountX):
                line = fid.readline()   # read a line
                line_items = line.split()
                for col in range(CountY):
                    TransmittanceTopXY[row, col] = float(line_items[col])
            
            sub_dir = os.path.join(imgpath, datalist['Image'].loc[i].split(os.sep)[0])
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
            np.save(os.path.join(imgpath, datalist['Image'].loc[i]), TransmittanceTopXY)

            if file_name.find('001.T.CW') != -1:
                print('saving reflectance image ...')
                img = np.log10(TransmittanceTopXY + 1e-10)
                # img = np.nan_to_num(img)
                isns.imshow(img, cmap='gist_heat')
                plt.savefig(os.path.join(image_dir, 
                                         datalist['Image'].loc[i].split(os.sep)[0] + '_' + datalist['Image'].loc[i].split(os.sep)[1] + '_Top.png'), 
                                         bbox_inches='tight')
                plt.close('all')

            # =======================================================
            line = fid.readline()   # skip blank line
            line = fid.readline()   # 3DCWTransmittanceBottom
            TransmittanceBottom = float(line.split()[1])
            line = fid.readline()   # skip CountX, Count Y
            line = fid.readline()   # skip 3DCWTransmittanceTopXY
            TransmittanceBottomXY = np.zeros((CountX,CountY))
            for row in range(CountX):
                line = fid.readline()   # read a line
                line_items = line.split()
                for col in range(CountY):
                    TransmittanceBottomXY[row, col] = float(line_items[col])

            if file_name.find('001.T.CW') != -1:
                print('saving transmittance image ...')
                img = np.log10(TransmittanceBottomXY + 1e-10)
                # img = np.nan_to_num(img)
                isns.imshow(img, cmap='gist_heat')
                plt.savefig(os.path.join(image_dir, 
                                         datalist['Image'].loc[i].split(os.sep)[0] + '_' + datalist['Image'].loc[i].split(os.sep)[1] + '_Bottom.png'), 
                                         bbox_inches='tight')
                plt.close('all')


            # =======================================================
            pdRow = {'TotalPhotonNum':TotalPhotonNum, 'AbsorpPhotonNum':AbsorpPhotonNum, 
                     'TransmitPhotonNum':TransmitPhotonNum, 'Runtime':Runtime, 
                     'SpecularReflectance':SpecularReflectance,
                     '3DCWTransmittance':Transmittance, 'Reflectance':TransmittanceTop, 
                     'Transmittance':TransmittanceBottom}
            df_MOSE_results = df_MOSE_results.append(pdRow, ignore_index=True)

    fid.close()

    df_MOSE_results = df_MOSE_results.convert_dtypes()

    return df_MOSE_results

# ==================================================================
if __name__=='__main__':

    data_path = "rawDataCW_v5"
    DataListFile = "TrainDataCW_v5.csv"
    img_path = "ImageCW_v5"
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    datalist = pd.read_csv(os.path.join(data_path, DataListFile))

    results = read_mose_results(datalist, data_path, img_path)

    datalist = datalist.join(results)
    datalist.to_csv(os.path.join(img_path, DataListFile[:-4]+'_Results.csv'), index=False)

    print('done')




