# calc the mean and std of a dataset

import numpy as np
import pandas as pd
import os

img_path = "ImageCW_Train_51"
DataListFile = "TrainDataCW_MCML_51.csv"

labels = pd.read_csv(os.path.join(img_path, DataListFile))

img_mean = 0.0
for i in range(len(labels)):
    file_name = os.path.join(img_path, labels.iloc[i,0]) + '.npy'
    img = np.load(file_name)

    img_mean += np.mean(img)
img_mean /= len(labels)

img_std = 0.0
for i in range(len(labels)):
    file_name = os.path.join(img_path, labels.iloc[i,0]) + '.npy'
    img = np.load(file_name)

    img_std += np.square(img-img_mean).sum()

h, w = img.shape
img_std = np.sqrt(img_std/(len(labels)*h*w))

print(img_mean)
print(img_std)

print('Done')

# The results

# 2021-09-21
# g_train = [0.65, 0.75, 0.85, 0.95]
# g_val   = [0.6, 0.7, 0.8, 0.9]
# Dataset MCML 301x301 (299x299), mean = 0.04370, std = 0.53899
# Dataset MCML 501x501 (499x499), mean = 0.01578, std = 0.32363
# Dataset MCML 251x251 (249x249), mean = 0.01584, std = 0.30017
# Dataset MCML 101x101 (99x99),   mean = 0.36190, std = 1.59218
# Dataset MCML 41x41   (39x39),   mean = 1.69022, std = 3.75663

# 2021-09-27 new dataset at resolution 0.002
# 401x401 (399x399), 0.8cm x 0.8 cm, mean = 0.53076, std = 2.29144
# 201x201 (199x199), 0.4cm x 0.4 cm, mean = 1.64004, std = 4.40112
# 101x101 (99x99),   0.2cm x 0.2 cm, mean = 4.20267, std = 8.26528
# 51x51   (49x49),   0.1cm x 0.1 cm, mean = 9.19103, std = 15.37926


# not used 
# g_train = [0.6, 0.7, 0.8, 0.9]
# g_val   = [0.55, 0.65, 0.75, 0.85, 0.95]
# Dataset MCML 301x301, mean = 0.04315, std = 0.55034


# 2021-09-13
# Dataset V6, large phantom, mean = 0.0022, std = 0.2915

# 2021-09-16
# Dataset V5, large phantom, mean = 0.0039, std = 0.2198