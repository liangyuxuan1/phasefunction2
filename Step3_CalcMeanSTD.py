# calc the mean and std of a dataset

import numpy as np
import pandas as pd
import os

img_path = "ImageCW_Train_101"
DataListFile = "TrainDataCW_MCML_101.csv"

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


# not used 
# g_train = [0.6, 0.7, 0.8, 0.9]
# g_val   = [0.55, 0.65, 0.75, 0.85, 0.95]
# Dataset MCML 301x301, mean = 0.04315, std = 0.55034


# 2021-09-13
# Dataset V6, large phantom, mean = 0.0022, std = 0.2915

# 2021-09-16
# Dataset V5, large phantom, mean = 0.0039, std = 0.2198