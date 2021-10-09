import numpy as np
import pandas as pd
import os
import seaborn as sns
import seaborn_image as isns
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

imgSize = 201

img_path = f"ImageCW_Val_{imgSize}"
DataListFile = f"ValDataCW_MCML_{imgSize}.csv"

output_path = 'testing_results'
if not os.path.exists(output_path):
    os.mkdir(output_path)

labels = pd.read_csv(os.path.join(img_path, DataListFile))

# Stp1 : Calculate the feature matrix for different scales

singleFig, subCategory, category = [], [], []

gFlag = 0.0
gCount = 1
tissueFlag = ' '
tissueCount = 1

for i in range(len(labels)):
    file_name = os.path.join(img_path, labels.iloc[i, 0]) + '.npy'
    gValue = labels.iloc[i, 3]
    tissue = labels.iloc[i, 4]
    ndr = labels.iloc[i, 6]

    img = np.load(file_name)
    imgFeature = img[ndr]

    singleFig.append(imgFeature)

    if gValue != gFlag:
        if i != 0:
            subCategory[-1] = subCategory[-1] / gCount
            gCount = 1
        subCategory.append(imgFeature)
    else:
        subCategory[-1] = subCategory[-1] + imgFeature
        gCount = gCount + 1
    gFlag = gValue

    if tissue != tissueFlag:
        if i != 0:
            category[-1] = category[-1] / tissueCount
            tissueCount = 1
        category.append(imgFeature)
    else:
        category[-1] = category[-1] + imgFeature
        tissueCount = tissueCount + 1
    tissueFlag = tissue

subCategory[-1] = subCategory[-1] / gCount
category[-1] = category[-1] / tissueCount

singleFig = np.array(singleFig)  # Image levels    1760*199
subCategory = np.array(subCategory)  # subcategory levels   44*199
category = np.array(category)  # category levels   11*199

# Stp2 : Calculate the similarity matrix

"""
Calculate the similarity matrix by cdist of scipy, the distance of parameters can be selected from
'euclidean', 'correlation', ' cosine'......
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
"""
metric = 'euclidean'
singleFigDis = cdist(singleFig, singleFig, metric=metric)
subCategoryDis = cdist(subCategory, subCategory, metric=metric)
categoryDis = cdist(category, category, metric=metric)

# Stp3 : Plot the similarity matrix

print("Saving the RDM figures...")
isns.imshow(np.flip(singleFigDis, axis=1))
plt.savefig(os.path.join(output_path, f'RDM_singleFig_{imgSize}' + '.png'), bbox_inches='tight', dpi=300)

isns.imshow(np.flip(subCategoryDis, axis=1))
plt.savefig(os.path.join(output_path, f'RDM_subCategory_{imgSize}' + '.png'), bbox_inches='tight', dpi=300)

isns.imshow(np.flip(categoryDis, axis=1))
plt.savefig(os.path.join(output_path, f'RDM_category_{imgSize}' + '.png'), bbox_inches='tight', dpi=300)
plt.close('all')