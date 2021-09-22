# seaborn basics
# https://cloud.tencent.com/developer/article/1651107
# https://cloud.tencent.com/developer/article/1721380

# twinx
# https://kedar.hashnode.dev/how-to-combine-two-line-charts-in-seaborn-and-python

# seaborn advances
# https://www.dataforeverybody.com/category/python-visualization/
# https://www.dataforeverybody.com/seaborn-plot-figure-size/
# https://www.dataforeverybody.com/seaborn-legend-change-location-size/

import numpy as np
import os, glob
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

imgSize1 = 301
checkpoint_path1 = f'training_results_MCML_{imgSize1}'
df1 = pd.read_csv(os.path.join(checkpoint_path1, f'Test_Results_{imgSize1}.csv'))
df1['ImgSize'] = '3x3 @ 0.01'

imgSize2 = 501
checkpoint_path2 = f'training_results_MCML_{imgSize2}'
df2 = pd.read_csv(os.path.join(checkpoint_path2, f'Test_Results_{imgSize2}.csv'))
df2['ImgSize'] = '5x5 @ 0.01'

output_path = 'testing_results'
if not os.path.exists(output_path):
    os.mkdir(output_path)

df = df1.append(df2, ignore_index=True)

# --------------------------------------------------------------------------------------
# line plot of each dataset

# dataset 301
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df1)
#ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
ax.legend(title='')
plt.xlabel('')
plt.ylabel('MSE')

figFile = os.path.join(output_path, f'Test_Results_{imgSize1}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

# dataset 501
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df2)
#ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
ax.legend(title='')
plt.xlabel('')
plt.ylabel('MSE')

figFile = os.path.join(output_path, f'Test_Results_{imgSize2}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

# --------------------------------------------------------------------------------------
# box plot
fig, ax = plt.subplots(figsize=(10,4), dpi=100)
ax = sns.boxplot(x="Tissue", y="Error", hue='ImgSize', data=df, linewidth=1, width=0.5, fliersize=1)
#ax.legend(title='Image Size', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
ax.legend(title='FoV')
plt.xlabel('')
plt.ylabel('MSE')

figFile = os.path.join(output_path, f'Test_Results_Boxplot.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()


# point plot
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
ax = sns.pointplot(x="g", y="Error", hue="ImgSize", data=df, markers='.', capsize=0.1, errwidth=1.5)
#ax = sns.boxplot(x="g", y="Error", hue="ImgSize", data=df, linewidth=1, width=0.5)
ax.legend(title='FoV', loc='upper left')
plt.xlabel('Anistropy Factor')
plt.ylabel('MSE')

figFile = os.path.join(output_path, f'Test_Results_Lineplot.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()


print('done')

