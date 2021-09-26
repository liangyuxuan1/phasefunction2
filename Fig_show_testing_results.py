# seaborn basics
# https://cloud.tencent.com/developer/article/1651107
# https://cloud.tencent.com/developer/article/1721380

# twinx
# https://kedar.hashnode.dev/how-to-combine-two-line-charts-in-seaborn-and-python

# seaborn advances
# https://www.dataforeverybody.com/category/python-visualization/
# https://www.dataforeverybody.com/seaborn-plot-figure-size/
# https://www.dataforeverybody.com/seaborn-legend-change-location-size/

# seabor colors 
# https://seaborn.pydata.org/tutorial/color_palettes.html

import numpy as np
import os, glob
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

import matplotlib
matplotlib.use("Agg")

#=================================================================================

imgSize1 = 301
checkpoint_path1 = f'training_results_MCML_{imgSize1}'
df1 = pd.read_csv(os.path.join(checkpoint_path1, f'Test_Results_{imgSize1}.csv'))
df1['ImgSize'] = '3x3 @ 0.01'

imgSize2 = 501
checkpoint_path2 = f'training_results_MCML_{imgSize2}'
df2 = pd.read_csv(os.path.join(checkpoint_path2, f'Test_Results_{imgSize2}.csv'))
df2['ImgSize'] = '5x5 @ 0.01'

imgSize3 = 251
checkpoint_path3 = f'training_results_MCML_{imgSize3}'
df3 = pd.read_csv(os.path.join(checkpoint_path3, f'Test_Results_{imgSize3}.csv'))
df3['ImgSize'] = '5x5 @ 0.02'

imgSize4 = 101
checkpoint_path4 = f'training_results_MCML_{imgSize4}'
df4 = pd.read_csv(os.path.join(checkpoint_path4, f'Test_Results_{imgSize4}.csv'))
df4['ImgSize'] = '1x1 @ 0.01'

imgSize5 = 41
checkpoint_path5 = f'training_results_MCML_{imgSize5}'
df5 = pd.read_csv(os.path.join(checkpoint_path5, f'Test_Results_{imgSize5}.csv'))
df5['ImgSize'] = '0.4x0.4 @ 0.01'


output_path = 'testing_results'
if not os.path.exists(output_path):
    os.mkdir(output_path)

df = df2.append(df1, ignore_index=True)
df = df.append(df4, ignore_index=True)
df = df.append(df5, ignore_index=True)
df = df.append(df3, ignore_index=True)

# --------------------------------------------------------------------------------------
# line plot of each dataset: Error vs g, Error vs leakage

# dataset 301
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df1, palette=sns.color_palette('bright', 11))
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df1)
ax = sns.barplot(x="g", y="Error", hue='Tissue', data=df1, errwidth=0, palette=sns.color_palette('deep'))
ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Anistropy Factor')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_{imgSize1}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

# dataset 301
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df1, palette=sns.color_palette('bright', 11))
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df1)
ax = sns.barplot(x="g", y="OutlierPercent", hue='Tissue', data=df1, errwidth=0, palette=sns.color_palette('deep'))
ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Anistropy Factor')
plt.ylabel('Reflectance leakage (%)')
figFile = os.path.join(output_path, f'Test_Results_Leakage_{imgSize1}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

#-------------
# dataset 501
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df2)
ax = sns.barplot(x="g", y="Error", hue='Tissue', data=df2, errwidth=0, palette=sns.color_palette('deep'))
ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
plt.xlabel('Anistropy Factor')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_{imgSize2}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

# dataset 501
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df1, palette=sns.color_palette('bright', 11))
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df1)
ax = sns.barplot(x="g", y="OutlierPercent", hue='Tissue', data=df2, errwidth=0, palette=sns.color_palette('deep'))
ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Anistropy Factor')
plt.ylabel('Reflectance leakage (%)')
figFile = os.path.join(output_path, f'Test_Results_Leakage_{imgSize2}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

#--------------
# dataset 251
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df3)
ax = sns.barplot(x="g", y="Error", hue='Tissue', data=df3, errwidth=0, palette=sns.color_palette('deep'))
ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
plt.xlabel('Anistropy Factor')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_{imgSize3}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

# dataset 251
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df1, palette=sns.color_palette('bright', 11))
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df1)
ax = sns.barplot(x="g", y="OutlierPercent", hue='Tissue', data=df3, errwidth=0, palette=sns.color_palette('deep'))
ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Anistropy Factor')
plt.ylabel('Reflectance leakage (%)')
figFile = os.path.join(output_path, f'Test_Results_Leakage_{imgSize3}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

#--------------
# dataset 101
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df3)
ax = sns.barplot(x="g", y="Error", hue='Tissue', data=df4, errwidth=0, palette=sns.color_palette('deep'))
ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
plt.xlabel('Anistropy Factor')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_{imgSize4}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

# dataset 101
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df1, palette=sns.color_palette('bright', 11))
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df1)
ax = sns.barplot(x="g", y="OutlierPercent", hue='Tissue', data=df4, errwidth=0, palette=sns.color_palette('deep'))
ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Anistropy Factor')
plt.ylabel('Reflectance leakage (%)')
figFile = os.path.join(output_path, f'Test_Results_Leakage_{imgSize4}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

#--------------
# dataset 41
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df3)
ax = sns.barplot(x="g", y="Error", hue='Tissue', data=df5, errwidth=0, palette=sns.color_palette('deep'))
ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
plt.xlabel('Anistropy Factor')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_{imgSize5}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

# dataset 41
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df1, palette=sns.color_palette('bright', 11))
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df1)
ax = sns.barplot(x="g", y="OutlierPercent", hue='Tissue', data=df5, errwidth=0, palette=sns.color_palette('deep'))
ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Anistropy Factor')
plt.ylabel('Reflectance leakage (%)')
figFile = os.path.join(output_path, f'Test_Results_Leakage_{imgSize5}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

# --------------------------------------------------------------------------------------
# box plot, error vs tissue
fig, ax = plt.subplots(figsize=(10,4), dpi=100)
#ax = sns.boxplot(x="Tissue", y="Error", hue='ImgSize', data=df, linewidth=1, width=0.5, showfliers=False)
ax = sns.barplot(x="Tissue", y="Error", hue='ImgSize', data=df, capsize=0.1, errwidth=1, palette=sns.color_palette('deep'))
ax.legend(title='FoV')
plt.xticks(rotation=-45)
plt.xlabel('')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_Tissue_Error.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

# bar plot, compare the reflectance ratio
fig, ax = plt.subplots(figsize=(10,4), dpi=100)
ax = sns.barplot(x="Tissue", y="OutlierPercent", hue='ImgSize', data=df, palette=sns.color_palette('deep'))
#ax.legend(title='Image Size', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
ax.legend(title='FoV')
plt.xticks(rotation=-45)
plt.xlabel('')
plt.ylabel('Reflectance leakage (%)')
figFile = os.path.join(output_path, f'Test_Results_Leakage.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()


# point plot, compare error vs g
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
ax = sns.pointplot(x="g", y="Error", hue="ImgSize", data=df, markers='.', capsize=0.1, errwidth=1)
ax.legend(title='FoV')
plt.xlabel('Anistropy Factor')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_g_Error.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

print('done')

