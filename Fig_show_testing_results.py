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
df1['ImgSize'] = '6x6 @ 0.02'

imgSize2 = 201
checkpoint_path2 = f'training_results_MCML_{imgSize2}'
df2 = pd.read_csv(os.path.join(checkpoint_path2, f'Test_Results_{imgSize2}.csv'))
df2['ImgSize'] = '4x4 @ 0.02'

imgSize3 = 101
checkpoint_path3 = f'training_results_MCML_{imgSize3}'
df3 = pd.read_csv(os.path.join(checkpoint_path3, f'Test_Results_{imgSize3}.csv'))
df3['ImgSize'] = '2x2 @ 0.02'

imgSize4 = 401
checkpoint_path4 = f'training_results_MCML_{imgSize4}'
df4 = pd.read_csv(os.path.join(checkpoint_path4, f'Test_Results_{imgSize4}.csv'))
df4['ImgSize'] = '4x4 @ 0.01'

imgSize5 = 100
checkpoint_path5 = f'training_results_MCML_{imgSize5}'
df5 = pd.read_csv(os.path.join(checkpoint_path5, f'Test_Results_{imgSize5}.csv'))
df5['ImgSize'] = '4x4 @ 0.04'

output_path = 'testing_results'
if not os.path.exists(output_path):
    os.mkdir(output_path)

df_FoV = df3.append(df2, ignore_index=True)
df_FoV = df_FoV.append(df1, ignore_index=True)

df_Res = df5.append(df2, ignore_index=True)
df_Res = df_Res.append(df4, ignore_index=True)

df = df3.append(df2, ignore_index=True)
df = df.append(df1,  ignore_index=True)
df = df.append(df5,  ignore_index=True)
df = df.append(df4,  ignore_index=True)

# Table Overview of Results, mean of MSE and standard error of MSE
# [df3['Error'].sem(),  df2['Error'].sem(),  df1['Error'].sem(),  df4['Error'].sem(),  df5['Error'].sem()]

results = np.array([[101, df3['Error'].mean(), df3['Error'].std(), df3['delta_g'].mean(), df3['delta_g'].std()],
                    [201, df2['Error'].mean(), df2['Error'].std(), df2['delta_g'].mean(), df2['delta_g'].std()],
                    [301, df1['Error'].mean(), df1['Error'].std(), df1['delta_g'].mean(), df1['delta_g'].std()],
                    [401, df4['Error'].mean(), df4['Error'].std(), df4['delta_g'].mean(), df4['delta_g'].std()],
                    [100, df5['Error'].mean(), df5['Error'].std(), df5['delta_g'].mean(), df5['delta_g'].std()]
            ])

np.savetxt(os.path.join(output_path, 'Test_Results_Overview.txt'), results, fmt='%.3f')

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
# dataset 201
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df2)
ax = sns.barplot(x="g", y="Error", hue='Tissue', data=df2, errwidth=0, palette=sns.color_palette('deep'))
ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
plt.xlabel('Anistropy Factor')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_{imgSize2}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

# dataset 201
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
# dataset 101
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df3)
ax = sns.barplot(x="g", y="Error", hue='Tissue', data=df3, errwidth=0, palette=sns.color_palette('deep'))
ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
plt.xlabel('Anistropy Factor')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_{imgSize3}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

# dataset 101
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
# dataset 401
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df3)
ax = sns.barplot(x="g", y="Error", hue='Tissue', data=df4, errwidth=0, palette=sns.color_palette('deep'))
ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
plt.xlabel('Anistropy Factor')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_{imgSize4}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

# dataset 401
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
# dataset 100
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
# ax = sns.lineplot(x="g", y="Error", hue='Tissue', data=df3)
ax = sns.barplot(x="g", y="Error", hue='Tissue', data=df5, errwidth=0, palette=sns.color_palette('deep'))
ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
plt.xlabel('Anistropy Factor')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_{imgSize5}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

# dataset 100
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
fig, ax = plt.subplots(figsize=(8,4), dpi=300)
#ax = sns.boxplot(x="Tissue", y="Error", hue='ImgSize', data=df, linewidth=1, width=0.5, showfliers=False)
ax = sns.barplot(x="Tissue", y="Error", hue='ImgSize', data=df_FoV, capsize=0.1, errwidth=1, palette=sns.color_palette('deep'))
ax.legend(title='', loc='upper left')
plt.xticks(rotation=-45)
plt.xlabel('')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_tissueError_FoV.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(8,4), dpi=300)
#ax = sns.boxplot(x="Tissue", y="Error", hue='ImgSize', data=df, linewidth=1, width=0.5, showfliers=False)
ax = sns.barplot(x="g", y="Error", hue='Tissue', data=df_FoV, capsize=0.03, errwidth=1, palette=sns.color_palette('deep'))
ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
plt.xticks(rotation=-45)
plt.xlabel('')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_tissueError_g_FoV.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()


# bar plot, compare the reflectance ratio
fig, ax = plt.subplots(figsize=(8,4), dpi=300)
ax = sns.barplot(x="Tissue", y="OutlierPercent", hue='ImgSize', data=df_FoV, errwidth=0, palette=sns.color_palette('deep'))
#ax.legend(title='Image Size', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
ax.legend(title='')
plt.xticks(rotation=-45)
plt.xlabel('')
plt.ylabel('Reflectance leakage (%)')
figFile = os.path.join(output_path, f'Test_Results_Leakage_FoV.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

# point plot, compare error vs g

# color palette as dictionary
palette = {"6x6 @ 0.02":"tab:purple",
           "4x4 @ 0.02":"tab:red", 
           "2x2 @ 0.02":"tab:blue"}

fig, ax = plt.subplots(figsize=(4,3), dpi=300)
ax = sns.pointplot(x="g", y="Error", hue="ImgSize", data=df_FoV, markers='.', capsize=0.1, errwidth=1, palette=palette)
ax.legend(title='', loc='upper left')
#ax.set_ylim(0, 0.06)
plt.xlabel('Anistropy Factor')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_g_FoV.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

# --------------------------------------------------------------------------------------
# box plot, error vs tissue
fig, ax = plt.subplots(figsize=(8,4), dpi=300)
#ax = sns.boxplot(x="Tissue", y="Error", hue='ImgSize', data=df, linewidth=1, width=0.5, showfliers=False)
ax = sns.barplot(x="Tissue", y="Error", hue='ImgSize', data=df_Res, capsize=0.1, errwidth=1, palette=sns.color_palette('deep'))
ax.legend(title='', loc='upper left')
plt.xticks(rotation=-45)
plt.xlabel('')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_tissueError_Res.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(8,4), dpi=300)
#ax = sns.boxplot(x="Tissue", y="Error", hue='ImgSize', data=df, linewidth=1, width=0.5, showfliers=False)
ax = sns.barplot(x="g", y="Error", hue='Tissue', data=df_Res, capsize=0.03, errwidth=1, palette=sns.color_palette('deep'))
ax.legend(title='', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
plt.xticks(rotation=-45)
plt.xlabel('')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_tissueError_g_Res.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()


# bar plot, compare the reflectance ratio
fig, ax = plt.subplots(figsize=(8,4), dpi=300)
ax = sns.barplot(x="Tissue", y="OutlierPercent", hue='ImgSize', data=df_Res, errwidth=0, palette=sns.color_palette('deep'))
#ax.legend(title='Image Size', bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)
ax.legend(title='')
plt.xticks(rotation=-45)
plt.xlabel('')
plt.ylabel('Reflectance leakage (%)')
figFile = os.path.join(output_path, f'Test_Results_Leakage_Res.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

# point plot, compare error vs g

# color palette as dictionary
palette = {"4x4 @ 0.01":"tab:orange",
           "4x4 @ 0.02":"tab:red", 
           "4x4 @ 0.04":"tab:green"}

fig, ax = plt.subplots(figsize=(4,3), dpi=300)
ax = sns.pointplot(x="g", y="Error", hue="ImgSize", data=df_Res, markers='.', capsize=0.1, errwidth=1, palette=palette)
ax.legend(title='', loc='upper left')
#ax.set_ylim(0, 0.06)
plt.xlabel('Anistropy Factor')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_g_Res.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

#=========================
# point plot, compare error vs g
fig, ax = plt.subplots(figsize=(4,3), dpi=300)
ax = sns.pointplot(x="g", y="Error", hue="ImgSize", data=df, markers='.', capsize=0.1, errwidth=1)
ax.legend(title='', loc='upper left')
#ax.set_ylim(0, 0.06)
plt.xlabel('Anistropy Factor')
plt.ylabel('MSE')
figFile = os.path.join(output_path, f'Test_Results_all.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()


print('done')

