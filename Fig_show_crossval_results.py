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

imgSize = 251
checkpoint_path = f'training_results_MCML_{imgSize}'

output_path = 'testing_results'
if not os.path.exists(output_path):
    os.mkdir(output_path)

files = glob.glob(os.path.join(checkpoint_path, 'Train_Val_Results*.csv'))
df = pd.DataFrame()
for file in files:
    df = df.append(pd.read_csv(file), ignore_index=True)

df.insert(df.shape[1], 'AIC', 0)
df.insert(df.shape[1], 'BIC', 0)
df.insert(df.shape[1], 'n', 0)  # number of samples
df.insert(df.shape[1], 'k', 0)  # number of NN free parameters

df['k'] = 512 * df['NoG']*3 + df['NoG']*3
df.loc[df['Events']=='Train', 'n'] = 11*4*200*0.8
df.loc[df['Events']=='Validation', 'n'] = 11*4*200*0.2

# https://en.wikipedia.org/wiki/Akaike_information_criterion
# https://en.wikipedia.org/wiki/Bayesian_information_criterion

df['AIC'] = df['n'] * np.log(df['Error']*df['n']) + 2*df['k']
df['BIC'] = df['n'] * np.log(df['Error']) + df['k']*np.log(df['n'])

print(df)

# sns.set(color_codes=True)
# sns.set_palette('Set2')
# sns.set_style('whitegrid')
# sns.set(color_codes=True, style='whitegrid')

df2 = df[df['Events']=='Validation']

'''
# ------------------------------------------------------------------------
# 怎样画三个坐标，参考这里
# https://matplotlib.org/2.0.2/examples/axes_grid/demo_parasite_axes2.html

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

offset = 60
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis["right"] = new_fixed_axis(loc="right", axes=par2, offset=(offset, 0))
par2.axis["right"].toggle(all=True)

p1, = host.plot(df2['NoG'], df2['Error'], label="Cross-Validation")
p2, = par1.plot(df2['NoG'], df2['AIC'], label="AIC")
p3, = par2.plot(df2['NoG'], df2['BIC'], label="BIC")

par1.set_ylim(500, 20000)
par2.set_ylim(25000, 65000)

host.legend()

plt.show()
'''
# ------------------------------------------------------------------------
# 如果用lineplot，参考这里
# https://kedar.hashnode.dev/how-to-combine-two-line-charts-in-seaborn-and-python

fig, ax1 = plt.subplots(figsize=(8,4))
ax2 = ax1.twinx()
sns.pointplot(data=df2, x='NoG', y='AIC', ax=ax1, color='tab:blue', capsize=0.1, errwidth=1)
sns.pointplot(data=df2, x='NoG', y='BIC', ax=ax2, color='tab:orange', capsize=0.1, errwidth=1)
ax1.set(xlabel='Number of Gaussian Components')

# 怎样添加legend， 参考这个
# https://matplotlib.org/2.0.2/examples/axes_grid/demo_parasite_axes2.html
plt.plot(np.nan, color='tab:blue', label = 'AIC')
plt.plot(np.nan, color='tab:orange', label = 'BIC')
plt.legend()

figFile = os.path.join(output_path, f'Fig_AIC_BIC_{imgSize}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()


# ------------------------------------------------------------------------
# 怎样删除legend上面的title
# https://stackoverflow.com/questions/43151440/remove-seaborn-barplot-legend-title

fig, ax = plt.subplots(figsize=(8,4))
sns.pointplot(data=df, x='NoG', y='Error', hue='Events', markers='.', capsize=0.1, errwidth=1)
# ax = sns.lineplot(x="NoG", y="Error", hue="Events", style='Events', data=df, markers=True, dashes=False)
lgd = ax.legend()
lgd.set_title('')
plt.ylabel('MSE')
plt.xlabel('Number of Gaussian Components')

figFile = os.path.join(output_path, f'Fig_Cross_Val_{imgSize}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

df2 = df[df['Events']=='Validation']
ds = df2.groupby(['NoG', 'Events'])
print(ds.mean()['Error'])


'''
sns.lineplot(data=df, x='NoG', y='Error', hue='Events', style='Events', dashes=False, markers=True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

sns.boxplot(data=df, x='NoG', y='Error', hue='Events')
plt.show()

sns.boxenplot(data=df, x='NoG', y='Error', hue='Events')
plt.show()

sns.barplot(data=df, x='NoG', y='Error', hue='Events')
plt.show()
'''

print('done')

