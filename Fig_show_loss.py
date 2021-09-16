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

loss_file = os.path.join(os.path.dirname(__file__),'train_loss_run_1.csv')
df = pd.read_csv(loss_file)

fig, ax = plt.subplots(figsize=(6,4), dpi=100)
ax = sns.lineplot(x="Epoch", y="Error", hue='Events', data=df)
ax.legend(title='', loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')

figFile = loss_file[:-3] + 'png'
plt.savefig(figFile, bbox_inches='tight')
plt.show()


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

