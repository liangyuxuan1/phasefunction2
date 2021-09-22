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

imgSize = 301

checkpoint_path = f'training_results_MCML_{imgSize}'

df = pd.read_csv(os.path.join(checkpoint_path, f'Test_Results_{imgSize}.csv'))

fig, ax = plt.subplots(figsize=(6,4), dpi=100)
ax = sns.lineplot(x="g", y="Error", hue="Tissue", data=df)
ax.legend(title='', loc='upper left')
plt.xlabel('Anistropy Factor')
plt.ylabel('MSE')

figFile = os.path.join(checkpoint_path, f'Test_Results_{imgSize}.png')
plt.savefig(figFile, bbox_inches='tight')
plt.show()

print('done')

