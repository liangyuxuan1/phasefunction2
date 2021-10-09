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

from scipy import stats

#=================================================================================

imgSize1 = 301
checkpoint_path1 = f'training_results_MCML_{imgSize1}'
df1 = pd.read_csv(os.path.join(checkpoint_path1, f'Test_Results_{imgSize1}.csv'))
df1['ImgSize'] = '0.6x0.6 @ 0.002'

imgSize2 = 201
checkpoint_path2 = f'training_results_MCML_{imgSize2}'
df2 = pd.read_csv(os.path.join(checkpoint_path2, f'Test_Results_{imgSize2}.csv'))
df2['ImgSize'] = '0.4x0.4 @ 0.002'

imgSize3 = 101
checkpoint_path3 = f'training_results_MCML_{imgSize3}'
df3 = pd.read_csv(os.path.join(checkpoint_path3, f'Test_Results_{imgSize3}.csv'))
df3['ImgSize'] = '0.2x0.2 @ 0.002'

imgSize4 = 401
checkpoint_path4 = f'training_results_MCML_{imgSize4}'
df4 = pd.read_csv(os.path.join(checkpoint_path4, f'Test_Results_{imgSize4}.csv'))
df4['ImgSize'] = '0.4x0.4 @ 0.001'

imgSize5 = 100
checkpoint_path5 = f'training_results_MCML_{imgSize5}'
df5 = pd.read_csv(os.path.join(checkpoint_path5, f'Test_Results_{imgSize5}.csv'))
df5['ImgSize'] = '0.4x0.4 @ 0.004'

output_path = 'testing_results'
if not os.path.exists(output_path):
    os.mkdir(output_path)

df_FoV = df3.append(df2, ignore_index=True)
df_FoV = df_FoV.append(df1, ignore_index=True)

df_Res = df5.append(df2, ignore_index=True)
df_Res = df_Res.append(df4, ignore_index=True)

df = df1.append(df2, ignore_index=True)
df = df.append(df3,  ignore_index=True)
df = df.append(df4,  ignore_index=True)
df = df.append(df5,  ignore_index=True)

# --------------------------------------------------------------------------------------
# two sample t test
# https://www.marsja.se/how-to-perform-a-two-sample-t-test-with-python-3-different-methods/

# Subset data
df301 = df1.query('g == 0.9')['Error']    # 0.6x0.6 @ 0.002
df201 = df2.query('g == 0.9')['Error']    # 0.4x0.4 @ 0.002
df101 = df3.query('g == 0.9')['Error']    # 0.2x0.2 @ 0.002
df401 = df4.query('g == 0.9')['Error']    # 0.4x0.4 @ 0.001
df100 = df5.query('g == 0.9')['Error']    # 0.4x0.4 @ 0.004

# Checking the Normality of Data
# Results: the MSE are not normally distributed. 
print(stats.shapiro(df301))
print(stats.shapiro(df201))
print(stats.shapiro(df101))
print(stats.shapiro(df401))
print(stats.shapiro(df100))

# Checking the Homogeneity of Variances Assumption
print(stats.levene(df301, df201))
print(stats.levene(df201, df101))
print(stats.levene(df301, df101))

# two sample t test, can not use because the data is not normall distributed.
res = stats.ttest_ind(df301, df201, equal_var=True)
print(res)
res = stats.ttest_ind(df201, df101, equal_var=False)
print(res)

#===============================================================
# https://www.reneshbedre.com/blog/mann-whitney-u-test.html
# Mann-Whitney U test (Wilcoxon rank sum test )
res = stats.mannwhitneyu(df301, df201, alternative = 'two-sided')
print(res)
res = stats.mannwhitneyu(df201, df101, alternative = 'two-sided')
print(res)
res = stats.mannwhitneyu(df301, df101, alternative = 'two-sided')
print(res)


# False discovery rate correction
# https://towardsdatascience.com/multiple-hypothesis-testing-correction-for-data-scientist-46d3a3d1611d



print('done')

