import numpy as np
from collections import namedtuple
import seaborn as sns
import seaborn_image as isns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

# matplotlib.use("Agg")

# ==================================================================
if __name__=='__main__':

    DataListFile = "TrainDataCW_v6_Results.csv"
    img_path = "ImageCW_v6"

    df = pd.read_csv(os.path.join(img_path, DataListFile))
    df = pd.DataFrame(df, columns=['Tissue', 'Reflectance', 'Transmittance'])
    df = df.reset_index().melt(id_vars='Tissue', 
                               value_vars=['Reflectance', 'Transmittance'], 
                               var_name='View',  value_name='vals')
    
    fig, ax = plt.subplots(figsize=(6,4), dpi=100)
    ax = sns.pointplot(x="Tissue", y="vals", hue='View', data=df)
    ax.legend(title='', loc='upper right')
    plt.xlabel('')
    plt.ylabel('')
    
    figFile = os.path.join(img_path, 'Fig_Reflectance.png')
    plt.savefig(figFile, bbox_inches='tight')
    plt.show()

    print('done')




