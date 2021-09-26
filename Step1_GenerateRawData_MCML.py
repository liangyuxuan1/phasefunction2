# Generate training, validation and test datalist files for MCML

import numpy as np
import os
import pandas as pd

def generate_mci(parameters, gV, num_of_samples, filename):
    fid = open(filename, 'w')
    fid.write('1.0                      	# file version\n')
    
    num_of_tissues, _ = parameters.shape
    num_of_runs = num_of_tissues*len(gV)*num_of_samples

    fid.write(f'{num_of_runs}                        	# number of runs\n\n')

    for ti in range(num_of_tissues):
        ua, rus, tissue = parameters['ua'].loc[ti], parameters['rus'].loc[ti], parameters['Tissue'].loc[ti]
        for gi in range(len(gV)):
            g = gV[gi]
            us = rus/(1-g)
            for i in range(1, num_of_samples+1):
                run = ti*len(gV)*num_of_samples + gi*num_of_samples + i
                fid.write(f'#### SPECIFY DATA FOR RUN {run}\n')
                fid.write('#InParm                    	# Input parameters. cm is used.\n')
                fid.write(f'{tissue}_g{gi:>1d}_{i:>04d}.mco 	A	      	# output file name, ASCII.\n')
                fid.write('10000000                  	# No. of photons\n')
                fid.write('0.01	0.01               	    # dz, dr [cm]\n')
                fid.write('10 20 10                	# No. of dz, dr, da.\n\n')
                fid.write('1                        	# Number of layers\n')
                fid.write('#n	mua	mus	g	d         	# One line for each layer\n')
                fid.write('1                         	# n for medium above\n')
                fid.write(f'1.37 {ua:>.6f} {us:>.6f} {g:>.6f} 100                      	# Number of layers\n')
                fid.write('1                        	# n for medium below\n')
                fid.write('\n')

    fid.close()

# ==============================================================

# 2021-09-17,V7 dataset
# using optical parameters in Ren Shenghan PlosOne paper
# Using modified Cuda MCML to output Rd_xy and Tr_xy

# absorption coefficient, [0.01, 10] mm^-1
# scattering coefficient, [0.1, 100] mm^-1

# Tissue parameters are calculated according to :
# George Alexandrakis, Fernando R Rannou and Arion F Chatziioannou, 
# Tomographic bioluminescence imaging by use of a combined optical-PET (OPET) system: a computer simulation feasibility study
# Phys. Med. Biol. 50 (2005) 4225–4241

tissueParams = 'TissueParams.csv'

df = pd.read_csv(tissueParams)
# change the unit of ua and us from mm-1 to cm-1, as required by MCML
df['ua'] = df['ua']*10 
df['rus'] = df['rus']*10 

n  = 1.37               # refractive index, no need to vary for single layer slab
trainNum = 200          # training number of runs (images) for each set of parameters
valNum   = 40

# train less and validation more leads to large validation error
# g_train = [0.6, 0.7, 0.8, 0.9]
# g_val   = [0.55, 0.65, 0.75, 0.85, 0.95]

g_train = [0.65, 0.75, 0.85, 0.95]
g_val   = [0.6, 0.7, 0.8, 0.9]

train_path  = 'RawData_MCML_Train_41'
val_path    = 'RawData_MCML_Val_41'

if not os.path.exists(train_path):
    os.mkdir(train_path)

if not os.path.exists(val_path):
    os.mkdir(val_path)

os.system(f'rm {train_path}{os.sep}*.mco')
generate_mci(df, g_train, trainNum, os.path.join(train_path, 'train.mci'))

os.system(f'rm {val_path}{os.sep}*.mco')
generate_mci(df, g_val, valNum, os.path.join(val_path, 'val.mci'))

print('done')
