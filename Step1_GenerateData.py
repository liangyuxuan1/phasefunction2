# Generate training, validation and test datalist files for MCML

import numpy as np
import os


def generate_mci(ua_list, us_list, g_list, num_of_samples, filename):
    fid = open(filename, 'w')
    fid.write('1.0                      	# file version\n')
    
    num_of_runs = len(ua_list)*len(us_list)*len(g_list)*num_of_samples

    fid.write(f'{num_of_runs}                        	# number of runs\n\n')

    run = 1
    for ia, ua in enumerate(ua_list):
        for js, urs in enumerate(us_list):
            for kg, g in enumerate(g_list):
                # change reduced scatter coefficient to us
                us = urs/(1-g)

                for i in range(1, num_of_samples+1):
                    fid.write(f'#### SPECIFY DATA FOR RUN {run}\n')
                    fid.write('#InParm                    	# Input parameters. cm is used.\n')
                    fid.write(f'a{ia:>02d}_s{js:>02d}_g{kg:>02d}_{i:>02d}.mco 	A	      	# output file name, ASCII.\n')
                    fid.write('10000000                  	# No. of photons\n')
                    fid.write('0.01	0.01               	# dz, dr [cm]\n')
                    fid.write('1 250 90                	# No. of dz, dr, da.\n\n')
                    fid.write('1                        	# Number of layers\n')
                    fid.write('#n	mua	mus	g	d         	# One line for each layer\n')
                    fid.write('1                         	# n for medium above\n')
                    fid.write(f'1.37 {ua:>.6f} {us:>.6f} {g:>.6f} 10                      	# Number of layers\n')
                    fid.write('1                        	# n for medium below\n')
                    fid.write('\n')

                    run = run+1

    fid.close()

# ==============================================================

# parameters refer to 
# Ivanˇciˇc, M., Nagliˇc, P., Pernuˇs, F., Likar, B. & B ̈urmen, M. (2018).  
# Efficient estimation of subdiffusiveoptical parameters in real time from spatially resolved reflectance by artificial neural networks,
# Optics Letters43(12): 2901–2904.

ua_train = np.linspace(start=0.01, stop=12.0,  num=7)
delta = (np.max(ua_train)-np.min(ua_train)) / (len(ua_train)-1)
ua_val   = ua_train[:-1] + delta/4.0
ua_test  = ua_train[:-1] + delta/2.0 
print(ua_train)
print(ua_val)
print(ua_test)

# reduced scattering coefficient: us'=us(1-g1)
us_train = np.linspace(start=5.0,  stop=35.0,  num=10)
delta = (np.max(us_train)-np.min(us_train)) / (len(us_train)-1)
us_val   = us_train[:-1] + delta/4.0
us_test  = us_train[:-1] + delta/2.0
print(us_train)
print(us_val)
print(us_test)

# gamma: = (1-g2)/(1-g1)
gamma_train = np.linspace(start=1.99,  stop=2.31,  num=10)
gamma_val   = np.linspace(start=2.01,  stop=2.29,  num=5)
gamma_test  = np.linspace(start=2.04,  stop=2.26,  num=4)

# g = g1
g_train = np.linspace(start=0.75,  stop=0.95,  num=5)
delta = (np.max(g_train)-np.min(g_train)) / (len(g_train)-1)
g_val   = g_train[:-1] + delta/4.0
g_test  = g_train[:-1] + delta/2.0

print(g_train)
print(g_val)
print(g_test)


train_path = 'train'
val_path = 'val'
test_path = 'test'

if not os.path.exists(train_path):
    os.mkdir(train_path)

if not os.path.exists(val_path):
    os.mkdir(val_path)

if not os.path.exists(test_path):
    os.mkdir(test_path)

generate_mci(ua_train, us_train, g_train, 1, os.path.join(train_path, 'train.mci'))
generate_mci(ua_val, us_val, g_val, 1, os.path.join(val_path, 'val.mci'))
generate_mci(ua_test, us_test, g_test, 1, os.path.join(test_path, 'test.mci'))


print('done')






