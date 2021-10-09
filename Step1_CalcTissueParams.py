# calc the mean and std of a dataset

import numpy as np
import pandas as pd
import os

# Tissue parameters are calculated according to :
# George Alexandrakis, Fernando R Rannou and Arion F Chatziioannou, 
# Tomographic bioluminescence imaging by use of a combined optical-PET (OPET) system: a computer simulation feasibility study
# Phys. Med. Biol. 50 (2005) 4225–4241

rawParams = 'TissueParamRaw.csv'

df = pd.read_csv(rawParams)

lamda = 670     # wavelength, nm

# Reduced scattering coefficients
# Equation 1 in reference
df['rus'] = df['a'] * np.power(lamda, -df['b'])

# absorption coefficients
# Equation 2 in reference

# ua_HbO2 and ua_Hb at wavelength 670 nm, https://omlc.org/spectra/hemoglobin/summary.html
ua_HbO2 = (2.303*294*150/64500)/10.0       
ua_Hb   = (2.303*2795.12*150/64500)/10.0
# ua of water: R.M.Pope and E. S. Fry,"Absorption spectrum(380-700nm)of pure water.II.Integrating cavity measurements," Appl.Opt.,36,8710--8723,(1997).
ua_W    = 0.00439/10.0

df['ua'] = df['SB']*(df['x']*ua_Hb + (1-df['x'])*ua_HbO2) + df['SW']*ua_W

# us = reduced_us/(1-g)

df['lamda'] = df['rus']/df['ua']

print(df)

df.to_csv('TissueParams.csv', index=False)
