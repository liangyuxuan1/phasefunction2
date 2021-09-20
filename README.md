# Phase function estimation using CudaMCML

The [GPU accelerated MCML](https://www.atomic.physics.lu.se/biophotonics/research/monte-carlo-simulations/gpu-monte-carlo/) is modified to also record the diffused reflectance and transmittance in Cartesian coordinate. The release program in the directory of ``CudaMCML``was build with Cuda 11.1 under Xubuntu 18.04. 

Refer to: E. Alerstam, T. Svensson, and S. Andersson-Engels, Parallel computing with graphics processing units for high speed Monte Carlo simulation of photon migration, J. Biomedical Optics Letters 13, 060504 (2008) 

## Step 1: Generate raw data using MCML

[Step1_CalcTissueParams.py](Step1_CalcTissueParams.py): Calculate the absorption and reduced scattering coefficients of tissues. 

Refer to: George Alexandrakis, Fernando R Rannou and Arion F Chatziioannou,Tomographic bioluminescence imaging by use of a combined optical-PET (OPET) system: a computer simulation feasibility study,  Phys. Med. Biol. 50, 4225–4241 (2005)

[Step1_GenerateRawData_MCML.py](Step1_GenerateRawData_MCML.py): Generate MCI files for training (``train.mci``) and validation (``val.mci``), respectively. Then go to the raw data directories (``RawData_MCML_Train`` and ``RawData_MCML_Val``, created by the program) and run ``./CudaMCML train.mci`` and ``./CudaMCML val.mci`` respectively in the terminal. Make sure to copy ``CudaMCML`` and ``safeprimes_base32.txt`` to the data directories before run ``CudaMCML``.


## Step2: Extract Rd_xy from the raw data

[Step2_Change_MCML_Data.py](Step2_Change_MCML_Data.py): Extract Rd_xy from the outputs of CudaMCML and save them to the directories of ``ImageCW_train`` and ``ImageCW_Val`` as numpy array. 

## Step3: Calculate the mean and std of the training dataset

[Step3_CalcMeanSTD.py](Step3_CalcMeanSTD.py)

The mean and std values are used to normalize the datasets (training, validation and testing) before training the neural networks.

## Step4: Parameter selection

[Step4_Train_Val.py](Step4_Train_Val.py)

We used ResNet-18 as the backbone and modified it to estimate the phase function using a Gaussian mixture model.The number of neurons in the last Fc layer of ResNet-18 was set to the number of Gaussian components (NoG). The optimal NoG was determined by varying the NoG from 2 to 10 and then training the models separately. The NoG value of the model with the smallest validation error was selected as the best NoG.

