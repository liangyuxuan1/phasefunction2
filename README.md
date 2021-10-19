# Estimation of phase function from diffuse reflectance images by deep convolutional neural networks

Summer intern project (remote) supervised by [Prof. Ge Wang](https://biotech.rpi.edu/centers/bic/people/faculty/ge-wang), Rensselaer Polytechnic Institute, USA. May-September, 2021. 

This repository contains the code for my summer intern project. Our paper is submitting to Physics in Medicine & Biology. A draft of the paper is available here. 

- Yuxuan Liang, Chuang Niu, Chen Wei, Shenghan Ren, Wenxiang Cong and Ge Wang, Estimation of phase function from diffuse reflectance images by deep convolutional neural networks, submitting to *Physics in Medicine & Biology*. 

## Step 1: Generate raw data using CUDAMCML

The [GPU accelerated MCML](https://www.atomic.physics.lu.se/biophotonics/research/monte-carlo-simulations/gpu-monte-carlo/) is modified to also record the diffused reflectance and transmittance in Cartesian coordinate. The release program in the directory of ``CudaMCML``was build with Cuda 11.1 under Xubuntu 18.04. 

[Step1_CalcTissueParams.py](Step1_CalcTissueParams.py): Calculate the absorption and reduced scattering coefficients of tissues. The input parameters is in [TissueParamRaw.csv](TissueParamRaw.csv) and the output is in [TissueParams.csv](TissueParams.csv).

[Step1_GenerateRawData_MCML.py](Step1_GenerateRawData_MCML.py): Generate MCI files for training (``train.mci``) and validation (``val.mci``), respectively. Then go to the raw data directories (``RawData_MCML_Train`` and ``RawData_MCML_Val``, created by the program) and run ``./CudaMCML train.mci`` and ``./CudaMCML val.mci`` respectively in the terminal. Make sure to copy ``CudaMCML`` and ``safeprimes_base32.txt`` to the data directories before run ``CudaMCML``.

It needs about 1 hour for training data (8800 CudaMCML runs, 301x301 pixels) and about 10 minutes for validation data (1760 runs, 301x301 pixels). 

## Step2: Extract Rd_xy from the raw data

[Step2_Change_MCML_Data.py](Step2_Change_MCML_Data.py): Extract Rd_xy from the outputs of CudaMCML and save them to the directories of ``ImageCW_train`` and ``ImageCW_Val`` as numpy array. 

Needs about 18 minutes for both training and validation datasets.

## Step3: Calculate the mean and std of the training dataset

[Step3_CalcMeanSTD.py](Step3_CalcMeanSTD.py)

The mean and std values are used to normalize the datasets (training, validation and testing) before training the neural networks.

## Step4: Parameter selection

[Step4_Train_Val.py](Step4_Train_Val.py)

We used ResNet-18 as the backbone and modified it to estimate the phase function using a Gaussian mixture model.The number of neurons in the last Fc layer of ResNet-18 was set to the number of Gaussian components (NoG). The optimal NoG was determined by varying the NoG from 2 to 10 and then training the models separately. The NoG value of the model with the smallest validation error was selected as the best NoG.

Depending on the NoG, training a model (30 epochs) costs about 20 minutes (NoG=2) to 1 hour (NoG=10).


## Config Github proxy

https://blog.csdn.net/weixin_39827315/article/details/110661140