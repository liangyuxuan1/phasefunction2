# Estimation of phase function from diffuse reflectance images by deep convolutional neural networks

Summer intern project (remote) supervised by [Prof. Ge Wang](https://biotech.rpi.edu/centers/bic/people/faculty/ge-wang), Rensselaer Polytechnic Institute, USA. May-September, 2021. 

This repository contains the code for my summer intern project. Our paper is submitting to Physics in Medicine & Biology. A draft of the paper is available [here](phase_manuscript_20211019.pdf). 

- Yuxuan Liang, Chuang Niu, Chen Wei, Shenghan Ren, Wenxiang Cong and Ge Wang, Estimation of phase function from diffuse reflectance images by deep convolutional neural networks, submitting to *Physics in Medicine & Biology*. 


## Development Environment 
- Xubuntu 18.04
- Cuda 11.1
- Cudnn 8.1.1
- Python 3.8
- Pytorch 1.8.2

## Usage

### Step 1: Generate raw data using CUDAMCML

The diffuse reflectance images of tissues are simulated using [CUDAMCML](https://www.atomic.physics.lu.se/biophotonics/research/monte-carlo-simulations/gpu-monte-carlo/), which is modified to also record the diffused reflectance and transmittance in Cartesian coordinate. The release program in the directory of ``CudaMCML``was build with Cuda 11.1 under Xubuntu 18.04. 

[Step1_CalcTissueParams.py](Step1_CalcTissueParams.py): Calculate the absorption and reduced scattering coefficients of tissues. The input parameters is in [TissueParamRaw.csv](TissueParamRaw.csv) and the output is in [TissueParams.csv](TissueParams.csv). Here the unit of the absorption and reduced scattering coefficients is mm$^{-1}$. It is changed to cm$^{-1}$ in the following program. 

[Step1_GenerateRawData_MCML.py](Step1_GenerateRawData_MCML.py): Generate MCI files for training (``train.mci``) and test (``val.mci``) datasets, respectively. Then go to the raw data directories (``RawData_MCML_Train`` and ``RawData_MCML_Val``, created by the program) and run ``./CudaMCML train.mci`` and ``./CudaMCML val.mci`` respectively in the terminal. Make sure to copy ``CudaMCML`` and ``safeprimes_base32.txt`` to the data directories before run ``CudaMCML``.

Five reflectance image datasets are constructed to investigate the effects of FOV and spatial resolution on the estimation accuracy of the phase function.

### Step2: Extract reflectance images from the raw data

[Step2_Change_MCML_Data.py](Step2_Change_MCML_Data.py): Extract Rd_xy from the outputs of CudaMCML and save them to the directories of ``ImageCW_train`` and ``ImageCW_Val`` as numpy array. 


### Step3: Calculate the mean and std of the training dataset

[Step3_CalcMeanSTD.py](Step3_CalcMeanSTD.py)

The mean and std values are used to normalize the datasets before training and test the neural networks.

### Step4: Parameter selection

[Step4_Train_Val_LoG.py](Step4_Train_Val_LoG.py)

We use ResNet-18 as the backbone and modify it to estimate the phase function using a Gaussian mixture model. The number of neurons in the last Fc layer of ResNet-18 is set to 3 times the number of Gaussian components (NoG). The optimal NoG is determined by a leave-one-$g$-out (LOGO) cross-validation method. 


### Step5: Model training

[Step5_Train.py](Step5_Train.py)

Five models are trained on each dataset respectively. 

### Step6: Model test

[Step6_Test.py](Step6_Test.py)

The models are tested on the corresponding test dataset and their average accuracy is reported.

Programs named as `Fig*.py` are used to draw the figures in the paper. 