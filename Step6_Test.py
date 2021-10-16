# PyTorch has two primitives to work with data: torch.utils.data.DataLoader and torch.utils.data.Dataset. 
# Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset.

import torch
from torch import nn
from torch.utils.data import DataLoader, dataset
from torchvision import transforms

# pip install torch-summary
from torchsummary import summary

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# pip install matplotlib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import seaborn as sns
import numpy as np
import time
import shutil

# pip install pandas
import pandas as pd

# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.append('./src')
from preprocessing import DataPreprocessor
from CustomImageDataset_Pickle import CustomImageDataset_Pickle
import checkpoints
from logger import double_logger
import tester
from NetworkModels import Resnet18

class gtNormalize(object):
    def __init__(self, minV, maxV):
        self.minV = torch.tensor(minV)
        self.maxV = torch.tensor(maxV)
    
    def __call__(self, gt):
        # normalize gt to [0.01, 1] to facilitate the calculation of relative error
        k = torch.div(1.0-0.01, self.maxV - self.minV)
        gt = 0.01 + k*(gt - self.minV)
        return gt

    def restore(self, gt):
        # restore the normalized values
        k = torch.div(1.0-0.01, self.maxV - self.minV)
        gt = (gt - 0.01)/k + self.minV
        return gt

# Optimizing the Model Parameters
# To train a model, we need a loss function and an optimizer.
def kl_divergence(dis_a, dis_b):
    disa = dis_a + 1e-6
    disb = dis_b + 1e-6
    loga = torch.log(disa)
    logb = torch.log(disb)  
    part1 = dis_a*loga
    part2 = dis_a*logb
    result = torch.mean(torch.sum(part1-part2, dim=1))
    assert torch.isnan(result).sum() == 0
    return result

def HG_theta(g, theta):
    # calculate 2*pi*p(cos(theta))
    bSize = g.size()[0] 
    p = torch.zeros(bSize, theta.size()[0]).cuda()
    for i in range(bSize):
        p[i,:] = 0.5*(1-g[i]*g[i])/((1+g[i]*g[i]-2*g[i]*torch.cos(theta))**(3.0/2.0))
        p[i,:] *= torch.sin(theta)
        # print(torch.sum(p[i,:]))
    return p

def normfun(x, mean, sigma):
    pi = np.pi
    pi = torch.tensor(pi)
    std = sigma + 1e-6
    G_x = torch.exp(-((x - mean)**2)/(2*std**2)) / (std * torch.sqrt(2*pi))
    return G_x

def GMM(nnOut, theta):
    pi = torch.tensor(np.pi)
    w = nnOut[:, 0:num_of_Gaussian]                         # weight [0, 1], sum(w)=1
    w_sum = torch.sum(w, dim=1)
    m = nnOut[:, num_of_Gaussian:num_of_Gaussian*2]*pi      # mean [0, 1]*pi
    d = nnOut[:, num_of_Gaussian*2:num_of_Gaussian*3]       # std [0, 1]

    bSize = nnOut.size()[0]
    gmm = torch.zeros(bSize, theta.size()[0]).cuda()
    for i in range(bSize):
        for j in range(num_of_Gaussian):
            gmm[i,:] += (w[i, j]/w_sum[i]) * normfun(theta, m[i, j], d[i, j])
        sumGmm = torch.sum(gmm[i,:]) * theta[1]     # discretization bin = pi/len(theta) radian
        gmm[i,:] /= sumGmm      # normalize to gurrantee the sum=1
    return gmm

def loss_func_mse(prediction, gt):
    gmm = GMM(prediction, theta)
    
    # gx = gtNorm.restore(gt.to("cpu"))
    # gt = gt.to(device)
    # g = gx[:, 2]
    g = gt[:,2]
    p_theta = HG_theta(g, theta)

    # loss1 = kl_divergence(gmm, p_theta)
    # loss2 = kl_divergence(p_theta, gmm)
    # loss_phase = (loss1 + loss2)/2.0
    
    loss_phase = nn.MSELoss()(gmm, p_theta)

    # loss_phase = (kl_divergence(gmm, p_theta) + kl_divergence(p_theta, gmm))/2.0

    # uas = prediction[:, -2:]
    # gt_uas = gt[:, :2]
    # loss_uas = nn.MSELoss()(uas, gt_uas)  

    #loss = loss_phase + loss_uas

    estimated_g = torch.sum(gmm*torch.cos(theta), dim=1)/1000.0
    delta_g = torch.div(torch.abs(estimated_g-g), g)*100.0

    loss = loss_phase

    return loss, p_theta, gmm, estimated_g, delta_g


# ==============================================================================================================
    # Need to calculate the mean and std of the dataset first.

    # imageCW, 500x500, g=0.5:0.01:0.95, training number = 70, mean = 0.0050, std = 0.3737
    # imageCW, 500x500, g=-1:0.025:1, training number = 100, mean = 0.0068, std = 1.2836
    # imageCW, 500*500, 14 materials, training number = 500, mean = 0.0040, sta = 0.4645
    # imageCW, 500*500, 12 materials, training number = 500, mean = 0.0047, sta = 0.5010
    # gt = [ua, us, g], min = [0.0010, 0.0150, 0.1550], max = [0.2750, 100.92, 0.9550]

    # imageCW_v3, 500x500, training number = 80, mean = 0.0026, std = 0.9595
    # imageCW_v4, 500x500, training number = 50, mean = 0.0026, std = 0.9595
    # trainDataCW_v3_ExcludeExtremes, 500x500, training number = 80, mean = 0.0028, std = 0.8302

    # imageCW_v4, 500x500, training number = 200, mean = 0.0045, std = 0.3633
    # imageCW_v4_fat, 500x500, training number = 200, mean = 0.0068, std = 0.3823

    # imageCW_v5, 500x500, number=1000, mean=0.0035, std=0.2197

    # Dataset V6, large phantom, mean = 0.0022, std = 0.2915

    # 2021-09-21
    # g_train = [0.65, 0.75, 0.85, 0.95]
    # g_val   = [0.6, 0.7, 0.8, 0.9]
    # Dataset MCML 301x301 (299x299), mean = 0.04370, std = 0.53899
    # Dataset MCML 501x501 (499x499), mean = 0.01578, std = 0.32363
    # Dataset MCML 251x251 (249x249), mean = 0.01584, std = 0.30017

    # 2021-09-28
    # 301, dr=0.002, ndr=150, FoV=0.6x0.6, mean = 0.86591, std = 3.01413
    # 201, dr=0.002, ndr=100, FoV=0.4x0.4, mean = 1.64004, std = 4.40112
    # 101, dr=0.002, ndr=50,  FoV=0.2x0.2, mean = 4.20267, std = 8.26528
    # 401, dr=0.001, ndr=200, FoV=0.4x0.4, mean = 1.63391, std = 4.67807
    # 100, dr=0.004, ndr=50,  FoV=0.4x0.4, mean = 1.65235, std = 4.12344

def test(imgSize, index=None):

    if imgSize == 301:
        meanPixelVal = 0.86591   
        stdPixelVal  = 3.01413

    if imgSize == 201:
        meanPixelVal = 1.64004   
        stdPixelVal  = 4.40112

    if imgSize == 101:
        meanPixelVal = 4.20267   
        stdPixelVal  = 8.26528

    if imgSize == 401:
        meanPixelVal = 1.63391   
        stdPixelVal  = 4.67807

    if imgSize == 100:
        meanPixelVal = 1.65235   
        stdPixelVal  = 4.12344
    
    test_img_path       = f"ImageCW_Val_{imgSize}"
    test_DataListFile   = f"ValDataCW_MCML_{imgSize}.csv"

    checkpoint_path     = f'training_results_MCML_{imgSize}'

    preprocessing_transformer = transforms.Normalize(meanPixelVal, stdPixelVal)
    inverse_preprocessing_transformer = transforms.Normalize(-meanPixelVal, 1.0/stdPixelVal)

    test_labels     = pd.read_csv(os.path.join(test_img_path, test_DataListFile))

    test_pickle_file_name  = 'test.pkl'
    
    print('Preprocessing...')
    DataPreprocessor().dump(test_labels, test_img_path, checkpoint_path, test_pickle_file_name, preprocessing_transformer)
    print('Preprocessing finished')

    test_data = CustomImageDataset_Pickle(
        img_labels = test_labels,
        file_preprocessed = os.path.join(checkpoint_path, test_pickle_file_name)
    )

    # Define model
    model = Resnet18(num_classes=num_of_Gaussian*3)

    Tst = tester.Tester()
    print(f'NoG: {num_of_Gaussian}, Start testing')

    df_results = pd.DataFrame()
    for run in range(5):
        model_name = f'best_model_NoG_11_run_{run}.pt'          

        df_loss, features = Tst.run(test_data, model, loss_func_mse, checkpoint_path, model_name, inverse_preprocessing_transformer)
        df_loss['Run']=run
        df_results = df_results.append(df_loss, ignore_index=True)

        np.save(os.path.join(checkpoint_path, f'Test_Results_Features_{imgSize}_Run_{run}.npy'), features)
        df_loss.to_csv(os.path.join(checkpoint_path, f'Test_Results_{imgSize}_Run_{run}.csv'))

        index = [df_loss['Error'].idxmin(), df_loss['Error'].idxmax()]
        Tst.run(test_data, model, loss_func_mse, checkpoint_path, model_name, inverse_preprocessing_transformer, 'images_testing', index, run)

    df_results.to_csv(os.path.join(checkpoint_path, f'Test_Results_{imgSize}.csv'))

#====================================================================
if __name__=='__main__':
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    theta = np.arange(0, np.pi, 0.001)
    theta = torch.from_numpy(theta).to(device)

    num_of_Gaussian = 11

    test(imgSize=201)
    test(imgSize=100)
    test(imgSize=301)
    test(imgSize=101)
    test(imgSize=401)

    print('Done')