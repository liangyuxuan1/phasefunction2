# PyTorch has two primitives to work with data: torch.utils.data.DataLoader and torch.utils.data.Dataset. 
# Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset.

import torch
from torch import nn
from torch.utils.data import DataLoader, dataset
from torchvision import transforms
from sklearn.model_selection import KFold

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
import trainer
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

    loss = loss_phase

    return loss


# ==============================================================================================================
if __name__=='__main__':

    torch.backends.cudnn.benchmark = True

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

    # 2021-09-27 new dataset at resolution 0.002
    # 401x401 (399x399), 0.8cm x 0.8 cm, mean = 0.53076, std = 2.29144
    # 201x201 (199x199), 0.4cm x 0.4 cm, mean = 1.64004, std = 4.40112
    # 101x101 (99x99),   0.2cm x 0.2 cm, mean = 4.20267, std = 8.26528
    # 51x51   (49x49),   0.1cm x 0.1 cm, mean = 9.19103, std = 15.37926

    # 2021-09-28
    # 301, dr=0.002, ndr=150, FoV=0.6x0.6, mean = 0.86591, std = 3.01413
    # 201, dr=0.002, ndr=100, FoV=0.4x0.4, mean = 1.64004, std = 4.40112
    # 101, dr=0.002, ndr=50,  FoV=0.2x0.2, mean = 4.20267, std = 8.26528
    # 401, dr=0.001, ndr=200, FoV=0.4x0.4, mean = 1.63391, std = 4.67807
    # 100, dr=0.004, ndr=50,  FoV=0.4x0.4, mean = 1.65235, std = 4.12344

    imgSize = 201

    # resolution = 0.002
    if imgSize == 301:
        meanPixelVal = 0.86591   
        stdPixelVal  = 3.01413
        batch_size = 220

    if imgSize == 201:
        meanPixelVal = 1.64004   
        stdPixelVal  = 4.40112
        batch_size   = 220

    if imgSize == 101:
        meanPixelVal = 4.20267   
        stdPixelVal  = 8.26528
        batch_size   = 220

    if imgSize == 401:
        meanPixelVal = 1.63391   
        stdPixelVal  = 4.67807
        batch_size = 220

    if imgSize == 100:
        meanPixelVal = 1.65235   
        stdPixelVal  = 4.12344
        batch_size = 220

    train_img_path      = f"ImageCW_Train_{imgSize}"
    train_DataListFile  = f"TrainDataCW_MCML_{imgSize}.csv"
    # val_img_path        = f"ImageCW_Val_{imgSize}"
    # val_DataListFile    = f"ValDataCW_MCML_{imgSize}.csv"

    checkpoint_path = f'training_results_MCML_{imgSize}'

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    logger = double_logger(log_path=checkpoint_path).getLogger()

    preprocessing_transformer = transforms.Normalize(meanPixelVal, stdPixelVal)
    inverse_preprocessing_transformer = transforms.Normalize(-meanPixelVal, 1.0/stdPixelVal)

    train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                            transforms.RandomVerticalFlip(0.5)
    ])

    train_pickle_file_name  = 'train.pkl'
    val_pickle_file_name    = 'val.pkl'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} device")

    theta = np.arange(0, np.pi, 0.001)
    theta = torch.from_numpy(theta).to(device)

    labels = pd.read_csv(os.path.join(train_img_path, train_DataListFile))
    gV = pd.unique(labels['g'])

    Trn = trainer.Trainer()
    for fold, g in enumerate(gV):
        train_labels = labels.iloc[(labels['g']!=g).tolist()]
        val_labels   = labels.iloc[(labels['g']==g).tolist()]
        logger.info(f'Fold: {fold}, val: g={g}')

        print('Preprocessing...')
        DataPreprocessor().dump(train_labels, train_img_path, checkpoint_path, train_pickle_file_name, preprocessing_transformer)
        DataPreprocessor().dump(val_labels,   train_img_path, checkpoint_path, val_pickle_file_name, preprocessing_transformer)
        print('Preprocessing finished')

        train_data = CustomImageDataset_Pickle(
            img_labels = train_labels,
            file_preprocessed = os.path.join(checkpoint_path, train_pickle_file_name),
            transform = train_transformer
        )
        val_data = CustomImageDataset_Pickle(
            img_labels = val_labels,
            file_preprocessed = os.path.join(checkpoint_path, val_pickle_file_name)
        )

        # Define data loaders for training and validating data in this fold
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
        val_dataloader   = DataLoader(val_data, batch_size=batch_size, pin_memory=True, num_workers=8)

        df_loss_best = pd.DataFrame(columns=['Fold', 'NoG', 'Events', 'Error'])
        for num_of_Gaussian in range(2, 13):
            # Define model
            model = Resnet18(num_classes=num_of_Gaussian*3)
            model_struct = summary(model, (1, imgSize-2, imgSize-2), verbose=0)
            model_struct_str = str(model_struct)
            logger.info('Model structure:\n {}'.format(model_struct_str))

            optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=5e-3)
            # optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

            # bestmodel_name      = f'best_model_NoG_{num_of_Gaussian}' 
            result_file_name    = f'train_loss_fold_{fold}_NoG_{num_of_Gaussian}'
            logger.info(f'Fold: {fold}, NoG: {num_of_Gaussian}, Start training')
            val_loss_min, train_loss, df_loss = Trn.train_and_val(train_dataloader, val_dataloader, model, loss_func_mse, 
                                                        optimizer, scheduler, num_epochs=30)

            train_result = {'Fold':fold, 'NoG':num_of_Gaussian, 'Events':'Train', 'Error':train_loss}
            val_result   = {'Fold':fold, 'NoG':num_of_Gaussian, 'Events':'Validation', 'Error':val_loss_min}
            df_loss_best = df_loss_best.append(train_result, ignore_index=True)
            df_loss_best = df_loss_best.append(val_result,   ignore_index=True)

            df_loss.to_csv(os.path.join(checkpoint_path, f'{result_file_name}.csv'), index=False)

            fig, ax = plt.subplots(figsize=(6,4), dpi=100)
            ax = sns.lineplot(x="Epoch", y="Error", hue='Events', data=df_loss)
            ax.legend(title='', loc='upper right')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')

            figFile = os.path.join(checkpoint_path, f'{result_file_name}.png')
            plt.savefig(figFile, bbox_inches='tight')

        #---end of for num_of_Gaussian
        print(df_loss_best)
        df_loss_best.to_csv(os.path.join(checkpoint_path, f'Train_Val_Results_fold_{fold}.csv'), index=False)

    #---end of training
    print('Done')