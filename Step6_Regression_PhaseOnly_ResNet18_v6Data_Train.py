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
# matplotlib.use("Agg")
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
        p[i,:] = 0.5*(1-g[i]*g[i])/((1+g[i]*g[i]-2*g[i]*torch.cos(theta))**(3.0/2.0) + 1e-6)
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
        sumGmm = torch.sum(gmm[i,:]) * 0.01     # discretization bin = 0.01 radian
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

# In a single training loop, the model makes predictions on the training dataset (fed to it in batches), 
# and backpropagates the prediction error to adjust the model’s parameters.
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    current = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current += len(X)
        if (batch+1) % 10 == 0:
            print(f"loss: {loss.item():>0.6f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    print(f"loss: {train_loss:>0.6f}  [{current:>5d}/{size:>5d}]")

    scheduler.step()

    return train_loss

# We also check the model’s performance against the test dataset to ensure it is learning.
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, np.zeros(3)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            
            # pred_uas = pred[:, -2:]
            # gt_uas = y[:, :2]
            # pred_error = (pred_uas - gt_uas).abs()/gt_uas
            # small_error_num = (pred_error <= 0.1).prod(1).sum().item()
            # large_error_num = (pred_error >= 0.5).prod(1).sum().item()
            # medium_error_num = len(pred_error) - small_error_num - large_error_num
            # correct += [small_error_num, medium_error_num, large_error_num]
    test_loss /= num_batches
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct[0]):>0.1f}%, {(100*correct[1]):>0.1f}%, {(100*correct[2]):>0.1f}%, Avg loss: {test_loss:>10f} \n")
    print(f"Validation Avg loss: {test_loss:>0.6f}")

    return test_loss, correct

# show test results and add the figure in writter
# https://tensorflow.google.cn/tensorboard/image_summaries?hl=zh-cn

def show_result_samples(dataset, showFig=False):
    cols, rows = 4, 2
    numEachUaGroup = len(dataset)/(cols*rows)
    sample_idx = np.zeros(cols*rows, dtype=np.int32)
    for i in range (cols*rows):
       sample_idx[i] = np.random.randint(numEachUaGroup) + i*numEachUaGroup
    
    model.eval()

    figure = plt.figure(figsize=(18, 9))
    label = dataset.img_labels
    for i in range(cols * rows):
        idx = sample_idx[i]
        x, gt = dataset[idx]
        x = x.reshape(1,*x.shape)
        gt = gt.reshape(1,-1)
        x, gt = x.to(device), gt.to(device)

        pred = model(x)
        loss = loss_fn(pred, gt)

        pred = pred.detach()

        gmm = GMM(pred[:, 0:num_of_Gaussian*3], theta)
        # gt = gtNorm.restore(gt.to("cpu"))
        g = label.iloc[idx, 3]
        g = torch.tensor([g])
        g = g.to(device)
        p_theta = HG_theta(g, theta)

        ua = label.iloc[idx, 1]
        us = label.iloc[idx, 2]

        figure.add_subplot(rows, cols, i+1)
        figtitle = 'ua=%.3f, us=%.2f, g=%.2f \n loss=%.4f' %(ua, us, g, loss.item())
        plt.title(figtitle)
        plt.axis("on")
        gmm, p_theta = gmm.to("cpu"), p_theta.to("cpu")
        gmm = gmm.numpy()
        p_theta = p_theta.numpy()
        px = theta.to("cpu")
        px = px.numpy()
        plt.plot(px, gmm.squeeze())
        plt.plot(px, p_theta.squeeze())

    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    if showFig:
        plt.show()
    return figure

def show_Results(dataset, figure_path, save_figure=False):
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)

    model.eval()

    results = np.zeros((len(dataset), 9))   # ['ua', 'pred_ua', 'ua_re', 'us', 'pred_us', 'us_re', 'g', 'phase_mse', 'numPixels']
    label = dataset.img_labels
    for i in range(len(dataset)):
        x, _ = dataset[i]
        img = x*stdPixelVal + meanPixelVal
        x = x.reshape(1,*x.shape)
        x = x.to(device)
        pred = model(x)
        # loss = loss_fn(pred, gt)

        gmm = GMM(pred[:, 0:num_of_Gaussian*3], theta)
        g = label.iloc[i, 3]
        g = torch.tensor([g])
        g = g.to(device)
        p_theta = HG_theta(g, theta)
        phase_mse = nn.MSELoss()(gmm, p_theta)
        phase_mse = phase_mse.detach()
        phase_mse = phase_mse.to("cpu")

        pred = pred.detach()
        pred = pred.squeeze()
        pred = pred.to("cpu")

        ua, us = label.iloc[i, 1], label.iloc[i, 2]
        # tmp = torch.tensor([pred[-2], pred[-1], np.random.rand()])
        # pred_rst = gtNorm.restore(tmp)      # change the network output to parameter range
        # pred_ua, pred_us = pred_rst[0], pred_rst[1]
        # ua_re, us_re = np.abs(ua-pred_ua)/ua, np.abs(us-pred_us)/us
        pred_ua, pred_us = 0, 0
        ua_re, us_re = 0, 0

        results[i,:] = [ua, pred_ua, ua_re, us, pred_us, us_re, label.iloc[i, 3], phase_mse, label.iloc[i, 4]]

        if save_figure and (i % 10 == 0):
            fig = plt.figure(figsize=(8, 4))
            plt.axis("off")
            figtitle = 'ua=%.3f, us=%.2f, g=%.2f, Phase MSE=%.4f \n' %(ua, us, g, phase_mse)
            plt.title(figtitle)

            fig.add_subplot(1, 2, 1)
            plt.axis("off")
            img = np.float_power(img.squeeze(), 0.1)
            plt.imshow((img), cmap="hot")

            fig.add_subplot(1, 2, 2)
            plt.axis("on")
            gmm, p_theta = gmm.detach(),  p_theta.detach()
            gmm, p_theta = gmm.to("cpu"), p_theta.to("cpu")
            gmm, p_theta = gmm.numpy(),   p_theta.numpy()
            px = theta.to("cpu")
            px = px.numpy()
            plt.plot(px, gmm.squeeze(), label='est')
            plt.plot(px, p_theta.squeeze(), label='gt')
            plt.legend()
            
            figFileName = 'Fig_%04d.png' % (i/10 + 1)
            figFile = os.path.join(figure_path, figFileName)
            plt.savefig(figFile, bbox_inches='tight')
            plt.close('all')
       
    return results

def write_results_exel(results, filename):
    data_df = pd.DataFrame(results)
    data_df.columns = ['ua', 'pred_ua', 'ua_RE', 'us', 'pred_us', 'us_RE', 'g', 'phase_MSE', 'numPixels']
    # need to install openpyxl: pip install openpyxl
    # and import openpyxl
    tb_writer = pd.ExcelWriter(filename)
    data_df.to_excel(tb_writer, 'page_1', float_format='%.4f')
    tb_writer.save()
    tb_writer.close()

def write_results_txt(results, filename):
    fid = open(filename, 'w')
    for i in range(np.size(results, 0)):
        for j in range(np.size(results, 1)):
            fid.write('%04f\t' % results[i,j])
        fid.write('%04f\n' % np.mean(results[i,:]))
    fid.write('\n%04f\n\n' % np.mean(results))

    model_struct = summary(model, (1, 500, 500), verbose=0)
    model_struct_str = str(model_struct)
    fid.write(model_struct_str)

    fid.close()

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

    meanPixelVal = 0.0022   # using statistics of all v6 data
    stdPixelVal  = 0.2915

    img_path = "ImageCW_v6"
    DataListFile = "TrainDataCW_v6_Results.csv"
    tmp_processed_data_dir = "temp_processed_data"
    checkpoint_path = 'training_results'

    if not os.path.exists(tmp_processed_data_dir):
        os.mkdir(tmp_processed_data_dir)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    logger = double_logger(log_path=checkpoint_path).getLogger()

    preprocessing_transformer = transforms.Normalize(meanPixelVal, stdPixelVal)
    inverse_preprocessing_transformer = transforms.Normalize(-meanPixelVal, 1.0/stdPixelVal)

    train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                            transforms.RandomVerticalFlip(0.5)
    ])


    labels = pd.read_csv(os.path.join(img_path, DataListFile))
    gV = pd.unique(labels['g'])

    total_pickle_file_name = 'total_train.pkl'
    test_pickle_file_name  = 'test.pkl'
    print('Preprocessing...')
    # DataPreprocessor().dump(labels, img_path, tmp_processed_data_dir, total_pickle_file_name, preprocessing_transformer)
    print('Preprocessing finished')

    total_data = CustomImageDataset_Pickle(
        img_labels = labels,
        file_preprocessed = os.path.join(tmp_processed_data_dir, total_pickle_file_name),
        transform = train_transformer
    )

    train_data, val_data = torch.utils.data.random_split(dataset=total_data, 
                                                        lengths=[int(0.8*len(total_data)), int(0.2*len(total_data))])

    # Create data loaders.
    batch_size = 160
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    val_dataloader   = DataLoader(val_data, batch_size=batch_size, pin_memory=True, num_workers=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} device")

    theta = np.arange(0, np.pi, 0.01)
    theta = torch.from_numpy(theta).to(device)
    num_of_Gaussian = 5

    # Define model
    model = Resnet18(num_classes=num_of_Gaussian*3)
    # model_struct = summary(model, (1, 500, 500), verbose=0)
    # model_struct_str = str(model_struct)
    # logger.info('Model structure:\n {}'.format(model_struct_str))

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    Trn = trainer.Trainer()
    bestmodel_name = f'best_model_NoG_{num_of_Gaussian}' 
    logger.info(f'NoG: {num_of_Gaussian}, Training {bestmodel_name}')
    val_loss_min, train_loss, df_loss = Trn.run(train_dataloader, val_dataloader, model, loss_func_mse, 
                                                optimizer, scheduler, num_epochs=30,
                                                model_dir=checkpoint_path, model_name=bestmodel_name)
    df_loss.to_csv(os.path.join(checkpoint_path, f'train_loss.csv'), index=False)

    #---end of for cross-validation
    print('Done')