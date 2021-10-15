import logging
import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_image as isns

class Tester:
    """
    This class tests a trained model
    """
    @property
    def logger(self):
        return logging.getLogger(__name__)

    def test(self, dataset, model, loss_fn, inverse_transform, figure_path, save_fig, index, device):
        if save_fig:
            if not os.path.exists(figure_path):
                os.mkdir(figure_path)

        model.eval()

        df = dataset.img_labels
        with torch.no_grad():
            features = []
            if index is None:
                idx = np.arange(len(dataset))
            else:
                idx = index

            for temp, i in enumerate(idx):
                X, gt = dataset[i]
                img = inverse_transform(X)
                img = img.squeeze().numpy()

                X = X.reshape(1,*X.shape)
                gt = gt.reshape(1,-1)
                X, y = X.to(device), gt.to(device)
                pred, feature = model(X)
                loss, pHG, pGMM, estimated_g, delta_g = loss_fn(pred, y)
                pHG, pGMM = pHG.to('cpu'), pGMM.to('cpu')
                pHG, pGMM = pHG.squeeze(), pGMM.squeeze()
                pHG, pGMM = pHG.numpy(), pGMM.numpy()

                feature = feature.to('cpu').squeeze()
                feature = feature.numpy()
                features.append(feature)

                gt = gt.numpy()
                #df['Error'].iloc[i] = loss.item()
                df.loc[i, 'Error'] = loss.item()
                df.loc[i, 'estimated_g'] = estimated_g.item()
                df.loc[i, 'delta_g'] = delta_g.item()

                filename = df['Image'].iloc[i]

                if save_fig:
                    if (filename.find('0001') != -1) or (index is not None):
                        print(f'Saving {filename},  Error: {loss.item()} ')

                        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
                        ua = df['ua'].iloc[i]
                        us = df['us'].iloc[i]
                        g  = df['g'].iloc[i]
                        plt.axis("off")
                        figtitle = 'ua=%.3f, us=%.3f, g=%.2f, MSE=%.6f\n' %(ua, us, g, loss.item())
                        ax.set_title(figtitle)

                        ax1 = fig.add_subplot(1, 2, 1)
                        plt.axis("off")
                        # img = np.log10(img + np.abs(np.min(img)) + 1e-10)
                        # plt.imshow(img, cmap='gist_heat')
                        img = np.power(img, 0.5)
                        isns.imshow(img, ax=ax1, cmap='gist_heat', vmin=0, dx=df['dr'].iloc[i], units='cm')

                        fig.add_subplot(1, 2, 2)
                        plt.axis("on")
                        theta = np.arange(0, np.pi, 0.001)
                        plt.plot(theta, pHG, label='HG')
                        plt.plot(theta, pGMM, label='GMM')
                        plt.legend()

                        plt.savefig(os.path.join(figure_path, filename[:-3]+'png'), bbox_inches='tight')

                        ax.set_title('')
                        plt.savefig(os.path.join(figure_path, filename[:-4]+'_notitle.png'), bbox_inches='tight')


                        # save individual parts
                        fig, ax = plt.subplots(figsize=(4,3), dpi=300)
                        isns.imshow(img, ax=ax, cmap='gist_heat', vmin=0, dx=df['dr'].iloc[i], units='cm')
                        plt.savefig(os.path.join(figure_path, filename[:-4]+'_image.png'), bbox_inches='tight')

                        fig, ax = plt.subplots(figsize=(4,3), dpi=300)
                        plt.axis("on")
                        plt.plot(theta, pHG, label='HG')
                        plt.plot(theta, pGMM, label='GMM')
                        plt.legend()
                        plt.savefig(os.path.join(figure_path, filename[:-4]+'_phase.png'), bbox_inches='tight')

                        plt.close('all')

            features = np.array(features)
                                    
        return df, features

    def run(self, dataset, network, loss_func, model_dir, model_name, inverse_transform, figure_path_name=None, index = None, device=None):
        if figure_path_name is None:
            save_fig = False
            figure_path = ''
        else:
            save_fig = True
            figure_path = os.path.join(model_dir, figure_path_name)

        # Determine device use GPU if available
        device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device {device}")

        self.logger.info(f"Testing {len(dataset)} images...")

        # load the model
        checkpoint = torch.load(os.path.join(model_dir, model_name))
        # restore state_dict from checkpoint to model
        network.load_state_dict(checkpoint['state_dict'])

        # copy network to device [cpu /gpu] if available
        network.to(device=device)

        loss_df, features = self.test(dataset, network, loss_func, inverse_transform, figure_path, save_fig, index, device)

        return loss_df, features
