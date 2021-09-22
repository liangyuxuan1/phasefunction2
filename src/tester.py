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

    def test(self, dataset, model, loss_fn, inverse_transform, figure_path, save_fig, device):
        if save_fig:
            if not os.path.exists(figure_path):
                os.mkdir(figure_path)

        model.eval()

        df = dataset.img_labels
        with torch.no_grad():
            for i in range(len(dataset)):
                X, gt = dataset[i]
                img = inverse_transform(X)
                img = img.squeeze().numpy()

                X = X.reshape(1,*X.shape)
                gt = gt.reshape(1,-1)
                X, y = X.to(device), gt.to(device)
                pred = model(X)
                loss, pHG, pGMM = loss_fn(pred, y)
                pHG, pGMM = pHG.to('cpu'), pGMM.to('cpu')
                pHG, pGMM = pHG.squeeze(), pGMM.squeeze()
                pHG, pGMM = pHG.numpy(), pGMM.numpy()

                gt = gt.numpy()
                df['Error'].iloc[i] = loss.item()

                filename = df['Image'].iloc[i]

                if save_fig:
                    if filename.find('0001') != -1:
                        print(f'Saving {filename},  Error: {loss.item()} ')

                        fig = plt.figure(figsize=(10, 4))
                        ua = df['ua'].iloc[i]
                        us = df['us'].iloc[i]
                        g  = df['g'].iloc[i]
                        plt.axis("off")
                        figtitle = 'ua=%.3f, us=%.2f, g=%.2f, Phase MSE=%.4f \n' %(ua, us, g, loss.item())
                        plt.title(figtitle)

                        ax1 = fig.add_subplot(1, 2, 1)
                        plt.axis("off")
                        img = np.log10(img + np.abs(np.min(img)) + 1e-10)
                        # plt.imshow(img, cmap='gist_heat')
                        isns.imshow(img, ax=ax1, cmap='gist_heat', vmin=-10, vmax=2, dx=df['dr'].iloc[i], units='cm')

                        fig.add_subplot(1, 2, 2)
                        plt.axis("on")
                        theta = np.arange(0, np.pi, 0.001)
                        plt.plot(theta, pHG, label='HG')
                        plt.plot(theta, pGMM, label='GMM')
                        plt.legend()

                        # plt.savefig(os.path.join(figure_path, filename[:-4]+'_phase.png'), bbox_inches='tight')
                        plt.savefig(os.path.join(figure_path, filename[:-3]+'png'), bbox_inches='tight')

                        plt.close('all')
                        
        return df

    def run(self, dataset, network, loss_func, model_dir, model_name, inverse_transform, figure_path_name=None, device=None):
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

        loss_df = self.test(dataset, network, loss_func, inverse_transform, figure_path, save_fig, device)
        return loss_df
