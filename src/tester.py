import logging
import os
import time
import torch
import numpy as np
import pandas as pd

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

        test_loss = pd.DataFrame(columns=['ua','us','g','Error'])
        with torch.no_grad():
            for i in range(len(dataset)):
                X, gt = dataset[i]
                img = inverse_transform(X)

                X = X.reshape(1,*X.shape)
                gt = gt.reshape(1,-1)
                X, y = X.to(device), gt.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)

                gt = gt.numpy()
                test_loss = test_loss.append({'ua':gt[0,0],'us':gt[0,1], 'g':gt[0,2], 'Error':loss.item()}, ignore_index=True)
                
        return test_loss

    def run(self, dataset, network, loss_func, model_dir, model_name, inverse_transform, figure_path_name=None, device=None):
        if figure_path_name is None:
            save_fig = False
            figure_path = ''
        else:
            save_Fig = True
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
