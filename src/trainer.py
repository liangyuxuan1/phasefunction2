import logging
import os
import time
import torch
import numpy as np
import pandas as pd
import checkpoints

class Trainer:
    """
    This class runs the training epochs
    """
    @property
    def logger(self):
        return logging.getLogger(__name__)

    def train(self, dataloader, model, loss_fn, optimizer, device):
        model.train()

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        train_loss = 0
        current = 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred, _ = model(X)
            loss = loss_fn(pred, y)
            train_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current += len(X)
            if (batch+1) % 10 == 0:
                self.logger.info(f"Batch:{(batch+1):>3d}, loss: {loss.item():>0.6f}, [{current:>5d}/{size:>5d}]")

        train_loss /= num_batches
        self.logger.info(f"Train loss: {train_loss:>0.6f}")

        return train_loss

    def validation(self, dataloader, model, loss_fn, device):
        model.eval()

        num_batches = len(dataloader)
        val_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred, _ = model(X)
                loss = loss_fn(pred, y)
                val_loss += loss.item()
                
        val_loss /= num_batches
        self.logger.info(f"Validation loss: {val_loss:>0.6f}")

        return val_loss

    def train_and_val(self, train_dataloader, val_dataloader, network, loss_func, optimizer, scheduler, num_epochs, model_dir=None, model_name=None, device=None):
        # Determine device use GPU if available
        device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.logger.info(f"Using device {device}")

        self.logger.info(f"Using dataloader with {len(train_dataloader)} batches, the batch size is {train_dataloader.batch_size}")

        # copy network to device [cpu /gpu] if available
        network.to(device=device)

        val_loss_min = torch.tensor(np.Inf)
        train_loss_with_val_loss_min = torch.tensor(np.Inf)

        tic = time.time()
        df_loss = pd.DataFrame(columns=['Epoch', 'Events', 'Error'])
        for epoch in range(num_epochs):
            self.logger.info(f'Epoch {epoch+1} ...')

            train_loss = self.train(train_dataloader, network, loss_func, optimizer, device)
            val_loss = self.validation(val_dataloader, network, loss_func, device)
            scheduler.step()

            train_result = {'Epoch':epoch+1, 'Events':'Train', 'Error':train_loss}
            val_result   = {'Epoch':epoch+1, 'Events':'Validation', 'Error':val_loss}
            df_loss = df_loss.append(train_result,  ignore_index=True)
            df_loss = df_loss.append(val_result,    ignore_index=True)

            if val_loss < val_loss_min:
                self.logger.info('Validation loss decreased ({:.6f} --> {:.6f})'.format(val_loss_min, val_loss))
                # save checkpoint as best model
                val_loss_min = val_loss
                train_loss_with_val_loss_min = train_loss

                if model_dir is not None:
                    self.logger.info('Saving model ...')
                    best_model_file = os.path.join(model_dir, model_name+'.pt')

                    # create checkpoint variable and add important data
                    checkpoint = {
                        'epoch'     : epoch + 1,
                        'state_dict': network.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'train_loss' : train_loss,
                        'val_loss'  : val_loss
                    }

                    # save checkpoint
                    checkpoints.save_ckp(checkpoint, best_model_file)

            timecost = time.time() - tic
            self.logger.info('Epoch [%d/%d] complete, time elapsed: %.0fh %.0fm %.0fs' % 
                            (epoch + 1, num_epochs, timecost//3600, (timecost%3600)//60, timecost%60))

        timecost = time.time() - tic
        self.logger.info('Train %d epochs complete, time cost: %.0fh %.0fm %.0fs\n' % 
                        (num_epochs, timecost//3600, (timecost%3600)//60, timecost%60))

        return val_loss_min, train_loss_with_val_loss_min, df_loss

    def train_only(self, train_dataloader, network, loss_func, optimizer, scheduler, num_epochs, model_dir=None, model_name=None, device=None):
        # Determine device use GPU if available
        device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.logger.info(f"Using device {device}")

        self.logger.info(f"Using dataloader with {len(train_dataloader)} batches, the batch size is {train_dataloader.batch_size}")

        # copy network to device [cpu /gpu] if available
        network.to(device=device)

        train_loss_min = torch.tensor(np.Inf)

        tic = time.time()
        df_loss = pd.DataFrame(columns=['Epoch', 'Events', 'Error'])
        for epoch in range(num_epochs):
            self.logger.info(f'Epoch {epoch+1} ...')

            train_loss = self.train(train_dataloader, network, loss_func, optimizer, device)
            scheduler.step()

            train_result = {'Epoch':epoch+1, 'Events':'Train', 'Error':train_loss}
            df_loss = df_loss.append(train_result,  ignore_index=True)

            if train_loss < train_loss_min:
                self.logger.info('Train loss decreased ({:.6f} --> {:.6f})'.format(train_loss_min, train_loss))
                # save checkpoint as best model
                train_loss_min = train_loss

                if model_dir is not None:
                    self.logger.info('Saving model ...')
                    best_model_file = os.path.join(model_dir, model_name+'.pt')

                    # create checkpoint variable and add important data
                    checkpoint = {
                        'epoch'     : epoch + 1,
                        'state_dict': network.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'train_loss' : train_loss,
                    }

                    # save checkpoint
                    checkpoints.save_ckp(checkpoint, best_model_file)

            timecost = time.time() - tic
            self.logger.info('Epoch [%d/%d] complete, time elapsed: %.0fh %.0fm %.0fs' % 
                            (epoch + 1, num_epochs, timecost//3600, (timecost%3600)//60, timecost%60))

        timecost = time.time() - tic
        self.logger.info('Train %d epochs complete, time cost: %.0fh %.0fm %.0fs\n' % 
                        (num_epochs, timecost//3600, (timecost%3600)//60, timecost%60))

        return train_loss_min, df_loss
