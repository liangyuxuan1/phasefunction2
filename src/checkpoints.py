import torch

# The training process is conducted over several iterations (epochs). 
# During each epoch, the model learns parameters to make better predictions. 
# We print the model’s accuracy and loss at each epoch; we’d like to see the accuracy increase and the loss decrease with every epoch.

# Show the loss curves of training and testing
# https://blog.csdn.net/weixin_42204220/article/details/86352565   does not work properly
# pip install tensorboardX
# from tensorboardX import SummaryWriter

# https://zhuanlan.zhihu.com/p/103630393 , this works
# 不要安装pytorch profiler, 如果安装了，pip uninstall torch-tb-profiler. 否则tensorboard load 数据有问题

# How To Save and Load Model In PyTorch With A Complete Example
# https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee


# Saving function
def save_ckp(state, checkpoint_path):
    """
    state: checkpoint we want to save
    checkpoint_path: path to save checkpoint
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)

# Loading function
def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # load loss from checkpoint
    train_loss  = checkpoint['train_loss']
    val_loss    = checkpoint['val_loss']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], train_loss, val_loss
