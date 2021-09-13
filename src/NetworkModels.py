from torch import nn
import torchvision.models as models

class Resnet18(nn.Module):
    """
    The resnet 18 model, modified to accomodate one channel input and n classes output
    """
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x: The input tensor with 3 dimensions
        :return:
        """

        return self.model(x)