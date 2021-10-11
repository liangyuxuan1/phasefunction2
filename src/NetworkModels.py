from torch import nn
import torchvision.models as models

class Resnet18(nn.Module):
    """
    The resnet 18 model, modified to accomodate one channel input and n classes output
    """
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18()
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        feature = x.view(x.size(0),-1)

        pred = self.fc(x)
    
        return pred, feature