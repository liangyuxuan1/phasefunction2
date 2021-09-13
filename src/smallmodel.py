from torch import nn

# Creating Models
# To define a neural network in PyTorch, we create a class that inherits from nn.Module. 
# We define the layers of the network in the __init__ function and specify how data will pass through the network in the forward function. 
# To accelerate operations in the neural network, we move it to the GPU if available.
class NeuralNetwork(nn.Module):
    def __init__(self, num_output):
        # super(NeuralNetwork, self).__init__()
        self.num_output = num_output
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2, stride=2)
        )

        self.convPhase = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1,1))
        )

        self.fc = nn.Sequential(
            nn.Linear(256, num_output),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.convPhase(x)
        x = x.view(x.size(0), -1)
        pred = self.fc(x)

        return pred
