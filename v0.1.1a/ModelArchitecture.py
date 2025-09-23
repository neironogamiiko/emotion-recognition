from torch import nn

class EmotionCNN(nn.Module):
    def __init__(self, num_classes:int):
        super().__init__()
        self.num_classes = num_classes
        self.cnn_layers = nn.Sequential(
            #    BLOCK 1    #
            nn.Conv2d(in_channels=3,
                      out_channels=10,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            #    BLOCK 2    #
            nn.Conv2d(in_channels=10,
                      out_channels=10,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(.2),
            #    BLOCK 3    #
            nn.Conv2d(in_channels=10,
                      out_channels=10,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(.2),
            #    BLOCK 4    #
            nn.Conv2d(in_channels=10,
                      out_channels=10,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classification_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=10*7*7,
                      out_features=self.num_classes)
        )
    def forward(self, x):
        return self.classification_layers(self.cnn_layers(x))