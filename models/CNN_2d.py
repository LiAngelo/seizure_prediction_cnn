from torch import nn
from .BasicModule import BasicModule

class CNN_2d(BasicModule):

    def __init__(self, num_classes=2):
        super(CNN_2d, self).__init__()

        self.model_name = 'cnn_2d'

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(11, 1), stride=(4, 1), padding=(2, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1)),
            nn.Conv2d(16, 32, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1)),
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1)),


        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size(0), 64 * 6 * 6)
        x = self.classifier(x)
        return x






















