from torch import nn
from .BasicModule import BasicModule


class CNN_1d(BasicModule):

    def __init__(self, num_classes=2):
        super(CNN_1d, self).__init__()

        self.model_name = 'cnn_1d'

        self.features = nn.Sequential(

            nn.Conv1d(6, 16, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 6, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 6)
        x = self.classifier(x)
        return x
