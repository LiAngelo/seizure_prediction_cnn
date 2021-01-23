from torch import nn
from .BasicModule import BasicModule

class CNN_1d(BasicModule):

    def __init__(self, num_classes=2):
        super(CNN_1d, self).__init__()

        self.model_name = 'cnn_1d'

        self.features = nn.Sequential(
            ## nn.Conv1d(1, 16, kernel_size=11, stride=4, padding=2),
            ## nn.ReLU(inplace=True),
            ## nn.MaxPool1d(kernel_size=3, stride=2),
            ## nn.Conv1d(16, 32, kernel_size=5, padding=2),
            ## nn.ReLU(inplace=True),
            ## nn.MaxPool1d(kernel_size=3, stride=2),
            ## nn.Conv1d(32, 64, kernel_size=3, padding=1),
            ## nn.ReLU(inplace=True),
            ## nn.MaxPool1d(kernel_size=3, stride=2),

            # nn.Conv1d(224, 64, 3),
            # nn.ReLU(),
            # nn.Dropout(),

            nn.Conv1d(6, 6, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(6, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(6, 6, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            ## nn.Dropout(),
            ## nn.Linear(64 * 6 * 6, 256),
            ## nn.ReLU(inplace=True),
            ## nn.Dropout(),
            ## nn.Linear(256, num_classes),
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        print(x.size(0))
        # x = x.view(x.size(0), 64 * 6 * 6)
        x = x.view(x.size(0), 128 * 6 * 6)
        x = self.classifier(x)
        return x






















