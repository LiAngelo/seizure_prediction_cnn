from torch import nn
from .BasicModule import BasicModule
import torch


class TextCNN(BasicModule):

    def __init__(self, num_classes=2):
        super(TextCNN, self).__init__()

        self.model_name = 'TextCNN'
        self.filter_sizes = (2, 3, 4)

        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=224,
                                    out_channels=100,
                                    kernel_size=h),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=6 - h + 1))
            for h in self.filter_sizes
        ])

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(100 * len(self.filter_sizes), 2)

    def forward(self, x):
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))
        out = self.dropout(out)
        out = self.fc(out)
        return out
