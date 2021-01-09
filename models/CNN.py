from torch import nn
from .BasicModule import BasicModule

class CNN(BasicModule):

    def __init__(self):
        super(CNN, self).__init__()

        self.model_name = 'cnn'

