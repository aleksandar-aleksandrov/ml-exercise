import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models.alexnet

class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.batch1 = nn.BatchNorm2d(16)
        #self.dropout1 = nn.Dropout()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch2 = nn.BatchNorm2d(32)
        #self.dropout2 = nn.Dropout()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU(inplace=True)
        self.batch3 = nn.BatchNorm2d(64)
        #self.dropout3 = nn.Dropout()

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=0)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch4 = nn.BatchNorm2d(96)

        self.conv5 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch5 = nn.BatchNorm2d(128)
        #self.dropout4 = nn.Dropout()

        self.dense1 = nn.Linear(10368, 500)
        self.relu5 = nn.ReLU(inplace=True)
        #self.batch4 = nn.Dropout()#nn.BatchNorm1d(1200)
        #self.dropout5 = nn.Dropout()

        #self.dense2 = nn.Linear(1200, 600)
        #self.relu6 = nn.ReLU(inplace=True)
        #self.batch5 = nn.BatchNorm1d(600)
        #self.dropout6 = nn.Dropout()

        #self.dense3 = nn.Linear(600, 300)
        #self.relu7 = nn.ReLU(inplace=True)
        #self.batch6 = nn.BatchNorm1d(300)
        #self.dropout7 = nn.Dropout()

        self.dense4 = nn.Linear(500, 30)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################
        x = self.batch1(self.relu1(self.conv1(x)))
        #print(x.shape)
        x = self.pool2(self.batch2(self.relu2(self.conv2(x))))
        #print(x.shape)
        x = self.batch3(self.relu3(self.conv3(x)))
        #print(x.shape)
        x = self.pool4(self.batch4(self.relu4(self.conv4(x))))
        #print(x.shape)
        x = self.pool5(self.batch5(self.relu5(self.conv5(x))))

        x = x.view(-1, 10368)

        x = self.relu5(self.dense1(x))
        #x = self.relu6(self.dense2(x))
        #x = self.relu7(self.dense3(x))
        x = self.dense4(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


from torch.nn import Conv2d, ReLU, BatchNorm2d, MaxPool2d


"""
            Conv2d(64, 64, 2),
            ReLU(),
            BatchNorm2d(64),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(64, 96, 2),
            ReLU(),
            BatchNorm2d(96),

            Conv2d(96, 96, 2),
            ReLU(),
            BatchNorm2d(96),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(96, 128, 2),
            ReLU(),
            BatchNorm2d(128),
            """


class KeypointModelSeq(nn.Module):

    def __init__(self):
        super(KeypointModelSeq, self).__init__()
        self.features = nn.Sequential(
            Conv2d(1, 16, 3),
            ReLU(),
            BatchNorm2d(16),

            Conv2d(16, 16, 3),
            ReLU(),
            BatchNorm2d(16),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(16, 24, 2),
            ReLU(),
            BatchNorm2d(24),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.clf = nn.Sequential(
            nn.Linear(11616, 1000),
            nn.ReLU(),
            nn.Dropout(.1),

            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout(.1),

            nn.Linear(100, 30)
        )

    def forward(self, x):
        x = self.features(x)

        x = x.view(-1, 11616)

        return self.clf(x)

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
