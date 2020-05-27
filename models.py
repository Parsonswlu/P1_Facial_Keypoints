## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
#         self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # 1 input image channel (grayscale), 32 output channels/feature maps, 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 + 1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)        
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2) 
        
        # second conv layer: 32 inputs, 32 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output tensor will have dimensions: (32, 106, 106)
        # after another pool layer this becomes (32, 53, 53)
        self.conv2 = nn.Conv2d(32, 32, 5)
        
        # third conv layer: 32 inputs, 32 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (53-5)/1 +1 = 49
        # the output tensor will have dimensions: (32, 49, 49)
        # after another pool layer this becomes (32, 24, 24); round down 24.5
        self.conv3 = nn.Conv2d(32, 32, 5)
        
        # fourth conv layer: 32 inputs, 32 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (25-5)/1 +1 = 21
        # the output tensor will have dimensions: (32, 21, 21)
        # after another pool layer this becomes (32, 10, 10); round down 10.5
        self.conv4 = nn.Conv2d(32, 32, 5)
        
        # 32 outputs * the 10*10 filtered/pooled map size
        self.fc1 = nn.Linear(32*10*10, 3200)
        
        # dropout with p=0.2
        self.fc1_drop = nn.Dropout(p=0.2)
        
        # finally, create 136 output channels (for the 136 classes)
        self.fc2 = nn.Linear(3200, 136)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # four conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # prep for linear layer
        x = x.view(x.size(0), -1)

        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
