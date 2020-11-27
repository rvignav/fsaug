import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy import ndimage
import math
import numbers
import random

class Net(nn.Module):
    '''
    Feature space augmentation network architecture.
    '''
    def __init__(self, conv1_out=64, kernel_size1=5, kernel_size2=1, conv2_out=128):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=kernel_size1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=kernel_size2)
        self.fc1 = nn.Linear(4608, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def loss_function(self, out, target):
        return F.cross_entropy(out, target)