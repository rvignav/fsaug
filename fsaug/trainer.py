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
from torchsummary import summary

class Trainer():
    '''
    Train a Net object with custom config and normalization options.
    '''
    def __init__(self, batch_size_train=64, batch_size_test=1000, learning_rate=1e-4, log_interval=50):
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.learning_rate = learning_rate
        self.log_interval = log_interval
        self.train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('/files/', train=True, download=True,
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.5,), (0.5,))
                                        ])),
            batch_size=batch_size_train, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('/files/', train=False, download=True,
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.5,), (0.5,))
                                        ])),
            batch_size=batch_size_test, shuffle=True)

        self.test_data = torchvision.datasets.MNIST('/files/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.5,), (0.5,))
                                    ]))

        self.inp, _ = test_data[0]
        self.inp = inp.unsqueeze(0)
        self.inp = inp.cpu().detach().numpy()[0][0]

    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = network(data)
            loss = network.loss_function(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(self.train_loader.dataset),
                100. * batch_idx / len(self.train_loader), loss.item()))
    
    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
            data = data.cuda()
            target = target.cuda()
            target = target.view(batch_size_test)
            output = network(data)
            test_loss += network.loss_function(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

    def run(n_epochs=20):
        test()
        for epoch in range(1, n_epochs + 1):
            train(epoch)
            test()

    def log_details():
        summary(network, (1, 28, 28))
    
    def eval():
        test()