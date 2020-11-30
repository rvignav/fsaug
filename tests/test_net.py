from fsaug.net import Net
from torchsummary import summary

network = Net()
summary(network, (1, 28, 28))