# We have to implement somethin in this file?

import torch
from torchvision import models


# **resnet18**. This class acts as a wrapper around the pre-trained ResNet-18 model available in `torchvision.models`.
# It exposes the individual layers and blocks of the ResNet-18 architecture as class attributes, allowing for more
#  flexible use of its features.
class resnet18(torch.nn.Module):
    # **resnet18.__init__**. This is the constructor method. It loads a pre-trained ResNet-18 model from `torchvision` 
    # (if `pretrained=True`) and assigns its initial convolutional layer, batch normalization, ReLU, max pooling, and 
    # the four main sequential blocks (`layer1` through `layer4`) as attributes of the class instance.
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet18(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    # **resnet18.forward**. This method defines the forward pass through this ResNet-18 wrapper. It processes the input 
    # sequentially through the initial layers and blocks (from `conv1` up to `layer4`). It extracts and returns the 
    # feature maps after `layer3` (`feature3`), after `layer4` (`feature4`), and a global average pooled feature 
    # (`tail`) derived from `feature4`.
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


# **resnet101**. This class acts as a wrapper around the pre-trained ResNet-101 model from `torchvision.models`.
#  Similar to the `resnet18` class, it provides access to the individual layers and blocks of the ResNet-101 
# architecture.
class resnet101(torch.nn.Module):
    # **resnet101.__init__**. This is the constructor method. It loads a pre-trained ResNet-101 model from
    #  `torchvision` (if `pretrained=True`) and assigns its initial convolutional layer, batch normalization,
    #  ReLU, max pooling, and the four main sequential blocks (`layer1` through `layer4`) as attributes of
    # the class instance.
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet101(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    # **resnet101.forward**. This method defines the forward pass through this ResNet-101 wrapper. It processes
    # the input sequentially through the initial layers and blocks (from `conv1` up to `layer4`). It extracts
    # and returns the feature maps after `layer3` (`feature3`), after `layer4` (`feature4`), and a global
    # average pooled feature (`tail`) derived from `feature4`.
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail

# **build_contextpath**. This function serves as a factory. Based on the input string `name`, it creates and 
# returns an instance of either the `resnet18` or `resnet101` wrapper class, loaded with pre-trained weights.
def build_contextpath(name):
    model = {
        'resnet18': resnet18(pretrained=True),
        'resnet101': resnet101(pretrained=True)
    }
    return model[name]
