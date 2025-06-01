# We have to implement somethin in this file?

import torch
from torch import nn
from .build_contextpath import build_contextpath
import warnings
warnings.filterwarnings(action='ignore')

class DropConnectConv2d(nn.Conv2d):  # CHANGED HERE
    def __init__(self, *args, drop_prob=0.2, **kwargs):  # CHANGED HERE
        super().__init__(*args, **kwargs)
        self.drop_prob = drop_prob  # CHANGED HERE

    def forward(self, input):  # CHANGED HERE
        if self.training:
            mask = (torch.rand_like(self.weight) > self.drop_prob).float()
            weight = self.weight * mask
        else:
            weight = self.weight * (1 - self.drop_prob)
        return nn.functional.conv2d(input, weight, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)

# questo è ognuno dei blocchi dentro Spatial Path con le trasformazioni conv+bn+relu
# **ConvBlock**. This class implements a standard convolutional block consisting of a 2D convolution, Batch Normalization, 
# and ReLU activation. It is typically used for downsampling the feature map.
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
        #                        stride=stride, padding=padding, bias=False)
        self.conv1 = DropConnectConv2d(in_channels, out_channels,  # CHANGED HERE
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       bias=False,
                                       drop_prob=0.3)  # CHANGED HERE
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)  # CHANGED HERE

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn(x))
        x = self.dropout(x)  # CHANGED HERE
        return x


"""
class ConvBlock(torch.nn.Module):
    # **ConvBlock.__init__**. This is the constructor method. It initializes the convolutional layer (`nn.Conv2d`),
    # the batch normalization layer (`nn.BatchNorm2d`), and the ReLU activation layer (`nn.ReLU`).
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        #self.relu = nn.ReLU() # nn.leakyRelu? nn.leakyReLU(negative_slope=0.1, inplace=True)
        self.relu = nn.leakyReLU(negative_slope=0.1, inplace=True) 

    # **ConvBlock.forward**. This method defines the forward pass of the block. It applies the convolution, then 
    # the batch normalization, and finally the ReLU activation to the input.
    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))"""

#sussueguri di blocchi in spatial path
# **Spatial_path**. This module processes the input image to extract rich spatial details.
# It consists of a sequence of three `ConvBlock` instances.
class Spatial_path(torch.nn.Module):
    # **Spatial_path.__init__**. This is the constructor method. It initializes the three `ConvBlock` layers with 
    # specific input/output channels to create a spatial processing path.
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    # **Spatial_path.forward**. This method defines the forward pass. It passes the input sequentially through 
    # the three convolutional blocks defined in the constructor.
    def forward(self, input):
        x = self.convblock1(input) # Qui PyTorch chiama automaticamente ConvBlock.forward(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

# **AttentionRefinementModule**. This module refines features from a specific path (likely the context path) by
#  applying channel-wise attention. It uses global average pooling to get channel statistics, processes them, and 
# multiplies the resulting attention mask with the original feature map.
class AttentionRefinementModule(torch.nn.Module):
    # **AttentionRefinementModule.__init__**. This is the constructor method. It initializes the necessary layers for 
    # attention: a 1x1 convolution, batch normalization, sigmoid activation, and adaptive average pooling.
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    # **AttentionRefinementModule.forward**. This method defines the forward pass of the module. It computes global average
    # pooling on the input, applies convolution/BN/sigmoid to get attention weights, and finally multiplies these weights 
    # with the original input feature map.
    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.sigmoid(self.bn(x))
        # x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x

# **FeatureFusionModule**. This module combines features from the spatial path and the context path (or parts of it). 
# It concatenates them, processes them through a `ConvBlock` (without downsampling), and then applies a refinement step 
# similar to the Attention Refinement Module, adding the result back to the original fused feature.
class FeatureFusionModule(torch.nn.Module):
    # **FeatureFusionModule.__init__**. This is the constructor method. It initializes the layers needed for 
    # feature fusion, including a `ConvBlock` (with stride 1), 1x1 convolutions, ReLU, Sigmoid, and adaptive 
    # average pooling. `in_channels` is set to be the sum of the channels from the expected inputs.
    def __init__(self, num_classes, in_channels):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # resnet101 3328 = 256(from spatial path) + 1024(from context path) + 2048(from context path)
        # resnet18  1024 = 256(from spatial path) + 256(from context path) + 512(from context path)
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        #self.relu = nn.ReLU() # nn.leakyReLU(negative_slope=0.1, inplace=True)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True) # nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    # **FeatureFusionModule.forward**. This method defines the forward pass for fusion. It concatenates the two 
    # input feature maps along the channel dimension, passes them through the `ConvBlock`, applies global average 
    # pooling, processes through the 1x1 conv/relu/conv/sigmoid sequence to get attention, multiplies this with 
    # the `ConvBlock` output (`feature`), and adds this result back to the `feature`.
    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


# **BiSeNet**. This is the main class implementing the BiSeNet architecture for semantic segmentation. It combines 
# a Spatial Path (`Spatial_path`) to preserve spatial details and a Context Path (`build_contextpath` output) to 
# capture global semantic information. It uses Attention Refinement Modules (ARM) to refine context features and 
# a Feature Fusion Module (FFM) to combine features from both paths.
class BiSeNet(torch.nn.Module):
    # **BiSeNet.__init__**. This is the constructor method. It initializes the `Spatial_path`, the chosen 
    # `Context_path` (based on the `context_path` argument), the two `AttentionRefinementModule`s, two supervision 
    # layers (typically used during training), and the `FeatureFusionModule`. It also initializes the final 1x1
    #  convolution and calls the `init_weight` method to initialize weights. It populates a list `self.mul_lr` with 
    # modules for potential differentiated learning rate configuration.
    def __init__(self, num_classes, context_path):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path()

        # build context path
        self.context_path = build_contextpath(name=context_path)

        # build attention refinement module  for resnet 101
        if context_path == 'resnet101':
            self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
            self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
            # supervision block
            self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
            self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(num_classes, 3328)

        elif context_path == 'resnet18':
            # build attention refinement module  for resnet 18
            self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
            self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
            # supervision block
            self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
            self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
            self.dropout = nn.Dropout(p=0.2) # dropout AUMENTA P    
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(num_classes, 1024)
        else:
            print('Error: unspport context_path network \n')

        # build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

        self.init_weight()

        self.mul_lr = []
        self.mul_lr.append(self.saptial_path)
        self.mul_lr.append(self.attention_refinement_module1)
        self.mul_lr.append(self.attention_refinement_module2)
        self.mul_lr.append(self.supervision1)
        self.mul_lr.append(self.supervision2)
        self.mul_lr.append(self.feature_fusion_module)
        self.mul_lr.append(self.conv)

    # **BiSeNet.init_weight**. This custom method initializes the weights of the convolutional and batch 
    # normalization layers within this model, specifically *excluding* the layers that belong to the 
    # `context_path`. It uses Kaiming initialization for convolutions and constant initialization for batch norm.
    def init_weight(self):
        for name, m in self.named_modules():
            if 'context_path' not in name:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-5
                    m.momentum = 0.1
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    # **BiSeNet.forward**. This method defines the forward pass of the entire BiSeNet model. It processes the 
    # input through the spatial path and the context path, applies ARMs to context features, upsamples and 
    # concatenates them. It then passes the spatial path output and concatenated context features through the FFM. 
    # The FFM output is upsampled to the original input size and processed by the final convolution layer. 
    # During training (`self.training == True`), it also computes and returns outputs from supervision layers 
    # applied to intermediate context features before fusion.
    def forward(self, input):
        # output of spatial path
        sx = self.saptial_path(input)

        # output of context path
        cx1, cx2, tail = self.context_path(input)
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        # upsampling
        cx1 = torch.nn.functional.interpolate(cx1, size=sx.size()[-2:], mode='bilinear')
        cx2 = torch.nn.functional.interpolate(cx2, size=sx.size()[-2:], mode='bilinear')
        cx = torch.cat((cx1, cx2), dim=1)

        if self.training == True:
            cx1_sup = self.supervision1(cx1)
            cx2_sup = self.supervision2(cx2)
            cx1_sup = torch.nn.functional.interpolate(cx1_sup, size=input.size()[-2:], mode='bilinear')
            cx2_sup = torch.nn.functional.interpolate(cx2_sup, size=input.size()[-2:], mode='bilinear')

        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)

        result = self.dropout(result)

        # upsampling
        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')
        result = self.conv(result)

        if self.training == True:
            return result, cx1_sup, cx2_sup

        return result


#DROPCONNECT
#CUTOUT
#BATCHNORMALIZATION
