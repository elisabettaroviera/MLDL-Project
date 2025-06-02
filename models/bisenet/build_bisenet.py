import torch
from torch import nn
from .build_contextpath import build_contextpath
import warnings
warnings.filterwarnings(action='ignore')

class DropConnectConv2d(nn.Conv2d):
    def __init__(self, *args, drop_prob=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_prob = drop_prob

    def forward(self, input):
        if self.training:
            mask = (torch.rand_like(self.weight) > self.drop_prob).float()
            weight = self.weight * mask
        else:
            weight = self.weight * (1 - self.drop_prob)
        return nn.functional.conv2d(input, weight, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = DropConnectConv2d(in_channels, out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       bias=False,
                                       drop_prob=0.3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn(x))
        x = self.dropout(x)
        return x


class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x


class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.sigmoid(self.bn(x))
        x = torch.mul(input, x)
        return x


# === ASPP MODULE (Atrous Spatial Pyramid Pooling) === #CHANGE HERE
class ASPP(nn.Module):  # CHANGE HERE
    def __init__(self, in_channels, out_channels):  # CHANGE HERE
        super(ASPP, self).__init__()  # CHANGE HERE
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # CHANGE HERE
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)  # CHANGE HERE
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)  # CHANGE HERE
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)  # CHANGE HERE
        self.out_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)  # CHANGE HERE
        self.bn = nn.BatchNorm2d(out_channels)  # CHANGE HERE
        self.relu = nn.ReLU(inplace=True)  # CHANGE HERE

    def forward(self, x):  # CHANGE HERE
        x1 = self.conv1(x)  # CHANGE HERE
        x2 = self.conv2(x)  # CHANGE HERE
        x3 = self.conv3(x)  # CHANGE HERE
        x4 = self.conv4(x)  # CHANGE HERE
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # CHANGE HERE
        x_out = self.relu(self.bn(self.out_conv(x_cat)))  # CHANGE HERE
        return x_out  # CHANGE HERE


# === CBAM (Channel-Spatial Attention Module) === #CHANGE HERE
class CBAM(nn.Module):  # CHANGE HERE
    def __init__(self, channels, reduction=16):  # CHANGE HERE
        super(CBAM, self).__init__()  # CHANGE HERE
        # Channel Attention  # CHANGE HERE
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # CHANGE HERE
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # CHANGE HERE
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)  # CHANGE HERE
        self.relu = nn.ReLU()  # CHANGE HERE
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)  # CHANGE HERE
        self.sigmoid_channel = nn.Sigmoid()  # CHANGE HERE

        # Spatial Attention  # CHANGE HERE
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)  # CHANGE HERE
        self.sigmoid_spatial = nn.Sigmoid()  # CHANGE HERE

    def forward(self, x):  # CHANGE HERE
        # Channel Attention  # CHANGE HERE
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))  # CHANGE HERE
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))  # CHANGE HERE
        channel_out = self.sigmoid_channel(avg_out + max_out)  # CHANGE HERE
        x = x * channel_out  # CHANGE HERE

        # Spatial Attention  # CHANGE HERE
        avg_out = torch.mean(x, dim=1, keepdim=True)  # CHANGE HERE
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # CHANGE HERE
        spatial_out = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))  # CHANGE HERE
        x = x * spatial_out  # CHANGE HERE
        return x  # CHANGE HERE


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)

        # === AGGIUNGI CBAM per migliorare l'attenzione === #CHANGE HERE
        self.cbam = CBAM(num_classes)  # CHANGE HERE

        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)

        # Applica CBAM per affinare la fusione #CHANGE HERE
        feature = self.cbam(feature)  # CHANGE HERE

        x = self.avgpool(feature)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


class BiSeNet(torch.nn.Module):
    def __init__(self, num_classes, context_path):
        super().__init__()
        self.saptial_path = Spatial_path()
        self.context_path = build_contextpath(name=context_path)

        if context_path == 'resnet101':
            self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
            self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
            self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
            self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
            self.feature_fusion_module = FeatureFusionModule(num_classes, 3328)

        elif context_path == 'resnet18':
            self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
            self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
            self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
            self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
            self.dropout = nn.Dropout(p=0.3)
            self.feature_fusion_module = FeatureFusionModule(num_classes, 1024)
        else:
            print('Error: unspport context_path network \n')

        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

        # === AGGIUNGI ASPP alla fine del context path === #CHANGE HERE
        self.aspp = ASPP(512, 512) if context_path == 'resnet18' else ASPP(2048, 2048)  # CHANGE HERE

        self.init_weight()

        self.mul_lr = []
        self.mul_lr.append(self.saptial_path)
        self.mul_lr.append(self.attention_refinement_module1)
        self.mul_lr.append(self.attention_refinement_module2)
        self.mul_lr.append(self.supervision1)
        self.mul_lr.append(self.supervision2)
        self.mul_lr.append(self.feature_fusion_module)
        self.mul_lr.append(self.conv)
        self.mul_lr.append(self.aspp)  # CHANGE HERE

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

    def forward(self, input):
        sx = self.saptial_path(input)
        cx1, cx2, tail = self.context_path(input)
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)

        # === Applica ASPP sul context finale === #CHANGE HERE
        cx2 = self.aspp(cx2)  # CHANGE HERE

        cx2 = torch.mul(cx2, tail)
        cx1 = torch.nn.functional.interpolate(cx1, size=sx.size()[-2:], mode='bilinear')
        cx2 = torch.nn.functional.interpolate(cx2, size=sx.size()[-2:], mode='bilinear')
        cx = torch.cat((cx1, cx2), dim=1)

        if self.training == True:
            cx1_sup = self.supervision1(cx1)
            cx2_sup = self.supervision2(cx2)
            cx1_sup = torch.nn.functional.interpolate(cx1_sup, size=input.size()[-2:], mode='bilinear')
            cx2_sup = torch.nn.functional.interpolate(cx2_sup, size=input.size()[-2:], mode='bilinear')

        result = self.feature_fusion_module(sx, cx)
        result = self.dropout(result)
        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')
        result = self.conv(result)

        if self.training == True:
            return result, cx1_sup, cx2_sup

        return result
