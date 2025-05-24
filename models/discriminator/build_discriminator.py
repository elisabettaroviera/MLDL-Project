import torch.nn as nn

class FCDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 2, 1),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.model(x)
