import torch
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class my_UNet(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=10, features=[32, 64, 128, 256, 512, 1024]
    ):
        super(my_UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        # self.dropout = nn.Dropout(0.5)

        #Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #Up part of UNET
        for feature in reversed(features):
           self.ups.append(
               nn.ConvTranspose2d(
                   feature*2, feature,  kernel_size=2, stride=2
               ),
           )
           self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            # x = self.dropout(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
             x = self.ups[idx](x)
             skip_connection = skip_connections[idx//2]
             concat_skip = torch.cat((skip_connection, x), dim=1)
             # concat_skip = self.dropout(concat_skip)
             x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)

if __name__ == "__main__":
    batch = torch.randn(64, 3, 128, 128)
    model = my_UNet()
    output = model(batch)
    print(output.size())

