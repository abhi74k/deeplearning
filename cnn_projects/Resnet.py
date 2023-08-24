from torch import nn


class IdentityBlock(nn.Module):

    def __init__(self, filters, in_channels):
        super().__init__()

        F1, F2, F3 = filters

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=F1, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=F1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=F2)
        self.relu2 = nn.ReLU()

        # The input to the third block is same as the inpput to the first block
        self.conv3 = nn.Conv2d(in_channels=F2, out_channels=F3, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=F3)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x_cache = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += x_cache  # Skip connection
        x = self.relu3(x)

        return x


class ConvolutionBlock(nn.Module):

    def __init__(self, filters, in_channels, s):
        super().__init__()

        F1, F2, F3 = filters

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=F1, kernel_size=1, stride=s, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=F1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=F1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=F2, out_channels=F3, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=F3)
        self.relu3 = nn.ReLU()

        # Below are the units along the skip connection
        # The input channels are reduced by half due to stride=2
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=F3, kernel_size=1, stride=s, padding=0)
        self.bn4 = nn.BatchNorm2d(num_features=F3)

    def forward(self, x):
        x_cache = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x_cache = self.conv4(x_cache)
        x_cache = self.bn4(x_cache)

        x += x_cache
        x = self.relu3(x)

        return x


class ResNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.network = nn.Sequential(

            nn.ZeroPad2d(padding=3),  # 64x64 -> 67x67

            # Stage 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=0),  # 67x67 -> 31x31
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # 31x31 -> 15x15

            # Stage 2
            ConvolutionBlock(filters=[64, 64, 256], in_channels=64, s=1),
            IdentityBlock(filters=[64, 64, 256], in_channels=256),
            IdentityBlock(filters=[64, 64, 256], in_channels=256),

            # Stage 3
            ConvolutionBlock(filters=[128, 128, 512], in_channels=256, s=2),  # 15x15 -> 8x8
            IdentityBlock(filters=[128, 128, 512], in_channels=512),
            IdentityBlock(filters=[128, 128, 512], in_channels=512),
            IdentityBlock(filters=[128, 128, 512], in_channels=512),

            # Stage 4
            ConvolutionBlock(filters=[256, 256, 1024], in_channels=512, s=2),  # 8x8 -> 4x4
            IdentityBlock(filters=[256, 256, 1024], in_channels=1024),
            IdentityBlock(filters=[256, 256, 1024], in_channels=1024),
            IdentityBlock(filters=[256, 256, 1024], in_channels=1024),
            IdentityBlock(filters=[256, 256, 1024], in_channels=1024),
            IdentityBlock(filters=[256, 256, 1024], in_channels=1024),

            # Stage 5
            ConvolutionBlock(filters=[512, 512, 2048], in_channels=1024, s=2),  # 4x4 -> 2x2
            IdentityBlock(filters=[512, 512, 2048], in_channels=2048),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 2x2 -> 1x1

            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=num_classes.shape[0])
        )

    def forward(self, x):
        return self.network(x)
