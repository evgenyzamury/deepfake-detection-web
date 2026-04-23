import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import transforms


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 bias: bool = False
                 ):
        super(ConvBlock, self).__init__()

        self.conv2d = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias)

        self.batchnorm2d = nn.BatchNorm2d(out_channels)

        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batchnorm2d(x)
        x = self.act(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_1x1: int,
                 reduce_3x3: int,
                 out_3x3: int,
                 reduce_5x5: int,
                 out_5x5: int,
                 out_1x1_pooling: int
                 ):
        super(InceptionBlock, self).__init__()

        # 1x1 conv
        self.branch_1 = ConvBlock(in_channels, out_1x1, 1, 1, 0)

        # 1x1 conv -> 3x3 conv
        self.branch_2 = nn.Sequential(ConvBlock(
            in_channels, reduce_3x3, 1, 1, 0), ConvBlock(reduce_3x3, out_3x3, 3, 1, 1))

        # 1x1 conv -> 5x5 conv
        self.branch_3 = nn.Sequential(ConvBlock(
            in_channels, reduce_5x5, 1, 1, 0), ConvBlock(reduce_5x5, out_5x5, 5, 1, 2))

        # 3x3 maxpool -> 1x1 conv
        self.branch_4 = nn.Sequential(nn.MaxPool2d(
            kernel_size=3, stride=1, padding=1), ConvBlock(in_channels, out_1x1_pooling, 1, 1, 0))

    def forward(self, x):
        return torch.cat([self.branch_1(x), self.branch_2(x), self.branch_3(x), self.branch_4(x)], dim=1)


class InceptionV1(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(InceptionV1, self).__init__()

        self.conv_1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv_2 = nn.Sequential(
            ConvBlock(64, 64, 1, 1, 0), ConvBlock(64, 192, 3, 1, 1))
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_3_a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception_3_b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_4_a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception_4_b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception_4_c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception_4_d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception_4_e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_5_a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception_5_b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv_1(x)
        # print('conv_1',x.shape)
        x = self.maxpool_1(x)
        # print('maxpool_1',x.shape)

        x = self.conv_2(x)
        # print('conv_2',x.shape)
        x = self.maxpool_2(x)
        # print('maxpool_2',x.shape)

        x = self.inception_3_a(x)
        # print('3_a',x.shape)

        x = self.inception_3_b(x)
        # print('3_b',x.shape)

        x = self.maxpool_3(x)
        # print('3_b_max',x.shape)

        x = self.inception_4_a(x)
        # print('4_a',x.shape)

        x = self.inception_4_b(x)
        # print('4_b',x.shape)

        x = self.inception_4_c(x)
        # print('4_c',x.shape)

        x = self.inception_4_d(x)
        # print('4_d',x.shape)

        x = self.inception_4_e(x)
        # print('4_e',x.shape)

        x = self.maxpool_4(x)
        # print('maxpool',x.shape)

        x = self.inception_5_a(x)
        # print('5_a',x.shape)

        x = self.inception_5_b(x)
        # print('5_b',x.shape)

        x = self.avgpool(x)
        # print('AvgPool',x.shape)

        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x


preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5194, 0.4280, 0.3847], std=[0.2861, 0.2640, 0.2636]),
])

inception_model = InceptionV1(3, 2)
inception_model.eval()
inception_model.load_state_dict(torch.load('inception-dirty-clear-v2.pth', map_location=torch.device('cpu')))
# используем cpu тк публикуем на сервер где только cpu

print('==== inception_model init ==== ')