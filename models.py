import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
        super(ConvBlock, self).__init__()
        block = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True)]
        block += [nn.BatchNorm2d(out_channels=out_ch)]
        block += [nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class UpConvBlock(nn.Module):
    """
    Upsampling Convolution Block
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
        super(UpConvBlock, self).__init__()
        block = [nn.Upsample(scale_factor=2)]
        block += [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True)]
        block += [nn.BatchNorm2d(out_channels=out_ch)]
        block += [nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """
    Residual Block
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1, dilation=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1, dilation=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        identity = self.conv1(x)
        out = self.bn1(identity)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class NestedUNet(nn.Module):
    """
    Implementation of nested Unet (Unet++)
    """
    
    def __init__(self, in_ch=3, out_ch=1, n=32):
        super(NestedUNet, self).__init__()

        filters = [n, n * 2, n * 4, n * 8, n * 16]

        self.conv0_0 = ResBlock(in_ch, filters[0])
        self.conv1_0 = ResBlock(filters[0], filters[1])
        self.conv2_0 = ResBlock(filters[1], filters[2])
        self.conv3_0 = ResBlock(filters[2], filters[3])
        self.conv4_0 = ResBlock(filters[3], filters[4])

        self.conv0_1 = ResBlock(filters[0] + filters[1], filters[0])
        self.conv1_1 = ResBlock(filters[1] + filters[2], filters[1])
        self.conv2_1 = ResBlock(filters[2] + filters[3], filters[2])
        self.conv3_1 = ResBlock(filters[3] + filters[4], filters[3])

        self.conv0_2 = ResBlock(filters[0] * 2 + filters[1], filters[0])
        self.conv1_2 = ResBlock(filters[1] * 2 + filters[2], filters[1])
        self.conv2_2 = ResBlock(filters[2] * 2 + filters[3], filters[2])

        self.conv0_3 = ResBlock(filters[0] * 3 + filters[1], filters[0])
        self.conv1_3 = ResBlock(filters[1] * 3 + filters[2], filters[1])

        self.conv0_4 = ResBlock(filters[0] * 4 + filters[1], filters[0])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.final1 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final2 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final3 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final4 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        # self.final5 = nn.Conv2d(4, out_ch, kernel_size=1)

    def forward(self, x):

        x0_0 = self.conv0_0(x)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat((x0_0, self.up(x1_0)), 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat((x1_0, self.up(x2_0)), 1))
        x0_2 = self.conv0_2(torch.cat((x0_0, x0_1, self.up(x1_1)), 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat((x2_0, self.up(x3_0)), 1))
        x1_2 = self.conv1_2(torch.cat((x1_0, x1_1, self.up(x2_1)), 1))
        x0_3 = self.conv0_3(torch.cat((x0_0, x0_1, x0_2, self.up(x1_2)), 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat((x3_0, self.up(x4_0)), 1))
        x2_2 = self.conv2_2(torch.cat((x2_0, x2_1, self.up(x3_1)), 1))
        x1_3 = self.conv1_3(torch.cat((x1_0, x1_1, x1_2, self.up(x2_2)), 1))
        x0_4 = self.conv0_4(torch.cat((x0_0, x0_1, x0_2, x0_3, self.up(x1_3)), 1))

        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)
        # output5 = self.final5(torch.cat((output1, output2, output3, output4), 1))

        # return [output1, output2, output3, output4, output5]
        return [output1, output2, output3, output4]


if __name__ == '__main__':
    a = torch.ones((16, 6, 256, 256))
    Unet = NestedUNet(in_ch=6, out_ch=1)
    output = Unet(a)