import torch.nn as nn
import torch

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 第1层
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # 第2层
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # 第3层
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)  
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

# 注: GPU中tensor的排列方式为NCHW
class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        # output.shape -> torch.Size([batch, 64, 152, 152])
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        # output.shape -> torch.Size([batch, 64, 152, 152])
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        # output.shape -> torch.Size([batch, 64, 152, 152])
        self.relu = nn.ReLU(inplace=True)
        # output.shape -> torch.Size([batch, 64, 76, 76])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Conv2_x: 输出channel 256
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        # Conv2_x: 输出channel 512
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        # Conv3_x: 输出channel 1024
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        # Conv4_x: 输出channel 2048
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # (2048, 21)
        
        # self.modules应该是继承自父类nn.Module
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, channel, block_num, stride=1):
        """
        channel = [64, 128, 256, 512]
        blocks_num = [3 ,4, 6, 3]
        """
        downsample = None
        # 解释downsample:
        # class Bottleneck(nn.Module):
        # expansion = 4

        # def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        #     ...
        # 如上:
        # 在1个block内, 分identify和非identify两部分.
        # 非identity部分的输出channel会变为out_channel * block.expansion
        # 所以，identity的channel必须也是out_channel * block.expansion才能和非identity完成相加
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion  # 64 -> 256 -> 512 -> 1024
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)    # torch.Size([batch, 64, 152, 152])
        x = self.bn1(x)      # torch.Size([batch, 64, 152, 152])
        x = self.relu(x)     # torch.Size([batch, 64, 152, 152])
        x = self.maxpool(x)  # torch.Size([batch, 64, 76, 76])

        x = self.layer1(x)  # torch.Size([batch, 256, 76, 76])
        x = self.layer2(x)  # torch.Size([batch, 512, 38, 38])
        x = self.layer3(x)  # torch.Size([batch, 1024, 19, 19])
        x = self.layer4(x)  # torch.Size([batch, 2048, 10, 10])

        if self.include_top:
            x = self.avgpool(x)      # torch.Size([batch, 2048, 1, 1])
            x = torch.flatten(x, 1)  # torch.Size([batch, 2048])
            x = self.fc(x)           # torch.Size([batch, 1000])
        
        return x

def resnet50(num_classes=1000, include_top=True):
    # Q: 传入参数Bottleneck并非一个类实例, 而是一个类?
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
