import torch
from torch import nn, Tensor



class Backbone(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Backbone, self).__init__()  # Q：子类调用父类的方法
        net = resnet50()
        self.out_channels = [1024, 512, 512, 256, 256, 256]  # 6个特征图的channel数目

        # 使用预训练模型的权重
        if pretrain_path is not None:
            net.load_state_dict(torch.load(pretrain_path))
        
        # 从resnet中取前7个模块作为feature extractor
        # 参考：https://blog.csdn.net/pengchengliu/article/details/113878358
        self.feature_extractor = nn.Sequential(*list(net.children())[:7])

        # 对Conv4的第1个Block进行修改
        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD300(nn.Module):
    def __init__(self, backbone=None, num_classes=21):
        super(SSD300, self).__init__()
        if backbone==None:
            raise Exception('backbone is None')
        if not hasattr(backbone, 'out_channels'):
            raise Exception('backbone not has attribute: out_channels')
        self.feature_extractor = backbone

        self.num_classes = num_classes
        # out_channels = [1024, 512, 512, 256, 256, 256]
        self._build_additional_features(self.feature_extractor.out_channels)
    

    def _build_additional_features(self, input_size):
        
