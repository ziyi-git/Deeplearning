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
        self.num_defaults = [4, 6, 6, 6, 4, 4]  # Q：为何这样设置
        location_extractors = []
        confidence_extractors = []

        # 由feature map提取location: [x, y, w, h], confidence: [prob_0, prob_1, ..., prob_20]
        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            location_extractors.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            confidence_extractors.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))
        
        # Q：将list转为ModuleList才能被torch识别？为何不是nn.Sequential，猜测nn.Sequential强调顺序执行，nn.ModuleList不强调
        self.loc = nn.ModuleList(location_extractors)
        self.conf = nn.ModuleList(confidence_extractors)

        default_box = dboxes300_coco()  # 所有default box在300*300图像上的坐标
        self.compute_loss = Loss(default_box)
        self.encoder = Encoder(default_box)  # Q：似乎没有用到
        self.postprocess = PostProcess(default_box)
    

    def _build_additional_features(self, input_size):
        additional_blocks = []
        middle_channels = [256, 256, 128, 128, 128]
        for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
            # padding, stride = (1, 2) + kernel size=3可认为将feature map缩放为1/2
            # padding, stride = (0, 1) + kernel size=3可认为feature map尺寸未变
            # Q：kernel size=3时，padding=0，如何对feature map边界处的点操作？
            padding, stride = (1, 2) if i < 3 else (0, 1)
            # Q：为何把channel先变为middle_ch，后再变为output_ch
            layer = nn.Sequential(
                nn.Conv2d(input_ch, middle_ch, kernel_size=1, bias=False),  # WH不变，C变为middle_ch，Q：bias=False如何理解
                nn.BatchNorm2d(middle_ch),
                nn.ReLU(),
                nn.Conv2d(middle_ch, output_ch, kernel_size=3, bias=False),
                nn.BatchNorm2d(output_ch),
                nn.ReLU()
            )
            additional_blocks.append(layer)
        self.addtional_blocks = nn.ModuleList(additional_blocks)


    def forward(self, image, targets=None):  # targets=None应该是在inference时使用
        x = self.feature_extractor(image)  # 输出backbone的conv4

        # Feature Map1(38x38x1024) -> Feature Map2(19x19x512) -> Feature Map3(10x10x512)
        # Feature Map4(5x5x256) -> Feature Map5(3x3x256) -> Feature Map6(1x1x256)
        # Q：torch.jit.annotate(the_type, the_value)
        # the_type：返回的张量的类型为List[Tensor]，the_value：表达式
        # 注意不要和nn.ModuleList搞混，nn.ModuleList也是一个列表但保存的是算子op，这里的List保存的Tensor张量
        # 在两个op之间的张量不用由torch.jit.annotate来显示的转，应该在torch内部有对应的操作，但在这里是一个显式
        # 的操作，所以需要声明。
        detection_features = torch.jit.annotate(List[Tensor], [])
        detection_features.append(x)
        for layer in self.addtional_blocks:
            x = layer(x)
            detection_features.append(x)

