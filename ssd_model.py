import torch
from torch import nn, Tensor



class Backbone(nn.Module):  # Q：nn.Module如何理解？计算图的父类？
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

        default_box = dboxes300_coco()  # 所有default box在300*300图像上的坐标, 8732 * 4(xywh)
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
    
    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:  # ?
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, features, loc_extractor, conf_extractor):
        locs = []
        confs = []

        for f, l, c in zip(features, loc_extractor, conf_extractor):
            # 将feature map处理为[x, y, w, h]形式
            # [batch, n * 4, feat_size, feat_size] -> [batch, 4, feat_size * feat_size]
            # 注：view()没有拷贝张量，没有在内存中改变张量的形状，而是重新定义了张量访问的规则
            locs.append(l(f).view(f.size(0), 4, -1))
            confs.append(c(f).view(f.size(0), self.num_classes, -1))
        
        # 为何使用continuous()，参考：https://zhuanlan.zhihu.com/p/64551412
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, image, targets=None):  # targets=None应该是在inference时使用
        x = self.feature_extractor(image)  # 输出backbone的conv4

        # Feature Map1(38x38x1024) -> Feature Map2(19x19x512) -> Feature Map3(10x10x512)
        # Feature Map4(5x5x256) -> Feature Map5(3x3x256) -> Feature Map6(1x1x256)
        # Q：torch.jit.annotate(the_type, the_value)
        # the_type：返回的张量的类型为List[Tensor]，the_value：表达式
        # 注意不要和nn.ModuleList搞混，nn.ModuleList也是一个列表但保存的是算子op，这里的List保存的Tensor张量
        # 在两个op之间的张量不用由torch.jit.annotate来显示的转，应该在torch内部有对应的操作，但在这里是一个显式
        # 的操作，所以需要声明。
        # 简言之：如果需要显示地操作张量，就要使用torch.jit
        detection_features = torch.jit.annotate(List[Tensor], [])
        detection_features.append(x)
        for layer in self.addtional_blocks:
            x = layer(x)
            detection_features.append(x)

        # locs: [batch, 4, 8732]
        # confs: [batch, 21, 8732]
        # 8732 = 38x38x4 + 19x19x4 + 10x10x4 + 5x5x4 + 3x3x4 + 1x1x4
        locs, confs = self.bbox_view(detection_features, self.loc, self.conf)

        if self.training:  # Q：是SSD300从父类nn.Module继承的成员变量？
            if targets if None:
                raise ValueError('In training mode, targets should be passed')
            bboxes_out = targets['boxes']  # Q：targets['boxes']在何处处理成了8732x4的形式
            bboxes_out = bboxes.transpose(1, 2).contiguous()  # [batch, 8732, 4] -> [batch, 4, 8732]
            labels_out = tagets['labels']

            # parms: ploc, pconf, gloc, gconf
            loss = self.compute_loss(locs, confs, bboxes_out, labels_out)
            return {'total_loss': loss}

class Loss(nn.Module):
    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0 / dboxes.scale_xy
        self.scale_wh = 1.0 / dboxes.scale_wh

        self.location_loss = nn.SmoothL1Loss(reduction=None)

        self.dboxes = nn.Parameter(dboxes(order='xywh').transpose(0, 1).unsqueeze(dim=0), requires_grad=False)
        self.confidence_loss = nn.CrossEntropyLoss(reduction=None)
    
    def _location_vec(self, loc):
        """
            2022年12月28日 23:59
            那些不明白的事情就像是圆心，我一直站在圆周上望着它，却不知道心路在圆周上的何处。
            线索一定在圆环上，唯有绕着圆走才会找到它。
            运气好时在不远处，运气差时需要走过整个圆周。
            不必无时无刻绕着圆走，每天回来走一点，总归是能找到它的。

            Q：这个函数我不懂，也想了整整两天，心气消磨了不少。
            回顾起来，在理解这个函数的作用时，我有很多线索都没有搞明白。
            location回归这一部分到底是在干什么？经过卷基层得出的ploc是不是说要找到一些box能够和
            ground truth达到最好的IOU，但是不用去管类别。ground truth boxes又是如何得到的？
            数据集的标签中只有少数的标记框，如何把这少数的标记框变成8732x4的？
            
            两个参考：
            1. https://blog.csdn.net/qq_37450561/article/details/89335413
            2. https://www.pianshen.com/article/5509670154/
        """
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :]
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()
    
    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nx8732
            gloc, glabel: Nx4x8732, Nx8732
            N应指batch
        """
        # tagets['label']，获取正样本的个数
        mask = torch.gt(glabel, 0)  # Q：> 0变为正样本1？
        pos_num = mask.sum(dim=1) # 每张图片中正样本个数 Tensor: [N]

        vec_gd = self._location_vec(gloc)
        # 只计算正样本的损失
        loc_loss = self.location_loss(ploc, vec_gd).sum(dim=1)  # Tensor: [N, 8732]
        loc_loss = (mask.float() * loc_loss).sum(dim=1)  # Tensor: [N]

        # hard negtive mining Tensor: [N, 8732]
        con = self.confidence_loss(plabel, glabel)
        
        # 只选择负样本
        con_neg = con.clone()
        con_neg[mask] = 0.0
        _, con_idx = con_neg.sort(dim=1, descending=True)  # 应该是把正样本都放到了最后
        _, con_rank = con_idx.sort(dim=1)  # ???

        # 负样本为正样本的3倍，且数目不超过8732
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1))
        neg_mask = torch.lt(con_rank, neg_num)

        con_loss = (con * (mask.float() + neg_mask.float())).sum(dim=1)

        total_loss = loc_loss + con_loss
        num_mask = torch.gt(pos_num, 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss * num_mask / pos_num).mean(dim=0)  # 只计算存在正样本的图像损失
        return ret




