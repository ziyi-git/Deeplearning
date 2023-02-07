import torch
from torch import nn, Tensor
from torch.jit.annotations import List

from .res50_backbone import resnet50
from .utils import dboxes300_coco, Encoder, PostProcess

class Backbone(nn.Module):  # Q：nn.Module如何理解？计算图的父类？
    def __init__(self, pretrain_path=None):
        # 子类Backbone继承父类nn.Module
        super(Backbone, self).__init__()  # Q：子类调用父类的方法
        net = resnet50()
        self.out_channels = [1024, 512, 512, 256, 256, 256]  # 6个特征图的channel数目

        # 使用预训练模型的权重
        if pretrain_path is not None:
            net.load_state_dict(torch.load(pretrain_path))
        
        # 从resnet中取前7个模块作为feature extractor
        # 参考：https://blog.csdn.net/pengchengliu/article/details/113878358
        # list(net.children())[:4] -> Conv1:
        # [Conv2d, BatchNorm2d, ReLU, MaxPool2d] -> 输出torch.Size([batch, 64, 152, 152])
        # list(net.children())[4] -> Conv2_x:
        # Sequential(Bottleneck, Bottleneck, Bottleneck) -> 输出torch.Size([batch, 256, 76, 76])
        # list(net.children())[5] -> Conv3_x:
        # Sequential(Bottleneck, Bottleneck, Bottleneck, Bottleneck) -> 输出torch.Size([batch, 512, 38, 38])
        # list(net.children())[6] -> Conv4_x:
        # Sequential(Bottleneck, Bottleneck, Bottleneck, Bottleneck, Bottleneck)
        self.feature_extractor = nn.Sequential(*list(net.children())[:7])

        # 对Conv4的第1个Block进行修改:
        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = (1, 1)  # Q: 在Bottleneck中这里本来就为(1, 1), 为何还要修改?
        # conv4_x输出由torch.Size([batch, 1024, 19, 19])变为torch.Size([batch, 1024, 38, 38])
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

class SSD300(nn.Module):
    def __init__(self, backbone=None, num_classes=21):
        super(SSD300, self).__init__()
        if backbone is None:
            raise Exception("backbone is None")
        if not hasattr(backbone, "out_channels"):
            raise Exception("backbone not has attribute: out_channels")
        self.feature_extractor = backbone

        self.num_classes = num_classes
        # out_channels = [1024, 512, 512, 256, 256, 256]
        self._build_additional_features(self.feature_extractor.out_channels)
        # feature map上每个cell产生的default box数目
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
        self.compute_loss = Loss(default_box)  # 注意: default_nox默认为ltrb格式, 在Loss中会被处理为xywh格式
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
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_ch, output_ch, kernel_size=3, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(output_ch),
                nn.ReLU(inplace=True)
            )
            additional_blocks.append(layer)
        self.additional_blocks = nn.ModuleList(additional_blocks)
    
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
            # locs:
            # torch.Size([batch, num_defualt * 4, feat_size, feat_size]) -> 
            # torch.Size([batch, 4, num_defualt * feat_size * feat_size]) 
            # confs:
            # torch.Size([batch, num_defualt * num_classes, feat_size, feat_size]) -> 
            # torch.Size([batch, num_classes, num_defualt * feat_size * feat_size])
            # 注：view()没有拷贝张量，没有在内存中改变张量的形状，而是重新定义了张量访问的规则
            locs.append(l(f).view(f.size(0), 4, -1))
            confs.append(c(f).view(f.size(0), self.num_classes, -1))
        
        # 为何使用continuous()，参考：https://zhuanlan.zhihu.com/p/64551412
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, image, targets=None):  # targets=None应该是在inference时使用
        x = self.feature_extractor(image)  # torch.Size([batch, 1024, 38, 38])

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
        for layer in self.additional_blocks:
            x = layer(x)
            detection_features.append(x)

        # locs: [batch, 4, 8732]
        # confs: [batch, 21, 8732]
        locs, confs = self.bbox_view(detection_features, self.loc, self.conf)

        if self.training:  # Q：是SSD300从父类nn.Module继承的成员变量？
            if targets is None:
                raise ValueError('In training mode, targets should be passed')
            bboxes_out = targets['boxes']
            bboxes_out = bboxes_out.transpose(1, 2).contiguous()  # torch.Size([batch, 4, 8732])
            labels_out = targets['labels']

            # parms: ploc, pconf, gloc, gconf
            # locs此时为[delta_x, delta_y, delta_w, delta_h]形式
            # bboxes_out此时为[x, y, w, h]形式
            loss = self.compute_loss(locs, confs, bboxes_out, labels_out)
            return {'total_losses': loss}
        
        results = self.postprocess(locs, confs)
        return results

class Loss(nn.Module):
    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0 / dboxes.scale_xy
        self.scale_wh = 1.0 / dboxes.scale_wh

        # self.location_loss = nn.SmoothL1Loss(reduction=None)
        self.location_loss = nn.SmoothL1Loss(reduction="none")

        self.dboxes = nn.Parameter(dboxes(order='xywh').transpose(0, 1).unsqueeze(dim=0), requires_grad=False)
        # self.confidence_loss = nn.CrossEntropyLoss(reduction=None)
        self.confidence_loss = nn.CrossEntropyLoss(reduction="none")
    
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

        2023年1月24日 17:00
        1. 在预测时, 8732个box中的每一个都只负责输出一个固定位置的结果.
        2. 每个位置输出[delta_x, delta_y, delta_w, delta_h], 代表在这个位置上的default box的多远处存在一个物体
        3. 为什么要对gloc做如下处理:
        #  a. 如1和2所述, gloc[i]是对于default_boxes[i]的输出, 代表以default_boxes[i]为基础的多大偏移处存在目标
        #  b. 那么, ploc[i]也要相对default_boxes[i]做出输出, 代表以default_boxes[i]为基础的多大偏移处存在目标
        #  c. 综合a和b, gloc和ploc都是围绕default box来进行预测, 锚点就是default box, 此时终于明白anchor的含义
        #  d. 按理说, 只需要[delta_x, delta_y, delta_w, delta_h]就可以了, 为什么还要在本函数中做一次变换?可以用
        #     一个案例来说明, 这实际是为了均衡大目标和小目标的loss
        #  e. 假设在一幅图像中存在一个大目标和一个小目标, 在大目标和小目标处均存在1个ploc, gloc和dbox.
        #     大目标(xmin, ymin, xmax, ymax):
        #     ploc_1 = [0.4, 0.4, 0.8, 0.7063]
        #     gloc_1 = [0.37, 0.41, 0.8105, 1]
        #     dbox_1 = [0.3, 0.3, 0.8, 0.8] -> ploc_1和gloc_1与dbox_1的IoU都为0.5
        #     小目标(xmin, ymin, xmax, ymax):
        #     ploc_1 = [0.4, 0.4, 0.8, 0.7063]
        #     gloc_1 = [0.37, 0.41, 0.8105, 1]
        #     dbox_1 = [0.3, 0.3, 0.8, 0.8] -> ploc_2和gloc_2与dbox_2的IoU都为0.5
        #     采用L1 loss, 当直接使用[delta_x, delta_y, delta_w, delta_h]方式
        #     大目标loss为0.4858, 小目标loss为0.06405, 大 / 小 = 7.58
        #     采用L1 lss, 当使用本函数所用变换
        #     大目标loss为6.992, 小目标loss为1.969, 大 / 小 = 3.549
        #     可以看到缩小了大小目标的损失比
        """
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :]  # 实际对xy的差异做了标准化
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()  # Q: log是为了处理大小目标误差的不平衡
        return torch.cat((gxy, gwh), dim=1).contiguous()
    
    def forward(self, ploc, plabel, gloc, glabel):
        """ 
        ploc, plabel: torch.Size([batch, 4, 8732]), torch.Size([batch, 8732])
        gloc, glabel: torch.Size([batch, 4, 8732]), torch.Size([batch, 8732])
        N应指batch
        """ 
        # tagets['label']，获取正样本的个数
        # mask:
        # tensor([[False, False, False,  ..., False, False, False],
        #        [False, False, False,  ..., False, False, False],
        #        [False, False, False,  ...,  True,  True,  True],
        #        [False, False, False,  ..., False, False, False]])
        mask = torch.gt(glabel, 0)
        pos_num = mask.sum(dim=1) # 一行(对应一幅图像)中正样本个数 torch.Size([batch])

        vec_gd = self._location_vec(gloc)
        # 只计算正样本的损失
        loc_loss = self.location_loss(ploc, vec_gd).sum(dim=1)  # Tensor.Size([batch, 8732])
        loc_loss = (mask.float() * loc_loss).sum(dim=1)  # Tensor.Size([batch])

        # hard negtive mining
        con = self.confidence_loss(plabel, glabel)  # Tensor.Size([batch, 8732])
        
        # 只选择负样本
        con_neg = con.clone()
        con_neg[mask] = 0.0
        # 2023年1月27日 16:38
        # 面对技巧型的代码, 应该先从本质上入手: 1.先搞清楚目的 -> 2.然后推演一遍自己会如何做 -> 3.最后理解别人如何做的
        # 1.这里的目的是为了从con_neg中选择每一行前面3 * pos_num个最大的loss
        # 2.按照降序对con_neg进行原位排序, 即获得每一行中各个元素的序号
        # 3.用一个例子理解别人如何做的:
        #   a. 假设con_neg是一个torch.Size([2, 6])的矩阵, batch为2, 6个loss
        #      con_neg = [[11, 21, 15,  9,  7, 14],
        #                 [25, 33, 11, 18, 24, 16]]
        #   b. new_con_neg, con_idx = con_neg.sort(dim=1, descending=True)
        #      new_con_neg = [[21, 15, 14, 11,  9,  7],
        #                     [33, 25, 24, 18, 16, 11]]
        #          con_idx = [[1, 2, 5, 0, 3, 4],
        #                     [1, 0, 4, 3, 5, 2]]
        #      此时con_idx中每个元素指降序后, 该元素在con_neg中的序号, 例如21在原来的con_neg中序号为1
        #   c. new_con_idx , con_rank = con_idx.sort(dim=1)
        #      new_con_idx = [[0, 1, 2, 3, 4, 5],
        #                     [0, 1, 2, 3, 4, 5]]
        #         con_rank = [[3, 0, 1, 4, 5, 2],
        #                     [1, 0, 5, 3, 2, 4]]
        #      这里通过一个升序又将元素顺序恢复, 而con_rank返回的是在con_idx中每个元素应该排在哪一位.
        #      例如元素15在原数组con_neg中"序号"为2, con_idx表示经过"序号"为2的元素升序后"排位"为1,
        #      con_rank则按照序号恢复了"序号"排序, 而con_rank表示每个序号在经过升序后的数组中的"排位"
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)
        # 负样本为正样本的3倍，且数目不超过8732
        # unsqueeze(-1): torch.Size([batch]) -> torch.Size([batch, 1])
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        # 将con_rank和neg_num主元素比较, 如果con_rank[i][j] < neg_num[i][j]
        # 则为True, 意为返回排为前3 * pos_num的loss被保留
        neg_mask = torch.lt(con_rank, neg_num)
        # 选择所有的正样本和3 * pos_num个负样本, torch.Size([batch])
        con_loss = (con * (mask.float() + neg_mask.float())).sum(dim=1)

        total_loss = loc_loss + con_loss
        # 将不存在正样本的图像的pos_num变为0
        # tensor([10, 14, 11, 0, 22]) -> tensor([1, 1, 1, 0, 1])
        num_mask = torch.gt(pos_num, 0).float()
        # 将小于1e-6的元素变为1e-6, 这里是为了防止后面分母为0
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss * num_mask / pos_num).mean(dim=0)  # 只计算存在正样本的图像损失
        return ret