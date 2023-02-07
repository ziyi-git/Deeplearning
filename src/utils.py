from math import sqrt
import itertools

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch.jit.annotations import Tuple, List

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def calc_iou_tensor(boxes1, boxes2):
    area1 = box_area(boxes1)  # torch.Size([nboxes1])
    area2 = box_area(boxes2)  # torch.Size([nboxes2])

    # 约定行为X轴(指向右), 列为Y轴(指向下), channel为Z轴(指向内)
    # None维度的长度为1
    # ===================== case1 ===========================
    # boxes1:
    # [[xmin1, ymin1, xmax1, ymax1],
    #  [xmin2, ymin2, xmax2, ymax2]]
    # boxes2:
    # [[xmin1, ymin1, xmax1, ymax1]]
    # 广播:
    # boxes1[:, None, :]将boxes1绕Y轴逆时针旋转90度, torch.Size([2, 1, 4])
    # boxes2[:, :]将boxes2先绕Y轴逆时针旋转90度, 再沿Y轴复制2次, torch.Size([2, 1, 4])
    # ===================== case2 ===========================
    # boxes1:
    # [[xmin1, ymin1, xmax1, ymax1],
    #  [xmin2, ymin2, xmax2, ymax2]]
    # boxes2:
    # [[xmin1, ymin1, xmax1, ymax1],
    #  [xmin2, ymin2, xmax2, ymax2],
    #  [xmin3, ymin3, xmax3, ymax3]]
    # boxes1[:, None, :]将boxes1绕Y轴逆时针旋转90度, 再沿X轴复制3次, torch.Size([2, 3, 4])
    # boxes2[:, :]将boxes2先绕Y轴逆时针旋转90度, 再绕Z轴逆时针旋转90度, 最后沿Y轴复制2次, torch.Size([2, 3, 4])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # 相交区域左上角
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # 相交区域右下角

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # 广播(只有2个维度时，仅仅考虑XY平面内旋转):
    # area1[:, None]将area1顺时针旋转即转置, 再沿X轴复制nboxes2次, torch.Size([nboxes1, nboxes2])
    # area2沿Y轴复制nboxes1次, torch.Size([nboxes1, nboxes2])
    # ious[m, n]: boxes1第m个box与boxes2第n个box的IoU
    iou = inter / (area1[:, None] + area2 - inter)
    return iou

class Encoder(object):
    def __init__(self, dboxes):
        self.dboxes = dboxes(order='ltrb')
        self.dboxes_xywh = dboxes(order='xywh').unsqueeze(dim=0)  # Q：增加了一个batch的维度？
        self.nboxes = self.dboxes.size(0)  # 注意区分self.nboxes和encode中的nboxes
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh
    
    def encode(self, bboxes_in, labels_in, criteria=0.5):
        """
        1. 选择与GT boxes(bboxes_in)的IoU大于criteria的default boxes(dboxes)提取出来
        2. 将这些default boxes的坐标替换为对应GT boxes的坐标
        """
        # 计算每个GT与default box的iou, torch.Size([nboxes, 8732]):
        # b(*): GT boxes, d(*): default boxes
        # [[b(1)-d(1),      b(1)-d(2),      ..., b(1)-d(8732),
        #   b(2)-d(1),      b(2)-d(2),      ..., b(2)-d(8732),
        #                                    ...
        #                                    ...
        #   b(nboxes)-d(1), b(nboxes)-d(2), ..., b(nboxes)-d(8732)]]
        ious = calc_iou_tensor(bboxes_in, self.dboxes)
        # best_dbox_ious: 对ious每列内取最大值, 获得每个default box匹配到的最大IoU, torch.Size([8732])
        # [best_d(1), best_d(2), ..., best_d(8732)]
        # best_dbox_idx: 最大IoU匹配的GT box的idx, 1 <= idx <= nboxes
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        # best_bbox_ious：对ious每行内取最大值, 获得每个GT box匹配到的最大IoU, torch.Size([nboxes])
        # [best_b(1), best_b(2), ..., best_b(nboxes)]
        # best_bbox_idx：最大IoU匹配到的default box的idx, 1 <= idx <= 8732
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)  # 每一个GT box匹配到的最大IoU

        # 将与每一个GT boxe之间IoU最大的default box的IoU置为2.0
        # Q：为何是2.0？按理IoU最大是1.0
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)
        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx  # Q: 这一行好像没起作用？

        masks = best_dbox_ious > criteria  # torch.Size([8732])
        labels_out = torch.zeros(self.nboxes, dtype=torch.int64)  # torch.Size([8732])
        # labels_out[masks]将和GT box匹配IoU>0.5的部分保留下来
        # best_dbox_idx是8732个default boxes每一个对应的最大匹配IoU的GT box的序号
        # best_dbox_idx[masks]将满足IoU>0.5的序号提取出来，注意：可能存在多个default box
        # 对应同一个GT box，所以best_dbox_idx[masks]的形式可能是[1, 1, 2, 3, 3, 3, 4, ...]
        labels_out[masks] = labels_in[best_dbox_idx[masks]]
        bboxes_out = self.dboxes.clone()
        # 关键步骤:
        # 1. 将default boxes中满足条件(即和GT boxes的IoU > criteria)的部分提取出来, 注意: 可能有多个
        #    default box和同一个GT box的IoU > criteria.
        # 2. 把上述default box的坐标, 替换成对应的GT box的坐标.
        #    例如:
        #    defaul_box_1和default_box_2都和GT_box_7的IoU > criteria, 则将它们的坐标都替换为GT_box_7
        bboxes_out[masks] = bboxes_in[best_dbox_idx[masks], :]

        # 转换为xywh格式
        x = 0.5 * (bboxes_out[:, 0] + bboxes_out[:, 2])
        y = 0.5 * (bboxes_out[:, 1] + bboxes_out[:, 3])
        w = bboxes_out[:, 2] - bboxes_out[:, 0]
        h = bboxes_out[:, 3] - bboxes_out[:, 1]
        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h

        return bboxes_out, labels_out
    
    def scale_back_batch(self, bboxes_in, scores_in):
        """
            将box格式从xywh转换回ltrb, 将预测目标score通过softmax处理
            Do scale and transform from xywh to ltrb
            suppose input N x 4 x num_bbox | N x label_num x num_bbox
            bboxes_in: 是网络预测的xywh回归参数
            scores_in: 是预测的每个default box的各目标概率
        """
        if bboxes_in.device == torch.device("cpu"):
            self.dboxes = self.dboxes.cpu()
            self.dboxes_xywh = self.dboxes_xywh.cpu()
        else:
            self.dboxes = self.dboxes.cuda()
            self.dboxes_xywh = self.dboxes_xywh.cuda()

        # Returns a view of the original tensor with its dimensions permuted.
        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)
        # print(bboxes_in.is_contiguous())

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]   # 预测的x, y回归参数
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]   # 预测的w, h回归参数

        # 将预测的回归参数叠加到default box上得到最终的预测边界框
        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        # transform format to ltrb
        l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l  # xmin
        bboxes_in[:, :, 1] = t  # ymin
        bboxes_in[:, :, 2] = r  # xmax
        bboxes_in[:, :, 3] = b  # ymax

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_batch(self, bboxes_in, scores_in, criteria=0.45, max_output=200):
        # 将box格式从xywh转换回ltrb（方便后面非极大值抑制时求iou）, 将预测目标score通过softmax处理
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        outputs = []
        # 遍历一个batch中的每张image数据
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            outputs.append(self.decode_single_new(bbox, prob, criteria, max_output))
        return outputs

    def decode_single_new(self, bboxes_in, scores_in, criteria, num_output=200):
        """
        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
        """
        device = bboxes_in.device
        num_classes = scores_in.shape[-1]

        # 对越界的bbox进行裁剪
        bboxes_in = bboxes_in.clamp(min=0, max=1)

        # [8732, 4] -> [8732, 21, 4]
        bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores_in)

        # remove prediction with the background label
        # 移除归为背景类别的概率信息
        bboxes_in = bboxes_in[:, 1:, :]
        scores_in = scores_in[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        bboxes_in = bboxes_in.reshape(-1, 4)
        scores_in = scores_in.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        # 移除低概率目标，self.scores_thresh=0.05
        inds = torch.nonzero(scores_in > 0.05, as_tuple=False).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[inds], scores_in[inds], labels[inds]

        # remove empty boxes
        ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
        keep = (ws >= 0.1 / 300) & (hs >= 0.1 / 300)
        keep = keep.nonzero(as_tuple=False).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]

        # non-maximum suppression
        keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)

        # keep only topk scoring predictions
        keep = keep[:num_output]
        bboxes_out = bboxes_in[keep, :]
        scores_out = scores_in[keep]
        labels_out = labels[keep]

        return bboxes_out, labels_out, scores_out

    # perform non-maximum suppression
    def decode_single(self, bboxes_in, scores_in, criteria, max_output, max_num=200):
        """
        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
        """
        # Reference to https://github.com/amdegroot/ssd.pytorch
        bboxes_out = []
        scores_out = []
        labels_out = []

        # 非极大值抑制算法
        # scores_in (Tensor 8732 x nitems), 遍历返回每一列数据，即8732个目标的同一类别的概率
        for i, score in enumerate(scores_in.split(1, 1)):
            # skip background
            if i == 0:
                continue

            # [8732, 1] -> [8732]
            score = score.squeeze(1)

            # 虑除预测概率小于0.05的目标
            mask = score > 0.05
            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0:
                continue

            # 按照分数从小到大排序
            score_sorted, score_idx_sorted = score.sort(dim=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []

            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                # 获取排名前score_idx_sorted名的bboxes信息 Tensor:[score_idx_sorted, 4]
                bboxes_sorted = bboxes[score_idx_sorted, :]
                # 获取排名第一的bboxes信息 Tensor:[4]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                # 计算前score_idx_sorted名的bboxes与第一名的bboxes的iou
                iou_sorted = calc_iou_tensor(bboxes_sorted, bboxes_idx).squeeze()

                # we only need iou < criteria
                # 丢弃与第一名iou > criteria的所有目标(包括自己本身)
                score_idx_sorted = score_idx_sorted[iou_sorted < criteria]
                # 保存第一名的索引信息
                candidates.append(idx)

            # 保存该类别通过非极大值抑制后的目标信息
            bboxes_out.append(bboxes[candidates, :])   # bbox坐标信息
            scores_out.append(score[candidates])       # score信息
            labels_out.extend([i] * len(candidates))   # 标签信息

        if not bboxes_out:  # 如果为空的话，返回空tensor，注意boxes对应的空tensor size，防止验证时出错
            return [torch.empty(size=(0, 4)), torch.empty(size=(0,), dtype=torch.int64), torch.empty(size=(0,))]

        bboxes_out = torch.cat(bboxes_out, dim=0).contiguous()
        scores_out = torch.cat(scores_out, dim=0).contiguous()
        labels_out = torch.as_tensor(labels_out, dtype=torch.long)

        # 对所有目标的概率进行排序（无论是什 么类别）,取前max_num个目标
        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]

# Q: object和nn.Module区别？ 
class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2):
        self.fig_size = fig_size
        self.feat_size = feat_size  # [38, 19, 10, 5, 3, 1]
        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh
        self.steps = steps  # [8, 16, 32, 64, 100, 300]
        self.scales = scales  # [21, 45, 99, 153, 207, 261, 315]

        # feature map(38x38)为例，每个cell在300x300原图上跨度为8x8，相当于将300平均分成300/8=37.5份。如果
        # 进行归一化，每个cell在原图上的中心坐标为[0.5/37.5, 0.5 / 37.5], [1.5/37.5, 1.5/37.5], ...
        fk = fig_size / np.array(steps)  # 300 / [8, 16, 32, 64, 100, 300] = [37.5, 18.75, 9.375, 4.6875,  3.,  1.]
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        for idx, sfeat in enumerate(self.feat_size):
            sk1 = scales[idx] / fig_size  # 将box的长宽转为[0-1]
            sk2 = scales[idx + 1] / fig_size
            sk3 = sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]  # aspect ratio为1:1的两个default box

            for alpha in aspect_ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
        
            # 计算当前feature map对应到原图上的所有default boxes
            for w, h in all_sizes:
                 for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))
            
        # 将default_boxes转为tensor形式
        self.dboxes = torch.as_tensor(self.default_boxes, dtype=torch.float32)
        self.dboxes.clamp_(min=0, max=1)  # 部分坐标值超出1，将这些值裁剪到1

        # 转换坐标表示[x, y, w, h] -> [xmin, ymin, xmax, ymax]
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]  # xmin
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]  # ymin
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]  # xmax
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]  # ymax
    
    # @property是把方法变为属性，防止属性被修改
    @property
    def scale_xy(self):
        return self.scale_xy_
    
    @property
    def scale_wh(self):
        return self.scale_wh_
    
    def __call__(self, order='ltrb'):
        if order == 'ltrb':
            return self.dboxes_ltrb
        
        if order == 'xywh':
            return self.dboxes

def dboxes300_coco():
    figsize=300  # 输入网络的图像大小
    feat_size = [38, 19, 10, 5, 3, 1]  # 每个预测层的feature map尺寸
    steps = [8, 16, 32, 64, 100, 300]  # 每个特征层上的一个cell在原图上的跨度，8代表1个cell跨越原图的8*8区域吗？
    scales = [21, 45, 99, 153, 207, 261, 315]  # 每个特征层上预测的default box的scale，Q：要重点理解？？？
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes

def nms(boxes, scores, iou_threshold):
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)

def nms(boxes, scores, iou_threshold):
    # for loop:
    #     寻找最大score的box
    #     将与最大score的box之间IoU超过iou_threshold的box的score置为.0
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)

def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Parameters
    ----------
    boxes : 候选边界框         torch.Size([nboxes, 4])
    scores: 候选边界框对应的概率 torch.Size([nboxes])
    idxs  : 候选边界框的类别    torch.Size([nboxes])
    
    Returns
    keep  : 
    """
    # 如果候选边界框数目为0则返回空, tensor([], dtype=torch.int64)
    if boxes.numel() == 0:
        torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    max_coordinate = boxes.max()  # 返回所有坐标值中的最大值, tensor([1.])
    # 为每个box分配1个偏移, 如果类别相同则偏移也相同, 确保不同类别内部进行nms.
    offsets = idxs.to(boxes) * (max_coordinate + 1)  # to()保证idxs和boxes的dtype一致
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep

class PostProcess(nn.Module):
    def __init__(self, dboxes):
        super(PostProcess, self).__init__()
        # [num_anchors, 4] -> [1, num_anchors, 4]
        self.dboxes_xywh = nn.Parameter(dboxes(order="xywh").unsqueeze(dim=0), requires_grad=False)
        
        self.scale_xy = dboxes.scale_xy  # 0.1
        self.scale_wh = dboxes.scale_wh  # 0.2

        self.criteria = 0.5
        self.max_output = 100

    def scale_back_batch(self, bboxes_in, scores_in):
        """
        bboxes_in: [delta_x, delta_y, delta_w, delta_h]格式
        1. 将bboxes_in的格式从[delta_x, delta_y, delta_w, delta_h]转换为[l, t, r, b]
        2. 对scores_in进行softmax回归, 注意是21个系数经过softmax又变成了21个类别概率
        """
        # Q: 为何调换维度?
        # torch.Size([batch, 4, 8732]) -> torch.Size([batch, 8732, 4])
        bboxes_in = bboxes_in.permute(0, 2, 1)
        # torch.Size([batch, 21, 8732]) -> torch.Size([batch, 8732, 21])
        scores_in = scores_in.permute(0, 2, 1)

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]

        # Q: 似乎为ssd_model.Loss._location_vec中的逆变换
        # self.scale_xy = 1.0 / dboxes.scale_xy
        # self.scale_wh = 1.0 / dboxes.scale_wh
        # gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :]
        # gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()  # Q: 猜测log是为了处理大小目标误差的不平衡
        # delta_x = loc_x - dbox_x
        # delta_x = delta_x / dbox_w
        # delta_x = delta_x * 5
        # ---------------------
        # b_x = b_x * 0.2
        # b_x = b_x * dbox_w
        # b_x = b_x + dbox_x
        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l
        bboxes_in[:, :, 1] = t
        bboxes_in[:, :, 2] = r
        bboxes_in[:, :, 3] = b

        return bboxes_in, F.softmax(scores_in, dim=-1)
    
    def decode_single_new(self, bboxes_in, scores_in, criteria, num_output):
        device = bboxes_in.device
        num_classes = scores_in.shape[-1]

        # 可能是为了防止在scale_back_batch中的转换发生了越界
        bboxes_in = bboxes_in.clamp(min=0, max=1)

        # torch.Size([8732, 4]) -> torch.Size([8732, 84]) ->torch.Size([8732, 21, 4])
        bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

        labels = torch.arange(num_classes, device=device)
        # torch.Size([num_classes]) -> torch.Size([8732, num_classes])
        labels = labels.view(1, -1).expand_as(scores_in)

        # 移除背景目标
        # Q: 为何移除背景目标, 是因为背景目标在inference时不需要关注?
        bboxes_in = bboxes_in[:, 1:, :]  # [8732, 21, 4] -> [8732, 20, 4]
        scores_in = scores_in[:, 1:]  # [8732, 21] -> [8732, 20]
        labels = labels[:, 1:]  # [8732, 21] -> [8732, 20]

        # bboxes_in:
        # tensor([[b1_l,      b1_t,      b1_r,      b1_b     ],
        #          ...,
        #         [b8732x4_l, b8732x4_t, b8732x4_r, b8732x4_b]])
        # scores_in:
        # tensor([s1, ...,  s8732x4])
        # labels_in:
        # tensor([l1, ...,  l8732x4])
        bboxes_in = bboxes_in.reshape(-1, 4)  # [8732, 20, 4] -> [8732 x 20, 4]
        scores_in = scores_in.reshape(-1)  # [8732, 20] -> [8732 x 20]
        labels = labels.reshape(-1) # [8732, 20] -> [8732 x 20]

        # 移除小概率目标
        inds = torch.where(torch.gt(scores_in, 0.05))[0]
        bboxes_in, scores_in, labels = bboxes_in[inds, :], scores_in[inds], labels[inds]

        # 移除空boxes
        ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
        keep = (ws >= 1 / 300) & (hs >= 1 / 300)  # Q: 在回归时的随机性导致的吗?
        keep = torch.where(keep)[0]
        bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]

        keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)

        keep = keep[:num_output]
        bboxes_out = bboxes_in[keep, :]
        scores_out = scores_in[keep]
        labels_out = labels[keep]

        return bboxes_out, labels_out, scores_out

    
    def forward(self, bboxes_in, scores_in):
        # 通过预测的boxes回归参数得到最终预测坐标, 将预测目标score通过softmax处理
        # bboxes_in: torch.Size([batch, 4, 8732])
        # scores_in: torch.Size([batch, 21, 8732])
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        # outputs:
        # [(boxes1, labels_1, scores_1), (boxes2, labels_2, scores_2), ..., 
        #  (boxe_batch_size, labels_batch_size, scores_batch_size)]
        # boxes_1  -> torch.Size([self.max_output, 4])
        # labels_1 -> torch.Size([self.max_output])
        # scores_1 -> torch.Size([self.max_output])
        outputs = torch.jit.annotate(List[Tuple[Tensor, Tensor, Tensor]], [])
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)  # torch.Size([1, 8732, 4]) -> torch.Size([8732, 4])
            prob = prob.squeeze(0)  # torch.Size([1, 8732, 21]) -> torch.Size([8732, 21])
            outputs.append(self.decode_single_new(bbox, prob, self.criteria, self.max_output))
        return outputs