import random

import torch
import torchvision.transforms as t
from torchvision.transforms import functional as F

from src import dboxes300_coco, calc_iou_tensor, Encoder

class Compose(object):
    """
    组合多个transform函数
    """
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target=None):
        for trans in self.transforms:
            image, target = trans(image, target)
        return image, target

class ToTensor(object):
    """
    将PIL图像转为Tensor
    Q: 为何没有__init__函数
    """
    def __call__(self, image, target):
        # torch.as_tensor()或者torch.tensor()无法推断image(type为PIL.Image.Image)的dtype
        # 关于contiguous:
        # 1. 约定tensor对象在torch的底层中是连续存储的, 存储于storage
        #    t1 = torch.arange(12).reshape(3, 4):
        #    tensor([[ 0,  1,  2,  3],
        #            [ 4,  5,  6,  7],
        #            [ 8,  9, 10, 11]])
        #    t1.storage():
        #    0 1 2 3 4 5 6 7 8 9 10 11
        #    t1.stride():
        #    (4, 1) 访问下一行相邻元素offset为4, 访问下一列相邻元素offset为1
        # 2. 转置操作生成的新变量仍共享storage, 但会拥有自己的offset
        #    t2 = t1.transpose(0, 1):
        #    tensor([[ 0,  4,  8],
        #            [ 1,  5,  9],
        #            [ 2,  6, 10],
        #            [ 3,  7, 11]])
        #    t2.stride():
        #    (1, 4) 访问下一行相邻元素offset为1, 访问下一列相邻元素offset为4
        #    t1.data_ptr() == t2.data_ptr():
        #    True
        # 3. 此时使用view()访问t2会报错, 原因是view要求t2中行内连续元素的地址也是连续的, 要求offset为1但实际不连续如0和4的offset为4
        # 4. 使用contiguous()重新开辟一块内存, 使用新的stride和新的内存排列顺序
        #    t3 = t1.transpose(0, 1).contiguous():
        #    tensor([[ 0,  4,  8],
        #            [ 1,  5,  9],
        #            [ 2,  6, 10],
        #            [ 3,  7, 11]])
        #    t3.stride:
        #    (3, 1)
        #    t1.data_ptr() == t3.data_ptr():
        #    False
        # 参考:
        # https://zhuanlan.zhihu.com/p/64551412
        # https://www.jianshu.com/p/ebd7f6395bf4
        image = F.to_tensor(image).contiguous()
        return image, target

class RandomHorizontalFlip(object):
    """
    随机水平翻转图像以及bboxes, 该方法应放在ToTensor之后
    Q: 为何只做水平翻转不做其他翻转？
    """
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        # 不是每一张图像都翻转，这里是随机翻转
        if random.random() < self.prob:
            image = image.flip(-1)  # 水平翻转
            bbox = target["boxes"]
            bbox[:, [0, 2]] = 1.0 - bbox[:, [2, 0]]
            target["bboxes"] = bbox
        return image, target

class SSDCropping(object):
    def __init__(self):
        self.sample_options = (
            None,         # do nothing
            (0.1, None),  # min IoU, max IoU
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),  # no IoU requirements
        )
        self.dboxes = dboxes300_coco()
    
    def __call__(self, image, target):
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, target
            
            htot, wtot = target["height_width"]

            min_iou, max_iou = mode
            min_iou = float("-inf") if min_iou is None else min_iou
            max_iou = float("+inf") if max_iou is None else max_iou

            # 重复5次确保获得合适的w/h
            for _ in range(5):
                w = random.uniform(0.3, 1.0)
                h = random.uniform(0.3, 1.0)

                if w/h < 0.5 or w/h > 2:
                    continue
                
                # 确保裁剪区域不超过图像边界
                left = random.uniform(0, 1.0 - w)
                top = random.uniform(0, 1.0 - h)
                right = left + w
                bottom = top + h

                # 裁剪区域和bboxes中每个box的IoU
                bboxes = target["boxes"]  # torch.Size([nboxes, 4])
                # ious[m, 1]: bboxes第m个box与torch.tensor([[left, top, right, bottom]]的IoU
                ious = calc_iou_tensor(bboxes, torch.tensor([[left, top, right, bottom]]))  # torch.Size([nbboxes, 1])

                # 所有的IoU均不满足条件则重新进入循环
                if not ((ious > min_iou) & (ious < max_iou)).all():
                    continue

                # 所有box的中心坐标
                xc = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
                yc = 0.5 * (bboxes[:, 1] + bboxes[:, 3])

                # 如果全部box的中心坐标均不在裁剪区域内则重新进入循环
                masks = (xc > left) & (xc < right) & (yc > top) & (yc < bottom)
                if not masks.any():
                    continue

                # 如果box顶点在裁剪区域外则收缩对齐到裁剪区域顶点处
                bboxes[bboxes[:, 0] < left, 0] = left
                bboxes[bboxes[:, 1] < top, 1] = top
                bboxes[bboxes[:, 2] > right, 2] = right
                bboxes[bboxes[:, 3] > bottom, 3] = bottom

                # 仅保留中心坐标位于裁剪区域内的box
                bboxes = bboxes[masks, :]
                labels = target["labels"]
                labels = labels[masks]

                # 裁剪
                left_idx = int(left * wtot)
                top_idx = int(top * htot)
                right_idx = int(right * wtot)
                bottom_idx = int(bottom * htot)
                image = image.crop((left_idx, top_idx, right_idx, bottom_idx))

                # 将box的坐标由原图像转换至裁剪区域内
                bboxes[:, 0] = (bboxes[:, 0] - left) / w
                bboxes[:, 1] = (bboxes[:, 1] - top) / h
                bboxes[:, 2] = (bboxes[:, 2] - left) / w
                bboxes[:, 3] = (bboxes[:, 3] - top) / h

                target['boxes'] = bboxes
                target['labels'] = labels

                return image, target

class Resize(object):
    """
    对图像进行resize，在ToTensor之前进行。
    Q: 为何在ToTensor之前？
    """
    def __init__(self, size=(300, 300)):
        self.resize = t.Resize(size)

    def __call__(self, image, target):
        image = self.resize(image)
        return image, target

class ColorJitter(object):
    """
    在ToTensor之前进行
    """
    def __init__(self, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05):
        self.trans = t.ColorJitter(brightness, contrast, saturation, hue)
    
    def __call__(self, image, target):
        image = self.trans(image)
        return image, target

class Normalization(object):
    """
    ToTensor之后
    """
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.normalize = t.Normalize(mean=mean, std=std)
    
    def __call__(self, image, target):
        image = self.normalize(image)
        return image, target

class AssignGTtoDefaultBox(object):
    def __init__(self):
        self.default_box = dboxes300_coco()
        self.encoder = Encoder(self.default_box)  # Q: Encoder到底在干什么？
    
    def __call__(self, image, target):
        boxes = target['boxes']
        labels = target['labels']
        bboxes_out, labels_out = self.encoder.encode(boxes, labels)
        target['boxes'] = bboxes_out
        target['labels'] = labels_out

        return image, target