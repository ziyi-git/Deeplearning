# # import torch
# # from torch import nn, Tensor
# # from torch.jit.annotations import List

# # from res50_backbone import resnet50

# # # net = resnet50()


# # # print(type( *list(net.children()) ) )

# # # location_extractors = []

# # # location_extractors.append(nn.Conv2d(3, 4, kernel_size=3, padding=1))
# # # location_extractors.append(nn.Conv2d(5, 6, kernel_size=3, padding=1))
# # # location_extractors.append(nn.Conv2d(7, 8, kernel_size=3, padding=1))

# # # print(location_extractors)

# # # location_extractors = nn.ModuleList(location_extractors)

# # # print(location_extractors)

# # # t = torch.tensor(
# # #     [
# # #         [1, 2, 3],
# # #         [4, 5, 6]
# # #     ]
# # #     )

# # # t1 = t.transpose(0, 1)

# # # print('t.shape: ', t.shape)
# # # print('t.view(): ', t1.view(-1))

# # import itertools
# # import numpy as np
# # fig_size = 300
# # fk = fig_size / np.array([8, 16, 32, 64, 100, 300])
# # idx = 0
# # for i, j in itertools.product(range(38), repeat=2):
# #     # print(i, j)
# #     cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
# #     print(cx)

# fig_size = 300
# feat_size = [38, 19, 10, 5, 3, 1]

# # self.scale_xy_ = scale_xy
# # self.scale_wh_ = scale_wh

# steps = [8, 16, 32, 64, 100, 300]
# scales = [21, 45, 99, 153, 207, 261, 315]

# import numpy as np
# from math import sqrt
# import itertools
# # 300 / [8, 16, 32, 64, 100, 300] = [37.5, 18.75, 9.375, 4.6875,  3.,  1.]
# fk = fig_size / np.array(steps)  # Q：初步猜测是将feature map的尺寸归一化
# aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

# default_boxes = []
# for idx, sfeat in enumerate(feat_size):
#     sk1 = scales[idx] / fig_size
#     sk2 = scales[idx + 1] / fig_size
#     sk3 = sqrt(sk1 * sk2)
#     all_sizes = [(sk1, sk1), (sk3, sk3)]

#     for alpha in aspect_ratios[idx]:
#         w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
#         all_sizes.append((w, h))
#         all_sizes.append((h, w))
    
#     # print('all_sizes: ', all_sizes)
#             # 计算当前特征层对应原图上的所有default box
#     for w, h in all_sizes:
#         for i, j in itertools.product(range(sfeat), repeat=2):  # i -> 行（y）， j -> 列（x）
#             # 计算每个default box的中心坐标（范围是在0-1之间）
#             cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
#             default_boxes.append((cx, cy, w, h))
#             # print((cx, cy))
#             print((i, j))
#         exit()

# # print('len(default_boxes): {}'.format(len(default_boxes)))
# # print('fk: ', fk)

# class X(object):
#     def __init__(self, x=0.5):
#         self.x = x
    
#     def __call__(self, y):
#         z = y + self.x
#         return z

# a = X(x=0.3)
# b = a(0.6)
# print(b)

# import transforms
# from my_dataset import VOCDataSet
# VOC_root = "/Users/liuziyi/Workspace/Data/VOCdevkit"
# data_transform = {
#     "train": transforms.Compose([
#         transforms.SSDCropping(),
#         transforms.Resize(),
#         transforms.ColorJitter(),
#         transforms.ToTensor(),
#         transforms.RandomHorizontalFlip(),
#         transforms.Normalization(),
#         transforms.AssignGTtoDefaultBox()
#     ]),
#     "val": transforms.Compose([
#         transforms.Resize(),
#         transforms.ToTensor(),
#         transforms.Normalization()
#     ])
# }
# train_dataset = VOCDataSet(VOC_root, '2012', data_transform["train"], train_set='train.txt')
# image, target = train_dataset[0]
# print("done")

# from src import resnet50
# net = resnet50()
# y = net(image[None, :, :])
# print("done")

# ------------------------------------------------create_model----------------------------------------------------------
# from train_ssd300 import create_model
# import transforms
# from my_dataset import VOCDataSet
# import torch

# VOC_root = "/Users/liuziyi/Workspace/Data/VOCdevkit"
# data_transform = {
#     "train": transforms.Compose([
#         transforms.SSDCropping(),
#         transforms.Resize(),
#         transforms.ColorJitter(),
#         transforms.ToTensor(),
#         transforms.RandomHorizontalFlip(),
#         transforms.Normalization(),
#         transforms.AssignGTtoDefaultBox()
#     ]),
#     "val": transforms.Compose([
#         transforms.Resize(),
#         transforms.ToTensor(),
#         transforms.Normalization()
#     ])
# }
# train_dataset = VOCDataSet(VOC_root, '2012', data_transform["train"], train_set='train.txt')
# train_data_loader = torch.utils.data.DataLoader(train_dataset,
#                                                 batch_size=4,
#                                                 shuffle=True,
#                                                 num_workers=0,
#                                                 collate_fn=train_dataset.collate_fn,
#                                                 drop_last=True)

# for x in train_data_loader:
#     print('x')
# # image, target = train_dataset[0]


# device = torch.device("cpu")
# model = create_model()
# model.to(device)
# model.eval()
# y = model(image[None, :, :, :], targets=None)
# print("done")

# if __name__ == "__main__":
from train_ssd300 import main
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.device = "cpu"
parser.num_classes = 20
parser.data_path = "/Users/liuziyi/Workspace/Data"
parser.output_dir = "./"
parser.resume = ""
parser.start_epoch = 0
parser.epochs = 15
parser.batch_size = 4
main(parser)
print("done")


# # 给定一个default box的[xmin, ymin, xmax, ymax], 生成一个给定xmin和ymin的box，要求IoU为0.5
# import numpy as np
# # default_box = [0.3, 0.3, 0.8, 0.8]
# # xmin = 0.4
# # ymin = 0.5
# # xmax_candidates = np.arange(0.4001, 1.0, 0.001)
# # ymax_candidates = np.arange(0.5001, 1.0, 0.001)
# # for xmax in xmax_candidates:
# #     for ymax in ymax_candidates:
# #         l = max(xmin, default_box[0])
# #         t = max(ymin, default_box[1])
# #         r = min(xmax, default_box[2])
# #         b = min(ymax, default_box[3])
# #         inter = (r - l) * (b - t)
# #         iou = inter / ((default_box[2] - default_box[0]) * (default_box[3] - default_box[1]) + (xmax - xmin) * (ymax - ymin) - inter)
# #         if abs(iou - 0.5) < 0.01:
# #             print("l t r b: ",  l, t, r, b)
# #             break

# # 
# xmin = 0.37
# ymin = 0.41
# dboxes = [0.3, 0.3, 0.8, 0.8]

# xmax_candidates = np.arange(xmin, 1.0, 0.0001)
# xmax_candidates = np.stack( (np.zeros(len(xmax_candidates)), np.zeros(len(xmax_candidates)), xmax_candidates, np.zeros(len(xmax_candidates))), axis=0)

# ymax_candidates = np.arange(ymin, 1.0, 0.0001)
# ymax_candidates = np.stack( (np.zeros(len(ymax_candidates)), np.zeros(len(ymax_candidates)), np.zeros(len(ymax_candidates)), ymax_candidates), axis=0)

# plocs = xmax_candidates[:, None, :] + ymax_candidates[:, :, None]
# plocs[0, :, :] = xmin
# plocs[1, :, :] = ymin

# # dboxes = np.array([0.3, 0.3, 0.8, 0.8]).reshape(4, 1, 1)
# dboxes = np.array(dboxes).reshape(4, 1, 1)
# dboxes = dboxes.repeat(plocs.shape[1], axis=1)
# dboxes = dboxes.repeat(plocs.shape[2], axis=2)

# delta = np.zeros(dboxes.shape)
# delta[:2, :, :] = np.maximum(plocs[:2, :, :], dboxes[:2, :, :])
# delta[2:, :, :] = np.minimum(plocs[2:, :, :], dboxes[2:, :, :])

# area_d = (dboxes[2 ,:, :] - dboxes[0 ,:, :]) * (dboxes[3 ,:, :] - dboxes[1 ,:, :])
# area_p = (plocs[2 ,:, :] - plocs[0 ,:, :]) * (plocs[3 ,:, :] - plocs[1 ,:, :])
# inter = (delta[2 ,:, :] - delta[0 ,:, :]) * (delta[3 ,:, :] - delta[1 ,:, :])
# ious = inter / (area_d + area_p - inter)
# (x, y) = np.where(np.abs(ious - 0.5) < 0.01)
# print(plocs[:, x, y])
# print(dboxes[:, x, y])


# # xmin = 0.31
# # ymin = 0.31
# # dboxes = [0.3, 0.3, 0.55, 0.55]
# # d = [0.3, 0.3, 0.55, 0.55]
# # p = [0.31, 0.33, 0.5499, 0.4577]
# # g = [0.34, 0.32, 0.55, 0.4659]
# # d_xywh = np.array([(d[0] + d[2]) / 2.0, (d[1] + d[3]) / 2.0, d[2] - d[0], d[3] - d[1]])
# # p_xywh = np.array([(p[0] + p[2]) / 2.0, (p[1] + p[3]) / 2.0, p[2] - p[0], p[3] - p[1]])
# # g_xywh = np.array([(g[0] + g[2]) / 2.0, (g[1] + g[3]) / 2.0, g[2] - g[0], g[3] - g[1]])
# # np.abs((d_xywh - p_xywh) - (d_xywh - g_xywh)).sum() 0.06405000000000005
# # pd = np.array([10.0 * (p_xywh[0] - d_xywh[0]) / d_xywh[2], 10.0 * (p_xywh[1] - d_xywh[3]) / d_xywh[3], 5.0 * np.log(p_xywh[2] / d_xywh[2]), 5.0 * np.log(p_xywh[3] / d_xywh[3])])
# # gd = np.array([10.0 * (g_xywh[0] - d_xywh[0]) / d_xywh[2], 10.0 * (g_xywh[1] - d_xywh[3]) / d_xywh[3], 5.0 * np.log(g_xywh[2] / d_xywh[2]), 5.0 * np.log(g_xywh[3] / d_xywh[3])])
# # np.abs(pd - gd).sum() 1.9697616580921358

# # d = [0.3, 0.3, 0.8, 0.8]
# # p = [0.4, 0.4, 0.8, 0.7063]
# # g = [0.37, 0.41, 0.8105, 1]
# # d_xywh = np.array([(d[0] + d[2]) / 2.0, (d[1] + d[3]) / 2.0, d[2] - d[0], d[3] - d[1]])
# # p_xywh = np.array([(p[0] + p[2]) / 2.0, (p[1] + p[3]) / 2.0, p[2] - p[0], p[3] - p[1]])
# # g_xywh = np.array([(g[0] + g[2]) / 2.0, (g[1] + g[3]) / 2.0, g[2] - g[0], g[3] - g[1]])
# # np.abs((d_xywh - p_xywh) - (d_xywh - g_xywh)).sum() 0.4858000000000001
# # pd = np.array([10.0 * (p_xywh[0] - d_xywh[0]) / d_xywh[2], 10.0 * (p_xywh[1] - d_xywh[3]) / d_xywh[3], 5.0 * np.log(p_xywh[2] / d_xywh[2]), 5.0 * np.log(p_xywh[3] / d_xywh[3])])
# # gd = np.array([10.0 * (g_xywh[0] - d_xywh[0]) / d_xywh[2], 10.0 * (g_xywh[1] - d_xywh[3]) / d_xywh[3], 5.0 * np.log(g_xywh[2] / d_xywh[2]), 5.0 * np.log(g_xywh[3] / d_xywh[3])])
# # np.abs(pd - gd).sum() 6.992017106646442