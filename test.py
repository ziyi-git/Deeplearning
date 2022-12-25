import torch
from torch import nn, Tensor
from torch.jit.annotations import List

from res50_backbone import resnet50

net = resnet50()


print(*list(net.children())[:7])