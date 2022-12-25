import torch
from torch import nn, Tensor
from torch.jit.annotations import List

from res50_backbone import resnet50

net = resnet50()


print(type( *list(net.children()) ) )

location_extractors = []

location_extractors.append(nn.Conv2d(3, 4, kernel_size=3, padding=1))
location_extractors.append(nn.Conv2d(5, 6, kernel_size=3, padding=1))
location_extractors.append(nn.Conv2d(7, 8, kernel_size=3, padding=1))

print(location_extractors)

location_extractors = nn.ModuleList(location_extractors)

print(location_extractors)