import torch
from torch import nn
import numpy

class MyLayer(nn.Module):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()

layer = MyLayer()
# 保存计算结果到变量
result = layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))
# 输出计算结果
print(result)
