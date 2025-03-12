import torch
from torch import nn
import numpy

x = numpy.array([1,2,3,4,5])
class MyLayer(nn.Module):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()

layer = MyLayer()
# 保存计算结果到变量
result = layer(torch.tensor([1., 2., 3., 4., 5.]))
# 输出计算结果
print(layer)
print(result)

class MyListDense(nn.Module):
    def __init__(self):
        super(MyListDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x
net = MyListDense()
print(net)