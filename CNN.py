import torch
from torch import nn

class MyLayer(nn.Module):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
    def forward(self, x):
        return x - x.mean()

layer = MyLayer()
layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))