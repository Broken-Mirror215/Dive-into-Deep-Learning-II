import torch
import torch.nn.functional as F
from torch import nn


class MyLinear(nn.Module):
    # 初始化使用的线性模型 y = wx+b这样子
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self,X):
        linear = torch.matmul(X,self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5,3)
print(linear.weight)
print(linear(torch.randn(2,5)))
