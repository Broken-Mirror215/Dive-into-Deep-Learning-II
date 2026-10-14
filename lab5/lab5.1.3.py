import torch
from torch import nn
from torch.nn import functional as F

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数，因此在训练期间保持不变
        self.rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20) # 全连接层 对输入进行乘权重 再加偏置。


    def forward(self, x):
        X = self.linear(x)
        # 使用创建的常量参数和relu和mm函数
        X = F.relu(torch.mm(X,self.rand_weight)+1)
        # 复用全连接层相当于这两个全连接层共享参数
        X = self.linear(X)

        # 控制流
        while X.abs().sum() >1:
            X /= 2
        return X.sum()

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),
                                 nn.Linear(64,32),nn.ReLU(),)
        self.linear = nn.Linear(32, 16)

    def forward(self,X):
        return self.linear(self.net(X))

X = torch.rand(2, 20)
chimera = nn.Sequential(NestMLP(), nn.Linear(16,20), FixedHiddenMLP())
print(chimera(X))
