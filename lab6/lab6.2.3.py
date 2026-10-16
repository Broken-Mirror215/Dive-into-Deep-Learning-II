import torch
from torch import nn


def corr2d(X, K):  # @ save
    # """二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        return corr2d(x,self.weight) + self.bias


X =torch.ones((6,8))
X[:,2:6] = 0
print(X)

K = torch.tensor([[1.0,-1.0]])
Y = corr2d(X,K)
print(Y)

print(corr2d(X.t(),K))