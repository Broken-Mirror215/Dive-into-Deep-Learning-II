import torch
from torch import nn

def comp_conv2d(conv2d, X):
    # 这里的1 , 1表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape) # 1 1 8 8
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])

# 这里每一边都加了一行/列，利用的是nn的最后一个参数，第三个参数是卷积核心大小喵。
conv2d = nn.Conv2d(1,1,kernel_size=3,padding=1)
X = torch.rand(size=(8,8))
print(comp_conv2d(conv2d,X).shape)
3
