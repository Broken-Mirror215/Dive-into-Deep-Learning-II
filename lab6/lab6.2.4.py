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


X =torch.ones((6,8))
X[:,2:6] = 0
#print(X)

K = torch.tensor([[1.0,-1.0]])
Y = corr2d(X,K)
#print(Y)

# print(corr2d(X.t(),K))



# 构造一个二维的卷积层，
conv2d = nn.Conv2d(1,1,kernel_size=(1,2),bias = False)

# 这个二维卷积层使用四维输入和输出格式，(批量大小，通道，高度，宽度）
X = X.reshape((1,1,6,8))
Y = Y.reshape((1,1,6,7))

lr =3e-2 # 这是学习率

for i in range (10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    with torch.no_grad():
        conv2d.weight -= lr * conv2d.weight.grad
    if (i+1)%2==0:
        print(f'epoch {i+1}, loss {l.sum():.3f}',)

print(conv2d.weight.detach().reshape(1, 2))
