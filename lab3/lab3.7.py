import torch
from torch import nn
from d2l import torch as d2l


# 3.7 softmax 回归的简洁实现
# 与 3.6 相同：使用 Fashion-MNIST，每张图片 28×28，共 10 个类别。
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# 3.7.1 初始化模型参数
# Sequential 按顺序执行其中的层。
# Flatten 保留批量维度，把 (批量大小, 1, 28, 28) 变成 (批量大小, 784)。
# Linear 为每张图片计算 10 个类别分数，输出形状为 (批量大小, 10)。
# 它相当于 3.6 的 X @ W + b，但内部 weight 的形状为 (10, 784)，
# 实际计算 X @ weight.T + bias。
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        # 沿用 3.6 的初始化方式，把偏置设为 0。
        nn.init.zeros_(m.bias)


# 对网络中的各个模块调用初始化函数。
net.apply(init_weights)


# 3.7.2 定义损失函数
# 重点：这里 net 输出的是原始类别分数（logits），还没有转成概率！
# CrossEntropyLoss 接收分数和真实标签，以数值稳定的方式完成
# log-softmax 和负对数似然计算，因此不要在它前面再加 softmax。
# reduction='none' 表示返回每张图片各自的损失，形状为 (批量大小,)。
loss = nn.CrossEntropyLoss(reduction='none')


# 3.7.3 定义优化算法
# net.parameters() 提供网络中的可训练参数，包括 Linear 的权重和偏置。
# 这替代了 3.6 中手写的 updater 和 d2l.sgd。
lr = 0.1
trainer = torch.optim.SGD(net.parameters(), lr=lr)


# 3.7.4 训练模型
# train_ch3 检测到 PyTorch 优化器后，每个批次执行：
#   l = loss(net(X), y)
#   trainer.zero_grad()   # 清空上一批的梯度
#   l.mean().backward()   # 对平均损失求梯度
#   trainer.step()        # 更新权重和偏置
# 10 个 epoch 表示遍历整个训练集 10 遍，每个批次都会更新参数。
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

print("训练后的测试准确率：", d2l.evaluate_accuracy(net, test_iter))

# 只判断类别时，直接对分数取 argmax 即可，softmax 不改变大小顺序。
# 如果需要查看每个类别的概率，可以使用：
# probabilities = torch.softmax(net(X), dim=1)

d2l.plt.show()
