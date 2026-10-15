import torch
from torch import nn


net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print("网络输出：")
print(net(X))

print("\n第二个全连接层的参数：")
print(net[2].state_dict())

# 这部分是查看目标参数
print("\n偏置的类型、参数对象和底层数值：")
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

print("\n权重梯度是否为 None：")
print(net[2].weight.grad is None)


# 这部分是一次性访问所有参数
print("\n第一个全连接层的参数名和形状：")
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print("整个网络的参数名和形状：")
print(*[(name, param.shape) for name, param in net.named_parameters()])

print("\n通过参数名访问第二个全连接层的偏置：")
print(net.state_dict()['2.bias'].data)



# 嵌套块，其中Sequential 是组织层和层之间的输入输出有序性。
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print("\n嵌套网络的输出：")
print(rgnet(X))

print("\n嵌套网络的结构：")
print(rgnet)

print("\n第二个小块中第一个全连接层的偏置：")
print(rgnet[0][1][0].bias.data)
