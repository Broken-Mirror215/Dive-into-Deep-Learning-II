from pathlib import Path
from runpy import run_path

import torch
from torch import nn


# 文件名含有点，按路径读取前两节定义的模块。
DenseBlock = run_path(str(Path(__file__).with_name("lab7.7.2.py")))["DenseBlock"]
transition_block = run_path(str(Path(__file__).with_name("lab7.7.3.py")))["transition_block"]


b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)

num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    num_channels += num_convs * growth_rate
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels //= 2

net = nn.Sequential(
    b1,
    *blks,
    nn.BatchNorm2d(num_channels),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10),
)


if __name__ == "__main__":
    X = torch.rand(1, 1, 96, 96)
    net.eval()
    with torch.no_grad():
        for layer in net:
            X = layer(X)
            print(layer.__class__.__name__, "output shape:", X.shape)
