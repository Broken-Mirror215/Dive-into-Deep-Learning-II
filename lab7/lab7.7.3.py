from pathlib import Path
from runpy import run_path

import torch
from torch import nn


def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2),
    )


if __name__ == "__main__":
    DenseBlock = run_path(str(Path(__file__).with_name("lab7.7.2.py")))["DenseBlock"]
    X = torch.randn(4, 3, 8, 8)
    Y = DenseBlock(2, 3, 10)(X)
    print("过渡层输出形状:", transition_block(23, 10)(Y).shape)
