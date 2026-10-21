import torch
from torch import nn


def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1),
    )


class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super().__init__()
        self.net = nn.Sequential(*(
            conv_block(input_channels + i * num_channels, num_channels)
            for i in range(num_convs)
        ))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 保留旧特征，沿通道维加入新特征
        return X


if __name__ == "__main__":
    X = torch.randn(4, 3, 8, 8)
    blk = DenseBlock(num_convs=2, input_channels=3, num_channels=10)
    print("稠密块输出形状:", blk(X).shape)
