from pathlib import Path
from runpy import run_path

import torch
from torch import nn


Residual = run_path(str(Path(__file__).with_name("lab7.6.2.py")))["Residual"]


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b2 = nn.Sequential(*resnet_block(64, 64, num_residuals=2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, num_residuals=2))
b4 = nn.Sequential(*resnet_block(128, 256, num_residuals=2))
b5 = nn.Sequential(*resnet_block(256, 512, num_residuals=2))
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(512, 10))


if __name__ == "__main__":
    X = torch.rand(1, 1, 224, 224)
    net.eval()
    with torch.no_grad():
        for layer in net:
            X = layer(X)
            print(layer.__class__.__name__, "output shape:", X.shape)
