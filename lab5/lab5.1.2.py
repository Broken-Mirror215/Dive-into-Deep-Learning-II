import torch
from torch import nn

class MySequential(nn.Module):
    def __init__(self,*args):
        super().__init__()
        for idx,module in enumerate(args):
            # 按顺序注册子模块，让 PyTorch 能管理它们的参数。
            self._modules[str(idx)] = module

    def forward(self, x):
        for block in self._modules.values():
            # 更新 x，让下一层接收当前层的输出。
            x = block(x)
        return x
    


net = MySequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

X = torch.rand(2, 20)
print(net)
print('输出形状:', net(X).shape)
