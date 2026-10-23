import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


def main():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    print("词表大小:", len(vocab))

    # 每个索引变成一个长度为词表大小的向量，只有对应位置为 1。
    indices = torch.tensor([0, 2])
    print("索引 0 和 2 的独热编码:")
    print(F.one_hot(indices, len(vocab)))

    # 教材示例：先转置，再进行独热编码，以便按时间步遍历输入。
    X = torch.arange(10).reshape((2, 5))
    print("\n示例 X:", X)
    print("X 的形状（批量大小，时间步数）:", X.shape)
    print("X.T 的形状（时间步数，批量大小）:", X.T.shape)
    print("独热编码后的形状:", F.one_hot(X.T, len(vocab)).shape)

    # 对真实批次做同样的转换，浮点类型用于后续与模型权重做矩阵乘法。
    X, Y = next(iter(train_iter))
    inputs = F.one_hot(X.T, len(vocab)).type(torch.float32)
    print("\n真实批次 X、Y 的形状:", X.shape, Y.shape)
    print("RNN 输入形状（时间步数，批量大小，词表大小）:", inputs.shape)
    print("RNN 输入的数据类型:", inputs.dtype)


if __name__ == "__main__":
    main()
