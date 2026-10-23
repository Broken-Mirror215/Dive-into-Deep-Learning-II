import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


def get_params(vocab_size, num_hiddens, device):
    """初始化 RNN 的五个参数张量，并开启梯度跟踪。"""
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def main():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    num_hiddens = 512
    device = d2l.try_gpu()
    params = get_params(len(vocab), num_hiddens, device)

    print("词表大小:", len(vocab))
    print("隐藏单元数:", num_hiddens)
    print("设备:", device)
    for name, param in zip(("W_xh", "W_hh", "b_h", "W_hq", "b_q"), params):
        print(f"{name}: shape={tuple(param.shape)}, "
              f"requires_grad={param.requires_grad}")


if __name__ == "__main__":
    main()
