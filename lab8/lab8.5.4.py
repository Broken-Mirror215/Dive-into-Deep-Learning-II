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


def init_rnn_state(batch_size, num_hiddens, device):
    # 状态统一放在元组中，供 rnn 中的 H, = state 解包。
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn(inputs, state, params):
    """按时间步更新隐藏状态，并返回所有时间步的输出和最终状态。"""
    # inputs 的形状：（时间步数，批量大小，词表大小）。
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        # 当前输入乘 W_xh，上一隐藏状态乘 W_hh。
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    # 每一步的输出为（批量大小，词表大小），沿第 0 维依次拼接。
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch:
    """把参数初始化、隐藏状态初始化和前向计算组合成一个模型。"""

    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        # （批量大小，时间步数）→（时间步数，批量大小，词表大小）。
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


@torch.no_grad()
def predict_ch8(prefix, num_preds, net, vocab, device):
    """保留给定前缀，并在后面逐步生成 num_preds 个词元。"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    # 预热：用真实前缀更新隐藏状态，保留前缀中的实际字符。
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])

    # 生成：每次把上一个字符作为输入，选出分数最高的下一个词元。
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(y.argmax(dim=1).item())

    # outputs 保存的是索引，需要反向查表得到字符。
    return ''.join(vocab.idx_to_token[i] for i in outputs)


def main():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    num_hiddens = 512
    device = d2l.try_gpu()
    net = RNNModelScratch(len(vocab), num_hiddens, device,
                          get_params, init_rnn_state, rnn)

    prefix, num_preds = 'time traveller ', 10
    print("前缀:", repr(prefix))
    print("生成结果:", predict_ch8(prefix, num_preds, net, vocab, device))
    print("当前模型尚未训练，生成结果暂时没有语言意义。")


if __name__ == "__main__":
    main()
