

import torch
from d2l import torch as d2l


# 9.2.2.1 初始化模型参数
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        # 输入权重、隐状态权重和偏置，各门分别拥有自己的参数。
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # 输入门：写入多少新信息
    W_xf, W_hf, b_f = three()  # 遗忘门：保留多少旧记忆
    W_xo, W_ho, b_o = three()  # 输出门：输出多少记忆信息
    W_xc, W_hc, b_c = three()  # 候选记忆元：准备写入的新信息

    # 将隐状态映射成词表中每个字符的预测分数（logits）。
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o,
              W_xc, W_hc, b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# 9.2.2.2 定义模型
def init_lstm_state(batch_size, num_hiddens, device):
    # H 是隐状态，C 是记忆元；两者的形状均为 (批量大小, 隐藏单元数)。
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))


def lstm(inputs, state, params):
    (W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o,
     W_xc, W_hc, b_c, W_hq, b_q) = params
    H, C = state
    outputs = []

    # RNNModelScratch 已将字符索引转成独热向量，并把时间步放在第一维。
    # inputs: (时间步数, 批量大小, 词表大小)，X: (批量大小, 词表大小)。
    for X in inputs:
        I = torch.sigmoid(X @ W_xi + H @ W_hi + b_i)
        F = torch.sigmoid(X @ W_xf + H @ W_hf + b_f)
        O = torch.sigmoid(X @ W_xo + H @ W_ho + b_o)
        C_tilda = torch.tanh(X @ W_xc + H @ W_hc + b_c)

        # * 是逐元素乘法，@ 是矩阵乘法。
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)

        Y = H @ W_hq + b_q
        outputs.append(Y)

    # 预测形状：(时间步数 * 批量大小, 词表大小)。
    # H 和 C 一起交给下一个批次；训练器负责按采样方式重置或分离状态。
    return torch.cat(outputs, dim=0), (H, C)


# 9.2.2.3 训练与预测
def main(num_epochs=500, show_plot=True):
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    lr = 1

    model = d2l.RNNModelScratch(vocab_size, num_hiddens, device,
                              get_lstm_params, init_lstm_state, lstm)
    # 复用第 8 章的交叉熵损失、梯度裁剪、SGD 和字符预测流程。
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

    # 书中使用 Notebook；独立脚本显式显示训练曲线。
    if show_plot:
        d2l.plt.show()


if __name__ == "__main__":
    main()
