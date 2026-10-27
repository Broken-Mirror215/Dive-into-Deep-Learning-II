"""《动手学深度学习》9.6.2：解码器。

教材：https://zh.d2l.ai/chapter_recurrent-modern/encoder-decoder.html
保留教材的通用接口，并通过小张量示例演示调用方式。
"""

import torch
from torch import nn


class Decoder(nn.Module):
    """解码器接口：先初始化状态，再根据输入和状态计算输出。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        # 将编码器输出转换为解码器需要的状态，可结合有效长度等信息。
        raise NotImplementedError('解码器子类需要实现 init_state 方法')

    def forward(self, X, state):
        # 具体实现通常返回 (输出, 新状态)，供后续时间步继续解码。
        raise NotImplementedError('解码器子类需要实现 forward 方法')


class DemoDecoder(Decoder):
    """接口演示：把编码上下文加到每个解码输入时间步。"""

    def init_state(self, enc_outputs, *args):
        # 此示例直接把编码结果作为状态；实际模型可以构造更复杂的状态。
        return enc_outputs

    def forward(self, X, state):
        # X: (批量大小, 目标序列长度, 特征数)。
        # state: (批量大小, 特征数)，增加时间维后广播到各时间步。
        output = X + state.unsqueeze(1)
        # 此示例保持上下文不变；循环神经网络解码器会逐步更新隐藏状态。
        return output, state


def main():
    decoder = DemoDecoder()
    # 假设编码器为两个样本各给出一个 4 维上下文向量。
    enc_outputs = torch.tensor([[1., 2., 3., 4.],
                                [5., 6., 7., 8.]])
    state = decoder.init_state(enc_outputs)
    # 用 3 个时间步的浮点特征演示解码器输入。
    X = torch.zeros((2, 3, 4))
    output, new_state = decoder(X, state)

    print('本节演示 init_state → forward 的调用过程。')
    print('编码器输出形状：', tuple(enc_outputs.shape))
    print('解码器初始状态形状：', tuple(state.shape))
    print('解码器输入形状：', tuple(X.shape))
    print('解码器输出形状：', tuple(output.shape))
    print('返回状态形状：', tuple(new_state.shape))
    print('输出：\n', output)

    # 同一个接口也可以每次只处理一个时间步，并接收上次返回的状态。
    step_input = torch.ones((2, 1, 4))
    step_output, _ = decoder(step_input, new_state)
    print('单步输入形状：', tuple(step_input.shape))
    print('单步输出形状：', tuple(step_output.shape))


if __name__ == '__main__':
    main()
