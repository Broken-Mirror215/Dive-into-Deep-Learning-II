"""《动手学深度学习》9.6.3：合并编码器和解码器。

教材：https://zh.d2l.ai/chapter_recurrent-modern/encoder-decoder.html
保留教材的通用接口，并通过小张量示例演示调用方式。
"""

import torch
from torch import nn


class Encoder(nn.Module):
    """编码器接口：具体子类实现输入序列到编码结果的转换。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X, *args):
        # *args 可传递有效长度等额外信息，由具体编码器决定如何使用。
        raise NotImplementedError('编码器子类需要实现 forward 方法')


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


class EncoderDecoder(nn.Module):
    """将编码器和解码器连接为一个完整的 nn.Module。"""

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        # 赋值为子模块后，PyTorch 会统一管理两部分的参数、设备及训练模式。
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        # 原样返回解码器的结果；这里不限定输出和状态的数据结构。
        return self.decoder(dec_X, dec_state)


class DemoEncoder(Encoder):
    """接口演示：对时间维取平均，得到每个样本的固定大小上下文。"""

    def forward(self, X, *args):
        # 本示例的 X 是浮点特征，形状为 (批量大小, 源序列长度, 特征数)。
        # 对应机器翻译时，词元编号需要先经过嵌入层才能变成浮点特征。
        return X.mean(dim=1)


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
    net = EncoderDecoder(DemoEncoder(), DemoDecoder())
    enc_X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    # 源序列长 3，解码输入序列长 2，二者可以不同。
    dec_X = torch.zeros((2, 2, 4))
    output, state = net(enc_X, dec_X)

    print('本节演示编码器和解码器之间的连接。')
    print('流程：enc_X → encoder → init_state → decoder(dec_X, state)')
    print('编码器输入形状：', tuple(enc_X.shape))
    print('解码器输入形状：', tuple(dec_X.shape))
    print('解码器输出形状：', tuple(output.shape))
    print('解码器返回状态形状：', tuple(state.shape))
    print('输出：\n', output)

    # 解码输入可以是多个时间步，也可以只传入当前时间步。
    one_step_output, _ = net(enc_X, dec_X[:, :1])
    print('单步解码输出形状：', tuple(one_step_output.shape))
    print('示例中的平均和相加用于展示接口；具体翻译模型将在 9.7 节实现。')


if __name__ == '__main__':
    main()
