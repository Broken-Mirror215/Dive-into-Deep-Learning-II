"""《动手学深度学习》9.6.1：编码器。

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


class DemoEncoder(Encoder):
    """接口演示：对时间维取平均，得到每个样本的固定大小上下文。"""

    def forward(self, X, *args):
        # 本示例的 X 是浮点特征，形状为 (批量大小, 源序列长度, 特征数)。
        # 对应机器翻译时，词元编号需要先经过嵌入层才能变成浮点特征。
        return X.mean(dim=1)


def main():
    encoder = DemoEncoder()
    X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    context = encoder(X)  # 使用模块调用，会自动执行 forward。

    print('本节演示编码器接口，使用时间维平均作为示例编码。')
    print('输入序列形状（批量大小, 源序列长度, 特征数）：', tuple(X.shape))
    print('编码结果形状（批量大小, 特征数）：', tuple(context.shape))
    print('编码结果：\n', context)

    longer_X = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
    longer_context = encoder(longer_X)
    print('改变源序列长度后的输入形状：', tuple(longer_X.shape))
    print('编码结果形状：', tuple(longer_context.shape))


if __name__ == '__main__':
    main()
