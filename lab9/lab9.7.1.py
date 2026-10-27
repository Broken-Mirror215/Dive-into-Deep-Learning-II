

import torch
from torch import nn
from d2l import torch as d2l


class Seq2SeqEncoder(d2l.Encoder):
    """用嵌入层和多层单向 GRU 实现 9.6 节的编码器接口。"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        # 每一行是一个词元的可学习向量，权重形状为 (词表大小, 嵌入维度)。
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # GRU 默认使用 (时间步数, 批量大小, 特征数) 的输入布局。
        # dropout 作用于相邻 GRU 层之间的输出，不作用于最后一层输出。
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 输入 X 是词元编号，形状为 (批量大小, 时间步数)，类型为 long。
        X = self.embedding(X)
        # 查表后的形状：(批量大小, 时间步数, 嵌入维度)。
        X = X.permute(1, 0, 2)
        # 调整后的形状：(时间步数, 批量大小, 嵌入维度)。
        # 不传入初始隐状态时，GRU 会使用全零初始状态。
        output, state = self.rnn(X)
        # output：最后一层在所有时间步的输出，形状为 (T, B, H)。
        # state：每一层在最后时间步的隐状态，形状为 (L, B, H)。
        # *args 保留与 EncoderDecoder 的兼容性，本节按教材处理全部时间步。
        return output, state


def main():
    torch.manual_seed(42)
    vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
    encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
    encoder.eval()

    # 教材示例：批量大小为 4，每个样本包含 7 个词元编号。
    # 编号 0 查表后得到可学习的嵌入向量，并不表示向量的分量全为零。
    X = torch.zeros((4, 7), dtype=torch.long)
    with torch.no_grad():
        embedded = encoder.embedding(X)
        output, state = encoder(X)

    print('输入编号形状（B, T）：', tuple(X.shape))
    print('嵌入层权重形状（V, E）：', tuple(encoder.embedding.weight.shape))
    print('嵌入向量形状（B, T, E）：', tuple(embedded.shape))
    print('GRU 输入形状（T, B, E）：', tuple(embedded.permute(1, 0, 2).shape))
    print('所有时间步输出形状（T, B, H）：', tuple(output.shape))
    print('最终隐状态形状（L, B, H）：', tuple(state.shape))
    print('最后一层最终隐状态形状（B, H）：', tuple(state[-1].shape))
    print('output[-1] 与 state[-1] 是否一致：',
          torch.allclose(output[-1], state[-1]))

    # 改变源序列长度时，output 的时间维随之变化，state 的形状保持不变。
    longer_X = torch.zeros((4, 10), dtype=torch.long)
    with torch.no_grad():
        longer_output, longer_state = encoder(longer_X)
    print('\n输入长度改为 10 后：')
    print('输入形状：', tuple(longer_X.shape))
    print('output 形状：', tuple(longer_output.shape))
    print('state 形状：', tuple(longer_state.shape))


if __name__ == '__main__':
    main()
