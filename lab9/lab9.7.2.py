

import torch
from torch import nn
from d2l import torch as d2l


class Seq2SeqEncoder(d2l.Encoder):
    """与 9.7.1 相同的嵌入层和多层单向 GRU 编码器。"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X).permute(1, 0, 2)
        # output: (源时间步数, 批量大小, 隐藏单元数)。
        # state: (层数, 批量大小, 隐藏单元数)。
        return self.rnn(X)


class Seq2SeqDecoder(d2l.Decoder):
    """通过词元嵌入、上下文拼接、GRU 和全连接层预测目标词元。"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 每个时间步同时输入词元向量 E 和上下文向量 H，因此输入维度为 E+H。
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        # 将隐藏向量映射为目标词表中每个词元的分数。
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # enc_outputs 是编码器返回的 (output, state)，取每一层的最终隐状态。
        # 直接传给 GRU 时，编码器和解码器的层数、隐藏单元数需要匹配。
        return enc_outputs[1]

    def forward(self, X, state):
        # 输入编号：(B, T) → 嵌入向量：(B, T, E) → GRU 布局：(T, B, E)。
        X = self.embedding(X).permute(1, 0, 2)
        # 按教材实现：取传入 state 的最后一层，广播到本次调用的每个时间步。
        # state[-1]: (B, H)，context: (T, B, H)。
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), dim=2)
        # 拼接后输入为 (T, B, E+H)，state 同时用于初始化 GRU 的隐藏状态。
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output 是原始分数 logits，形状为 (B, T, 目标词表大小)。
        # state 是更新后的各层隐状态，形状为 (L, B, H)，供下次调用继续使用。
        # 后续交叉熵损失直接接收 logits；需要查看概率时再执行 softmax。
        return output, state


def main():
    torch.manual_seed(42)
    vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
    encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
    decoder = Seq2SeqDecoder(vocab_size, embed_size, num_hiddens, num_layers)
    encoder.eval()
    decoder.eval()

    # 教材示例：两个模块均使用批量大小 4、长度 7 的词元编号序列。
    enc_X = torch.zeros((4, 7), dtype=torch.long)
    dec_X = torch.zeros((4, 7), dtype=torch.long)
    with torch.no_grad():
        enc_outputs = encoder(enc_X)
        initial_state = decoder.init_state(enc_outputs)
        embedded = decoder.embedding(dec_X).permute(1, 0, 2)
        context = initial_state[-1].repeat(dec_X.shape[1], 1, 1)
        output, state = decoder(dec_X, initial_state)
        probabilities = output.softmax(dim=-1)

    print('编码器输入形状（B, 源时间步数）：', tuple(enc_X.shape))
    print('解码器初始隐状态形状（L, B, H）：', tuple(initial_state.shape))
    print('解码器输入形状（B, 目标时间步数）：', tuple(dec_X.shape))
    print('词元嵌入形状（T, B, E）：', tuple(embedded.shape))
    print('广播后的上下文形状（T, B, H）：', tuple(context.shape))
    print('拼接后 GRU 输入形状（T, B, E+H）：',
          tuple(torch.cat((embedded, context), dim=2).shape))
    print('解码器输出 logits 形状（B, T, V）：', tuple(output.shape))
    print('解码器最终隐状态形状（L, B, H）：', tuple(state.shape))
    print('softmax 后各位置的概率和是否接近 1：',
          torch.allclose(probabilities.sum(dim=-1), torch.ones_like(output[..., 0])))

    # 目标长度可以与源长度不同；重新从编码器状态开始解码。
    with torch.no_grad():
        short_output, _ = decoder(dec_X[:, :3], initial_state)
        # 也可以只输入一个时间步，并传入上次返回的状态继续解码。
        step_output, next_state = decoder(dec_X[:, :1], state)
    print('\n目标长度改为 3 时的输出形状：', tuple(short_output.shape))
    print('继续单步解码的输出形状：', tuple(step_output.shape))
    print('单步解码返回的状态形状：', tuple(next_state.shape))


if __name__ == '__main__':
    main()
