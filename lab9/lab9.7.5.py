"""《动手学深度学习》PyTorch 版 9.7.5：序列到序列模型的预测。

教材：https://zh.d2l.ai/chapter_recurrent-modern/seq2seq.html
加载 9.7.4 保存的参数和词表，使用贪心解码逐词生成法语翻译。
运行示例：python lab9.7.5.py --sentence "go." --trace
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from d2l import torch as d2l


CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1] / 'data' / 'seq2seq-9.7.4' / 'model.pt'
)


class Seq2SeqEncoder(d2l.Encoder):
    """与训练时相同的编码器，结构参数从保存的配置中读取。"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X).permute(1, 0, 2)
        return self.rnn(X)


class Seq2SeqDecoder(d2l.Decoder):
    """与 9.7.4 保持一致，以便直接加载已训练的参数。"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), dim=2)
        output, state = self.rnn(X_and_context, state)
        return self.dense(output).permute(1, 0, 2), state


def restore_vocab(idx_to_token):
    """恢复训练时的词元编号，预测时沿用同一套词表。"""
    vocab = d2l.Vocab()
    vocab.idx_to_token = list(idx_to_token)
    vocab.token_to_idx = {
        token: index for index, token in enumerate(vocab.idx_to_token)
    }
    return vocab


def load_trained_model(checkpoint_path, device):
    """读取 9.7.4 保存的配置、模型权重和两种语言的词表。"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'找不到模型：{checkpoint_path}，请先运行 lab9.7.4.py 完成训练。')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    src_vocab = restore_vocab(checkpoint['src_idx_to_token'])
    tgt_vocab = restore_vocab(checkpoint['tgt_idx_to_token'])
    encoder = Seq2SeqEncoder(
        len(src_vocab), config['embed_size'], config['num_hiddens'],
        config['num_layers'], config['dropout'])
    decoder = Seq2SeqDecoder(
        len(tgt_vocab), config['embed_size'], config['num_hiddens'],
        config['num_layers'], config['dropout'])
    net = d2l.EncoderDecoder(encoder, decoder).to(device)
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    net.eval()
    return net, src_vocab, tgt_vocab, config


@torch.no_grad()  # 预测只进行前向计算，无需建立反向传播的计算图。
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False, trace=False):
    """按教材执行贪心预测，返回 (翻译字符串, 注意力权重列表)。"""
    if num_steps <= 0:
        raise ValueError('num_steps 必须为正整数')
    if save_attention_weights and not hasattr(net.decoder, 'attention_weights'):
        raise ValueError('本节解码器没有注意力权重，请保持 save_attention_weights=False')
    net.eval()  # 关闭训练时使用的 dropout。

    # 与训练时使用相同的大小写、空格和标点处理。
    # 例如 "Go." → ["go", "."]，再查表并添加 <eos>。
    tokens = d2l.preprocess_nmt(src_sentence).split()
    if not tokens:
        raise ValueError('请输入非空的英文句子')
    src_tokens = src_vocab[tokens] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor(
        [min(len(src_tokens), num_steps)], dtype=torch.long, device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(0)

    # 源句子只编码一次，随后由解码器逐个时间步生成词元。
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = torch.tensor([[tgt_vocab['<bos>']]], dtype=torch.long, device=device)
    output_seq, attention_weight_seq = [], []
    for step in range(num_steps):
        previous_token = int(dec_X.item())
        # dec_X: (1, 1)，Y: (1, 1, 目标词表大小)。
        # 新的 dec_state 会传给下一步，让模型记住此前生成的内容。
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 贪心解码：每一步选择分数最高的词元。
        # softmax 不改变最大值的位置，可以直接对 logits 求 argmax。
        dec_X = Y.argmax(dim=2)
        pred = int(dec_X.item())
        if trace:
            print(f'  第 {step + 1} 步：输入 '
                  f'{tgt_vocab.to_tokens(previous_token)} → 预测 '
                  f'{tgt_vocab.to_tokens(pred)}')
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']:
            break  # 结束标记不加入最终翻译。
        output_seq.append(pred)

    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def main():
    parser = argparse.ArgumentParser(description='9.7.5：使用已训练的模型预测翻译')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_PATH)
    parser.add_argument('--sentence', action='append', help='英文句子，可重复指定')
    parser.add_argument('--trace', action='store_true', help='显示逐词预测过程')
    parser.add_argument('--threads', type=int, default=1, help='PyTorch CPU 线程数')
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error('threads 必须为正整数')
    torch.set_num_threads(args.threads)
    device = d2l.try_gpu()
    net, src_vocab, tgt_vocab, config = load_trained_model(args.checkpoint, device)
    print(f'已加载训练 {config["num_epochs"]} 轮的模型；设备：{device}')
    sentences = args.sentence or ['go .', 'i lost .', "he's calm .", "i'm home ."]
    for sentence in sentences:
        if args.trace:
            print(f'\n英文：{sentence}')
        translation, _ = predict_seq2seq(
            net, sentence, src_vocab, tgt_vocab, config['num_steps'], device,
            trace=args.trace)
        print(f'{sentence} => {translation}')


if __name__ == '__main__':
    main()
