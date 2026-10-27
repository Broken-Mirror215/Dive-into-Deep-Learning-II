"""《动手学深度学习》PyTorch 版 9.7.4：训练 seq2seq 模型。

教材：https://zh.d2l.ai/chapter_recurrent-modern/seq2seq.html
保留前几节的模型和损失定义，使本文件可以独立运行。
默认按教材训练 300 轮；可用 --epochs 3 先观察训练流程。
"""

import argparse
import csv
import time
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # 脚本将曲线保存为图片，适用于终端和 PyCharm。
import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l


DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
OUTPUT_DIR = DATA_DIR / 'seq2seq-9.7.4'


class Seq2SeqEncoder(d2l.Encoder):
    """9.7.1：嵌入层加多层单向 GRU。"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # (B, T) → (B, T, E) → (T, B, E)。
        X = self.embedding(X).permute(1, 0, 2)
        # 与教材一致，GRU 处理所有时间步，*args 保留有效长度参数的接口。
        return self.rnn(X)


class Seq2SeqDecoder(d2l.Decoder):
    """9.7.2：把上下文与词元嵌入拼接后输入 GRU。"""

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
        # (T, B, H) → (B, T, V)，返回原始分数 logits。
        return self.dense(output).permute(1, 0, 2), state


def sequence_mask(X, valid_len, value=0):
    """9.7.3：原地屏蔽有效长度以外的时间步。"""
    mask = torch.arange(X.size(1), device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """9.7.3：屏蔽填充词元，再沿全部时间步取平均。"""

    def __init__(self):
        super().__init__(reduction='none')

    def forward(self, pred, label, valid_len):
        weights = sequence_mask(torch.ones_like(label), valid_len)
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        return (unweighted_loss * weights).mean(dim=1)


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """沿用 9.5 节的数据处理，并固定使用项目 data 目录的缓存。"""
    d2l.DATA_HUB['fra-eng'] = (
        'https://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip',
        '94646ad1522d915e7b0f9296181140edcf86a4f5',
    )
    archive_path = d2l.download('fra-eng', cache_dir=str(DATA_DIR))
    with zipfile.ZipFile(archive_path) as archive:
        text = archive.read('fra-eng/fra.txt').decode('utf-8')
    text = d2l.preprocess_nmt(text)

    source, target = [], []
    for line in text.splitlines():
        if len(source) >= num_examples:
            break
        pair = line.split('\t')
        if len(pair) != 2:
            continue
        src, tgt = pair[0].split(), pair[1].split()
        if src and tgt:
            source.append(src)
            target.append(tgt)
    if not source:
        raise ValueError('没有可用的英法句子对')

    reserved_tokens = ['<pad>', '<bos>', '<eos>']
    src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=reserved_tokens)
    tgt_vocab = d2l.Vocab(target, min_freq=2, reserved_tokens=reserved_tokens)
    src_array, src_valid_len = d2l.build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = d2l.build_array_nmt(target, tgt_vocab, num_steps)
    data_iter = d2l.load_array(
        (src_array, src_valid_len, tgt_array, tgt_valid_len), batch_size)
    return data_iter, src_vocab, tgt_vocab


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """按教材执行强制教学、带掩码的损失、梯度裁剪及 Adam 更新。"""
    def xavier_init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.GRU):
            # 初始化每一层的输入权重和循环权重；偏置保留默认初始化。
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()  # 开启训练模式，包括 GRU 层间的 dropout。
    history = []

    for epoch in range(num_epochs):
        start = time.perf_counter()
        metric = d2l.Accumulator(2)  # 带掩码的损失总和、有效词元数量。
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]

            # teacher forcing：用上一个“真实词元”预测下一个词元。
            # 输入：[<bos>, y1, y2, ...]；标签：[y1, y2, y3, ...]。
            # Y[:, :-1] 去掉最后一列，最后一列也可能是 <pad>。
            bos = torch.full((Y.shape[0], 1), tgt_vocab['<bos>'],
                             dtype=Y.dtype, device=device)
            dec_input = torch.cat((bos, Y[:, :-1]), dim=1)
            Y_hat, _ = net(X, dec_input, X_valid_len)
            # Y_hat: (B, T, V)，l: (B,)，每个样本有一个损失。
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # 先变成标量，再反向传播。
            d2l.grad_clipping(net, 1)  # 梯度范数超过 1 时按比例缩小。
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), Y_valid_len.sum())

        elapsed = time.perf_counter() - start
        # 保留教材的报告口径：l 内部已按 T 平均，此处再除有效词元数。
        # 因而这里显示的 loss 不是通常的“逐词元平均交叉熵”。
        epoch_loss = metric[0] / metric[1]
        tokens_per_sec = metric[1] / elapsed
        history.append({'epoch': epoch + 1, 'loss': epoch_loss,
                        'tokens_per_sec': tokens_per_sec})
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == num_epochs:
            print(f'epoch {epoch + 1:3d}/{num_epochs}: '
                  f'loss {epoch_loss:.4f}, {tokens_per_sec:.1f} tokens/sec',
                  flush=True)

    print(f'loss {history[-1]["loss"]:.3f}, '
          f'{history[-1]["tokens_per_sec"]:.1f} tokens/sec on {device}')
    return history


def save_results(net, src_vocab, tgt_vocab, history, config, output_dir):
    """保存曲线、训练记录和模型，便于后续学习预测部分。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'tokens_per_sec'])
        writer.writeheader()
        writer.writerows(history)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([row['epoch'] for row in history], [row['loss'] for row in history])
    ax.set(xlabel='epoch', ylabel='loss (textbook metric)',
           title='Seq2seq training: section 9.7.4')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'loss.png', dpi=150)
    plt.close(fig)

    torch.save({'model_state_dict': net.state_dict(),
                'src_idx_to_token': src_vocab.idx_to_token,
                'tgt_idx_to_token': tgt_vocab.idx_to_token,
                'config': config}, output_dir / 'model.pt')
    print(f'训练记录、损失曲线和模型已保存到：{output_dir}')


def main():
    parser = argparse.ArgumentParser(description='9.7.4：训练英法 seq2seq 模型')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数，教材为 300')
    parser.add_argument('--num-examples', type=int, default=600, help='训练句子对数量')
    parser.add_argument('--threads', type=int, default=1, help='PyTorch CPU 线程数')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if min(args.epochs, args.num_examples, args.threads) <= 0:
        parser.error('epochs、num-examples 和 threads 必须为正整数')

    torch.manual_seed(42)
    # 本节模型很小，CPU 线程过多时调度开销反而较高；可通过参数调整。
    torch.set_num_threads(args.threads)
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps, lr = 64, 10, 0.005
    device = d2l.try_gpu()
    train_iter, src_vocab, tgt_vocab = load_data_nmt(
        batch_size, num_steps, args.num_examples)
    print(f'设备：{device}；样本数：{len(train_iter.dataset)}；'
          f'英文/法文词表大小：{len(src_vocab)}/{len(tgt_vocab)}')

    # 先展示一个样本的输入和标签如何错开一位。
    _, _, Y, _ = next(iter(train_iter))
    bos = torch.full((Y.shape[0], 1), tgt_vocab['<bos>'], dtype=Y.dtype)
    dec_input = torch.cat((bos, Y[:, :-1]), dim=1)
    print('一个目标序列 Y：', tgt_vocab.to_tokens(Y[0].tolist()))
    print('对应解码输入：  ', tgt_vocab.to_tokens(dec_input[0].tolist()))
    print('训练流程：编码 → 强制教学解码 → 掩码损失 → 反向传播 → 裁剪 → 更新')

    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens,
                             num_layers, dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens,
                             num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    history = train_seq2seq(net, train_iter, lr, args.epochs, tgt_vocab, device)
    config = dict(embed_size=embed_size, num_hiddens=num_hiddens,
                  num_layers=num_layers, dropout=dropout, num_steps=num_steps,
                  batch_size=batch_size, lr=lr, num_epochs=args.epochs,
                  num_examples=len(train_iter.dataset))
    save_results(net, src_vocab, tgt_vocab, history, config, args.output_dir)


if __name__ == '__main__':
    main()
