"""《动手学深度学习》9.5.4：加载数据集。

教材：https://zh.d2l.ai/chapter_recurrent-modern/machine-translation-and-dataset.html
前面各节的数据读取、预处理和词元化函数保留在本文件中，可独立运行。
"""

import os
import zipfile

import torch
from d2l import torch as d2l


# 注册课程使用的英法数据集：下载地址和 SHA-1 校验值缺一不可。
# 使用 HTTPS 下载同一份文件，校验值与书中保持一致。
d2l.DATA_HUB['fra-eng'] = (
    'https://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip',
    '94646ad1522d915e7b0f9296181140edcf86a4f5',
)


def read_data_nmt():
    """下载并读取“英语－法语”数据集。"""
    # 将缓存固定在项目的 data 目录，避免启动目录变化时重复下载。
    cache_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data'))
    archive_path = d2l.download('fra-eng', cache_dir=cache_dir)
    # 每行由“英文句子\t法文句子”组成，保留制表符供后续分隔语言对。
    with zipfile.ZipFile(archive_path) as archive:
        return archive.read('fra-eng/fra.txt').decode('utf-8')


def preprocess_nmt(text):
    """统一空格和大小写，并在指定标点前补空格。"""
    def no_space(char, prev_char):
        return char in ',.!?' and prev_char != ' '

    # 法语中可能使用窄不间断空格（\u202f）或不间断空格（\xa0）。
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()

    # 例如 Go. -> go .，为下一节按空格进行词元化做准备。
    # 保留 \t 和 \n，不破坏语言对及各行之间的边界。
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """将最多 num_examples 对英法句子转换成两个词元列表。"""
    source, target = [], []
    for line in text.splitlines():
        # None 表示使用全部数据；0 表示不取样本。
        # 按已收集的句子对计数，避免多取一对或把无效行算入样本。
        if num_examples is not None and len(source) >= num_examples:
            break

        parts = line.split('\t')
        if len(parts) == 2:
            # split() 合并连续空格，并避免首尾空格产生空词元。
            src_tokens, tgt_tokens = parts[0].split(), parts[1].split()
            if not src_tokens or not tgt_tokens:
                continue
            # 同时添加，保证 source[i] 和 target[i] 是一对翻译。
            source.append(src_tokens)
            target.append(tgt_tokens)
    return source, target


def build_vocabs(source, target, min_freq=2):
    """分别构建源语言和目标语言词表，低频词映射为 <unk>。"""
    # <unk> 由 Vocab 自动添加；另外保留填充、开始和结束标记。
    reserved_tokens = ['<pad>', '<bos>', '<eos>']
    src_vocab = d2l.Vocab(source, min_freq=min_freq,
                          reserved_tokens=reserved_tokens)
    tgt_vocab = d2l.Vocab(target, min_freq=min_freq,
                          reserved_tokens=reserved_tokens)
    return src_vocab, tgt_vocab


def truncate_pad(line, num_steps, padding_token):
    """把编号列表截断或填充到 num_steps 个时间步。"""
    if num_steps <= 0:
        raise ValueError('num_steps 必须为正整数')
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines, vocab, num_steps):
    """将词元列表转为定长张量，并返回每个样本的有效长度。"""
    if num_steps <= 0:
        raise ValueError('num_steps 必须为正整数')

    # 先查表得到编号，再在每个句子末尾添加结束标记。
    lines = [vocab[line] + [vocab['<eos>']] for line in lines]
    # 与教材一致：先添加 <eos> 再截断，过长句子的 <eos> 也可能被截去。
    # long 类型可以直接作为后续嵌入层的输入；空输入保持 (0, num_steps)。
    array = torch.tensor(
        [truncate_pad(line, num_steps, vocab['<pad>']) for line in lines],
        dtype=torch.long).reshape(-1, num_steps)
    # 有效长度包含保留下来的 <eos>，不包含补齐的 <pad>。
    valid_len = (array != vocab['<pad>']).sum(dim=1)
    return array, valid_len


def main(num_examples=600, num_steps=10):
    # 演示使用前 600 对句子；num_examples=None 表示使用全部数据。
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab, tgt_vocab = build_vocabs(source, target)

    if source:
        first_ids = src_vocab[source[0]]
        print('第一句英文词元：', source[0])
        print('原始编号：', first_ids)
        print(f'填充到 {num_steps} 步：',
              truncate_pad(first_ids, num_steps, src_vocab['<pad>']))
        print('截断到 1 步：', truncate_pad(first_ids, 1, src_vocab['<pad>']))

    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    print('\n源语言张量形状：', tuple(src_array.shape))
    print('目标语言张量形状：', tuple(tgt_array.shape))
    print('编号张量的数据类型：', src_array.dtype)
    print('前 2 个英文样本：\n', src_array[:2])
    print('英文有效长度：', src_valid_len[:2])
    print('前 2 个法文样本：\n', tgt_array[:2])
    print('法文有效长度：', tgt_valid_len[:2])
    if source:
        print('第一句英文张量对应的词元：',
              src_vocab.to_tokens(src_array[0].tolist()))
        print('第一句法文张量对应的词元：',
              tgt_vocab.to_tokens(tgt_array[0].tolist()))
    # 此处尚未添加 <bos>；它在后续章节构造解码器输入时使用。


if __name__ == '__main__':
    main()
