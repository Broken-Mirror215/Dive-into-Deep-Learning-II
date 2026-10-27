"""《动手学深度学习》9.5.3：词表。

教材：https://zh.d2l.ai/chapter_recurrent-modern/machine-translation-and-dataset.html
前面各节的数据读取、预处理和词元化函数保留在本文件中，可独立运行。
"""

import os
import zipfile
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


def main(num_examples=None):
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab, tgt_vocab = build_vocabs(source, target)

    print(f'英法句子对数量：{len(source)}')
    print(f'源语言（英语）词表大小：{len(src_vocab)}')
    print(f'目标语言（法语）词表大小：{len(tgt_vocab)}')
    print('英文词表前 10 个词元：', src_vocab.idx_to_token[:10])
    print('法文词表前 10 个词元：', tgt_vocab.idx_to_token[:10])
    for token in ['<unk>', '<pad>', '<bos>', '<eos>']:
        print(f'{token}: 英文编号={src_vocab[token]}, 法文编号={tgt_vocab[token]}')

    if source:
        print('\n第一对句子的“词元 → 编号 → 词元”：')
        for name, tokens, vocab in [('英语', source[0], src_vocab),
                                     ('法语', target[0], tgt_vocab)]:
            indices = vocab[tokens]
            print(f'{name}词元：{tokens}')
            print(f'{name}编号：{indices}')
            print(f'反向查表：{vocab.to_tokens(indices)}')

    print('\n词表外词元的编号：', src_vocab['never_seen_token'])
    print('对应词元：', src_vocab.to_tokens(src_vocab['never_seen_token']))
    # 英法词表独立编号，相同的数字在两种语言中可能对应不同词元。


if __name__ == '__main__':
    main()
