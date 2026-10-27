"""《动手学深度学习》9.5.2：英法句子词元化与词元数量直方图。

教材：https://zh.d2l.ai/chapter_recurrent-modern/machine-translation-and-dataset.html
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


def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制源语言和目标语言的词元数量分布。"""
    d2l.set_figsize()
    fig = d2l.plt.figure()
    _, _, patches = d2l.plt.hist(
        [[len(tokens) for tokens in xlist],
         [len(tokens) for tokens in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    # 用斜线标记目标语言的数据，便于区分两组柱子。
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)
    d2l.plt.tight_layout()
    return fig


def main(num_examples=None, show_plot=True):
    """运行词元化实验，打印样例并将直方图保存到脚本所在目录。"""
    raw_text = read_data_nmt()
    text = preprocess_nmt(raw_text)
    source, target = tokenize_nmt(text, num_examples)

    print(f'词元化后的英法句子对数量：{len(source)}')
    print('前 6 对句子的词元：')
    for src, tgt in zip(source[:6], target[:6]):
        print('source:', src)
        print('target:', tgt)

    fig = show_list_len_pair_hist(
        ['source', 'target'], '# tokens per sequence', 'count', source, target)
    figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'lab9.5.2-token-lengths.png')
    fig.savefig(figure_path, dpi=150)
    print(f'直方图已保存：{figure_path}')
    # Agg 后端不能弹出图窗，此时可直接打开保存的图片。
    if show_plot and d2l.plt.get_backend().lower() != 'agg':
        d2l.plt.show()
    d2l.plt.close(fig)


if __name__ == '__main__':
    main()
