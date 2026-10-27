import os
from d2l import torch as d2l


# 注册课程使用的英法数据集：下载地址和 SHA-1 校验值缺一不可。
# 使用 HTTPS 下载同一份文件，校验值与书中保持一致。
d2l.DATA_HUB['fra-eng'] = (
    'https://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip',
    '94646ad1522d915e7b0f9296181140edcf86a4f5',
)


def read_data_nmt():
    """下载、解压并读取“英语－法语”数据集。"""
    data_dir = d2l.download_extract('fra-eng')
    # 每行由“英文句子\t法文句子”组成，保留制表符供后续分隔语言对。
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()


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


