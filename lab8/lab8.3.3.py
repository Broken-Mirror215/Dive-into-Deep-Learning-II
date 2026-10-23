import random
import torch
from d2l import torch as d2l

tokens = d2l.tokenize(d2l.read_time_machine())

# 将各行的单词列表展开，拼接成一个连续的语料列表。
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
print(vocab.token_freqs[:10])
