import random

import torch


def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成形状为 (batch_size, num_steps) 的输入和标签。"""
    # 从随机偏移处开始，保留后面的整段序列，注意这里使用切片。
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 为每段输入的最后一个词元预留其下一个词元，作为标签。
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        # 内层函数直接使用外层的 corpus 和 num_steps。
        return corpus[pos: pos + num_steps]

    # 只生成完整批次，丢弃不足一个批次的剩余子序列。
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        # 标签从下一个词元开始，比输入向后错开一步。
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成小批量，使相邻批次中同一行的子序列连续。"""
    # 随机选择起点，随后按原始顺序划分，不打乱子序列。
    offset = random.randint(0, num_steps)
    # 为标签预留一个词元，再保留能均分为 batch_size 行的长度。
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])

    # 每一行是一段连续序列，标签比输入向后错开一个词元。
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        # 所有行同时向后移动 num_steps 列，得到下一个批次。
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


if __name__ == "__main__":
    my_seq = list(range(35))
    print("随机抽样：")
    for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print("X:", X, "\nY:", Y)

    print("\n顺序分区：")
    for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
        print("X:", X, "\nY:", Y)
