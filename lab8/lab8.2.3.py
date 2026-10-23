import collections
from pathlib import Path
from runpy import run_path


def count_corpus(tokens):
    """统计一维或二维词元列表中各词元的出现次数。"""
    if tokens and isinstance(tokens[0], (list, tuple)):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """把文本词元与数字索引互相转换。"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        tokens = [] if tokens is None else tokens
        reserved_tokens = [] if reserved_tokens is None else reserved_tokens
        # 按照出现频率排序
        self._token_freqs = sorted(
            count_corpus(tokens).items(), key=lambda item: item[1], reverse=True
        )
        # 未知词元的索引是0
        self.idx_to_token = ["<unk>"] + list(reserved_tokens)
        self.token_to_idx = {
            token: index for index, token in enumerate(self.idx_to_token)
        }
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.token_to_idx[token] = len(self.idx_to_token)
                self.idx_to_token.append(token)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self[token] for token in tokens]
        return self.token_to_idx.get(tokens, self.unk)

    def to_tokens(self, indices):
        if isinstance(indices, (list, tuple)):
            return [self.to_tokens(index) for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


if __name__ == "__main__":
    # 文件名包含点，按路径复用 8.2.2 的读取和词元化函数。
    text_tools = run_path(str(Path(__file__).with_name("lab8.2.2.py")))
    lines = text_tools["read_time_machine"]()
    tokens = text_tools["tokenize"](lines)
    vocab = Vocab(tokens)
    print("前 10 个词元及索引:", list(vocab.token_to_idx.items())[:10])
    for i in (0, 10):
        print("文本:", tokens[i])
        print("索引:", vocab[tokens[i]])
