from pathlib import Path
import re

from d2l import torch as d2l


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
d2l.DATA_HUB["time_machine"] = (
    d2l.DATA_URL + "timemachine.txt",
    "090b5e7e70c295757f55df93cb0a180b9691891a",
)


def read_time_machine():
    """读取《时间机器》，返回清理标点并转成小写的文本行。"""
    filename = d2l.download("time_machine", cache_dir=str(DATA_DIR))
    with open(filename, encoding="utf-8") as file:
        return [re.sub(r"[^A-Za-z]+", " ", line).strip().lower()
                for line in file]

def tokenize(lines, token="word"):
    """将文本行拆分为单词或字符词元。"""
    if token == "word":
        return [line.split() for line in lines]
    if token == "char":
        return [list(line) for line in lines]
    raise ValueError(f"未知词元类型: {token}")


if __name__ == "__main__":
    lines = read_time_machine()
    tokens = tokenize(lines)
    for i in range(11):
        print(tokens[i])
