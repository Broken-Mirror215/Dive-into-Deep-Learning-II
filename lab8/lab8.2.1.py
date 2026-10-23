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


if __name__ == "__main__":
    lines = read_time_machine()
    print(f"文本总行数: {len(lines)}")
    print(lines[0])
    print(lines[10])
