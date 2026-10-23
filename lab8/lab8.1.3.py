from pathlib import Path
from runpy import run_path

import torch
from d2l import torch as d2l


def main():
    # 文件名包含点，按路径复用 8.1.2 的数据、模型和训练函数。
    experiment = run_path(str(Path(__file__).with_name("lab8.1.2.py")))
    time = experiment["time"]
    x = experiment["x"]
    tau = experiment["tau"]
    features = experiment["features"]
    labels = experiment["labels"]
    n_train = experiment["n_train"]

    net = experiment["get_net"]()
    experiment["train"](net, experiment["train_iter"], experiment["loss"],
                        epochs=5, lr=0.01)

    net.eval()
    with torch.no_grad():
        onestep_preds = net(features).reshape(-1)
        test_mse = ((onestep_preds[n_train:] - labels[n_train:, 0]) ** 2).mean()
    print(f"训练集外单步预测 MSE: {test_mse.item():.6f}")

    d2l.plot(
        [time, time[tau:]],
        [x.numpy(), onestep_preds.numpy()],
        "time", "x",
        legend=["data", "1-step preds"],
        xlim=[1, len(time)], figsize=(6, 3),
    )
    d2l.plt.show()


if __name__ == "__main__":
    main()
