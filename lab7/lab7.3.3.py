from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from d2l import torch as d2l


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride=strides, padding=padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
    )


net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(p=0.5),
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
)


def train_nin(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)

    net.apply(init_weights)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    num_batches = len(train_iter)
    progress_interval = max(1, num_batches // 10)
    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    print(f"training on {device}", flush=True)
    for epoch in range(1, num_epochs + 1):
        net.train()
        total_loss = total_correct = total_examples = 0
        print(f"epoch {epoch}/{num_epochs} started", flush=True)

        for batch_idx, (X, y) in enumerate(train_iter, start=1):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            batch_size = y.numel()
            total_loss += loss.item() * batch_size
            total_correct += (y_hat.argmax(dim=1) == y).sum().item()
            total_examples += batch_size

            if batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == num_batches:
                print(
                    f"epoch {epoch}/{num_epochs}, batch {batch_idx}/{num_batches}, "
                    f"train loss {total_loss / total_examples:.3f}, "
                    f"train acc {total_correct / total_examples:.3f}",
                    flush=True,
                )

        net.eval()
        test_correct = test_examples = 0
        print(f"epoch {epoch}/{num_epochs} evaluating", flush=True)
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                test_correct += (net(X).argmax(dim=1) == y).sum().item()
                test_examples += y.numel()

        train_loss = total_loss / total_examples
        train_acc = total_correct / total_examples
        test_acc = test_correct / test_examples
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(
            f"epoch {epoch}/{num_epochs} complete: loss {train_loss:.3f}, "
            f"train acc {train_acc:.3f}, test acc {test_acc:.3f}",
            flush=True,
        )

    return history


if __name__ == "__main__":
    lr, num_epochs, batch_size = 0.1, 10, 128
    data_dir = Path(__file__).resolve().parents[1] / "data"
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    train_data = datasets.FashionMNIST(data_dir, train=True, transform=transform, download=True)
    test_data = datasets.FashionMNIST(data_dir, train=False, transform=transform, download=True)
    train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = DataLoader(test_data, batch_size=batch_size, num_workers=0)
    history = train_nin(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    epochs = range(1, num_epochs + 1)
    for name, values in history.items():
        d2l.plt.plot(epochs, values, label=name)
    d2l.plt.xlabel("epoch")
    d2l.plt.legend()
    d2l.plt.show()
