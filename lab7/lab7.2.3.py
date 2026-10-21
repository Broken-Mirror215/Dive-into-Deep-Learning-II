from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from d2l import torch as d2l

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for num_convs, out_channels in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 10),
    )


ratio = 4
small_conv_arch = [(num_convs, out_channels // ratio)
                   for num_convs, out_channels in conv_arch]
net = vgg(small_conv_arch)


def train_vgg(net, train_iter, test_iter, num_epochs, lr, device):
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
    lr, num_epochs, batch_size = 0.05, 10, 128
    data_dir = Path(__file__).resolve().parents[1] / "data"
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    train_data = datasets.FashionMNIST(data_dir, train=True, transform=transform, download=True)
    test_data = datasets.FashionMNIST(data_dir, train=False, transform=transform, download=True)
    train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = DataLoader(test_data, batch_size=batch_size, num_workers=0)
    history = train_vgg(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    epochs = range(1, num_epochs + 1)
    for name, values in history.items():
        d2l.plt.plot(epochs, values, label=name)
    d2l.plt.xlabel("epoch")
    d2l.plt.legend()
    d2l.plt.show()
