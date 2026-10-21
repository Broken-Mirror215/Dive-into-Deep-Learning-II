from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from d2l import torch as d2l


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)
b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)
b3 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)
b4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)
b5 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
)
net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))


def train_googlenet(net, train_iter, test_iter, num_epochs, lr, device):
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
    transform = transforms.Compose([transforms.Resize(96), transforms.ToTensor()])
    train_data = datasets.FashionMNIST(data_dir, train=True, transform=transform, download=True)
    test_data = datasets.FashionMNIST(data_dir, train=False, transform=transform, download=True)
    train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = DataLoader(test_data, batch_size=batch_size, num_workers=0)
    history = train_googlenet(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    epochs = range(1, num_epochs + 1)
    for name, values in history.items():
        d2l.plt.plot(epochs, values, label=name)
    d2l.plt.xlabel("epoch")
    d2l.plt.legend()
    d2l.plt.show()
