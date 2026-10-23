import torch
from torch import nn
from d2l import torch as d2l


T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i:T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def get_net():
    net = nn.Sequential(
        nn.Linear(tau, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
    )
    net.apply(init_weights)
    return net


loss = nn.MSELoss(reduction="none")


def train(net, train_iter, loss, epochs, lr):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            batch_loss = loss(net(X), y)
            batch_loss.sum().backward()
            optimizer.step()
        net.eval()
        print(f"epoch {epoch + 1}, loss: {d2l.evaluate_loss(net, train_iter, loss):f}")


if __name__ == "__main__":
    d2l.plot(time, [x], "time", "x", xlim=[1, T], figsize=(6, 3))
    net = get_net()
    train(net, train_iter, loss, epochs=5, lr=0.01)
