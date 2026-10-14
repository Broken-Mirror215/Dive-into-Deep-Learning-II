import torch


# 4.7 前向传播、反向传播和计算图
# 用一个隐藏神经元和一个输出神经元，手算梯度，再与自动求导比较。
# 为了方便观察，省略偏置，使用平方损失。
#
# 预测路线：x -> z = w1*x -> h = ReLU(z) -> o = w2*h -> L
# 正则路线：w1、w2 -> s = lambd/2 * (w1² + w2²)
# 两条路线汇合：J = L + s


def main():
    x = torch.tensor(2.0)  # 输入数据
    y = torch.tensor(4.0)  # 真实答案
    w1 = torch.tensor(0.5, requires_grad=True)
    w2 = torch.tensor(3.0, requires_grad=True)
    lambd = 0.1
    lr = 0.01

    # 1. 前向传播：用当前权重计算预测和总目标，并建立计算图。
    z = w1 * x
    h = torch.relu(z)
    o = w2 * h
    L = (o - y).square() / 2
    s = lambd / 2 * (w1.square() + w2.square())
    J = L + s

    # 中间结果不是叶子张量，默认不会保留它们的 .grad。
    # retain_grad() 让我们能查看反向传播经过各个位置时的梯度。
    z.retain_grad()
    h.retain_grad()
    o.retain_grad()

    print('前向传播结果：')
    for name, value in [('z', z), ('h', h), ('o', o),
                        ('L', L), ('s', s), ('J', J)]:
        print(f'  {name} = {value.item():.6f}')

    # 2. 手算梯度：每个权重都有预测损失和正则化两条影响路线。
    # dJ/do = o-y = -1
    # dJ/dh = (o-y)*w2 = -3
    # z=1>0，所以 ReLU 在这里的导数是 1，dJ/dz = -3。
    # dJ/dw2 = (o-y)*h + lambd*w2 = -1 + 0.3 = -0.7
    # dJ/dw1 = (o-y)*w2*1*x + lambd*w1 = -6 + 0.05 = -5.95
    expected_w1_grad = -5.95
    expected_w2_grad = -0.7

    # 3. 反向传播：计算梯度并存入 .grad，此时权重还没有更新。
    J.backward()
    print('\n反向传播结果：')
    print(f'  dJ/do = {o.grad.item():.6f}')
    print(f'  dJ/dh = {h.grad.item():.6f}')
    print(f'  dJ/dz = {z.grad.item():.6f}')
    print(f'  dJ/dw1 = {w1.grad.item():.6f}，手算：{expected_w1_grad}')
    print(f'  dJ/dw2 = {w2.grad.item():.6f}，手算：{expected_w2_grad}')
    torch.testing.assert_close(w1.grad, torch.tensor(expected_w1_grad))
    torch.testing.assert_close(w2.grad, torch.tensor(expected_w2_grad))
    print('  自动求导与手算一致。')

    # 4. 手动执行一步 SGD：w <- w - lr * grad。
    # 更新参数时关闭梯度记录；更新后的预测需要重新前向计算。
    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad
        new_o = w2 * torch.relu(w1 * x)
        new_J = ((new_o - y).square() / 2
                 + lambd / 2 * (w1.square() + w2.square()))

    print('\n一步 SGD 更新后：')
    print(f'  w1 = {w1.item():.6f}，w2 = {w2.item():.6f}')
    print(f'  预测值：{o.item():.6f} -> {new_o.item():.6f}，真实值：{y.item()}')
    print(f'  总目标：{J.item():.6f} -> {new_J.item():.6f}')

    # backward() 会累加梯度；为下一次训练清空本次梯度。
    w1.grad = None
    w2.grad = None


if __name__ == '__main__':
    main()
