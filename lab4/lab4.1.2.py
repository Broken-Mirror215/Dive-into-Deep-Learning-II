import matplotlib.pyplot as plt
import torch


# 4.1.2 激活函数：ReLU、sigmoid、tanh
# 激活函数为网络引入非线性，让多层网络能够学习更复杂的关系。
# 本实验分别画出三个函数的曲线，以及输出对输入 x 的导数。
x = torch.linspace(-8.0, 8.0, steps=161, requires_grad=True)


# 1. ReLU：relu(x) = max(0, x)
# x < 0 时输出为 0，x > 0 时输出为 x。
# 对应的导数分别为 0 和 1；在 x = 0 处不可导，PyTorch 采用梯度 0。
y_relu = torch.relu(x)

# y_relu 是向量，传入全 1 的上游梯度，相当于对 y_relu.sum() 求导。
# 由于每个输出只依赖对应的输入，x.grad 就是各个位置的函数导数。
y_relu.backward(torch.ones_like(x))
grad_relu = x.grad.detach().clone()


# 2. sigmoid：sigmoid(x) = 1 / (1 + exp(-x))
# 数学上输出在 (0, 1) 内，sigmoid(0) = 0.5。
# 导数为 sigmoid(x) * (1 - sigmoid(x))，最大值是 0.25。
# 当 x 很大或很小时，曲线趋于平坦，梯度接近 0。
x.grad = None  # 清空上一次的梯度，避免 backward 将新梯度累加上去。
y_sigmoid = torch.sigmoid(x)
y_sigmoid.backward(torch.ones_like(x))
grad_sigmoid = x.grad.detach().clone()


# 3. tanh：tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
# 数学上输出在 (-1, 1) 内，tanh(0) = 0，输出以 0 为中心。
# 导数为 1 - tanh(x)**2，最大值是 1。
# 当 x 很大或很小时，同样会出现梯度接近 0 的情况。
x.grad = None
y_tanh = torch.tanh(x)
y_tanh.backward(torch.ones_like(x))
grad_tanh = x.grad.detach().clone()

# 每个激活函数都重新建立了计算图，所以这里不需要 retain_graph=True。
# clone() 保存各次计算得到的梯度，便于一起绘图比较。


# 在一个窗口中展示 6 张图：左边是函数值，右边是导数。
fig, axes = plt.subplots(3, 2, figsize=(10, 9))
curves = [
    ("ReLU", y_relu, grad_relu),
    ("sigmoid", y_sigmoid, grad_sigmoid),
    ("tanh", y_tanh, grad_tanh),
]

for row, (name, y, grad) in enumerate(curves):
    # detach() 将张量从计算图中分离，再转为 NumPy 数组供绘图使用。
    axes[row, 0].plot(x.detach().numpy(), y.detach().numpy())
    axes[row, 0].set_title(f"{name}(x)")
    axes[row, 0].set_ylabel("function value")

    axes[row, 1].plot(x.detach().numpy(), grad.numpy(), color="tab:orange")
    axes[row, 1].set_title(f"gradient of {name}(x)")
    axes[row, 1].set_ylabel("gradient")

    for ax in axes[row]:
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)

fig.tight_layout()
plt.show()
