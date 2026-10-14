# import 导入模块；后面通过 random.shuffle、torch.tensor 等名字使用它们。
import random
import torch

# from ... import ... as ...：导入一个名字，并给它起别名。
from d2l import torch as d2l


# def 定义函数，括号里是参数；冒号后缩进的代码属于函数体。
# 定义函数时不会执行函数体，下面调用 synthetic_data(...) 时才会执行。
def synthetic_data(w, b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    # return 结束本次函数调用；逗号分隔的两个结果会组成一个元组。
    return x, y.reshape(-1, 1)

true_w = torch.tensor([2.0, -3.4]) #what is type?
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
# 上面是解包赋值：第一个返回值交给 features，第二个交给 labels。

print("输入的形状：", features.shape)
print("答案的形状：", labels.shape)
print("第一条输入：", features[0])
print("第一条答案：", labels[0])


d2l.set_figsize()
# [:, 1] 是索引：取所有行的第 2 列。Python 的下标从 0 开始。
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    # range(n) 提供 0 到 n-1；list(...) 将这些数字收集成列表。
    indices = list(range(num_examples))
    random.shuffle(indices)
    # range(起点, 终点, 步长)：包含起点，不包含终点。
    for i in range(0, num_examples, batch_size):
        # [起点:终点] 是切片，不包含终点；min(...) 防止最后一批越界。
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)])
        # yield 产出一批并暂停，使函数成为生成器；下次迭代从暂停处继续。
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for x, y in data_iter(batch_size, features, labels):
    print(x, '\n', y)
    break

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义模型(线性)
def linreg(x, w, b):
    return torch.matmul(x, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    # shape 访问属性，.reshape(...) 调用方法。
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    # 小批量随机梯度下降
    # with 管理一个上下文；这里进入代码块时关闭梯度跟踪，退出时恢复。
    with torch.no_grad():
        for param in params:
            # -= 是减法赋值；这里对张量参数执行原地更新。
            param -= lr * param.grad / batch_size
            # zero_() 是 PyTorch 方法；末尾的 _ 是原地修改的命名约定。
            param.grad.zero_()

# 训练
lr = 0.03
num_epochs = 3
# 函数也是对象：这里给函数起另一个名字；加括号 net(...) 才是调用。
net = linreg
loss = squared_loss
# 外层循环训练多轮，内层循环遍历本轮的所有小批量；缩进决定循环范围。
for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
        l = loss(net(x, w, b), y)  # x和y的小批量损失
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        # f 字符串中的 {...} 会计算表达式；:f 按浮点格式显示，默认 6 位小数。
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

d2l.plt.show()
