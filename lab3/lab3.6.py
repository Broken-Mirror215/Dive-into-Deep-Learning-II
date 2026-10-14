import torch
from IPython import display
from d2l import torch as d2l

# 3.6.1
batch_size = 256
train_iter , test_iter = d2l.load_data_fashion_mnist(batch_size)

num_input = 784
num_output = 10

# 先初始随机化W
W = torch.normal(0,0.01,size =(num_input,num_output),requires_grad= True)
b = torch.zeros(num_output,requires_grad= True)

# 3.6.2
X = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

print(X.sum(0, keepdim=True))
print(X.sum(1, keepdim=True))

# 把10个图片变成概率 #用的方法是广播..
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

X = torch.normal(0,1,(2,5))
X_prob = softmax(X)
print(X_prob)
print(X_prob.sum(1))

#3.6.3 这个是把图片变成784个像素，通过参数算出10个类别的分数，再用softmax转成概率
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

#3.6.4
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6],
                      [0.3, 0.2, 0.5]])

print(y_hat[[0, 1], y])
# 这个是计算损失的函数...真实类别的概率越低，损失越大
def cross_entropy(y_hat, y):
        return - torch.log(y_hat[range(len(y_hat)), y])
print(cross_entropy(y_hat, y))
#统计预测正确的数量
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and  y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
print(accuracy(y_hat,y)/len(y))

#3.6.5 分类精度 #计算整个数据集上的准确率
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

print(evaluate_accuracy(net, test_iter))

#3.6.6 #sg根据梯度调整w,b
lr = 0.1
def updater (batch_size):
    return d2l.sgd([W,b], lr ,batch_size)

#把整个数据集学习10遍
num_epochs = 10

d2l.train_ch3(
    net,
    train_iter,
    test_iter,
    cross_entropy,
    num_epochs,
    updater
)

print ("训练后的测试准确率： " ,evaluate_accuracy(net,test_iter))

d2l.plt.show()
