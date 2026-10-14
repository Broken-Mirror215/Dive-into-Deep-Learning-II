import torch
from torch import nn
from d2l import torch as d2l

def dropout_layer(X,dropout):
    assert 0 <= dropout <= 1
    #本次丢掉所有的元素
    if dropout == 1:
        return torch.zeros_like(X)
    #本情况保留所有元素
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

num_inputs , num_outputs , num_hiddens1 , num_hiddens2 = 784 , 10 , 256 , 256

# 定义模型
dropout1 ,dropout2 = 0.2 , 0.5
class Net(nn.Module):
    def __init__(self,num_inputs , num_outputs , num_hiddens1 , num_hiddens2,is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(self.num_inputs , num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1 , num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2 , num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape(-1,self.num_inputs)))
        #只有在训练模型的时候使用暂退法
        if self.training==True:
            H1 = dropout_layer(H1,dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training==True:
            H2 = dropout_layer(H2,dropout2)
        out = self.lin3(H2)
        return out

net = Net(num_inputs , num_outputs , num_hiddens1 , num_hiddens2)

# 训练和测试
num_epochs , lr ,batch_size = 10 , 0.5 , 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter ,test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(),lr=lr)
d2l.train_ch3(net, train_iter , test_iter , loss , num_epochs , trainer)
d2l.plt.show()
