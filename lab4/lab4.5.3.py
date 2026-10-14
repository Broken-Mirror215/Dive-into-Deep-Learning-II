import torch
from d2l import torch as d2l


n_train , n_test , num_inputs , batch_size = 20 , 100 , 200 , 5
true_w , true_b = torch.ones((num_inputs,1)) * 0.01 , 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data,batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

# 初始化模型参数
def init_params():
    w = torch.normal(0,1,size=(num_inputs,1),requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return (w,b)

# 定义l2范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) /2


# 定义模型训练
def train(lambd):
    w,b = init_params()
    net , loss = lambda X: d2l.linreg(X,w,b) , d2l.squared_loss
    num_epochs , lr = 100 , 0.003
    animator = d2l.Animator(xlabel='epochs',ylabel='loss',yscale='log',
                            xlim=[5,num_epochs],legend=['train','test'])
    for epoch in range(num_epochs):
        for X,y in train_iter:
            # 给每个样本的损失加上 L2 惩罚；lambd=0 时不启用权重衰减
            # 标量惩罚广播到 (batch_size, 1)，sgd 中再除以实际批量大小
            l = loss(net(X),y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w,b],lr,X.shape[0])
        if (epoch+1) % 5 == 0:
            with torch.no_grad():
                train_loss = d2l.evaluate_loss(net, train_iter, loss)
                test_loss = d2l.evaluate_loss(net, test_iter, loss)
            animator.add(epoch + 1, (train_loss, test_loss))
    print('最终训练损失:', train_loss)
    print('最终测试损失:', test_loss)
    print('w的L2范数是:', torch.norm(w).item())

train(lambd=3)
d2l.plt.show()
