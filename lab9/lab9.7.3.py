
import torch
from torch import nn


def sequence_mask(X, valid_len, value=0):
    """原地屏蔽每个样本在有效长度以外的位置，返回修改后的 X。

    X 的前两维是 (批量大小, 时间步数)，后面可以有更多特征维度。
    valid_len 的形状为 (批量大小,)，表示每个样本保留多少个时间步。
    """
    maxlen = X.size(1)
    # (1, T) 与 (B, 1) 比较，广播得到 (B, T) 的布尔掩码。
    # True 表示有效位置；例如长度为 2 时，只保留下标 0、1。
    mask = torch.arange(maxlen, device=X.device)[None, :] < valid_len[:, None]
    # 对三维输入，选中的每个 (样本, 时间步) 的全部特征都会被替换。
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带掩码的交叉熵损失，排除目标序列中填充位置的贡献。"""

    def __init__(self):
        # 先保留每个词元的损失，才能按样本的有效长度进行屏蔽。
        super().__init__(reduction='none')

    def forward(self, pred, label, valid_len):
        # pred: (B, T, V)，是解码器输出的原始分数 logits。
        # label: (B, T)，是正确词元编号；valid_len: (B,)。
        weights = sequence_mask(torch.ones_like(label), valid_len)
        # CrossEntropyLoss 要求类别轴在第 1 维，因此调整为 (B, V, T)。
        # 它内部完成 log_softmax 等计算，这里直接传 logits。
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        # 教材沿全部 T 个时间步取平均，而不是除以各自的 valid_len。
        # 填充位置乘 0，最后返回每个样本的损失，形状为 (B,)。
        return (unweighted_loss * weights).mean(dim=1)


def main():
    # 教材示例一：二维序列，分别保留前 1 项和前 2 项。
    X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print('二维序列掩码（有效长度为 1、2）：')
    print(sequence_mask(X, torch.tensor([1, 2])))

    # 教材示例二：对三维张量，将无效时间步的全部特征替换成 -1。
    X = torch.ones(2, 3, 4)
    print('\n三维序列掩码（填充值为 -1）：')
    print(sequence_mask(X, torch.tensor([1, 2]), value=-1))

    # 教材健全性检查：相同预测与标签，只有有效长度不同。
    # 所有 logits 都为 1，十个类别的概率相同，单个词元损失为 ln(10)。
    pred = torch.ones(3, 4, 10, requires_grad=True)
    label = torch.ones((3, 4), dtype=torch.long)
    valid_len = torch.tensor([4, 2, 0])
    loss = MaskedSoftmaxCELoss()
    result = loss(pred, label, valid_len)
    expected = torch.log(torch.tensor(10.0)) * torch.tensor([1.0, 0.5, 0.0])
    print('\n带掩码的损失（有效长度为 4、2、0）：')
    print(result.detach())
    print('预期：第一个损失是第二个的两倍，第三个为 0。')
    assert torch.allclose(result, expected)

    # 改变填充位置的预测分数，确认这些位置不会影响损失。
    padding_mask = torch.arange(pred.size(1))[None, :] >= valid_len[:, None]
    changed_pred = pred.detach().clone()
    changed_pred[..., 0][padding_mask] = 100.0
    assert torch.allclose(loss(changed_pred, label, valid_len), result.detach())

    # 屏蔽的时间步也不应产生梯度，整个有效长度为 0 的样本同样如此。
    result.sum().backward()
    assert torch.equal(pred.grad[padding_mask],
                       torch.zeros_like(pred.grad[padding_mask]))
    assert pred.grad[0].abs().sum() > 0
    print('检查通过：填充位置不影响损失，其预测分数的梯度也为 0。')


if __name__ == '__main__':
    main()
