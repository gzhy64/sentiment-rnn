"""定义 LSTM 单元中的可学习层。"""

import torch
import torch.nn as nn


class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """创建遗忘门、输入门、候选记忆和输出门的可学习线性层。"""
        super().__init__()

        combined_size = input_size + hidden_size

        self.forget_linear = nn.Linear(combined_size, hidden_size)
        self.input_linear = nn.Linear(combined_size, hidden_size)
        self.candidate_linear = nn.Linear(combined_size, hidden_size)
        self.output_linear = nn.Linear(combined_size, hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        """根据当前词向量和上一时刻状态，计算并返回新的隐藏状态与记忆状态。"""
        combined = torch.cat([x_t, h_prev], dim=-1)

        forget_gate = torch.sigmoid(self.forget_linear(combined))
        input_gate = torch.sigmoid(self.input_linear(combined))
        candidate = torch.tanh(self.candidate_linear(combined))
        output_gate = torch.sigmoid(self.output_linear(combined))

        c_new = forget_gate * c_prev + input_gate * candidate
        h_new = output_gate * torch.tanh(c_new)

        return h_new, c_new



if __name__ == '__main__':
    cell = MyLSTMCell(input_size=2, hidden_size=3)

    # 分类
    classifier = nn.Linear(3, 2)

    learning_rate = 0.1
    optimizer = torch.optim.SGD(list(cell.parameters()) + list(classifier.parameters()), lr=learning_rate)

    # 每一行代表一个词的演示向量。
    sequence = torch.tensor([
        [0.5, -0.2],
        [-0.3, 0.7],
        [0.1, 0.4],
    ], dtype=torch.float32)

    h = torch.zeros(3, dtype=torch.float32)
    c = torch.zeros(3, dtype=torch.float32)

    for step, x_t in enumerate(sequence, start=1):
        h, c = cell(x_t, h, c)
        print(f"第{step}步隐藏状态：", h)

    logits = classifier(h)
    print("分类分数：", logits)
    print("分类分数形状：", logits.shape)

    target = torch.tensor(1, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, target)

    print("损失：", loss)
    print("损失形状：", loss.shape)

    optimizer.zero_grad()
    loss.backward()

    weight_before = classifier.weight[0, 0].item()
    grad_value = classifier.weight.grad[0, 0].item()

    optimizer.step()

    weight_after = classifier.weight[0, 0].item()

    print("更新前的权重：", weight_before)
    print("该权重的梯度：", grad_value)
    print("更新后的权重：", weight_after)
    print("按公式计算的结果：", weight_before - learning_rate * grad_value)

    with torch.no_grad():
        h = torch.zeros(3, dtype=torch.float32)
        c = torch.zeros(3, dtype=torch.float32)

        for x_t in sequence:
            h, c = cell(x_t, h, c)

        logits_after = classifier(h)
        loss_after = criterion(logits_after, target)

    print("更新前的损失：", loss.item())
    print("更新后的损失：", loss_after.item())
