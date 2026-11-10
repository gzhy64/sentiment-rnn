"""定义手写 LSTM 单元。"""

import torch
import torch.nn as nn


class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """创建遗忘门、输入门、候选记忆和输出门的线性层。"""
        super().__init__()
        combined_size = input_size + hidden_size
        self.forget_linear = nn.Linear(combined_size, hidden_size)
        self.input_linear = nn.Linear(combined_size, hidden_size)
        self.candidate_linear = nn.Linear(combined_size, hidden_size)
        self.output_linear = nn.Linear(combined_size, hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        """根据当前输入和上一时刻状态，返回新的隐藏状态与记忆状态。"""
        combined = torch.cat([x_t, h_prev], dim=-1)
        forget_gate = torch.sigmoid(self.forget_linear(combined))
        input_gate = torch.sigmoid(self.input_linear(combined))
        candidate = torch.tanh(self.candidate_linear(combined))
        output_gate = torch.sigmoid(self.output_linear(combined))
        c_new = forget_gate * c_prev + input_gate * candidate
        h_new = output_gate * torch.tanh(c_new)
        return h_new, c_new
