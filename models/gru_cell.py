"""定义手写 GRU 单元。"""

import torch
import torch.nn as nn


class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """创建重置门、更新门和生成候选状态的线性层。"""
        super().__init__()
        combined_size = input_size + hidden_size
        self.reset_linear = nn.Linear(combined_size, hidden_size)
        self.update_linear = nn.Linear(combined_size, hidden_size)
        self.candidate_input_linear = nn.Linear(input_size, hidden_size)
        self.candidate_hidden_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t, h_prev):
        """根据当前输入和上一时刻隐藏状态，返回新的隐藏状态。"""
        combined = torch.cat([x_t, h_prev], dim=-1)
        reset_gate = torch.sigmoid(self.reset_linear(combined))
        update_gate = torch.sigmoid(self.update_linear(combined))
        candidate = torch.tanh(
            self.candidate_input_linear(x_t)
            + reset_gate * self.candidate_hidden_linear(h_prev)
        )
        return update_gate * h_prev + (1 - update_gate) * candidate
