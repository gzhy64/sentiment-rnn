"""定义基于手写 LSTM 的文本情感分类模型。"""

import torch
import torch.nn as nn

from lstm_cell import MyLSTMCell


class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        """创建词向量层、LSTM 单元和二分类层，并保存隐藏状态维度。"""
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.cell = MyLSTMCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, inputs_ids):
        """将单条评论的词编号序列转换为两个情感类别的分数。"""
        word_vectors = self.embedding(inputs_ids)

        h = word_vectors.new_zeros((self.hidden_size,))
        c = word_vectors.new_zeros((self.hidden_size,))

        for x_t in word_vectors:
            h, c = self.cell(x_t, h, c)

        return self.classifier(h)


if __name__ == '__main__':
    embedding_dim = 128
    hidden_size = 64

    device = torch.device("cuda")

    model = LSTMSentimentClassifier(
        vocab_size=2476,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size
    ).to(device)

    print("词向量权重形状：", model.embedding.weight.shape)
    print("遗忘门权重形状：", model.cell.forget_linear.weight.shape)
    print("分类层权重形状：", model.classifier.weight.shape)

    input_ids = torch.tensor([31, 333, 237, 2, 210, 1117, 134], dtype=torch.long, device=device)
    logits = model(input_ids)

    print("分类分数：", logits)
    print("分类分数形状：", logits.shape)

    print("模型参数设备：", model.embedding.weight.device)
    print("输入设备：", input_ids.device)
    print("输出设备：", logits.device)
