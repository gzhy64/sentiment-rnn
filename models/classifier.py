"""定义由词向量、手写 LSTM 和分类层组成的情感分类模型。"""

import torch.nn as nn

from .lstm_cell import MyLSTMCell
from .gru_cell import MyGRUCell


class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        """创建词向量层、LSTM 单元和二分类层，并保存隐藏状态维度。"""
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.cell = MyLSTMCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids):
        """将单条评论的词编号序列转换为两个情感类别的分数。"""
        word_vectors = self.embedding(input_ids)
        h = word_vectors.new_zeros((self.hidden_size,))
        c = word_vectors.new_zeros((self.hidden_size,))
        for x_t in word_vectors:
            h, c = self.cell(x_t, h, c)
        return self.classifier(h)


class GRUSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        """创建词向量层、GRU 单元和二分类层。"""
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.cell = MyGRUCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids):
        """依次处理评论中的词向量，返回两个情感类别的分数。"""
        word_vectors = self.embedding(input_ids)
        h = word_vectors.new_zeros((self.hidden_size,))

        for x_t in word_vectors:
            h = self.cell(x_t, h)

        return self.classifier(h)


def build_model(model_name, vocab_size, embedding_dim, hidden_size):
    """根据模型类型创建 LSTM 或 GRU 情感分类器。"""
    model_names = {
        "lstm": LSTMSentimentClassifier,
        "gru": GRUSentimentClassifier,
    }
    if model_name not in model_names:
        raise ValueError(f"不支持的模型名称: {model_name}")

    return model_names[model_name](vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size)

