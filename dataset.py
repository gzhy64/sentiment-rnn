"""读取情感分类数据集。"""

import re
from collections import Counter
from pathlib import Path


def load_corpus(file_path):
    """读取语料文件，返回由整数标签和评论组成的样本列表。"""
    raw_text = Path(file_path).read_text(encoding="utf-8")
    pattern = r"<Polarity>([01])</Polarity>\s*<text>(.*?)</text>"
    matches = re.findall(pattern, raw_text, flags=re.DOTALL)

    samples = []
    for label, text in matches:
        samples.append((int(label), text))

    return samples


def tokenize(text):
    """将评论转为小写词列表，保留单词内部的英文撇号。"""
    return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())


def count_words(samples):
    """统计样本中每个词的总出现次数。"""
    word_counts = Counter()

    for _, text in samples:
        word_counts.update(tokenize(text))

    return word_counts


def build_vocab(word_counts):
    """根据训练集词频建立词到编号的词表，预留填充词和未知词编号。"""
    word_to_id  = {"<PAD>": 0, "<UNK>": 1}

    for word, _ in word_counts.most_common():
        word_to_id[word] = len(word_to_id)

    return word_to_id


def encode_text(text, word_to_id):
    """按词表将评论转换为编号序列，将未登录词映射到 <UNK>。"""
    unk_id = word_to_id["<UNK>"]
    token_ids = []

    for word in tokenize(text):
        token_ids.append(word_to_id.get(word, unk_id))

    return token_ids



if __name__ == '__main__':
    train_path = Path(__file__).resolve().parent / "corpus" / "trains.txt"
    train_samples = load_corpus(train_path)

    word_counts = count_words(train_samples)

    word_to_id = build_vocab(word_counts)

    first_text = train_samples[0][1]
    token_ids = encode_text(first_text, word_to_id)

    print(token_ids)

    import torch

    input_ids = torch.tensor(token_ids, dtype=torch.long)

    embedding = torch.nn.Embedding(
        num_embeddings=len(word_to_id),
        embedding_dim=2,
        padding_idx=word_to_id["<PAD>"]
    )

    word_vectors = embedding(input_ids)

    print("词编号张量形状：", input_ids)
    print("词向量形状：", word_vectors)
    print("Embedding 权重形状：", embedding.weight.shape)
    print("第一个词向量：", word_vectors[0])



