"""读取真实情感语料，完成分词、词频统计和词编号转换。"""

import re
from collections import Counter
from pathlib import Path


def load_corpus(file_path):
    """读取语料文件，返回整数标签与评论组成的非空样本列表。"""
    raw_text = Path(file_path).read_text(encoding="utf-8")
    pattern = r"<Polarity>([01])</Polarity>\s*<text>(.*?)</text>"
    matches = re.findall(pattern, raw_text, flags=re.DOTALL)
    if not matches:
        raise ValueError(f"未从语料文件读取到有效样本：{file_path}")
    return [(int(label), text) for label, text in matches]


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
    """根据训练集词频建立词表，预留填充词和未知词编号。"""
    word_to_id = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in word_counts.most_common():
        word_to_id[word] = len(word_to_id)
    return word_to_id


def encode_text(text, word_to_id):
    """按词表将非空评论转换为词编号，将未登录词映射到 <UNK>。"""
    tokens = tokenize(text)
    if not tokens:
        raise ValueError("评论必须包含可识别的词。")
    unk_id = word_to_id["<UNK>"]
    return [word_to_id.get(word, unk_id) for word in tokens]
