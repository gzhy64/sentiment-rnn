"""准备情感分类训练所需的数据与模型。"""

from pathlib import Path

import torch
import torch.nn as nn

from dataset import load_corpus, count_words, build_vocab, encode_text
from model import LSTMSentimentClassifier


def train_one_epoch(model, samples, word_to_id, criterion, optimizer, device):
    """遍历训练样本完成一轮参数更新，返回平均训练损失。"""
    model.train()
    total_loss = 0.0

    for label, text in samples:
        token_ids = encode_text(text, word_to_id)
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        target = torch.tensor(label, dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(samples)


def evaluate_accuracy(model, samples, word_to_id, device):
    """以评估模式计算给定样本的分类准确率。"""
    model.eval()
    correct = 0

    with torch.no_grad():
        for label, text in samples:
            token_ids = encode_text(text, word_to_id)
            input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)

            logits = model(input_ids)
            predicted_label = logits.argmax(dim=-1).item()
            correct += int(predicted_label == label)

    return correct / len(samples)


if __name__ == '__main__':
    torch.manual_seed(2026)
    device = torch.device("cuda")
    embedding_dim = 128
    hidden_size = 64

    train_path = Path(__file__).resolve().parent / "corpus" / "trains.txt"
    train_samples = load_corpus(train_path)

    dev_path = train_path.parent / "dev.txt"
    dev_samples = load_corpus(dev_path)

    word_counts = count_words(train_samples)
    word_to_id = build_vocab(word_counts)

    model = LSTMSentimentClassifier(
        vocab_size=len(word_to_id),
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 15

    best_accuracy = -1.0
    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_lstm.pth"

    for epoch in range(1, epochs + 1):
        average_loss = train_one_epoch(model, train_samples, word_to_id, criterion, optimizer, device)
        val_accuracy = evaluate_accuracy(model, dev_samples, word_to_id, device)

        print(
            f"第{epoch}轮：loss={average_loss:.4f}，"
            f"val_acc={val_accuracy:.2%}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "word_to_id": word_to_id,
                "embedding_dim": embedding_dim,
                "hidden_size": hidden_size,
                "epoch": epoch,
                "val_accuracy": val_accuracy,
            }

            torch.save(checkpoint, checkpoint_path)
            print(f"已保存最佳模型：第{epoch}轮，准确率={val_accuracy:.2%}")