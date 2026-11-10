"""提供训练和测试共用的设备选择、单轮训练与指标评估。"""

import torch

from dataset import encode_text


def get_device(name=None):
    """返回指定设备，省略时优先选择 CUDA，否则使用 CPU。"""
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def train_one_epoch(model, samples, word_to_id, criterion, optimizer, device):
    """完成一轮参数更新，返回更新过程中累计的平均损失和准确率。"""
    model.train()
    total_loss = 0.0
    correct = 0
    for label, text in samples:
        input_ids = torch.tensor(
            encode_text(text, word_to_id), dtype=torch.long, device=device
        )
        target = torch.tensor(label, dtype=torch.long, device=device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += int(logits.argmax(dim=-1).item() == label)
    return total_loss / len(samples), correct / len(samples)


def evaluate_predictions(model, samples, word_to_id, criterion, device):
    """以评估模式返回平均损失、真实标签列表和预测标签列表。"""
    model.eval()
    total_loss = 0.0
    labels = []
    predictions = []

    with torch.no_grad():
        for label, text in samples:
            input_ids = torch.tensor(
                encode_text(text, word_to_id), dtype=torch.long, device=device
            )
            target = torch.tensor(label, dtype=torch.long, device=device)
            logits = model(input_ids)
            total_loss += criterion(logits, target).item()
            labels.append(label)
            predictions.append(logits.argmax(dim=-1).item())

    return total_loss / len(samples), labels, predictions


def evaluate_metrics(model, samples, word_to_id, criterion, device):
    """返回平均损失和准确率，供每轮训练后的验证流程使用。"""
    loss, labels, predictions = evaluate_predictions(
        model, samples, word_to_id, criterion, device
    )
    accuracy = sum(a == b for a, b in zip(labels, predictions)) / len(labels)
    return loss, accuracy
