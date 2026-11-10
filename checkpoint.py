"""统一保存和恢复情感分类模型，兼容已有 LSTM 检查点。"""

from pathlib import Path

import torch

from models import build_model


def load_checkpoint(checkpoint_path, device):
    """按检查点记录的类型恢复模型，返回评估模型和检查点信息。"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint.get("config", {})
    model_name = config.get("model_name", "lstm")

    model = build_model(
        model_name=model_name,
        vocab_size=len(checkpoint["word_to_id"]),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_size=checkpoint["hidden_size"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def save_checkpoint(checkpoint_path, model, word_to_id, config, epoch, val_accuracy):
    """保存最佳模型、词表和训练配置，写完临时文件后替换目标文件。"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "word_to_id": word_to_id,
        "embedding_dim": config["embedding_dim"],
        "hidden_size": config["hidden_size"],
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "config": config,
    }
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)
