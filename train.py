"""组织 LSTM 训练，独立保存配置、逐轮指标和最佳模型。"""

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

import torch

from checkpoint import save_checkpoint
from dataset import build_vocab, count_words, load_corpus
from engine import evaluate_metrics, get_device, train_one_epoch
from models import build_model


def save_json(data, file_path):
    """先写临时 JSON，再替换正式文件，保留上一份完整记录直到写入成功。"""
    file_path = Path(file_path)
    temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(file_path)


def train_model(config, data_dir, output_dir, device):
    """按配置训练 LSTM，保存独立实验结果并返回训练记录路径。"""
    torch.manual_seed(config["seed"])
    train_path = Path(data_dir) / "trains.txt"
    dev_path = Path(data_dir) / "dev.txt"
    train_samples = load_corpus(train_path)
    dev_samples = load_corpus(dev_path)
    word_to_id = build_vocab(count_words(train_samples))

    model_name = config["model_name"]
    model = build_model(
        model_name=model_name,
        vocab_size=len(word_to_id),
        embedding_dim=config["embedding_dim"],
        hidden_size=config["hidden_size"],
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_classes = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    optimizer_class = optimizer_classes[config["optimizer"]]
    optimizer = optimizer_class(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    run_name = datetime.now().strftime(f"{model_name}_%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True)
    history_path = run_dir / "history.json"
    run_config = {
        **config,
        "train_file": str(train_path.resolve()),
        "dev_file": str(dev_path.resolve()),
        "train_samples": len(train_samples),
        "dev_samples": len(dev_samples),
        "vocab_size": len(word_to_id),
        "device": str(model.embedding.weight.device),
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
    }
    save_json(run_config, run_dir / "config.json")
    history = []
    save_json(history, history_path)
    best_accuracy = -1.0
    print("本次输出目录：", run_dir)

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_samples, word_to_id, criterion, optimizer, device
        )
        val_loss, val_accuracy = evaluate_metrics(
            model, dev_samples, word_to_id, criterion, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            }
        )
        save_json(history, history_path)
        print(
            f"第{epoch}轮：train_loss={train_loss:.6f}，"
            f"val_loss={val_loss:.6f}，train_acc={train_accuracy:.2%}，"
            f"val_acc={val_accuracy:.2%}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(
                run_dir / f"best_{model_name}.pth",
                model,
                word_to_id,
                run_config,
                epoch,
                val_accuracy,
            )
            print(f"已保存最佳模型：第{epoch}轮，准确率={val_accuracy:.2%}")

    print("训练完成，记录已保存：", history_path)
    return history_path


def main(argv=None):
    """读取运行参数，组织训练配置并启动一次独立实验。"""
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="训练手写 LSTM 情感分类模型")
    parser.add_argument("--data-dir", type=Path, default=project_dir / "corpus")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "runs")
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", help="例如 cuda:0 或 cpu；省略时自动选择。")
    parser.add_argument("--model", choices=("lstm", "gru"), default="lstm", help="选择循环神经网络类型")
    args = parser.parse_args(argv)

    config = {
        "model_name": args.model,
        "embedding_dim": args.embedding_dim,
        "hidden_size": args.hidden_size,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "seed": args.seed,
    }
    train_model(config, args.data_dir, args.output_dir, get_device(args.device))


if __name__ == "__main__":
    main()
