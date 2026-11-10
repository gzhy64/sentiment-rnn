"""使用指定检查点评估独立测试集。"""

import argparse
from pathlib import Path

import torch

from checkpoint import load_checkpoint
from dataset import load_corpus
from engine import evaluate_metrics, get_device


def evaluate_checkpoint(checkpoint_path, test_path, device):
    """加载指定模型并评估测试集，返回检查点信息和测试指标。"""
    checkpoint_path = Path(checkpoint_path).resolve()
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    test_samples = load_corpus(test_path)
    test_loss, test_accuracy = evaluate_metrics(
        model,
        test_samples,
        checkpoint["word_to_id"],
        torch.nn.CrossEntropyLoss(),
        device,
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "epoch": checkpoint["epoch"],
        "sample_count": len(test_samples),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }


def main(argv=None):
    """读取测试参数，调用测试函数并打印评估结果。"""
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="评估指定的 LSTM 检查点")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--test-file", type=Path, default=project_dir / "corpus" / "tests.txt"
    )
    parser.add_argument("--device", help="例如 cuda:0 或 cpu；省略时自动选择。")
    args = parser.parse_args(argv)
    results = evaluate_checkpoint(
        args.checkpoint, args.test_file, get_device(args.device)
    )
    print("模型文件：", results["checkpoint_path"])
    print("最佳模型轮次：", results["epoch"])
    print("测试集样本数量：", results["sample_count"])
    print(f"测试集损失：{results['test_loss']:.6f}")
    print(f"测试集准确率：{results['test_accuracy']:.2%}")


if __name__ == "__main__":
    main()
