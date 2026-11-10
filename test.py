"""评估指定模型，保存测试损失、准确率、F1 和混淆矩阵。"""

import argparse
import json
from pathlib import Path

import torch

from checkpoint import load_checkpoint
from dataset import load_corpus
from engine import evaluate_predictions, get_device


def classification_metrics(labels, predictions):
    """计算二分类混淆矩阵、每类 F1 和 Macro-F1，除零时取 0。"""
    if not labels or len(labels) != len(predictions):
        raise ValueError("真实标签和预测标签必须非空且数量一致。")

    matrix = [[0, 0], [0, 0]]
    for label, prediction in zip(labels, predictions):
        matrix[label][prediction] += 1

    tn, fp = matrix[0]
    fn, tp = matrix[1]

    positive_denominator = 2 * tp + fp + fn
    negative_denominator = 2 * tn + fp + fn
    positive_f1 = 2 * tp / positive_denominator if positive_denominator else 0.0
    negative_f1 = 2 * tn / negative_denominator if negative_denominator else 0.0

    return {
        "f1_macro": (negative_f1 + positive_f1) / 2,
        "f1_negative": negative_f1,
        "f1_positive": positive_f1,
        "confusion_matrix": matrix,
        "label_order": [0, 1],
        "label_names": ["负面", "正面"],
        "confusion_matrix_axes": {"rows": "true", "columns": "predicted"},
    }


def evaluate_checkpoint(checkpoint_path, test_path, device):
    """加载模型并评估测试集，返回检查点信息、损失和分类指标。"""
    checkpoint_path = Path(checkpoint_path).resolve()
    test_path = Path(test_path).resolve()
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    samples = load_corpus(test_path)

    test_loss, labels, predictions = evaluate_predictions(
        model,
        samples,
        checkpoint["word_to_id"],
        torch.nn.CrossEntropyLoss(),
        device,
    )
    accuracy = sum(a == b for a, b in zip(labels, predictions)) / len(labels)

    return {
        "checkpoint_path": str(checkpoint_path),
        "model_name": checkpoint.get("config", {}).get("model_name", "lstm"),
        "epoch": checkpoint["epoch"],
        "test_file": str(test_path),
        "sample_count": len(samples),
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        **classification_metrics(labels, predictions),
    }


def main(argv=None):
    """读取测试参数，打印指标并将结果 JSON 保存到检查点所在目录。"""
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="评估 LSTM/GRU 情感分类模型")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=project_dir / "corpus" / "tests.txt")
    parser.add_argument("--device", help="例如 cuda:0 或 cpu；省略时自动选择。")
    args = parser.parse_args(argv)

    results = evaluate_checkpoint(
        args.checkpoint, args.test_file, get_device(args.device)
    )

    result_path = Path(results["checkpoint_path"]).parent / "test_metrics.json"
    temporary_path = result_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(result_path)

    print("模型文件：", results["checkpoint_path"])
    print("最佳模型轮次：", results["epoch"])
    print("测试集样本数量：", results["sample_count"])
    print(f"测试集损失：{results['test_loss']:.6f}")
    print(f"测试集准确率：{results['test_accuracy']:.2%}")
    print(f"Macro-F1：{results['f1_macro']:.4f}")
    print(f"正面 F1：{results['f1_positive']:.4f}")
    print("混淆矩阵（行=真实，列=预测；类别顺序：0负面、1正面）：")
    for row in results["confusion_matrix"]:
        print(row)
    print("测试结果已保存：", result_path)


if __name__ == "__main__":
    main()