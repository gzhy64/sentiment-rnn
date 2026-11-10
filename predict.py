"""加载指定检查点，对真实评论进行情感预测。"""

import argparse
from pathlib import Path

import torch

from checkpoint import load_checkpoint
from dataset import encode_text, load_corpus
from engine import get_device


def predict_sentiment(text, model, word_to_id, device):
    """预测单条评论，返回情感类别编号和该类别的模型概率。"""
    input_ids = torch.tensor(
        encode_text(text, word_to_id), dtype=torch.long, device=device
    )
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(input_ids), dim=-1)
    label = probabilities.argmax(dim=-1).item()
    return label, probabilities[label].item()


def main(argv=None):
    """读取参数并预测评论，未指定文本时读取训练集第一条真实评论。"""
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="使用指定 LSTM 检查点预测情感")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--text", action="append", help="待预测评论，可重复指定；省略时读取训练样本。"
    )
    parser.add_argument("--device", help="例如 cuda:0 或 cpu；省略时自动选择。")
    args = parser.parse_args(argv)
    device = get_device(args.device)
    checkpoint_path = args.checkpoint.resolve()
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    label_names = {0: "负面", 1: "正面"}
    texts = args.text
    print("模型文件：", checkpoint_path)
    if texts is None:
        label, text = load_corpus(project_dir / "corpus" / "trains.txt")[0]
        texts = [text]
        print("读取训练集第一条评论，真实情感：", label_names[label])

    for text in texts:
        label, probability = predict_sentiment(
            text, model, checkpoint["word_to_id"], device
        )
        print("评论：", text)
        print(f"预测：{label_names[label]}，模型概率：{probability:.2%}")
        print()


if __name__ == "__main__":
    main()
