"""独立读取训练记录，在一张图片中对比训练与验证指标。"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator, PercentFormatter  # noqa: E402


def plot_history(run_dir):
    """读取指定实验文件夹的记录，并将训练曲线保存到该文件夹。"""
    run_dir = Path(run_dir).resolve()
    history_path = run_dir / "history.json"
    config_path = run_dir / "config.json"
    model_name = "LSTM"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_name = config.get("model_name", "lstm").upper()
    history = json.loads(history_path.read_text(encoding="utf-8"))

    if not history:
        raise ValueError("训练记录为空，无法绘图。")
    required_keys = {
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_accuracy",
    }
    if any(not required_keys.issubset(record) for record in history):
        raise ValueError("训练记录需要包含轮次及训练、验证的损失和准确率。")

    epochs = [record["epoch"] for record in history]
    best_record = max(history, key=lambda record: (record["val_accuracy"], -record["epoch"]))
    best_epoch = best_record["epoch"]
    best_accuracy = best_record["val_accuracy"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, (train_key, val_key, ylabel) in zip(
        axes,
        [
            ("train_loss", None, "Loss"),
            ("train_accuracy", "val_accuracy", "Accuracy"),
        ],
    ):
        train_values = [record[train_key] for record in history]
        ax.plot(epochs, train_values, label="Train", color="#0072B2", linewidth=2)
        if val_key is not None:
            val_values = [record[val_key] for record in history]
            ax.plot(epochs, val_values, label="Validation", color="#D55E00", linewidth=2)

        ax.set(xlabel="Epoch", ylabel=ylabel, title=f"{model_name} {ylabel}")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
        ax.legend()

        if ylabel == "Accuracy":
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.scatter(
                best_epoch,
                best_accuracy,
                marker="*",
                s=220,
                color="#D62728",
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )

            on_left = best_epoch <= (min(epochs) + max(epochs)) / 2
            ax.annotate(
                f"First best: {best_accuracy:.2%}\nEpoch: {best_epoch}",
                xy=(best_epoch, best_accuracy),
                xytext=(12, -20) if on_left else (-12, -20),
                textcoords="offset points",
                ha="left" if on_left else "right",
                va="top",
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.8",
                    "alpha": 0.9,
                },
            )
        else:
            ax.set_ylim(bottom=0)

    fig.tight_layout()
    figure_path = run_dir / "training_curves.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    return figure_path


def main(argv=None):
    """读取实验文件夹参数，在该文件夹中读取记录并保存曲线。"""
    parser = argparse.ArgumentParser(
        description="读取指定实验文件夹的 history.json 并绘图"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="包含 history.json 的实验文件夹路径。",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    print("读取训练记录：", run_dir / "history.json")

    figure_path = plot_history(run_dir)
    print("图片已保存：", figure_path)


if __name__ == "__main__":
    main()
