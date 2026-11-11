#  LSTM / GRU 文本情感分类

使用 PyTorch 的基础层实现英文餐厅评论二分类：`0` 表示负面，`1` 表示正面。
LSTM 和 GRU 的门控与状态更新均手写实现，词向量从随机初始化开始学习，
两种模型共用数据处理、训练评估、检查点读写、测试、预测和绘图流程。

当前 `runs` 保存了两次同配置实验：LSTM 测试准确率为 **76.33%**，GRU 为 **80.00%**。
具体配置、验证结果、F1 和混淆矩阵见下文“已保存的实验结果”。

## 项目结构与数据流

```text
.
├── models/
│   ├── __init__.py          # 导出模型、循环单元和 build_model
│   ├── classifier.py        # LSTM/GRU 分类器及统一模型创建入口
│   ├── lstm_cell.py         # 手写 LSTM 单元
│   └── gru_cell.py          # 手写 GRU 单元
├── dataset.py              # 语料读取、分词、词表和编码
├── engine.py               # 设备选择、单轮训练、评估与逐条预测
├── checkpoint.py           # 保存与恢复模型、词表和实验配置
├── train.py                # 训练入口，保存配置、历史和最佳检查点
├── test.py                 # 独立测试，保存准确率、F1 和混淆矩阵
├── predict.py              # 单条或多条评论的情感预测
├── plot_history.py         # 读取指定实验文件夹并在原目录保存曲线
├── examples/
│   ├── inspect_sample.py   # 用真实样本检查 LSTM 中间状态和梯度
│   └── check_gru.py         # 用真实样本检查 GRU 分类器前向与反向
├── environment.yml         # 服务器环境声明
├── corpus/                 # 训练、验证、测试语料
└── runs/                   # 各次实验的配置、权重、指标与图片
```

数据流为：评论文本 → 分词 → 词编号 → Embedding → 逐词经过 LSTM/GRU → 最后隐藏状态 → 二分类分数。
长度为 `N` 的评论依次得到 `[N]` 的词编号、`[N, embedding_dim]` 的词向量、
`[hidden_size]` 的最终隐藏状态和 `[2]` 的分类分数。
LSTM 额外维护同样为 `[hidden_size]` 的记忆状态；GRU 仅传递隐藏状态。

## 环境与数据

```bash
conda env create -f environment.yml
conda activate sentiment-rnn
```

### 语料与标签

| 文件 | 用途 | 样本数 | 负面 / 正面 |
| --- | --- | ---: | ---: |
| `corpus/trains.txt` | 建立词表和更新模型参数 | 1000 | 500 / 500 |
| `corpus/dev.txt` | 每轮验证及选择最佳模型 | 200 | 100 / 100 |
| `corpus/tests.txt` | 对选定模型进行独立测试 | 300 | 150 / 150 |

每条记录由情感标签和评论文本组成，下面是训练集第一条真实样本：

```xml
<Polarity>1</Polarity>
<text>great bagels made the old-fashioned way.</text>
```

分词时转为小写，保留单词内部的英文撇号。词表仅根据完整训练集的词频建立，
当前大小为 **2476**，其中 `<PAD>` 编号为 `0`，`<UNK>` 编号为 `1`。
验证、测试和预测使用训练时保存的词表；加载模型时不会重新建立词表。

## 运行方法

### 1. 训练与验证

运行 [train.py](train.py)。LSTM 和 GRU 的参数分别为：

```text
python train.py --model lstm
```

```text
python train.py --model gru
```

当前训练入口的默认配置如下，也是下文两次已有实验使用的配置：

| 配置 | 参数或设置位置 | 默认值 |
| --- | --- | --- |
| 模型 | `--model` | `lstm`，可选 `lstm`、`gru` |
| 词向量维度 | `--embedding-dim` | 1024 |
| 隐藏状态维度 | `--hidden-size` | 64 |
| 优化器 | `train.py` 中 `config["optimizer"]` | Adam |
| 学习率 | `--learning-rate` | 0.001 |
| 权重衰减 | `--weight-decay` | 0.001 |
| 训练轮数 | `--epochs` | 20 |
| 随机种子 | `--seed` | 2026 |
| 数据文件夹 | `--data-dir` | 项目根目录下的 `corpus` |
| 实验输出根目录 | `--output-dir` | 项目根目录下的 `runs` |
| 设备 | `--device` | 自动优先选择 CUDA，否则使用 CPU |

优化器创建由 `config["optimizer"]` 控制，支持 `SGD`、`Adam` 和 `AdamW`。
切换时应将名称写入该配置，并为所选优化器设置学习率与权重衰减；运行配置会随实验保存。

**最佳模型按验证准确率选择。** 只有严格高于历史最佳值时才保存，因此多轮并列最高时，保留最先达到该准确率的轮次；最低验证损失与最高验证准确率可能出现在不同轮次。
参数选择和模型选择使用验证集，独立测试在配置确定后执行。

### 2. 独立测试

运行 [test.py](test.py)，使用 `--checkpoint` 指定具体模型文件。
例如测试当前保存的 LSTM 或 GRU：

```text
python test.py --checkpoint /home/mel/zhy/sentiment-rnn/runs/lstm_20261110_221727/best_lstm.pth
```

```text
python test.py --checkpoint /home/mel/zhy/sentiment-rnn/runs/gru_20261110_223143/best_gru.pth
```

默认读取 `corpus/tests.txt`，可通过 `--test-file` 指定其他测试文件。
结果打印到控制台，并保存为**所选检查点同目录下的 `test_metrics.json`**。
该文件包含模型名称、检查点路径、保存轮次、测试文件、样本数、平均损失、准确率、
Macro-F1、正负类别 F1，以及带类别顺序和行列说明的混淆矩阵。

### 3. 评论预测

运行 [predict.py](predict.py)，同样使用 `--checkpoint` 选择 LSTM 或 GRU 权重。
省略 `--text` 时读取训练集第一条真实评论，并显示真实标签，便于检查数据流程。
指定文本时，例如使用这条真实评论：

```text
python predict.py --checkpoint /home/mel/zhy/sentiment-rnn/runs/gru_20261110_223143/best_gru.pth --text "great bagels made the old-fashioned way."
```

重复添加 `--text` 可以依次预测多条评论，模型仅加载一次。
输出包括情感类别和 softmax 概率；该概率表示模型的预测信心，不能直接当作预测正确率。

训练、测试、预测和 LSTM 学习检查入口支持 `--device cuda:0` 或 `--device cpu`。
训练、测试、预测和绘图入口可通过 `--help` 查看其支持的参数。

### 4. 独立绘图

运行 [plot_history.py](plot_history.py)，用 **`--run-dir` 指定实验文件夹**：

```text
python plot_history.py --run-dir /home/mel/zhy/sentiment-rnn/runs/lstm_20261110_221727
```

```text
python plot_history --run-dir /home/mel/zhy/sentiment-rnn/runs/gru_20261110_223143
```

脚本读取指定文件夹的 `history.json` 和可用的 `config.json`，
将一张 `training_curves.png` 保存到**同一个文件夹**：

- 左侧：训练损失曲线。
- 右侧：训练准确率和验证准确率曲线。
- 红色星号：首次达到最高验证准确率的轮次，旁边标注 epoch 与准确率。

绘图入口接收文件夹路径，输出位置由该文件夹确定。
验证损失仍保存在 `history.json` 中，当前图中不展示。
绘图独立于训练，修改绘图代码或绘图失败后，可直接使用已保存记录重画。

### 5. 真实样本学习检查

在 IDE 中以模块方式运行，工作目录为项目根目录：

| 模块名 | 用途 | 参数说明 |
| --- | --- | --- |
| `examples.inspect_sample` | 跟踪 LSTM 的词编号、词向量、隐藏状态、记忆状态、损失和梯度 | 可指定 LSTM 的 `--checkpoint`；省略时新初始化 LSTM |
| `examples.check_gru` | 用首条真实评论检查 GRU 分类器前向与反向 | 当前固定使用 1024 维词向量、64 维隐藏状态，并自动选择设备 |

`examples.inspect_sample` 按 LSTM 的 `h、c` 接口跟踪状态，仅适用于 LSTM 检查点。
两个学习脚本都读取真实训练评论；词表根据完整训练集建立，或使用已有检查点中的词表。

## 实验产物与指标约定

每次训练创建 `runs/<模型名>_YYYYMMDD_HHMMSS/`，例如：

```text
runs/gru_20261110_223143/
├── config.json          # 训练生成：配置、数据路径、样本数量及运行环境
├── history.json         # 训练生成：逐轮训练和验证指标
├── best_gru.pth         # 训练生成：首次达到最高验证准确率的检查点
├── test_metrics.json    # 测试生成：测试指标和混淆矩阵
└── training_curves.png  # 绘图生成：两个子图及最佳验证轮次星标
```

LSTM 对应的检查点名为 `best_lstm.pth`。

- `history.json` 每轮保存 `epoch`、`train_loss`、`val_loss`、`train_accuracy`、`val_accuracy`。
- 检查点保存 `state_dict`、词表、模型维度、保存轮次、验证准确率和完整配置；恢复时根据 `config.model_name` 创建模型。
- 加载模型时使用检查点记录的维度和词表，独立于当前训练入口的默认参数。
- JSON 与检查点均先写同目录临时文件，再替换正式文件，已完成轮次的记录可以独立用于绘图。
- 检查点用于恢复评估模型，没有保存优化器状态，当前未提供断点续训入口。
- 缺少配置的旧 LSTM 检查点可按 LSTM 恢复；历史实验缺少的信息不会由当前源码补写。

F1 取值为 `0～1`，`Macro-F1` 是负面与正面 F1 的算术平均，分母为零时对应 F1 取 `0`。
准确率和 F1 在 JSON 中均保存为小数。
混淆矩阵固定为**行表示真实类别、列表示预测类别**，行列顺序均为 `[0负面, 1正面]`：

| 真实类别 / 预测类别 | 预测负面（0） | 预测正面（1） |
| --- | --- | --- |
| 真实负面（0） | TN | FP |
| 真实正面（1） | FN | TP |

矩阵四个数之和等于样本数，对角线之和除以样本数等于准确率。
当前验证集有 200 条，准确率每增加一条正确预测提高 0.5 个百分点；
测试集有 300 条，对应步长约为 0.333 个百分点。

## 已保存的实验结果

以下结果读取自当前工作目录的两组 `config.json`、`history.json` 和 `test_metrics.json`。
两次实验均使用上述默认配置：词向量 1024 维、隐藏状态 64 维、Adam、学习率 0.001、
权重衰减 0.001、20 轮、seed 2026；数据划分和词表相同。
记录的运行环境为 Python 3.11.18、PyTorch 2.7.1+cu118、`cuda:0`。

| 模型 | 首次最佳轮次 | 最佳验证准确率 | 测试准确率 | 测试 Macro-F1 | 测试损失 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM | 17 | 76.00% | 76.33% | 0.7633 | 0.8396 |
| GRU | 7 | 77.00% | 80.00% | 0.7996 | 0.6253 |

LSTM 第 17、18 轮并列最高验证准确率，使用第 17 轮检查点；
GRU 第 7、8 轮并列最高，使用第 7 轮检查点。
测试表评估的是这两个检查点，并非最后一轮模型。

| 模型 | 测试负面 F1 | 测试正面 F1 | 测试混淆矩阵 `[[TN, FP], [FN, TP]]` |
| --- | ---: | ---: | --- |
| LSTM | 0.7625 | 0.7641 | `[[114, 36], [35, 115]]` |
| GRU | 0.8089 | 0.7902 | `[[127, 23], [37, 113]]` |

结果来源：

- LSTM：`runs/lstm_20261110_221727/`，测试正确 229/300 条。
- GRU：`runs/gru_20261110_223143/`，测试正确 240/300 条。
- 两个目录均包含对应配置、逐轮记录、最佳权重、测试 JSON 和训练曲线。

在这次同配置实验中，GRU 测试准确率高约 3.67 个百分点，对应多预测正确 11 条评论。
这是单一划分、单一随机种子的实验结果，更多随机种子实验可用于检查稳定性。
重新训练的结果以新生成的实验记录为准。
