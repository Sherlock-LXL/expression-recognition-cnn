# 表情识别 CNN / Expression Recognition CNN

一个使用 PyTorch 从零训练的七分类人脸表情识别项目：手动实现轻量级残差 CNN，并通过三个模型的等权 soft-voting ensemble，将 combined FER2013 test accuracy 提升至 66.59%。

A from-scratch PyTorch project for seven-class facial-expression recognition, featuring a hand-built lightweight residual CNN and an equal-weight three-model soft-voting ensemble that reaches 66.59% accuracy on the combined FER2013 test set.

---

## 中文说明

### 项目简介

Expression Recognition CNN 是一个使用 PyTorch 从零训练的 FER2013 风格七分类人脸表情识别项目。项目不使用预训练 ResNet，而是手动实现轻量级残差结构，用于学习和理解 CNN、残差连接、Batch Normalization、学习率调度以及模型集成。

模型识别以下类别：

```text
angry, disgust, fear, happy, neutral, sad, surprise
```

项目从普通四层 CNN 逐步发展到残差 CNN，并在 V3.0.0 中使用三个独立训练的残差模型进行 soft-voting ensemble。详细逐 epoch 训练记录见 [network_optimization_log.md](docs/network_optimization_log.md)，完整测试结果见 [test_accuracy.md](docs/test_accuracy.md)。

### V3.0.0 更新

V3.0.0 的主要变化：

- 将原来的单卷积层级替换为手动实现的 `ResidualBlock`。
- 每个残差块包含两层 3×3 卷积、Batch Normalization、shortcut 和残差相加。
- 输入输出通道不一致时，shortcut 使用 1×1 卷积和 Batch Normalization 完成通道匹配。
- 保留四阶段通道设计：`1 → 16 → 32 → 64 → 128`。
- 保留 V2 的数据增强、Adam 优化器和分阶段学习率。
- 独立训练并保存三个残差模型。
- 推理时对三个模型的 Softmax 概率进行等权平均，再输出最终类别与 Top-3 概率。
- 三个单模型的 combined test accuracy 分别达到 63.90%、63.33% 和 63.65%。
- 三模型 ensemble 的 combined test accuracy 达到 **66.59%**。

与 V2.0.0 的 61.87% 相比，V3 ensemble 提高了 **4.72 个百分点**；与 V1.0.0 的 56.56% 相比，提高了 **10.03 个百分点**。

### 残差模型结构

最终模型结构以 [`src/model.py`](src/model.py) 为准。这里的网络是项目作者手动实现的轻量级 ResNet-style CNN，不是 torchvision 中的标准 ResNet，也没有使用预训练参数。

每个残差块的主分支为：

```text
Conv2d(3×3, bias=False)
BatchNorm2d
ReLU
Conv2d(3×3, bias=False)
BatchNorm2d
```

shortcut 分支：

```text
in_channels == out_channels: Identity
in_channels != out_channels: Conv2d(1×1, bias=False) + BatchNorm2d
```

两个分支相加后再执行 ReLU。完整网络为：

```text
Input: [N, 1, 48, 48]

ResidualBlock: 1 → 16
MaxPool2d(2)
        ↓
[N, 16, 24, 24]

ResidualBlock: 16 → 32
MaxPool2d(2)
        ↓
[N, 32, 12, 12]

ResidualBlock: 32 → 64
MaxPool2d(2)
        ↓
[N, 64, 6, 6]

ResidualBlock: 64 → 128
MaxPool2d(2)
        ↓
[N, 128, 3, 3]

Flatten: 128 × 3 × 3 = 1152
Linear: 1152 → 128
ReLU
Linear: 128 → 7
        ↓
[N, 7] logits
```

单个 V3 模型包含 4 个残差块和 **453,447** 个可训练参数。最后一层输出原始 logits；训练时直接交给 `CrossEntropyLoss`，模型内部不包含 Softmax。

### 三模型 Soft-Voting Ensemble

最终 `app.py` 和 `src/test.py` 都加载以下三个模型：

```text
checkpoints/best_model_res_1.pth
checkpoints/best_model_res_2.pth
checkpoints/best_model_res_3.pth
```

对于同一批输入，每个模型先独立输出 logits，再计算 Softmax 概率：

```text
p1 = softmax(model_1(x))
p2 = softmax(model_2(x))
p3 = softmax(model_3(x))

p_ensemble = (p1 + p2 + p3) / 3
prediction = argmax(p_ensemble)
```

因此，当前代码实现的是**等权 soft voting / probability averaging**，不是三个模型各自选出类别后再进行多数表决的 hard voting。Top-3 结果也来自平均后的最终概率。

三个模型结构相同，但来自独立训练过程，因此在不同类别上具有一定互补性。例如模型 1 的 `fear` 表现较好，模型 2 的 `sad` 表现较好，而集成后整体准确率和多个类别准确率进一步提高。

### 性能结果

所有结果均来自 7,178 张 combined `PublicTest + PrivateTest` 图片，不是 standalone `PrivateTest` benchmark。

#### 版本与模型总体结果

| Version / Model | Overall accuracy | Macro per-class accuracy |
|---|---:|---:|
| V1.0.0 four-layer CNN | 56.56% | 49.74% |
| V2.0.0 BatchNorm + LR decay | 61.87% | 57.98% |
| V3.0.0 residual model 1 | 63.90% | 59.86% |
| V3.0.0 residual model 2 | 63.33% | 58.97% |
| V3.0.0 residual model 3 | 63.65% | 59.88% |
| **V3.0.0 three-model ensemble** | **66.59%** | **62.89%** |

V3 ensemble 相比表现最好的单模型 1 再提高 **2.69 个百分点**。

#### V3 单模型测试结果

| Class | Model 1 | Model 2 | Model 3 |
|---|---:|---:|---:|
| angry | 56.37% (540/958) | 54.70% (524/958) | 58.56% (561/958) |
| disgust | 42.34% (47/111) | 41.44% (46/111) | 45.05% (50/111) |
| fear | 45.80% (469/1024) | 28.81% (295/1024) | 37.30% (382/1024) |
| happy | 86.13% (1528/1774) | 83.31% (1478/1774) | 85.40% (1515/1774) |
| neutral | 64.07% (790/1233) | 61.88% (763/1233) | 63.83% (787/1233) |
| sad | 43.30% (540/1247) | 61.19% (763/1247) | 48.60% (606/1247) |
| surprise | 80.99% (673/831) | 81.47% (677/831) | 80.39% (668/831) |
| **Overall** | **63.90%** | **63.33%** | **63.65%** |

#### V1 / V2 / V3 ensemble 各类别对比

| Class | V1.0.0 | V2.0.0 | V3 ensemble |
|---|---:|---:|---:|
| angry | 39.25% (376/958) | 51.25% (491/958) | **58.77% (563/958)** |
| disgust | 23.42% (26/111) | 44.14% (49/111) | **48.65% (54/111)** |
| fear | 18.65% (191/1024) | 34.96% (358/1024) | **39.16% (401/1024)** |
| happy | 85.68% (1520/1774) | 81.51% (1446/1774) | **86.58% (1536/1774)** |
| neutral | 64.48% (795/1233) | 61.23% (755/1233) | **68.69% (847/1233)** |
| sad | 43.79% (546/1247) | **57.42% (716/1247)** | 55.09% (687/1247) |
| surprise | 72.92% (606/831) | 75.33% (626/831) | **83.27% (692/831)** |
| **Overall** | **56.56%** | **61.87%** | **66.59%** |

V3 ensemble 并非每个类别都超过所有历史版本。例如 `sad` 仍略低于 V2，但 `angry`、`disgust`、`fear`、`happy`、`neutral` 和 `surprise` 均达到三个正式版本中的最好结果。

#### V3 validation 结果

| Model | Best validation accuracy |
|---|---:|
| Residual model 1 | 64.74% |
| Residual model 2 | 64.32% |
| Residual model 3 | 63.76% |

以上数值均来自项目现有日志和测试记录。项目暂未提供 confusion matrix，因此不进一步推断具体类别之间的混淆关系。

### 数据集与预处理

项目使用 `ImageFolder` 风格的 FER2013 数据目录：

```text
data/train/<class>/
data/test/<class>/
```

原始 `data/train/` 包含 28,709 张图片，并使用 `SEED = 42` 固定训练集和验证集划分：

| Split | Source | Images |
|---|---|---:|
| train | `data/train/` 的随机子集 | 25,839 |
| validation | `data/train/` 的随机子集 | 2,870 |
| test | `data/test/` | 7,178 |

训练/验证划分使用 `random_split`，不是 stratified split。`data/test/` 合并了 3,589 张 `PublicTest` 和 3,589 张 `PrivateTest` 图片。

训练阶段的数据增强：

```text
Grayscale(1)
RandomHorizontalFlip(p=0.5)
RandomRotation(10)
ColorJitter(brightness=0.1, contrast=0.1)
ToTensor()
```

Validation 与 test：

```text
Grayscale(1)
ToTensor()
```

`data/` 中的图片需要已经是 48×48。项目没有使用 Normalize。`app.py` 对外部图片额外执行 `Resize((48, 48))`。

### 训练配置

- Loss：`CrossEntropyLoss`
- Optimizer：Adam
- Batch size：64
- Epochs：20
- Epoch 1–10 learning rate：`1e-3`
- Epoch 11–20 learning rate：`3e-4`
- Scheduler：`MultiStepLR(milestones=[10], gamma=0.3)`
- Model selection：validation accuracy
- Default training output：`checkpoints/best_model.pth`

`src/train.py` 每次训练一个残差模型，并保存 validation accuracy 最高的参数。V3 的三份正式权重是三个独立训练结果，已分别保存在 `best_model_res_1.pth`、`best_model_res_2.pth` 和 `best_model_res_3.pth`。

### Checkpoints

仓库当前保留 5 份正式模型文件：

| Checkpoint | Version | Purpose |
|---|---|---|
| `best_model_4_layers.pth` | V1.0.0 | 原始四层 CNN |
| `best_model_batchnorm_lrdecay.pth` | V2.0.0 | BatchNorm + learning-rate decay 最终模型 |
| `best_model_res_1.pth` | V3.0.0 | Ensemble residual model 1 |
| `best_model_res_2.pth` | V3.0.0 | Ensemble residual model 2 |
| `best_model_res_3.pth` | V3.0.0 | Ensemble residual model 3 |

BatchNorm 固定学习率的中间实验模型已经移除，不再作为正式 checkpoint 保留。

### 推理

`samples/` 保留四张不同格式的示例图片：

```text
sample_01.jpg
sample_02.tif
sample_03.png
sample_04.bmp
```

从项目根目录运行：

```bash
py app.py sample_01.jpg
```

`app.py` 会加载三份 V3 residual checkpoint，对输入执行灰度化、48×48 缩放和 Tensor 转换，然后平均三个模型的概率并输出 Top-3。

输入最好是单人、以人脸为中心的图片。当前版本不包含人脸检测或自动裁剪。

### Windows 桌面版

V3 提供独立 Windows 桌面程序：

```text
ExpressionRecognitionCNN-v3.0.0.exe
```

桌面版保留 V2 的图片选择、预览、Top-3 概率、高 DPI 和小窗口适配设计，但内部推理模型更新为 V3 三模型 soft-voting ensemble。用户无需安装 Python、PyTorch 或 CUDA。

发布后可从 [GitHub Releases](https://github.com/Sherlock-LXL/expression-recognition-cnn/releases) 下载。

### 项目结构

```text
expression-recognition-cnn/
├── README.md
├── LICENSE
├── requirements.txt
├── app.py
├── checkpoints/
│   ├── best_model_4_layers.pth
│   ├── best_model_batchnorm_lrdecay.pth
│   ├── best_model_res_1.pth
│   ├── best_model_res_2.pth
│   └── best_model_res_3.pth
├── docs/
│   ├── network_optimization_log.md
│   └── test_accuracy.md
├── samples/
│   ├── sample_01.jpg
│   ├── sample_02.tif
│   ├── sample_03.png
│   └── sample_04.bmp
└── src/
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    ├── model.py
    ├── test.py
    └── train.py
```

`data/` 不随仓库发布，并已由 `.gitignore` 排除。

### 使用方法

安装依赖：

```bash
pip install -r requirements.txt
```

训练一个新的 residual model：

```bash
py -m src.train
```

测试 V3 三模型 ensemble：

```bash
py -m src.test
```

对样例图片进行推理：

```bash
py app.py <filename>
```

也可以使用 `python` 代替 Windows Python Launcher 的 `py`。

### Requirements

核心依赖以 `requirements.txt` 为准：

```text
torch>=2.0
torchvision>=0.15
Pillow>=9.0
```

项目可以在 CPU 上运行。GPU 加速取决于本地 PyTorch 与 CUDA 环境。

### 方法说明与限制

- 模型选择使用 validation accuracy；combined test set 用于最终评估。
- V3 的三个 checkpoint 共享相同结构，但来自独立训练过程。
- Ensemble 对三个模型等权平均，没有学习额外的融合权重。
- Softmax 输出表示七个类别之间的相对置信度，不应视为现实世界中的绝对概率。
- 模型只接收 48×48 单通道输入。
- App 不包含人脸检测和自动裁剪。
- 测试结果是 combined `PublicTest + PrivateTest` accuracy，不是 standalone `PrivateTest` benchmark。
- 本项目用于 CNN、残差结构和 ensemble 学习，不以达到 SOTA 为目标。

### 项目分工与致谢

项目作者负责模型设计、PyTorch 实现、训练、实验、测试、推理应用和最终技术决策。

ChatGPT / OpenAI 用于 CNN 与 PyTorch 学习辅助、概念解释、代码 review/debug guidance、实验设计建议、结果分析和文档整理。项目核心模型、训练、实验、测试与应用均由项目作者完成。

### License

本项目采用 MIT License，详见 [`LICENSE`](LICENSE)。

---

## English

### Overview

Expression Recognition CNN is a seven-class, FER2013-style facial-expression recognition project built and trained from scratch with PyTorch. V3.0.0 replaces the previous plain convolutional stages with a hand-built lightweight residual CNN and combines three independently trained models through equal-weight soft voting.

The project predicts:

```text
angry, disgust, fear, happy, neutral, sad, surprise
```

It does not use torchvision's pretrained ResNet models. The residual blocks, training pipeline, experiments, evaluation, and inference application are implemented within this project.

Detailed records are available in [network_optimization_log.md](docs/network_optimization_log.md) and [test_accuracy.md](docs/test_accuracy.md).

### V3.0.0 Highlights

- Replaces each plain convolutional stage with a custom `ResidualBlock`.
- Uses two 3×3 convolutions and Batch Normalization in each main branch.
- Uses identity shortcuts when channel counts match.
- Uses a 1×1 convolution and Batch Normalization for channel-changing shortcuts.
- Retains four stages with channels `1 → 16 → 32 → 64 → 128`.
- Retains Adam, data augmentation, and staged learning-rate decay from V2.
- Trains and preserves three independent residual checkpoints.
- Averages the three models' Softmax probability distributions at inference time.
- Reaches 63.90%, 63.33%, and 63.65% combined test accuracy with the individual models.
- Reaches **66.59%** combined test accuracy with the final ensemble.

### Residual Architecture

Each residual block contains:

```text
Main branch:
Conv2d(3×3, bias=False)
BatchNorm2d
ReLU
Conv2d(3×3, bias=False)
BatchNorm2d

Shortcut:
Identity
or Conv2d(1×1, bias=False) + BatchNorm2d

Output:
ReLU(main + shortcut)
```

The complete network is:

```text
Input [N, 1, 48, 48]
ResidualBlock 1 → 16   + MaxPool2d
ResidualBlock 16 → 32  + MaxPool2d
ResidualBlock 32 → 64  + MaxPool2d
ResidualBlock 64 → 128 + MaxPool2d
Flatten 128×3×3
Linear 1152 → 128
ReLU
Linear 128 → 7
```

Each model contains four residual blocks and **453,447** trainable parameters. It returns raw logits and does not include Softmax inside the model definition.

### Soft-Voting Ensemble

The final inference and test paths load:

```text
best_model_res_1.pth
best_model_res_2.pth
best_model_res_3.pth
```

For an input `x`:

```text
p_ensemble = (
    softmax(model_1(x))
    + softmax(model_2(x))
    + softmax(model_3(x))
) / 3
```

This is probability averaging, or equal-weight **soft voting**. It is not hard majority voting over three predicted labels.

### Performance

| Version / Model | Overall accuracy | Macro per-class accuracy |
|---|---:|---:|
| V1.0.0 four-layer CNN | 56.56% | 49.74% |
| V2.0.0 BatchNorm + LR decay | 61.87% | 57.98% |
| V3 residual model 1 | 63.90% | 59.86% |
| V3 residual model 2 | 63.33% | 58.97% |
| V3 residual model 3 | 63.65% | 59.88% |
| **V3 three-model ensemble** | **66.59%** | **62.89%** |

The ensemble improves by 4.72 percentage points over V2 and by 2.69 points over the strongest V3 single model.

| Class | V1.0.0 | V2.0.0 | V3 ensemble |
|---|---:|---:|---:|
| angry | 39.25% | 51.25% | **58.77%** |
| disgust | 23.42% | 44.14% | **48.65%** |
| fear | 18.65% | 34.96% | **39.16%** |
| happy | 85.68% | 81.51% | **86.58%** |
| neutral | 64.48% | 61.23% | **68.69%** |
| sad | 43.79% | **57.42%** | 55.09% |
| surprise | 72.92% | 75.33% | **83.27%** |

These results use all 7,178 combined PublicTest and PrivateTest images. They are not standalone PrivateTest benchmark results.

### Dataset and Training

The project expects an `ImageFolder` layout:

```text
data/train/<class>/
data/test/<class>/
```

The 28,709 training images are split into 25,839 training and 2,870 validation images with `SEED = 42`. The split uses `random_split` and is not class-stratified. The combined test set contains 7,178 images.

Training augmentation:

```text
Grayscale(1)
RandomHorizontalFlip(p=0.5)
RandomRotation(10)
ColorJitter(brightness=0.1, contrast=0.1)
ToTensor()
```

Training configuration:

- `CrossEntropyLoss`
- Adam optimizer
- Batch size 64
- 20 epochs
- Learning rate `1e-3` for epochs 1–10
- Learning rate `3e-4` for epochs 11–20
- `MultiStepLR(milestones=[10], gamma=0.3)`
- Best-checkpoint selection by validation accuracy

### Checkpoints

The repository preserves five official checkpoint files:

| Checkpoint | Role |
|---|---|
| `best_model_4_layers.pth` | V1.0.0 final model |
| `best_model_batchnorm_lrdecay.pth` | V2.0.0 final model |
| `best_model_res_1.pth` | V3 ensemble model 1 |
| `best_model_res_2.pth` | V3 ensemble model 2 |
| `best_model_res_3.pth` | V3 ensemble model 3 |

The fixed-learning-rate BatchNorm intermediate checkpoint has been removed.

### Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train one residual model:

```bash
py -m src.train
```

Evaluate the three-model ensemble:

```bash
py -m src.test
```

Run Top-3 inference on an image stored under `samples/`:

```bash
py app.py sample_01.jpg
```

### Windows Application

The V3 Windows release asset is named:

```text
ExpressionRecognitionCNN-v3.0.0.exe
```

It preserves the V2 file picker, image preview, Top-3 display, high-DPI support, and responsive layout while replacing the inference backend with the V3 three-model ensemble. It runs on CPU and does not require a local Python, PyTorch, or CUDA installation.

Download it from [GitHub Releases](https://github.com/Sherlock-LXL/expression-recognition-cnn/releases).

### Limitations

- Inputs are converted to 1×48×48 grayscale tensors.
- The application does not perform face detection or automatic cropping.
- Ensemble probabilities are relative confidence values, not calibrated real-world probabilities.
- Reported accuracy uses combined PublicTest and PrivateTest.
- This is a learning and experimentation project rather than a state-of-the-art FER system.

### Acknowledgements

The project author completed the model design, PyTorch implementation, training, experiments, testing, inference application, and final technical decisions.

ChatGPT / OpenAI supported learning, concept explanations, code review and debugging guidance, experimental-design suggestions, result analysis, and documentation organization. The core models, training, experiments, tests, and application were completed by the project author.

### License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
