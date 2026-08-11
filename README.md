# Expression Recognition CNN

## 中文

### 项目简介

Expression Recognition CNN 是一个基于 PyTorch、从零训练的 FER2013 风格七分类人脸表情识别项目。模型接收 1×48×48 灰度人脸图像，并预测以下类别：

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

项目用于配合 CS231n 与 CNN 学习过程，通过逐步实验理解卷积网络、数据增强、Batch Normalization、优化方法以及 validation/test protocol。项目不使用大规模预训练分类模型，也不以达到当前最先进性能为目标。

### V2.0.0 更新

V2.0.0 在 V1 四层 CNN 的基础上完成以下更新：

- 在每个卷积层后加入 `BatchNorm2d`。
- 保留训练阶段的数据增强。
- 使用分阶段学习率：前 10 个 epoch 为 `1e-3`，第 11–20 个 epoch 为 `3e-4`。
- 最终模型不使用 Dropout。
- combined test accuracy 从 56.56% 提升到 61.87%。
- 原本表现较弱的多个类别得到改善，macro average per-class accuracy 从约 49.74% 提升到约 57.98%。

V2.0.0 的最终 checkpoint 为：

```text
checkpoints/best_model_learning_rate_decay.pth
```

该 checkpoint 对应四层 CNN、Batch Normalization、learning-rate decay，并且不包含 Dropout。正式测试和最终 `app.py` 推理均使用该模型。

### 模型结构

最终 V2 模型结构以 `src/model.py` 为准：

```text
Input: [N, 1, 48, 48]

Conv2d: 1 → 16
BatchNorm2d(16)
ReLU
MaxPool2d
        ↓
[N, 16, 24, 24]

Conv2d: 16 → 32
BatchNorm2d(32)
ReLU
MaxPool2d
        ↓
[N, 32, 12, 12]

Conv2d: 32 → 64
BatchNorm2d(64)
ReLU
MaxPool2d
        ↓
[N, 64, 6, 6]

Conv2d: 64 → 128
BatchNorm2d(128)
ReLU
MaxPool2d
        ↓
[N, 128, 3, 3]

Flatten
        ↓
[N, 1152]

Linear: 1152 → 128
ReLU
Linear: 128 → 7
        ↓
[N, 7] logits
```

最后一层直接输出 logits，并交给 `CrossEntropyLoss`。训练模型本身不包含 Softmax；Softmax 只在推理阶段用于生成 Top-3 类别概率。

### 数据集与预处理

项目使用 `ImageFolder` 风格的 FER2013 数据目录：

```text
data/train/<class>/
data/test/<class>/
```

目录中的类别顺序为：

```text
angry, disgust, fear, happy, neutral, sad, surprise
```

原始 `data/train/` 包含 28,709 张图片，并使用 `SEED = 42` 进行固定随机划分：

| Split | Source | Images |
|---|---|---:|
| train | `data/train/` 的随机子集 | 25,839 |
| validation | `data/train/` 的随机子集 | 2,870 |
| test | `data/test/` | 7,178 |

训练集与验证集使用 `random_split`，不是按类别进行的 stratified split。

`data/test/` 由 3,589 张 `PublicTest` 图片和 3,589 张 `PrivateTest` 图片合并组成。因此本文的测试结果是 combined `PublicTest + PrivateTest` accuracy，不是 FER2013 standalone `PrivateTest` benchmark。

训练阶段的数据增强为：

```text
Grayscale(num_output_channels=1)
RandomHorizontalFlip(p=0.5)
RandomRotation(10)
ColorJitter(brightness=0.1, contrast=0.1)
ToTensor()
```

Validation 与 test 预处理为：

```text
Grayscale(1)
ToTensor()
```

`data/` 中的图片需要已经是 48×48。项目没有使用 Normalize。`app.py` 对外部图片额外执行：

```text
Grayscale(num_output_channels=1)
Resize((48, 48))
ToTensor()
```

### 训练配置

- Loss：`CrossEntropyLoss`
- Optimizer：Adam
- Batch size：64
- Epochs：20
- Epoch 1–10 learning rate：`1e-3`
- Epoch 11–20 learning rate：`3e-4`
- Model selection：validation accuracy
- Checkpoint：保存 validation accuracy 最高的模型参数

当前训练代码使用 `MultiStepLR(milestones=[10], gamma=0.3)`，并在每个 epoch 结束后更新学习率。这使第 11 个 epoch 开始时的学习率从 `1e-3` 降为 `3e-4`。

### 性能对比

#### Validation 演化

| Version / Experiment | Best validation accuracy |
|---|---:|
| V1.0.0 baseline | 57.84% |
| Four-layer CNN + BatchNorm, fixed learning rate | 59.97% |
| Four-layer CNN + BatchNorm + learning-rate decay | 61.05% |

`61.05%` 对应 V2.0.0 最终方案，原始记录为 `61.045296...%`。

#### 总体测试结果

| Version | Overall accuracy | Macro average per-class accuracy |
|---|---:|---:|
| V1.0.0 | 56.56% | 49.74% |
| V2.0.0 | 61.87% | 57.98% |
| Change | +5.31 percentage points | +8.24 percentage points |

#### 各类别测试结果

| Class | V1.0.0 | V2.0.0 | Change |
|---|---:|---:|---:|
| angry | 39.25% (376/958) | 51.25% (491/958) | +12.00 pp |
| disgust | 23.42% (26/111) | 44.14% (49/111) | +20.72 pp |
| fear | 18.65% (191/1024) | 34.96% (358/1024) | +16.31 pp |
| happy | 85.68% (1520/1774) | 81.51% (1446/1774) | −4.17 pp |
| neutral | 64.48% (795/1233) | 61.23% (755/1233) | −3.25 pp |
| sad | 43.79% (546/1247) | 57.42% (716/1247) | +13.63 pp |
| surprise | 72.92% (606/831) | 75.33% (626/831) | +2.41 pp |

V2 在总体准确率提升的同时，明显改善了 V1 中表现较弱的 `angry`、`disgust`、`fear` 和 `sad`。`surprise` 小幅提高，`happy` 与 `neutral` 有一定下降。因此，V2 的主要变化不是所有类别同时提高，而是整体表现提高且类别间表现更加均衡。

当前项目没有提供 confusion matrix，因此不进一步推断具体类别之间的混淆关系。

### 优化与消融实验

- **LayerNorm**：在本次实验中没有观察到明确优于最终 baseline 的收益，因此没有进入 V2 最终模型。
- **BatchNorm**：在本次实验中加快了收敛，并将 best validation accuracy 提高到约 59.97%。
- **固定学习率的 BatchNorm**：后期 validation performance 存在较明显波动。
- **BatchNorm + learning-rate decay**：第 11 个 epoch 将学习率从 `1e-3` 降到 `3e-4`，best validation accuracy 提高到 61.05%，因此被选为最终 V2 方案。
- **Dropout(p=0.3)**：train loss 明显升高，说明正则化在该实验中生效，但 best validation accuracy 约为 60.35%，没有超过无 Dropout 的 61.05%。因此最终 V2 不使用 Dropout。

这些结论仅描述当前实验结果，不表示 LayerNorm 或 Dropout 在其他模型和训练设置中一定无效。现有逐 epoch 训练记录见 `docs/network_optimization_log.md`。

### 推理

`samples/` 当前保留四张内容不同的示例图片，分别使用 JPG、TIF、PNG 和 BMP 格式：

```text
sample_01.jpg
sample_02.tif
sample_03.png
sample_04.bmp
```

将待识别图片放入 `samples/`，然后从项目根目录运行：

```bash
py app.py <filename>
```

例如：

```bash
py app.py sample_01.jpg
```

最终 `app.py` 使用：

```text
checkpoints/best_model_learning_rate_decay.pth
```

程序会将输入转换为 `[1, 1, 48, 48]`，使用 `model.eval()` 和 `torch.no_grad()` 完成推理，并按概率从高到低输出 Top-3 类别。

输入最好是单人、以人脸为中心的图片。当前版本不包含人脸检测或自动裁剪功能。

### 项目结构

```text
expression-recognition-cnn/
├── .gitignore
├── LICENSE
├── README.md
├── app.py
├── requirements.txt
│
├── checkpoints/
│   ├── best_model_4_layers.pth
│   ├── best_model_batchnorm.pth
│   └── best_model_learning_rate_decay.pth
│
├── docs/
│   ├── network_optimization_log.md
│   └── test_accuracy.md
│
├── samples/
│   ├── sample_01.jpg
│   ├── sample_02.tif
│   ├── sample_03.png
│   └── sample_04.bmp
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── test.py
│   └── train.py
│
└── data/
    ├── train/
    └── test/
```

Checkpoint 说明：

- `best_model_4_layers.pth`：V1.0.0 四层 CNN。
- `best_model_batchnorm.pth`：BatchNorm 固定学习率中间实验。
- `best_model_learning_rate_decay.pth`：V2.0.0 最终模型。

`data/` 不随仓库发布，并已由 `.gitignore` 排除。

### 使用方法

安装依赖：

```bash
pip install -r requirements.txt
```

从项目根目录训练：

```bash
py -m src.train
```

测试 V2 最终 checkpoint：

```bash
py -m src.test
```

推理：

```bash
py app.py <filename>
```

也可以使用 `python` 替代 Windows Python Launcher 的 `py`。

### Requirements

核心依赖以 `requirements.txt` 为准：

```text
torch>=2.0
torchvision>=0.15
Pillow>=9.0
```

GPU 加速取决于本地 PyTorch 与 CUDA 环境；项目也可以在 CPU 上运行。

### 方法说明

- 模型选择只使用 validation set，并根据 validation accuracy 保存最佳 checkpoint。
- Combined test set 只用于最终评估，不参与模型选择。
- V2.0.0 的 test 结果已经被查看。后续版本开发不应根据这组 test 结果继续调整模型结构或超参数，而应继续只依赖 validation set。
- Softmax 输出表示模型在七个类别之间的相对置信度，不应解释为现实世界中的绝对概率。

### 项目限制

- 模型只接收 48×48 单通道输入。
- App 不包含人脸检测与裁剪。
- 测试结果来自 combined `PublicTest + PrivateTest`，不能作为 standalone `PrivateTest` benchmark。
- 本项目是 CNN 学习与实验项目，不以 SOTA 表情识别性能为目标。

### 项目分工与致谢

项目作者负责模型设计、PyTorch 实现、训练、实验、测试、推理 app 和最终技术决策。

ChatGPT / OpenAI 在项目中用于 CNN 与 PyTorch 学习辅助、概念解释、代码 review/debug guidance、实验设计建议、结果分析和文档整理。项目核心模型、训练、实验、测试与 app 均由项目作者完成。

### License

本项目采用 MIT License，详见 `LICENSE`。

---

## English

### Project Overview

Expression Recognition CNN is a seven-class, FER2013-style facial-expression recognition project built and trained from scratch with PyTorch. The model accepts 1×48×48 grayscale face images and predicts the following classes:

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

The project accompanies the study of CS231n and convolutional neural networks. Its experiments focus on convolutional architectures, data augmentation, Batch Normalization, optimization, and validation/test protocol. It does not use a large pretrained classification model and is not intended to achieve state-of-the-art performance.

### V2.0.0 Update

V2.0.0 makes the following changes to the four-layer V1 CNN:

- Adds `BatchNorm2d` after every convolutional layer.
- Retains training-time data augmentation.
- Uses a staged learning rate: `1e-3` for epochs 1–10 and `3e-4` for epochs 11–20.
- Does not use Dropout in the final model.
- Improves combined test accuracy from 56.56% to 61.87%.
- Improves several previously weak classes and raises macro average per-class accuracy from approximately 49.74% to 57.98%.

The final V2.0.0 checkpoint is:

```text
checkpoints/best_model_learning_rate_decay.pth
```

This checkpoint corresponds to the four-layer CNN with Batch Normalization and learning-rate decay, without Dropout. Official testing and the final `app.py` inference path use this model.

### Model Architecture

The final V2 architecture follows `src/model.py`:

```text
Input: [N, 1, 48, 48]

Conv2d: 1 → 16
BatchNorm2d(16)
ReLU
MaxPool2d
        ↓
[N, 16, 24, 24]

Conv2d: 16 → 32
BatchNorm2d(32)
ReLU
MaxPool2d
        ↓
[N, 32, 12, 12]

Conv2d: 32 → 64
BatchNorm2d(64)
ReLU
MaxPool2d
        ↓
[N, 64, 6, 6]

Conv2d: 64 → 128
BatchNorm2d(128)
ReLU
MaxPool2d
        ↓
[N, 128, 3, 3]

Flatten
        ↓
[N, 1152]

Linear: 1152 → 128
ReLU
Linear: 128 → 7
        ↓
[N, 7] logits
```

The final layer returns raw logits to `CrossEntropyLoss`. The training model does not contain Softmax; Softmax is used only during inference to produce Top-3 class probabilities.

### Dataset and Preprocessing

The project uses a FER2013-style `ImageFolder` directory layout:

```text
data/train/<class>/
data/test/<class>/
```

The directory-based class order is:

```text
angry, disgust, fear, happy, neutral, sad, surprise
```

The original `data/train/` directory contains 28,709 images. It is split with `SEED = 42` as follows:

| Split | Source | Images |
|---|---|---:|
| train | Random subset of `data/train/` | 25,839 |
| validation | Random subset of `data/train/` | 2,870 |
| test | `data/test/` | 7,178 |

The training/validation split uses `random_split`; it is not a class-stratified split.

`data/test/` combines 3,589 `PublicTest` images and 3,589 `PrivateTest` images. The reported result is therefore combined `PublicTest + PrivateTest` accuracy, not a standalone FER2013 `PrivateTest` benchmark.

Training-time augmentation is:

```text
Grayscale(num_output_channels=1)
RandomHorizontalFlip(p=0.5)
RandomRotation(10)
ColorJitter(brightness=0.1, contrast=0.1)
ToTensor()
```

Validation and test preprocessing is:

```text
Grayscale(1)
ToTensor()
```

Images under `data/` must already be 48×48. The project does not use Normalize. For external images, `app.py` additionally applies:

```text
Grayscale(num_output_channels=1)
Resize((48, 48))
ToTensor()
```

### Training Configuration

- Loss: `CrossEntropyLoss`
- Optimizer: Adam
- Batch size: 64
- Epochs: 20
- Epoch 1–10 learning rate: `1e-3`
- Epoch 11–20 learning rate: `3e-4`
- Model selection: validation accuracy
- Checkpoint: parameters with the highest validation accuracy

The current training code uses `MultiStepLR(milestones=[10], gamma=0.3)` and updates the scheduler after each epoch. The learning rate therefore changes from `1e-3` to `3e-4` at the start of epoch 11.

### Performance Comparison

#### Validation Evolution

| Version / Experiment | Best validation accuracy |
|---|---:|
| V1.0.0 baseline | 57.84% |
| Four-layer CNN + BatchNorm, fixed learning rate | 59.97% |
| Four-layer CNN + BatchNorm + learning-rate decay | 61.05% |

The `61.05%` result corresponds to the final V2.0.0 configuration; the raw value was `61.045296...%`.

#### Overall Test Results

| Version | Overall accuracy | Macro average per-class accuracy |
|---|---:|---:|
| V1.0.0 | 56.56% | 49.74% |
| V2.0.0 | 61.87% | 57.98% |
| Change | +5.31 percentage points | +8.24 percentage points |

#### Per-Class Test Results

| Class | V1.0.0 | V2.0.0 | Change |
|---|---:|---:|---:|
| angry | 39.25% (376/958) | 51.25% (491/958) | +12.00 pp |
| disgust | 23.42% (26/111) | 44.14% (49/111) | +20.72 pp |
| fear | 18.65% (191/1024) | 34.96% (358/1024) | +16.31 pp |
| happy | 85.68% (1520/1774) | 81.51% (1446/1774) | −4.17 pp |
| neutral | 64.48% (795/1233) | 61.23% (755/1233) | −3.25 pp |
| sad | 43.79% (546/1247) | 57.42% (716/1247) | +13.63 pp |
| surprise | 72.92% (606/831) | 75.33% (626/831) | +2.41 pp |

V2 improves overall accuracy and substantially improves `angry`, `disgust`, `fear`, and `sad`, which were relatively weak in V1. `surprise` improves slightly, while `happy` and `neutral` decrease. The result is therefore better described as higher overall accuracy with more balanced per-class performance, rather than an improvement in every class.

No confusion matrix is included, so the project does not make claims about specific class-to-class confusions.

### Optimization and Ablation

- **LayerNorm**: did not show a clear benefit over the final baseline in this experiment and was not selected for V2.
- **BatchNorm**: accelerated convergence in this experiment and raised best validation accuracy to approximately 59.97%.
- **BatchNorm with a fixed learning rate**: showed noticeable validation fluctuation later in training.
- **BatchNorm + learning-rate decay**: reduced the learning rate from `1e-3` to `3e-4` at epoch 11 and reached 61.05% best validation accuracy, so it was selected for final V2.
- **Dropout(p=0.3)**: increased training loss, indicating that regularization was active in this experiment, but reached only approximately 60.35% best validation accuracy. It did not exceed the 61.05% result without Dropout, so final V2 does not use Dropout.

These statements describe the current experiments only; they do not imply that LayerNorm or Dropout is ineffective in other models or training settings. Existing per-epoch training records are stored in `docs/network_optimization_log.md`.

### Inference

`samples/` currently contains four visually distinct sample images in JPG, TIF, PNG, and BMP formats:

```text
sample_01.jpg
sample_02.tif
sample_03.png
sample_04.bmp
```

Place an input image under `samples/` and run from the repository root:

```bash
py app.py <filename>
```

For example:

```bash
py app.py sample_01.jpg
```

The final `app.py` uses:

```text
checkpoints/best_model_learning_rate_decay.pth
```

The application converts the input to `[1, 1, 48, 48]`, performs inference with `model.eval()` and `torch.no_grad()`, and prints the Top-3 classes in descending probability order.

Input images work best when they contain one centered face. The current version does not include face detection or automatic face cropping.

### Project Structure

```text
expression-recognition-cnn/
├── .gitignore
├── LICENSE
├── README.md
├── app.py
├── requirements.txt
│
├── checkpoints/
│   ├── best_model_4_layers.pth
│   ├── best_model_batchnorm.pth
│   └── best_model_learning_rate_decay.pth
│
├── docs/
│   ├── network_optimization_log.md
│   └── test_accuracy.md
│
├── samples/
│   ├── sample_01.jpg
│   ├── sample_02.tif
│   ├── sample_03.png
│   └── sample_04.bmp
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── test.py
│   └── train.py
│
└── data/
    ├── train/
    └── test/
```

Checkpoint roles:

- `best_model_4_layers.pth`: V1.0.0 four-layer CNN.
- `best_model_batchnorm.pth`: intermediate BatchNorm experiment with a fixed learning rate.
- `best_model_learning_rate_decay.pth`: final V2.0.0 model.

`data/` is not distributed with the repository and is excluded by `.gitignore`.

### Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train from the repository root:

```bash
py -m src.train
```

Test the final V2 checkpoint:

```bash
py -m src.test
```

Run inference:

```bash
py app.py <filename>
```

`python` can be used instead of the Windows Python Launcher command `py`.

### Requirements

Core dependencies are defined in `requirements.txt`:

```text
torch>=2.0
torchvision>=0.15
Pillow>=9.0
```

GPU acceleration depends on the local PyTorch and CUDA environment. The project can also run on CPU.

### Methodology Notes

- Model selection uses only the validation set, with the best checkpoint selected by validation accuracy.
- The combined test set is used only for final evaluation and not for model selection.
- The V2.0.0 test result has now been observed. Future development should not tune architectures or hyperparameters against this test result and should continue to rely only on validation performance.
- Softmax outputs represent relative confidence among the seven classes and should not be interpreted as absolute real-world probabilities.

### Project Limitations

- The model accepts only 48×48 single-channel inputs.
- The application does not include face detection or cropping.
- Test results are measured on combined `PublicTest + PrivateTest` and are not a standalone `PrivateTest` benchmark.
- This is a CNN learning and experimentation project rather than an attempt to achieve state-of-the-art facial-expression recognition performance.

### Contribution and Acknowledgements

The project author completed the model design, PyTorch implementation, training, experiments, testing, inference application, and final technical decisions.

ChatGPT / OpenAI supported CNN and PyTorch learning, concept explanations, code review and debugging guidance, experimental-design suggestions, result analysis, and documentation organization. The project's core model, training, experiments, testing, and application were completed by the project author.

### License

This project is licensed under the MIT License. See `LICENSE` for details.
