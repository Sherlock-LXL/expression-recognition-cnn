# 基于 CNN 的表情识别 / Expression Recognition CNN

## 中文

### 1. 项目简介

**Expression Recognition CNN** 是一个使用 PyTorch 从零训练的七分类人脸表情识别项目。项目以 48×48 人脸图像为输入，通过卷积神经网络识别以下七类表情：

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

本项目的主要目标不是追求最复杂的网络结构，而是完整实践 CNN 的核心流程：数据处理、卷积网络搭建、训练、验证、模型优化、checkpoint 保存、测试集评估，以及对新图片进行实际推理。

最终模型采用四个卷积 block，并保存验证集表现最好的参数到 `checkpoints/best_model.pth`。

### 2. 项目特点

- 使用 PyTorch 从零搭建 CNN，而不是直接调用预训练分类模型。
- 训练、验证、测试三个阶段分开处理，其中验证集由训练目录按固定随机种子划分得到。
- 使用数据增强改善模型泛化。
- 通过多轮实验比较不同 CNN 深度，并记录优化过程。
- 使用 checkpoint 保存验证集表现最好的模型。
- 输出总体测试准确率以及七个表情类别各自的准确率。
- 提供 `app.py`，可以直接读取 `samples/` 中的新图片并输出预测结果。
- 支持 JPG、JPEG、PNG、TIF、WEBP 等常见图片格式，只要 Pillow 能正常读取即可。
- 推理时会将输入统一转换为模型训练所需的 1×48×48 灰度张量。

### 3. 项目结构

```text
expression-recognition-cnn/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── checkpoints/
│   └── best_model.pth
│
├── docs/
│   ├── network_optimization_log.md
│   └── test_accuracy.md
│
├── samples/
│   ├── sample_01.jpg
│   ├── sample_02.tif
│   ├── sample_03.jpeg
│   ├── sample_04.png
│   └── sample_05.bmp
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

`data/` 不上传 GitHub，并已在 `.gitignore` 中排除。

本项目使用 FER2013 风格的数据组织方式。数据集文件不随仓库发布，使用者需要自行准备数据并按照上述目录结构放置。

### 4. 模型结构

最终模型使用四个卷积 block。核心结构为：

```text
Input: [N, 1, 48, 48]

Conv2d: 1 → 16
ReLU
MaxPool2d
        ↓
[N, 16, 24, 24]

Conv2d: 16 → 32
ReLU
MaxPool2d
        ↓
[N, 32, 12, 12]

Conv2d: 32 → 64
ReLU
MaxPool2d
        ↓
[N, 64, 6, 6]

Conv2d: 64 → 128
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

训练时最后一层直接输出 logits，并交给 `CrossEntropyLoss`，因此训练阶段不在模型末尾添加 Softmax。

### 5. 数据与预处理

训练数据和测试数据使用 `ImageFolder` 风格的目录结构：

```text
data/train/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/

data/test/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

训练阶段使用的数据增强包括：

```text
Grayscale(1)
RandomHorizontalFlip
RandomRotation
ColorJitter（brightness / contrast）
ToTensor
```

验证和测试阶段不使用随机增强，只进行与模型输入一致的确定性预处理。

当前训练、验证和测试 transform 不包含 Resize，因此 `data/` 中的图片需要已经是 48×48。只有 `app.py` 会将外部图片显式缩放到 48×48。

当前本地数据划分如下：

| Split | Source | Images |
|---|---|---:|
| train | `data/train/` 的 90% 随机子集 | 25,839 |
| validation | `data/train/` 的 10% 随机子集 | 2,870 |
| test | `data/test/` | 7,178 |

训练集和验证集通过 `SEED = 42` 固定划分索引。该划分是普通随机划分，不是按类别进行的分层抽样。

当前 `data/test/` 同时包含 3,589 张 `PublicTest` 图片和 3,589 张 `PrivateTest` 图片。因此本文报告的测试准确率是两部分合并后的结果，不等同于只在 FER2013 `PrivateTest` 上计算的标准单独指标。

外部图片在 `app.py` 中统一处理为：

```text
任意可读取图片
→ Grayscale
→ Resize(48, 48)
→ ToTensor
→ [1, 48, 48]
→ 增加 batch 维度
→ [1, 1, 48, 48]
```

### 6. 训练与优化

项目在训练过程中依次尝试了不同深度的 CNN，并观察 train loss、validation loss 与 validation accuracy 的变化。

主要实验内容包括：

- 基础 CNN。
- 数据增强。
- 增加卷积层深度。
- 比较 2、3、4 层卷积结构的训练表现。
- 观察网络加深后的收益、优化难度与过拟合现象。
- 使用 validation accuracy 保存最佳 checkpoint。

各轮实验的训练指标记录保存在：

```text
docs/network_optimization_log.md
```

最终选择四层卷积模型作为本项目的收尾版本。

本次最终版本训练中，最佳 validation accuracy 约为：

```text
57.84%
```

优化日志中还保留了另一轮四层卷积探索实验，该轮最高 validation accuracy 为 58.92%。它属于优化过程记录；本文最终结果统一以日志中的“最终版本（四层网络）”以及当前发布的 checkpoint 测试结果为准。

### 7. 测试结果

加载 `checkpoints/best_model.pth` 后，在由 `PublicTest` 和 `PrivateTest` 合并组成的 7,178 张测试图片上的总体准确率为：

```text
Overall accuracy: 56.56%
```

各类别测试结果：

| Expression | Accuracy | Correct / Total |
|---|---:|---:|
| angry | 39.25% | 376 / 958 |
| disgust | 23.42% | 26 / 111 |
| fear | 18.65% | 191 / 1024 |
| happy | 85.68% | 1520 / 1774 |
| neutral | 64.48% | 795 / 1233 |
| sad | 43.79% | 546 / 1247 |
| surprise | 72.92% | 606 / 831 |

完整测试记录保存在：

```text
docs/test_accuracy.md
```

可以看到，不同表情类别的识别难度差异明显。`happy` 和 `surprise` 的识别效果较好，而 `fear`、`disgust` 的分类准确率较低。由于当前结果未提供混淆矩阵，因此这里只比较各类别准确率，不进一步判断具体类别之间的混淆关系。

### 8. 使用方法

#### 8.1 安装依赖

先安装依赖：

```bash
pip install -r requirements.txt
```

#### 8.2 训练模型

准备好 `data/train/` 后，在项目根目录运行：

```bash
py -m src.train
```

也可以使用：

```bash
python -m src.train
```

训练过程中会从 `data/train/` 中划分训练集和验证集，并将 validation accuracy 最高的模型参数保存到：

```text
checkpoints/best_model.pth
```

#### 8.3 测试模型

准备好 `data/test/` 和最佳 checkpoint 后，在项目根目录运行：

```bash
py -m src.test
```

也可以使用：

```bash
python -m src.test
```

程序会输出七个类别各自的准确率以及总体准确率。

#### 8.4 对新图片进行推理

确认项目中存在：

```text
checkpoints/best_model.pth
```

将需要识别的图片放入：

```text
samples/
```

然后在项目根目录运行：

```bash
py app.py <文件名>
```

例如：

```bash
py app.py sample_01.jpg
```

也可以使用：

```bash
python app.py sample_01.jpg
```

程序会自动完成：

```text
读取图片
→ 灰度化
→ Resize 到 48×48
→ 转换为 Tensor
→ 加载最佳 CNN 参数
→ Forward
→ Softmax
→ 输出预测结果
```

当前应用会按概率从高到低输出 Top-3 候选表情，可用于观察模型在多个相近类别之间的判断。

### 9. 示例图片

`samples/` 中提供五张已经实际测试过的示例图片：

```text
sample_01.jpg
sample_02.tif
sample_03.jpeg
sample_04.png
sample_05.bmp
```

这些图片用于演示不同常见图片格式均可以经过统一预处理后送入模型。

用户也可以将自己的图片放入 `samples/`，然后使用：

```bash
py app.py <文件名>
```

进行识别。

### 10. 项目限制

- 模型是在 48×48 人脸表情图像上训练的，因此输入图片最好以单个人脸为主体。
- 当前 `app.py` 不负责自动检测和裁剪人脸；如果输入是一张包含复杂背景或多个人物的大图，直接缩放到 48×48 可能明显降低识别效果。
- 模型在七个类别上的表现并不均衡，整体准确率不能代表每一类都具有相同的识别能力。
- Softmax 输出表示模型在当前七个类别之间的相对置信度，不应直接解释为现实世界中的绝对概率。
- 本项目定位为 CNN 学习与实践项目，不以追求当前最先进的表情识别性能为目标。

### 11. 项目分工与致谢

本项目由项目作者主导完成。

项目作者完成的工作包括：

- 项目选题与目标设计。
- PyTorch 项目结构搭建。
- 数据集组织与数据处理。
- CNN 模型代码实现。
- 训练循环、验证逻辑与测试逻辑实现。
- 多组网络结构实验与模型优化。
- checkpoint 保存与加载。
- 最终测试、结果判断与实际图片验证。
- `app.py` 推理流程实现。
- 项目最终整理与发布决策。

ChatGPT（OpenAI）在本项目中作为学习辅助工具参与，主要提供：

- CNN、PyTorch、卷积、池化、Autograd、checkpoint 等概念讲解。
- 代码逻辑检查与 debugging 指引。
- 模型实验设计与对照思路建议。
- 训练结果分析与过拟合判断。
- 测试与推理流程的教学式指导。
- 项目结构、README 与文档整理方面的协助。

最终代码选择、实验执行、训练结果、模型判断与项目发布均由项目作者负责。

---

## English

### 1. Project Overview

**Expression Recognition CNN** is a seven-class facial expression recognition project built and trained from scratch with PyTorch. The model takes 48×48 face images as input and predicts one of the following seven expressions:

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

The main goal of this project is not to build the most complex possible architecture, but to complete the full CNN workflow: data processing, convolutional network design, training, validation, model optimization, checkpoint saving, test-set evaluation, and inference on new images.

The final model uses four convolutional blocks, and the best validation checkpoint is saved to `checkpoints/best_model.pth`.

### 2. Project Features

- Builds a CNN from scratch with PyTorch instead of directly using a pretrained classification model.
- Separates training, validation, and testing, with the validation subset created from the training directory using a fixed random seed.
- Uses data augmentation to improve generalization.
- Compares CNNs of different depths through multiple experiments and records the optimization process.
- Saves the best validation model with a checkpoint.
- Reports both overall test accuracy and per-class accuracy for all seven expressions.
- Provides `app.py` for running inference directly on new images stored in `samples/`.
- Supports common image formats such as JPG, JPEG, PNG, TIF, and WEBP as long as Pillow can read them.
- Converts inference images into the 1×48×48 grayscale tensor format required by the trained model.

### 3. Project Structure

```text
expression-recognition-cnn/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── checkpoints/
│   └── best_model.pth
│
├── docs/
│   ├── network_optimization_log.md
│   └── test_accuracy.md
│
├── samples/
│   ├── sample_01.jpg
│   ├── sample_02.tif
│   ├── sample_03.jpeg
│   ├── sample_04.png
│   └── sample_05.bmp
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

`data/` is intentionally excluded from GitHub and is ignored by `.gitignore`.

This project uses a FER2013-style data layout. Dataset files are not distributed with the repository, so users need to prepare the data separately and organize it according to the directory structure above.

### 4. Model Architecture

The final model uses four convolutional blocks. Its core structure is:

```text
Input: [N, 1, 48, 48]

Conv2d: 1 → 16
ReLU
MaxPool2d
        ↓
[N, 16, 24, 24]

Conv2d: 16 → 32
ReLU
MaxPool2d
        ↓
[N, 32, 12, 12]

Conv2d: 32 → 64
ReLU
MaxPool2d
        ↓
[N, 64, 6, 6]

Conv2d: 64 → 128
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

During training, the final layer outputs raw logits directly to `CrossEntropyLoss`, so Softmax is not added to the end of the model during training.

### 5. Data and Preprocessing

The training and test datasets follow an `ImageFolder`-style directory layout:

```text
data/train/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/

data/test/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

Training-time augmentation includes:

```text
Grayscale(1)
RandomHorizontalFlip
RandomRotation
ColorJitter (brightness / contrast)
ToTensor
```

Validation and testing do not use random augmentation and only apply deterministic preprocessing consistent with the model input.

The current training, validation, and test transforms do not include Resize, so images under `data/` must already be 48×48. Only `app.py` explicitly resizes external images to 48×48.

The current local data split is:

| Split | Source | Images |
|---|---|---:|
| train | Random 90% subset of `data/train/` | 25,839 |
| validation | Random 10% subset of `data/train/` | 2,870 |
| test | `data/test/` | 7,178 |

Training and validation indices are fixed with `SEED = 42`. This is a standard random split rather than a class-stratified split.

The current `data/test/` contains 3,589 `PublicTest` images and 3,589 `PrivateTest` images. Therefore, the reported test accuracy is calculated on the combination of both subsets and is not the same as a standard metric calculated on FER2013 `PrivateTest` alone.

External images are normalized in `app.py` through the following pipeline:

```text
Any readable image
→ Grayscale
→ Resize(48, 48)
→ ToTensor
→ [1, 48, 48]
→ add batch dimension
→ [1, 1, 48, 48]
```

### 6. Training and Optimization

During training, CNNs with different depths were tested while monitoring train loss, validation loss, and validation accuracy.

The main experiments include:

- Baseline CNN.
- Data augmentation.
- Increasing convolutional depth.
- Comparing 2-, 3-, and 4-convolution architectures.
- Observing the benefits, optimization difficulty, and overfitting behavior caused by increasing depth.
- Saving the best checkpoint according to validation accuracy.

The training metrics from each experiment are stored in:

```text
docs/network_optimization_log.md
```

The four-convolution model was selected as the final version of this project.

In the final-version training run, the best validation accuracy was approximately:

```text
57.84%
```

The optimization log also contains an earlier four-convolution exploratory run that reached a maximum validation accuracy of 58.92%. That run remains part of the optimization history; the results presented here are based on the log section named "Final version (four-layer network)" and the test results associated with the currently published checkpoint.

### 7. Test Results

After loading `checkpoints/best_model.pth`, the overall accuracy on the 7,178-image test set formed by combining `PublicTest` and `PrivateTest` was:

```text
Overall accuracy: 56.56%
```

Per-class test results:

| Expression | Accuracy | Correct / Total |
|---|---:|---:|
| angry | 39.25% | 376 / 958 |
| disgust | 23.42% | 26 / 111 |
| fear | 18.65% | 191 / 1024 |
| happy | 85.68% | 1520 / 1774 |
| neutral | 64.48% | 795 / 1233 |
| sad | 43.79% | 546 / 1247 |
| surprise | 72.92% | 606 / 831 |

The complete test record is stored in:

```text
docs/test_accuracy.md
```

The results show clear differences in difficulty between expression classes. `happy` and `surprise` are recognized relatively well, while `fear` and `disgust` have lower classification accuracy. Since no confusion matrix is included, this section only compares per-class accuracy and does not make claims about which specific classes are confused with one another.

### 8. Usage

#### 8.1 Install Dependencies

Install dependencies first:

```bash
pip install -r requirements.txt
```

#### 8.2 Train the Model

After preparing `data/train/`, run the following command from the project root:

```bash
py -m src.train
```

Alternatively:

```bash
python -m src.train
```

The training process creates training and validation subsets from `data/train/` and saves the parameters with the highest validation accuracy to:

```text
checkpoints/best_model.pth
```

#### 8.3 Test the Model

After preparing `data/test/` and the best checkpoint, run the following command from the project root:

```bash
py -m src.test
```

Alternatively:

```bash
python -m src.test
```

The program reports per-class accuracy for all seven expressions and the overall accuracy.

#### 8.4 Run Inference on a New Image

Make sure the project contains:

```text
checkpoints/best_model.pth
```

Place the image to be recognized in:

```text
samples/
```

Then run from the project root:

```bash
py app.py <filename>
```

For example:

```bash
py app.py sample_01.jpg
```

You can also use:

```bash
python app.py sample_01.jpg
```

The program automatically performs:

```text
Load image
→ convert to grayscale
→ resize to 48×48
→ convert to Tensor
→ load the best CNN parameters
→ forward pass
→ Softmax
→ output predictions
```

The current application outputs the Top-3 candidate expressions in descending probability order, which makes it possible to inspect how the model distributes confidence across similar classes.

### 9. Sample Images

Five tested sample images are included in `samples/`:

```text
sample_01.jpg
sample_02.tif
sample_03.jpeg
sample_04.png
sample_05.bmp
```

These files demonstrate that several common image formats can be converted into the same model input format through the preprocessing pipeline.

Users can also place their own images in `samples/` and run:

```bash
py app.py <filename>
```

to perform recognition.

### 10. Project Limitations

- The model was trained on 48×48 facial-expression images, so inference works best when a single face is the main subject of the input image.
- The current `app.py` does not automatically detect or crop faces. If the input is a large image with a complex background or multiple people, resizing the entire image to 48×48 may significantly reduce recognition quality.
- Performance is not balanced across all seven expression classes, so overall accuracy does not mean that every class has the same recognition quality.
- Softmax outputs represent the model's relative confidence among the current seven classes and should not be interpreted as absolute real-world probabilities.
- This project is intended as a CNN learning and practice project rather than an attempt to achieve state-of-the-art facial-expression recognition performance.

### 11. Project Contribution and Acknowledgements

This project was led and completed by the project author.

Work completed by the project author includes:

- Project topic selection and goal design.
- PyTorch project structure.
- Dataset organization and preprocessing.
- CNN model implementation.
- Training loop, validation logic, and test logic.
- Multiple architecture experiments and model optimization.
- Checkpoint saving and loading.
- Final testing, result interpretation, and real-image verification.
- `app.py` inference pipeline implementation.
- Final project organization and publication decisions.

ChatGPT (OpenAI) participated as a learning assistant and mainly provided:

- Explanations of CNN, PyTorch, convolution, pooling, Autograd, checkpoints, and related concepts.
- Code-logic review and debugging guidance.
- Suggestions for controlled model experiments and comparisons.
- Analysis of training behavior and overfitting.
- Step-by-step guidance for testing and inference.
- Assistance with project structure, README writing, and documentation organization.

Final code choices, experiment execution, training results, model decisions, and project publication remain the responsibility of the project author.
