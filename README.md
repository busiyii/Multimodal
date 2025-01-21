# 多模态情感分类

这是一个多模态情感分类模型的官方仓库，结合文本和图像数据进行情感标签的预测。模型利用BERT进行文本编码，ResNet-18进行图像特征提取，并通过自定义的交叉注意力融合机制将两种模态的特征进行融合。

## 环境设置

为了确保项目的可复现性和顺利运行，建议使用Python虚拟环境。 您需要安装以下依赖包:

- torch==2.5.1+cu118

- torchvision==0.20.1+cu118

- pandas==2.2.3

- pillow==10.4.0

- scikit-learn==1.5.2

- matplotlib==3.9.2

- numpy==2.0.2

- transformers==4.47.1

- chardet==5.2.0

- tqdm==4.67.1

你可以简单地通过以下命令安装所需的依赖： 

```python
pip install -r requirements.txt
```

## 文件结构

```python
|-- data 
    |-- train.txt
    |-- test_without_label.txt 
    |-- <GUID>.txt  
    |-- <GUID>.jpg 
|-- outputs 
    |-- text_loss_curve.png
    |-- text_accuracy_curve.png 
    |-- image_loss_curve.png 
    |-- image_accuracy_curve.png 
    |-- multimodal_loss_curve.png
    |-- multimodal_accuracy_curve.png  
    |-- predictions.txt  
|-- src 
    |-- Multimodal.py
|-- requirements.txt
|-- README.md

```
## 执行指南

按照以下步骤训练模型并生成预测结果。

1. **准备数据**

    确保数据按照以下结构组织：

    - **训练数据**: `data/train.txt`，包含 `guid`、`tag` 及其他相关列。
    - **测试数据**: `data/test_without_label.txt`，包含 `guid` 及其他相关列。
    - **文本文件**: 每个 `guid` 对应一个 `<GUID>.txt` 文件，包含文本数据。
    - **图像文件**: 每个 `guid` 对应一个 `<GUID>.jpg` 图像文件。

2. **配置实验参数**

    在 `src/main.py` 中设置所需的实验模式，通过修改 `EXP_MODE` 变量：

    ```python
    EXP_MODE = 'text'  # 选项: 'text', 'image', 'multimodal'
    ```

    - `'text'`: 仅使用文本数据。
    - `'image'`: 仅使用图像数据。
    - `'multimodal'`: 同时使用文本和图像数据。

3. **运行训练脚本**

    ```bash
    python Multimodal.py
    ```

4. **监控训练过程**

    训练过程中，脚本会显示：

    - 当前的epoch编号
    - 训练损失
    - 验证损失和准确率

    训练和验证的损失及准确率曲线将保存在 `outputs/` 目录中。

5. **生成预测结果**

    训练完成后，脚本将在测试集上进行推理，并将预测结果保存到 `outputs/predictions.txt`。

## 参考使用的库

本项目使用了多个库来处理不同的任务：

- **PyTorch**: 核心深度学习框架，用于模型构建和训练。
- **Transformers**: 提供预训练的BERT模型用于文本编码。
- **Torchvision**: 用于图像预处理和预训练ResNet-18模型。
- **Pandas**: 数据处理与分析。
- **Scikit-learn**: 用于数据分割和预处理。
- **Pillow**: 图像处理。
- **Matplotlib**: 绘制训练和验证指标图表。
- **TQDM**: 预测过程中的进度条显示。
- **Chardet**: 检测文件编码，以处理各种文本格式。
