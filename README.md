# XNN

这是一个从零开始、使用多种语言构建神经网络的学习项目

旨在通过亲手实现核心算法，深入理解其底层工作原理

## 项目结构

```plain
.
├── data/         # 数据集，来自 kaggle 的 Fashion-MNIST CSV 数据
├── nn4c/         # C 实现
├── nn4numpy/     # Python 实现（基于 NumPy）
└── nn4torch/     # Python 实现（基于 PyTorch）
```

- **`nn4c`**: 仅依赖标准库，大幅参考了 [miniMNIST-c](https://github.com/konrad-gajdus/miniMNIST-c/) 的实现
- **`nn4numpy`**: 基于 `NumPy` 的实现，注释详细，功能完善
- **`nn4torch`**: 基于 `PyTorch` 的实现，高度抽象，代码简洁

## 当前实现与特性

目前面向 **Fashion-MNIST** 数据集分类任务，实现多层感知机

### nn4c

学习整理 [miniMNIST-c](https://github.com/konrad-gajdus/miniMNIST-c/) 项目得到，没有多余的抽象，直观展示了神经网络的底层机制

- **网络架构**:
  - 包含 **一个隐藏层** 的多层感知机
  - 使用 **ReLU** 作为隐藏层激活函数
- **核心算法**:
  - 完整的前向传播与反向传播
  - 使用**交叉熵损失**作为损失函数
  - 使用**带动量的梯度下降**进行优化

### nn4numpy

由 `nn4c` 重写而来，基于 `NumPy` 实现，代码清晰，注释详尽，易于理解

- **网络架构**:
  - 包含 **两个隐藏层** 的多层感知机
  - 使用 **ReLU** 作为隐藏层激活函数
- **核心算法**:
  - 完整的前向传播与反向传播
  - 使用**交叉熵损失**作为损失函数
  - 使用**带动量的梯度下降**进行优化
  - 集成 **Dropout** 正则化

### nn4torch

根据 `nn4numpy` 重写，基于 `PyTorch` 实现，大量使用封装的 API，代码简洁，性能更优，功能和结构上与 `nn4numpy` 保持一致

## 如何运行

```sh
# C
cd nn4c
make run  # or zig build run -Doptimize=ReleaseFast

# Python
cd nn4numpy
python main.py  # or uv run main.py
```
