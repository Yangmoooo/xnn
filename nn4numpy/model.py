import numpy as np
from utils import softmax


class Layer:
    """神经网络的线性层"""

    def __init__(self, input_size, output_size):
        """
        初始化权重和偏置
        """
        # He 初始化权重，适用于 ReLU 激活函数
        scale = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(input_size, output_size) * scale
        self.biases = np.zeros(output_size)

        # 用于权重和偏置更新的动量项
        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.biases)


class Network:
    """一个简单的多层感知机"""

    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        """
        初始化网络，包含两个隐藏层和一个输出层
        """
        self.hidden_layer1 = Layer(input_size, hidden_size)
        self.hidden_layer2 = Layer(hidden_size, hidden_size)
        self.output_layer = Layer(hidden_size, output_size)
        self.dropout_rate = dropout_rate
        self.U1 = None  # 隐藏层 1 的 Dropout 掩码
        self.U2 = None  # 隐藏层 2 的 Dropout 掩码

    def forward(self, input_data, training=True):
        """
        前向传播
        如果在训练模式下，则启用 dropout
        """
        # 隐藏层 1
        # 线性变换后用 ReLU 激活，即线性输出为 Z = W @ X + b，激活后为 output = ReLU(Z)
        hidden1_z = (
            np.dot(input_data, self.hidden_layer1.weights) + self.hidden_layer1.biases
        )
        hidden1_output = np.maximum(0, hidden1_z)
        if training:
            # Inverted Dropout
            # 首先生成与隐藏层同型的随机数矩阵
            # 然后与 dropout 率比较，按照概率得到掩码布尔矩阵，即每个神经元是否会被保留
            # 接着计算缩放比例，因为部分神经元被丢弃，所以需要强化被保留的神经元，以保持输出的期望值不变
            # 最后将输出乘以掩码，实现 dropout 效果
            self.U1 = (np.random.rand(*hidden1_output.shape) > self.dropout_rate) / (
                1.0 - self.dropout_rate
            )
            hidden1_output *= self.U1

        # 隐藏层 2
        hidden2_z = (
            np.dot(hidden1_output, self.hidden_layer2.weights)
            + self.hidden_layer2.biases
        )
        hidden2_output = np.maximum(0, hidden2_z)
        if training:
            self.U2 = (np.random.rand(*hidden2_output.shape) > self.dropout_rate) / (
                1.0 - self.dropout_rate
            )
            hidden2_output *= self.U2

        # 输出层
        final_z = (
            np.dot(hidden2_output, self.output_layer.weights) + self.output_layer.biases
        )

        return hidden1_output, hidden2_output, final_z

    def backward(
        self, input_data, hidden1_output, hidden2_output, output_grad, lr, momentum
    ):
        """
        反向传播并更新网络的权重和偏置
        """
        batch_size = input_data.shape[0]

        # --- 输出层更新 ---
        # output_grad 是最终的总损失 Loss 对输出层线性结果 Z = W @ X + b 的梯度，即 ∂L/∂Z
        # 代表“误差”已经从最终的损失函数传播到了 Z 这一步，即 Z 的变化对 Loss 的影响
        # 现在则是将“误差”由 Z 还原分配到 W 和 b，即求出 ∂L/∂W 和 ∂L/∂b
        # 对于权重的梯度 ∂L/∂W = ∂L/∂Z @ ∂Z/∂W，而 Z 对 W 的梯度 ∂Z/∂W 就是前一层的输出 X，即 hidden2_output
        # 对于偏置的梯度 ∂L/∂b = ∂L/∂Z @ ∂Z/∂b，而 Z 对 b 的梯度 ∂Z/∂b 恒为 1，但需要对批次求和
        output_weight_grad = hidden2_output.T @ output_grad
        output_bias_grad = np.sum(output_grad, axis=0)

        # 通过动量来优化梯度下降，使下降过程带有“惯性”，不仅基于当前的梯度，还考虑了之前累积的梯度
        # 有助于加速收敛，减少震荡，跳出局部最优
        # 若不使用动量，则直接减去 lr * (output_weight_grad / batch_size)
        self.output_layer.weight_momentum = (
            momentum * self.output_layer.weight_momentum
        ) + (lr * output_weight_grad / batch_size)
        self.output_layer.bias_momentum = (
            momentum * self.output_layer.bias_momentum
        ) + (lr * output_bias_grad / batch_size)

        self.output_layer.weights -= self.output_layer.weight_momentum
        self.output_layer.biases -= self.output_layer.bias_momentum

        # --- 隐藏层 2 更新 ---
        # 现在要将“误差”从输出层传播到隐藏层 2
        # 设隐藏层 2 的输出为 H2，则有 Z_out = H2 @ W_out + B_out
        # 而由链式法则，又有 ∂L/∂H2 = ∂L/∂Z_out @ ∂Z_out/∂H2
        # 其中 ∂L/∂Z_out 就是 output_grad，而 ∂Z_out/∂H2 则对应上式中的 W_out
        hidden2_grad = output_grad @ self.output_layer.weights.T
        # 应用 dropout 掩码的导数
        if self.U2 is not None:
            hidden2_grad *= self.U2
        # 应用 ReLU 导数
        hidden2_grad[hidden2_output <= 0] = 0

        hidden2_weight_grad = hidden1_output.T @ hidden2_grad
        hidden2_bias_grad = np.sum(hidden2_grad, axis=0)

        self.hidden_layer2.weight_momentum = (
            momentum * self.hidden_layer2.weight_momentum
        ) + (lr * hidden2_weight_grad / batch_size)
        self.hidden_layer2.bias_momentum = (
            momentum * self.hidden_layer2.bias_momentum
        ) + (lr * hidden2_bias_grad / batch_size)

        self.hidden_layer2.weights -= self.hidden_layer2.weight_momentum
        self.hidden_layer2.biases -= self.hidden_layer2.bias_momentum

        # --- 隐藏层 1 更新 ---
        hidden1_grad = hidden2_grad @ self.hidden_layer2.weights.T
        # 应用 dropout 掩码的导数
        if self.U1 is not None:
            hidden1_grad *= self.U1
        # 应用 ReLU 导数
        hidden1_grad[hidden1_output <= 0] = 0

        hidden1_weight_grad = input_data.T @ hidden1_grad
        hidden1_bias_grad = np.sum(hidden1_grad, axis=0)

        self.hidden_layer1.weight_momentum = (
            momentum * self.hidden_layer1.weight_momentum
        ) + (lr * hidden1_weight_grad / batch_size)
        self.hidden_layer1.bias_momentum = (
            momentum * self.hidden_layer1.bias_momentum
        ) + (lr * hidden1_bias_grad / batch_size)

        self.hidden_layer1.weights -= self.hidden_layer1.weight_momentum
        self.hidden_layer1.biases -= self.hidden_layer1.bias_momentum

    def predict(self, input_data):
        """
        对一批输入样本进行预测
        """
        # 在预测模式下进行前向传播 (training=False)
        _, _, final_z = self.forward(input_data, training=False)

        # 通过 softmax 获取概率
        probabilities = softmax(final_z)

        # 返回批次中每个样本具有最高概率的类别
        return np.argmax(probabilities, axis=-1)
