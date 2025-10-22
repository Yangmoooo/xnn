import numpy as np


def softmax(x):
    """

    Args:
        x (np.ndarray): n 维 numpy 分数数组

    Returns:
        np.ndarray: softmax 概率
    """
    # 减去最大值以实现数值稳定性
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))

    # 最后一个维度始终是类标签
    return e_x / e_x.sum(axis=-1, keepdims=True)
