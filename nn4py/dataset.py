import numpy as np


def load_mnist_csv(filename):
    """
    加载 MNIST-like CSV 文件

    Args:
        filename: CSV 文件路径

    Returns:
        tuple:
            - np.ndarray: 图像数组 (n_samples, 784)
            - np.ndarray: 标签数组 (n_samples,)
    """
    print(f"Loading data from {filename}...")
    data = np.loadtxt(filename, delimiter=",", dtype=np.float32)

    # 第一列是标签，其余是像素
    labels = data[:, 0].astype(np.uint8)
    images = data[:, 1:]

    images /= 255.0

    print(f"Found {len(images)} images.")
    return images, labels


def shuffle_data(images, labels):
    """
    打乱图像和标签

    Args:
        images (np.ndarray): 图像数组
        labels (np.ndarray): 标签数组

    Returns:
        tuple: 包含洗牌后图像和标签的元组
    """
    assert len(images) == len(labels)

    # 生成一个随机排列来洗牌图像和标签
    permutation = np.random.permutation(len(images))
    return images[permutation], labels[permutation]
