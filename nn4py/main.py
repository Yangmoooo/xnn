import time
import numpy as np
from dataset import load_mnist_csv, shuffle_data
from model import Network
from utils import softmax

# --- 超参数 ---
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCHS = 50
BATCH_SIZE = 64
DROPOUT_RATE = 0.5
TRAIN_SPLIT = 0.8
TRAIN_CSV_PATH = "./data/fashion-mnist/fashion-mnist_train.csv"
TEST_CSV_PATH = "./data/fashion-mnist/fashion-mnist_test.csv"


def evaluate(network, images, labels, batch_size):
    """在给定数据集上评估"""
    n_samples = len(images)
    if n_samples == 0:
        return 0.0, 0

    correct_predictions = 0
    for i in range(0, n_samples, batch_size):
        end = i + batch_size
        image_batch = images[i:end]
        label_batch = labels[i:end]

        predictions = network.predict(image_batch)
        correct_predictions += np.sum(predictions == label_batch)

    accuracy = correct_predictions / n_samples
    return accuracy, correct_predictions


def main():
    """运行训练和评估的主函数"""
    np.random.seed(int(time.time()))

    # --- 数据加载 ---
    train_images, train_labels = load_mnist_csv(TRAIN_CSV_PATH)
    test_images, test_labels = load_mnist_csv(TEST_CSV_PATH)

    print("Shuffling training data...")
    train_images, train_labels = shuffle_data(train_images, train_labels)

    # 将训练数据拆分为训练集和验证集
    split_index = int(len(train_images) * TRAIN_SPLIT)
    val_images, val_labels = train_images[split_index:], train_labels[split_index:]
    train_images, train_labels = train_images[:split_index], train_labels[:split_index]

    n_train = len(train_images)
    n_val = len(val_images)
    n_test = len(test_images)
    input_size = train_images.shape[1]

    print(f"Training on {n_train} samples, validating on {n_val} samples.")

    # --- 网络初始化 ---
    print("Initializing network...")
    network = Network(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        dropout_rate=DROPOUT_RATE,
    )

    # --- 训练循环 ---
    train_indices = np.arange(n_train)
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0

        # 每个 epoch 都洗牌索引
        np.random.shuffle(train_indices)

        for i in range(0, n_train, BATCH_SIZE):
            # 使用洗牌后的索引创建 batch
            end = i + BATCH_SIZE
            batch_indices = train_indices[i:end]
            image_batch = train_images[batch_indices]
            label_batch = train_labels[batch_indices]

            # 前向传播
            hidden1_output, hidden2_output, final_z = network.forward(
                image_batch, training=True
            )
            probabilities = softmax(final_z)

            # 计算交叉熵损失
            # 计算 batch 的损失，提取出真实标签对应的预测概率，然后取负对数
            log_probs = -np.log(
                probabilities[range(len(label_batch)), label_batch] + 1e-10
            )
            total_loss += np.sum(log_probs)

            # 反向传播
            # probabilities 是网络预测的概率分布，如 [0.1, 0.8, 0.05, ...]
            # one_hot_labels 是真实标签的独热编码，如 [0, 0, 1, ...] 表示真实标签是 2
            one_hot_labels = np.zeros_like(probabilities)
            one_hot_labels[range(len(label_batch)), label_batch] = 1

            # 梯度实际上是 **一个高维的切向量**，其指向函数值变化最快的方向，大小就是高维的斜率
            # output_grad 就是输出层的梯度，即总损失对线性输出 Z 的梯度
            # 两者相减实际上是 Softmax 与 CrossEntropyLoss 的复合函数的导数
            # 若某个输出神经元对应的不是真实标签，但预测概率很高，如 0.8 - 0 = 0.8，梯度为正，表明推高了损失，需要降低
            # 若对应的是真实标签，但预测概率很低，如 0.1 - 1 = -0.9，梯度为负，拉低了正确分类的信心，需要提高
            output_grad = probabilities - one_hot_labels

            network.backward(
                image_batch,
                hidden1_output,
                hidden2_output,
                output_grad,
                LEARNING_RATE,
                MOMENTUM,
            )

        # --- 验证 ---
        val_accuracy, _ = evaluate(network, val_images, val_labels, BATCH_SIZE)
        avg_loss = total_loss / n_train
        end_time = time.time()

        print(
            f"Epoch {epoch + 1:2d}, Accuracy: {val_accuracy * 100:6.2f}%, "
            f"Avg Loss: {avg_loss:.4f}, Time: {end_time - start_time:.2f} seconds"
        )

    # --- 在测试集上进行最终评估 ---
    print("Starting final evaluation on the test set...")
    test_accuracy, n_correct = evaluate(network, test_images, test_labels, BATCH_SIZE)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}% ({n_correct}/{n_test})")


if __name__ == "__main__":
    main()
