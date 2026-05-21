import numpy as np
from data import load_mnist, iterate_minibatches
from model import SimpleCNN
from layers import SGD, CrossEntropyLoss
from utils import accuracy, evaluate

def train(model, X_train, y_train, X_test, y_test,
          num_epochs=3,
          batch_size=64,
          learning_rate=0.01):
    """
    训练网络的主循环函数。

    参数:
        model: 模型实例
        X_train: 训练输入数据 [N, C, H, W]
        y_train: 训练真实标签 [N]
        X_test: 测试输入数据 [N, C, H, W]
        y_test: 测试真实标签 [N]
        num_epochs: 训练轮次 Epochs
        batch_size: 批大小 Batch Size
        learning_rate: 学习率
    """
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(learning_rate=learning_rate)

    print(f"开始训练模型 (Epochs={num_epochs}, Batch Size={batch_size}, LR={learning_rate})...")
    for epoch in range(num_epochs):
        batch_losses = []
        batch_accs = []

        # 遍历批数据进行单次迭代
        for batch_idx, (X_batch, y_batch) in enumerate(
            iterate_minibatches(X_train, y_train, batch_size=batch_size, shuffle=True)
        ):
            # 1. 前向传播（输入到 Softmax 激活层，输出为概率分布 probs）
            probs = model.forward(X_batch, verbose=False)

            # 2. 计算解耦后的交叉熵 Loss
            loss = loss_fn.forward(probs, y_batch)

            # 3. 对概率 probs 求梯度 dprobs
            dprobs = loss_fn.backward()

            # 4. 反向传播（因为 Softmax 是模型的最后一层，所以直接传入对 probs 的梯度进行反向传播）
            model.backward(dprobs)

            # 5. 更新参数
            optimizer.step(model.params_and_grads())

            # 6. 记录 Loss 和 Acc
            batch_losses.append(loss)
            batch_accs.append(accuracy(probs, y_batch))

            # 打印训练进度
            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"Batch {batch_idx}, "
                    f"Loss: {loss:.4f}, "
                    f"Batch Acc: {batch_accs[-1]:.4f}"
                )

        train_loss = float(np.mean(batch_losses))
        train_acc = float(np.mean(batch_accs))
        test_acc = evaluate(model, X_test, y_test, batch_size=128)

        print(
            f"\n=== [Epoch {epoch + 1} 结束] ===\n"
            f"  训练集平均 Loss: {train_loss:.4f}\n"
            f"  训练集平均 Accuracy: {train_acc:.4f}\n"
            f"  测试集 Accuracy: {test_acc:.4f}\n"
        )


if __name__ == "__main__":
    # 固定随机种子保证结果可复现
    np.random.seed(0)

    # 1. 从数据目录加载 MNIST 真实数据集
    X_train, y_train, X_test, y_test = load_mnist('data/MNISTData(1).mat')

    # 2. 初始化网络模型
    model = SimpleCNN()

    # 3. 进行训练和评估
    # 说明：由于这是一个使用 NumPy 纯手工实现的 CPU 卷积神经网络，
    # 包含大量的多层 nested for 循环。如果直接在完整的 60,000 张图像上运行，
    # 会非常耗时。为了进行快速效果验证，我们在这里使用一个 2000 个样本的小型训练子集，
    # 和 500 个样本的测试子集。若需要训练全量数据集，只需去除切片即可！
    
    print("\n为了进行快速验证演示，我们使用前 2000 个训练样本和前 500 个测试样本进行训练：")
    X_train_sub = X_train[:2000]
    y_train_sub = y_train[:2000]
    X_test_sub = X_test[:500]
    y_test_sub = y_test[:500]

    train(
        model,
        X_train_sub,
        y_train_sub,
        X_test_sub,
        y_test_sub,
        num_epochs=3,
        batch_size=64,
        learning_rate=0.01
    )
