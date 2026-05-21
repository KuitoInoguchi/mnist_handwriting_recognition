import numpy as np
import scipy.io

def load_mnist(data_path='data/MNISTData(1).mat'):
    """
    从 MATLAB .mat 文件加载 MNIST 数据集并进行预处理。

    参数:
        data_path: .mat 文件的路径

    返回:
        X_train: [60000, 1, 28, 28] float32, 像素在 0.0 到 1.0 之间
        y_train: [60000] int64, 标签编号 0~9
        X_test: [10000, 1, 28, 28] float32, 像素在 0.0 到 1.0 之间
        y_test: [10000] int64, 标签编号 0~9
    """
    print(f"正在从 {data_path} 加载 MNIST 数据集...")
    mat = scipy.io.loadmat(data_path)

    # 提取并转置图像数据
    # 原数据格式：X_Train 为 (28, 28, 60000)，D_Train 为 (10, 60000)
    X_train = mat['X_Train'].transpose(2, 0, 1)  # -> (60000, 28, 28)
    X_train = X_train[:, None, :, :].astype(np.float32)  # -> (60000, 1, 28, 28)

    X_test = mat['X_Test'].transpose(2, 0, 1)  # -> (10000, 28, 28)
    X_test = X_test[:, None, :, :].astype(np.float32)  # -> (10000, 1, 28, 28)

    # 提取标签（从 one-hot 编码的 (10, N) 转换为类编号索引 (N,)）
    y_train = np.argmax(mat['D_Train'], axis=0).astype(np.int64)
    y_test = np.argmax(mat['D_Test'], axis=0).astype(np.int64)

    print("数据加载与预处理完成。")
    print(f"  训练集: X_train.shape={X_train.shape}, y_train={y_train.shape}")
    print(f"  测试集: X_test.shape={X_test.shape}, y_test={y_test.shape}")

    return X_train, y_train, X_test, y_test


def iterate_minibatches(X, y, batch_size=64, shuffle=True):
    """
    生成 mini-batch 数据的生成器。

    参数:
        X: 输入数据 [N, C, H, W]
        y: 标签数据 [N]
        batch_size: 每个 Batch 的样本数量
        shuffle: 是否在每个 Epoch 开始前打乱数据
    """
    N = len(X)

    if shuffle:
        perm = np.random.permutation(N)
        X = X[perm]
        y = y[perm]

    for start in range(0, N, batch_size):
        end = start + batch_size
        X_batch = X[start:end]
        y_batch = y[start:end]

        yield X_batch, y_batch
