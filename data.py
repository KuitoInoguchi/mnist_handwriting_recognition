import numpy as np
import scipy.io

CLASS_TO_DIGIT = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.int64)


def one_hot_rows_to_digits(one_hot):
    """
    MNISTData(1).mat 中，one-hot 的行顺序是 1,2,...,9,0。
    这里转换为自然数字标签 0~9，避免结果展示时整体错位。
    """
    class_index = np.argmax(one_hot, axis=0).astype(np.int64)
    return CLASS_TO_DIGIT[class_index]


def load_mnist(data_path='data/MNISTData(1).mat', verbose=True):
    """
    从 MATLAB .mat 文件加载 MNIST 数据集并进行预处理。

    参数:
        data_path: .mat 文件的路径
        verbose: 是否打印加载信息

    返回:
        X_train: [60000, 1, 28, 28] float32, 像素在 0.0 到 1.0 之间
        y_train: [60000] int64, 标签编号 0~9
        X_test: [10000, 1, 28, 28] float32, 像素在 0.0 到 1.0 之间
        y_test: [10000] int64, 标签编号 0~9
    """
    if verbose:
        print(f"正在从 {data_path} 加载 MNIST 数据集...")

    mat = scipy.io.loadmat(data_path)
    required_keys = {'X_Train', 'D_Train', 'X_Test', 'D_Test'}
    missing_keys = required_keys - set(mat.keys())
    if missing_keys:
        raise KeyError(f"MNIST 数据文件缺少字段: {sorted(missing_keys)}")

    # 提取并转置图像数据
    # 原数据格式：X_Train 为 (28, 28, 60000)，D_Train 为 (10, 60000)
    X_train = mat['X_Train'].transpose(2, 0, 1)  # -> (60000, 28, 28)
    X_train = X_train[:, None, :, :].astype(np.float32)  # -> (60000, 1, 28, 28)

    X_test = mat['X_Test'].transpose(2, 0, 1)  # -> (10000, 28, 28)
    X_test = X_test[:, None, :, :].astype(np.float32)  # -> (10000, 1, 28, 28)

    # 兼容未归一化的 MNIST 文件；当前课件数据已经是 [0, 1]。
    if X_train.max() > 1.0 or X_test.max() > 1.0:
        X_train /= 255.0
        X_test /= 255.0

    # 提取标签（从 one-hot 编码的 (10, N) 转换为自然数字 0~9）
    y_train = one_hot_rows_to_digits(mat['D_Train'])
    y_test = one_hot_rows_to_digits(mat['D_Test'])

    if mat['D_Train'].sum(axis=0).min() != 1 or mat['D_Train'].sum(axis=0).max() != 1:
        raise ValueError("D_Train 不是合法 one-hot 标签矩阵")
    if mat['D_Test'].sum(axis=0).min() != 1 or mat['D_Test'].sum(axis=0).max() != 1:
        raise ValueError("D_Test 不是合法 one-hot 标签矩阵")

    if verbose:
        print("数据加载与预处理完成。")
        print(f"  训练集: X_train.shape={X_train.shape}, y_train={y_train.shape}")
        print(f"  测试集: X_test.shape={X_test.shape}, y_test={y_test.shape}")
        print(f"  训练标签分布(0~9): {np.bincount(y_train, minlength=10).tolist()}")
        print(f"  测试标签分布(0~9): {np.bincount(y_test, minlength=10).tolist()}")

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
