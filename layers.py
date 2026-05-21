import numpy as np
# ============================================================
# 1. 全连接层 Linear
# ============================================================

class Linear:
    """
    forward:
        X: [N, D_in]
        W: [D_in, D_out]
        b: [D_out]
        out = X @ W + b
        out: [N, D_out]

    backward:
        dY: [N, D_out]
        dW: [D_in, D_out]
        db: [D_out]
        dX: [N, D_in]
    """

    def __init__(self, in_features, out_features, weight_scale=None):
        self.in_features = in_features
        self.out_features = out_features

        # He 初始化的简化版本。对于 ReLU 网络，比固定乘 0.01 通常更稳。
        if weight_scale is None:
            weight_scale = np.sqrt(2.0 / in_features)

        self.W = (np.random.randn(in_features, out_features) * weight_scale).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)

        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        """
        X: [N, D_in]
        return: [N, D_out]
        """
        self.X = X
        out = X @ self.W + self.b
        return out

    def backward(self, dY):
        """
        dY: [N, D_out]
        return dX: [N, D_in]
        """
        X = self.X

        self.dW = X.T @ dY
        self.db = np.sum(dY, axis=0)
        dX = dY @ self.W.T

        return dX

    def params_and_grads(self):
        """
        返回本层所有可训练参数及其梯度。
        优化器会用这个函数统一更新参数。
        """
        return [
            (self.W, self.dW),
            (self.b, self.db),
        ]


# ============================================================
# 2. ReLU 层
# ============================================================

class ReLU:
    """
    forward:
        out = max(0, X)
    backward:
        X > 0 的位置梯度通过；
        X <= 0 的位置梯度变成 0。
    """

    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dY):
        dX = dY * (self.X > 0)
        return dX

    def params_and_grads(self):
        return []


# ============================================================
# 3. Flatten 层
# ============================================================

class Flatten:
    """
    forward:
        X: [N, C, H, W]
        out: [N, C * H * W]

    backward:
        dY: [N, C * H * W]
        dX: [N, C, H, W]

    Flatten 没有可训练参数。
    """

    def __init__(self):
        self.input_shape = None

    def forward(self, X):
        self.input_shape = X.shape
        N = X.shape[0]
        out = X.reshape(N, -1)
        return out

    def backward(self, dY):
        dX = dY.reshape(self.input_shape)
        return dX

    def params_and_grads(self):
        return []


# ============================================================
# 4. 平均池化层 AvgPool2D
# ============================================================

class AvgPool2D:
    """
    二维平均池化层。

    输入:
        X: [N, C, H, W]

    输出:
        out: [N, C, H_out, W_out]

    最常见设置:
        pool_size = 2
        stride = 2

    AvgPool 没有可训练参数。
    """

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        """
        X: [N, C, H, W]
        """
        self.X = X

        N, C, H, W = X.shape
        K = self.pool_size
        S = self.stride

        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1

        out = np.zeros((N, C, H_out, W_out), dtype=X.dtype)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        h_end = h_start + K
                        w_start = j * S
                        w_end = w_start + K

                        region = X[n, c, h_start:h_end, w_start:w_end]
                        out[n, c, i, j] = np.mean(region)

        return out

    def backward(self, dY):
        """
        dY: [N, C, H_out, W_out]
        return dX: [N, C, H, W]
        """
        X = self.X
        N, C, H, W = X.shape
        K = self.pool_size
        S = self.stride

        _, _, H_out, W_out = dY.shape

        dX = np.zeros_like(X)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        h_end = h_start + K
                        w_start = j * S
                        w_end = w_start + K

                        # dY[n, c, i, j] 是一个标量。
                        # 平均池化将梯度平均分配给该池化区域内的所有像素。
                        dX[n, c, h_start:h_end, w_start:w_end] += dY[n, c, i, j] / (K * K)

        return dX

    def params_and_grads(self):
        return []


# ============================================================
# 5. 卷积层 Conv2D
# ============================================================

class Conv2D:
    """
    朴素二维卷积层，采用 NCHW 格式。

    输入:
        X: [N, C_in, H, W]

    参数:
        W: [C_out, C_in, K, K]
        b: [C_out]

    输出:
        out: [N, C_out, H_out, W_out]

    注意：
        这里实现的是深度学习框架中通常所说的 convolution，
        数学上更接近 cross-correlation，也就是不翻转卷积核。
        这是 CNN 实现里的常见写法。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, weight_scale=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        K = kernel_size

        # fan_in 是一个输出值连接到的输入数量。
        # 对卷积层来说，fan_in = C_in * K * K。
        fan_in = in_channels * K * K

        if weight_scale is None:
            weight_scale = np.sqrt(2.0 / fan_in)

        self.W = (np.random.randn(out_channels, in_channels, K, K) * weight_scale).astype(np.float32)
        self.b = np.zeros(out_channels, dtype=np.float32)

        self.X = None
        self.X_padded = None
        self.dW = None
        self.db = None

    def forward(self, X):
        """
        X: [N, C_in, H, W]
        return out: [N, C_out, H_out, W_out]
        """
        self.X = X

        N, C_in, H, W = X.shape
        C_out, C_in_w, K, K_w = self.W.shape

        assert C_in == C_in_w, "输入通道数必须等于卷积核的输入通道数"
        assert K == K_w, "这里只处理正方形卷积核"

        S = self.stride
        P = self.padding

        if P > 0:
            X_padded = np.pad(
                X,
                pad_width=((0, 0), (0, 0), (P, P), (P, P)),
                mode="constant",
                constant_values=0
            )
        else:
            X_padded = X

        self.X_padded = X_padded

        H_padded = H + 2 * P
        W_padded = W + 2 * P

        H_out = (H_padded - K) // S + 1
        W_out = (W_padded - K) // S + 1

        out = np.zeros((N, C_out, H_out, W_out), dtype=X.dtype)

        for n in range(N):
            for oc in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        h_end = h_start + K
                        w_start = j * S
                        w_end = w_start + K

                        # region: [C_in, K, K]
                        region = X_padded[n, :, h_start:h_end, w_start:w_end]

                        # self.W[oc]: [C_in, K, K]
                        # 二者逐元素相乘后求和，再加上当前输出通道的 bias。
                        out[n, oc, i, j] = np.sum(region * self.W[oc]) + self.b[oc]

        return out

    def backward(self, dY):
        """
        dY: [N, C_out, H_out, W_out]
        return dX: [N, C_in, H, W]
        """
        X = self.X
        X_padded = self.X_padded

        N, C_in, H, W = X.shape
        C_out, _, K, _ = self.W.shape
        S = self.stride
        P = self.padding

        _, _, H_out, W_out = dY.shape

        dX_padded = np.zeros_like(X_padded)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        for n in range(N):
            for oc in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        h_end = h_start + K
                        w_start = j * S
                        w_end = w_start + K

                        # region: [C_in, K, K]
                        region = X_padded[n, :, h_start:h_end, w_start:w_end]

                        # grad 是当前输出位置传回来的标量梯度。
                        grad = dY[n, oc, i, j]

                        # 对卷积核的梯度：
                        # 当前输出值 forward 时使用了 region * W[oc]，
                        # 所以 W[oc] 的梯度要加上 grad * region。
                        self.dW[oc] += grad * region

                        # 对 bias 的梯度：
                        # b[oc] 直接加到了该输出通道的所有空间位置上。
                        self.db[oc] += grad

                        # 对输入区域的梯度：
                        # 当前输出值对输入 region 的导数是 W[oc]。
                        # 注意这里必须用 +=，因为同一个输入像素可能被多个卷积窗口覆盖。
                        dX_padded[n, :, h_start:h_end, w_start:w_end] += grad * self.W[oc]

        if P > 0:
            dX = dX_padded[:, :, P:-P, P:-P]
        else:
            dX = dX_padded

        return dX

    def params_and_grads(self):
        return [
            (self.W, self.dW),
            (self.b, self.db),
        ]


# ============================================================
# 6. Softmax + Cross Entropy Loss
# ============================================================

class SoftmaxCrossEntropyLoss:
    """
    softmax + cross entropy 合并实现。

    输入:
        logits: [N, C]
        y: [N]，每个元素是类别编号，例如 MNIST 中是 0~9。

    forward 返回:
        loss: 标量

    backward 返回:
        dlogits: [N, C]

    核心公式:
        dlogits = (probs - onehot(y)) / N
    """

    def __init__(self):
        self.probs = None
        self.y = None
        self.N = None
        self.loss = None

    def forward(self, logits, y):
        """
        logits: [N, C]
        y: [N]
        """
        self.y = y
        self.N = logits.shape[0]

        # 数值稳定版本 softmax。
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        self.probs = probs

        correct_probs = probs[np.arange(self.N), y]
        loss = -np.mean(np.log(correct_probs + 1e-12))

        self.loss = loss
        return loss

    def backward(self):
        """
        return dlogits: [N, C]
        """
        dlogits = self.probs.copy()

        # 对真实类别对应的位置减 1，相当于 probs - onehot(y)。
        dlogits[np.arange(self.N), self.y] -= 1

        # 因为 forward 的 loss 对 batch 取了平均，所以梯度也要除以 N。
        dlogits /= self.N

        return dlogits


# ============================================================
# 7. 一个简单 CNN 模型
# ============================================================

class SimpleCNN:
    """
    一个 LeNet 风格的小 CNN，适合 MNIST。

    输入:
        X: [N, 1, 28, 28]

    结构:
        Conv1: 1 -> 6, K=5, S=1, P=0
        ReLU
        AvgPool: K=2, S=2

        Conv2: 6 -> 16, K=5, S=1, P=0
        ReLU
        AvgPool: K=2, S=2

        Flatten: [N, 16, 4, 4] -> [N, 256]

        Linear: 256 -> 120
        ReLU
        Linear: 120 -> 84
        ReLU
        Linear: 84 -> 10

    输出:
        logits: [N, 10]
    """

    def __init__(self):
        self.layers = [
            Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            ReLU(),
            AvgPool2D(pool_size=2, stride=2),

            Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            ReLU(),
            AvgPool2D(pool_size=2, stride=2),

            Flatten(),

            Linear(in_features=16 * 4 * 4, out_features=120),
            ReLU(),

            Linear(in_features=120, out_features=84),
            ReLU(),

            Linear(in_features=84, out_features=10),
        ]

    def forward(self, X, verbose=False):
        """
        X: [N, 1, 28, 28]
        return logits: [N, 10]
        """
        out = X

        if verbose:
            print("input:", out.shape)

        for idx, layer in enumerate(self.layers):
            out = layer.forward(out)

            if verbose:
                print(f"after layer {idx} ({layer.__class__.__name__}):", out.shape)

        return out

    def backward(self, dlogits):
        """
        从最后一层开始，倒序调用每一层的 backward。
        """
        dout = dlogits

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout

    def params_and_grads(self):
        """
        收集所有层的参数和梯度。
        """
        params_grads = []

        for layer in self.layers:
            params_grads.extend(layer.params_and_grads())

        return params_grads


# ============================================================
# 8. SGD 优化器
# ============================================================

class SGD:
    """
    最简单的随机梯度下降优化器。

    参数更新公式:
        param -= lr * grad
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, params_and_grads):
        for param, grad in params_and_grads:
            if grad is None:
                continue
            param -= self.learning_rate * grad


# ============================================================
# 9. 数据 mini-batch 生成器
# ============================================================

def iterate_minibatches(X, y, batch_size=64, shuffle=True):
    """
    X: [N, 1, 28, 28]
    y: [N]

    每次 yield:
        X_batch: [B, 1, 28, 28]
        y_batch: [B]
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


# ============================================================
# 10. 准确率评估
# ============================================================

def accuracy(logits, y):
    """
    logits: [N, C]
    y: [N]
    """
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == y)
    return acc


def evaluate(model, X, y, batch_size=128):
    """
    在验证集或测试集上计算平均准确率。

    注意：这里为了简单，只计算 accuracy。
    如果数据很多，也应该按 batch 评估，避免一次性占用过多内存。
    """
    acc_list = []

    for X_batch, y_batch in iterate_minibatches(X, y, batch_size=batch_size, shuffle=False):
        logits = model.forward(X_batch, verbose=False)
        acc_list.append(accuracy(logits, y_batch))

    return float(np.mean(acc_list))


# ============================================================
# 11. 训练循环骨架
# ============================================================

def train(model, X_train, y_train, X_test, y_test,
          num_epochs=5,
          batch_size=64,
          learning_rate=0.01):
    """
    一个基本训练循环。

    每个 batch:
        1. forward
        2. loss forward
        3. loss backward 得到 dlogits
        4. model backward
        5. optimizer step 更新参数
    """

    loss_fn = SoftmaxCrossEntropyLoss()
    optimizer = SGD(learning_rate=learning_rate)

    for epoch in range(num_epochs):
        batch_losses = []
        batch_accs = []

        for batch_idx, (X_batch, y_batch) in enumerate(
            iterate_minibatches(X_train, y_train, batch_size=batch_size, shuffle=True)
        ):
            # 1. 前向传播
            logits = model.forward(X_batch, verbose=False)

            # 2. 计算 loss
            loss = loss_fn.forward(logits, y_batch)

            # 3. 从 softmax + cross entropy 得到 dlogits
            dlogits = loss_fn.backward()

            # 4. 反向传播
            model.backward(dlogits)

            # 5. 更新参数
            optimizer.step(model.params_and_grads())

            # 6. 记录当前 batch 的 loss 和 accuracy
            batch_losses.append(loss)
            batch_accs.append(accuracy(logits, y_batch))

            # 你可以先每隔若干 batch 打印一次，方便调试。
            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"Batch {batch_idx}, "
                    f"Loss {loss:.4f}, "
                    f"Batch Acc {batch_accs[-1]:.4f}"
                )

        train_loss = float(np.mean(batch_losses))
        train_acc = float(np.mean(batch_accs))
        test_acc = evaluate(model, X_test, y_test, batch_size=128)

        print(
            f"[Epoch {epoch + 1}] "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"test_acc={test_acc:.4f}"
        )


# ============================================================
# 12. 临时假数据测试入口
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)

    # 先用假数据测试网络是否能 forward/backward 跑通。
    # 真正接入 MNIST 前，强烈建议先跑这一步。
    X_dummy = np.random.randn(8, 1, 28, 28).astype(np.float32)
    y_dummy = np.random.randint(0, 10, size=(8,), dtype=np.int64)

    model = SimpleCNN()

    print("========== Shape Test ==========")
    logits = model.forward(X_dummy, verbose=True)
    print("logits:", logits.shape)

    loss_fn = SoftmaxCrossEntropyLoss()
    loss = loss_fn.forward(logits, y_dummy)
    print("loss:", loss)

    dlogits = loss_fn.backward()
    print("dlogits:", dlogits.shape)

    dX = model.backward(dlogits)
    print("dX:", dX.shape)

    # 检查参数梯度 shape 是否正确。
    print("========== Gradient Shape Check ==========")
    for layer in model.layers:
        for param, grad in layer.params_and_grads():
            print(layer.__class__.__name__, "param:", param.shape, "grad:", grad.shape)

    # 用假数据跑一两个 epoch 没有实际意义，只是测试训练循环不会崩。
    # 真正训练时，把下面的 dummy 数据换成 MNIST 数据。
    print("========== Dummy Training Test ==========")
    X_train_dummy = np.random.randn(32, 1, 28, 28).astype(np.float32)
    y_train_dummy = np.random.randint(0, 10, size=(32,), dtype=np.int64)

    X_test_dummy = np.random.randn(16, 1, 28, 28).astype(np.float32)
    y_test_dummy = np.random.randint(0, 10, size=(16,), dtype=np.int64)

    train(
        model,
        X_train_dummy,
        y_train_dummy,
        X_test_dummy,
        y_test_dummy,
        num_epochs=1,
        batch_size=4,
        learning_rate=0.001
    )