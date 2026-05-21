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

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        self.W = (np.random.randn(in_features, out_features) * 0.01).astype(np.float32)
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
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        K = kernel_size

        self.W = (np.random.randn(out_channels, in_channels, K, K) * 0.01).astype(np.float32)
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

                        region = X_padded[n, :, h_start:h_end, w_start:w_end]

                        grad = dY[n, oc, i, j]

                        self.dW[oc] += grad * region
                        self.db[oc] += grad
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
# 6. Softmax 激活层
# ============================================================

class Softmax:
    """
    Softmax 激活层。

    输入:
        X: [N, C]
    
    输出:
        probs: [N, C]
    """

    def __init__(self):
        self.probs = None

    def forward(self, X):
        """
        前向传播。
        X: [N, C]
        """
        # 数值稳定版本 softmax
        shifted = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(shifted)
        self.probs = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.probs

    def backward(self, dY):
        """
        反向传播。
        dY: [N, C]
        返回 dX: [N, C]
        """
        # dX = probs * (dY - sum(dY * probs, axis=1, keepdims=True))
        dX = self.probs * (dY - np.sum(dY * self.probs, axis=1, keepdims=True))
        return dX

    def params_and_grads(self):
        return []


# ============================================================
# 7. 交叉熵损失函数 CrossEntropyLoss
# ============================================================

class CrossEntropyLoss:
    """
    独立交叉熵损失函数（非合并版本）。

    输入:
        probs: [N, C]，每个类别的预测概率（通常是 Softmax 层的输出）
        y: [N]，真实标签的类编号
    """

    def __init__(self):
        self.probs = None
        self.y = None
        self.N = None

    def forward(self, probs, y):
        """
        计算交叉熵 Loss。
        probs: [N, C]
        y: [N]
        返回: loss 标量
        """
        self.probs = probs
        self.y = y
        self.N = probs.shape[0]

        correct_probs = probs[np.arange(self.N), y]
        loss = -np.mean(np.log(correct_probs + 1e-12))
        return loss

    def backward(self):
        """
        返回对输入 probs 的导数 dprobs: [N, C]
        """
        dprobs = np.zeros_like(self.probs)
        # 对真实类别 y 的导数是 -1 / (N * probs[y])
        dprobs[np.arange(self.N), self.y] = -1.0 / (self.N * (self.probs[np.arange(self.N), self.y] + 1e-12))
        return dprobs


# ============================================================
# 8. SGD 优化器
# ============================================================

class SGD:
    """
    随机梯度下降优化器。
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, params_and_grads):
        for param, grad in params_and_grads:
            if grad is None:
                continue
            param -= self.learning_rate * grad