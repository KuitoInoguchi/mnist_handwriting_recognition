import numpy as np

# ============================================================
# 辅助函数 im2col 和 col2im
# ============================================================

def im2col(x, filter_h, filter_w, stride=1, pad=0):
    """
    将图像块转换为矩阵的列。用于无循环的高速卷积和平均池化计算。
    """
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(x, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=x.dtype)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x_ in range(filter_w):
            x_max = x_ + stride * out_w
            col[:, :, y, x_, :, :] = img[:, :, y:y_max:stride, x_:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    将矩阵的列还原回图像块。用于无循环的反向传播计算。
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1), dtype=col.dtype)
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x_ in range(filter_w):
            x_max = x_ + stride * out_w
            img[:, :, y:y_max:stride, x_:x_max:stride] += col[:, :, y, x_, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


class Linear:
    def __init__(self, in_features, out_features, init_type='he'):
        self.in_features = in_features
        self.out_features = out_features

        if init_type == 'he':
            weight_scale = np.sqrt(2.0 / in_features)
        else:
            weight_scale = 0.01

        self.W = (np.random.randn(in_features, out_features) * weight_scale).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)

        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        out = X @ self.W + self.b
        return out

    def backward(self, dY):
        X = self.X
        self.dW = X.T @ dY
        self.db = np.sum(dY, axis=0)
        dX = dY @ self.W.T
        return dX

    def params_and_grads(self):
        return [
            (self.W, self.dW),
            (self.b, self.db),
        ]


class ReLU:
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


class Flatten:
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


class AvgPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        K = self.pool_size
        S = self.stride

        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1

        X_reshaped = X.reshape(N * C, 1, H, W)
        X_col = im2col(X_reshaped, K, K, stride=S, pad=0)
        out = np.mean(X_col, axis=1)

        return out.reshape(N, C, H_out, W_out)

    def backward(self, dY):
        N, C, H, W = self.X.shape
        K = self.pool_size
        S = self.stride

        dX_col = np.repeat(dY.reshape(-1, 1) / (K * K), K * K, axis=1)
        dX_reshaped = col2im(dX_col, (N * C, 1, H, W), K, K, stride=S, pad=0)

        return dX_reshaped.reshape(N, C, H, W)

    def params_and_grads(self):
        return []


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, init_type='he'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        K = kernel_size
        
        if init_type == 'he':
            fan_in = in_channels * K * K
            weight_scale = np.sqrt(2.0 / fan_in)
        else:
            weight_scale = 0.01

        self.W = (np.random.randn(out_channels, in_channels, K, K) * weight_scale).astype(np.float32)
        self.b = np.zeros(out_channels, dtype=np.float32)

        self.X = None
        self.X_col = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        N, C_in, H, W = X.shape
        C_out, _, K, _ = self.W.shape
        S = self.stride
        P = self.padding

        H_out = (H + 2 * P - K) // S + 1
        W_out = (W + 2 * P - K) // S + 1

        self.X_col = im2col(X, K, K, stride=S, pad=P)
        W_row = self.W.reshape(C_out, -1)

        out = self.X_col @ W_row.T + self.b

        out = out.reshape(N, H_out, W_out, C_out).transpose(0, 3, 1, 2)
        return out

    def backward(self, dY):
        N, C_in, H, W = self.X.shape
        C_out, _, K, _ = self.W.shape
        S = self.stride
        P = self.padding

        dY_flat = dY.transpose(0, 2, 3, 1).reshape(-1, C_out)
        self.db = np.sum(dY_flat, axis=0)

        dW_row = dY_flat.T @ self.X_col
        self.dW = dW_row.reshape(self.W.shape)

        dX_col = dY_flat @ self.W.reshape(C_out, -1)

        dX = col2im(dX_col, self.X.shape, K, K, stride=S, pad=P)
        return dX

    def params_and_grads(self):
        return [
            (self.W, self.dW),
            (self.b, self.db),
        ]


class Softmax:
    def __init__(self):
        self.probs = None

    def forward(self, X):
        shifted = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(shifted)
        self.probs = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.probs

    def backward(self, dY):
        # dX = probs * (dY - sum(dY * probs, axis=1, keepdims=True))
        dX = self.probs * (dY - np.sum(dY * self.probs, axis=1, keepdims=True))
        return dX

    def params_and_grads(self):
        return []



class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.y = None
        self.N = None

    def forward(self, probs, y):
        self.probs = probs
        self.y = y
        self.N = probs.shape[0]

        correct_probs = probs[np.arange(self.N), y]
        loss = -np.mean(np.log(correct_probs + 1e-12))
        return loss

    def backward(self):
        dprobs = np.zeros_like(self.probs)
        dprobs[np.arange(self.N), self.y] = -1.0 / (self.N * (self.probs[np.arange(self.N), self.y] + 1e-12))
        return dprobs

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