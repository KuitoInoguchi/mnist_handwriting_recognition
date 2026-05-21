from layers import Conv2D, ReLU, AvgPool2D, Flatten, Linear, Softmax

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
        Softmax

    输出:
        probs: [N, 10]
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
            Softmax(),
        ]

    def forward(self, X, verbose=False):
        """
        X: [N, 1, 28, 28]
        return probs: [N, 10]
        """
        out = X

        if verbose:
            print("input:", out.shape)

        for idx, layer in enumerate(self.layers):
            out = layer.forward(out)

            if verbose:
                print(f"after layer {idx} ({layer.__class__.__name__}):", out.shape)

        return out

    def backward(self, dprobs):
        """
        从最后一层开始，倒序调用每一层的 backward。
        """
        dout = dprobs

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
