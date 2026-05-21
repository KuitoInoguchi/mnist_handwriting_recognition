import numpy as np

from layers import AvgPool2D, Conv2D, Flatten, Linear, ReLU, Softmax


class SequentialCNN:
    """
    纯 NumPy 顺序 CNN 基类，负责串联各层的 forward/backward。
    """

    name = "sequential"

    def __init__(self, layers):
        self.layers = layers

    def forward(self, X, verbose=False):
        out = X

        if verbose:
            print("input:", out.shape)

        for idx, layer in enumerate(self.layers):
            out = layer.forward(out)

            if verbose:
                print(f"after layer {idx} ({layer.__class__.__name__}):", out.shape)

        return out

    def backward(self, dprobs):
        dout = dprobs

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout

    def params_and_grads(self):
        params_grads = []

        for layer in self.layers:
            params_grads.extend(layer.params_and_grads())

        return params_grads

    def parameter_count(self, include_bias=True):
        total = 0
        for layer in self.layers:
            if hasattr(layer, "W"):
                total += layer.W.size
            if include_bias and hasattr(layer, "b"):
                total += layer.b.size
        return int(total)

    def state_dict(self):
        state = {}
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, "W"):
                state[f"layer_{idx}_W"] = layer.W.copy()
            if hasattr(layer, "b"):
                state[f"layer_{idx}_b"] = layer.b.copy()
        return state

    def load_state_dict(self, state):
        for idx, layer in enumerate(self.layers):
            w_key = f"layer_{idx}_W"
            b_key = f"layer_{idx}_b"
            if hasattr(layer, "W") and w_key in state:
                layer.W[...] = state[w_key]
            if hasattr(layer, "b") and b_key in state:
                layer.b[...] = state[b_key]

    def architecture_rows(self):
        rows = []
        for idx, layer in enumerate(self.layers):
            params = 0
            shape = ""
            if hasattr(layer, "W"):
                params += layer.W.size
                shape = str(layer.W.shape)
            if hasattr(layer, "b"):
                params += layer.b.size
            rows.append({
                "index": idx,
                "layer": layer.__class__.__name__,
                "shape": shape,
                "params": params,
            })
        return rows


class LectureCNN(SequentialCNN):
    """
    课件同款 MNIST CNN：
    9x9x20 卷积 -> ReLU -> 2x2 平均池化 -> 展平(2000)
    -> 100 隐层 -> ReLU -> 10 类 Softmax。

    权值数（不含 bias）:
        9*9*20 + 2000*100 + 100*10 = 202620
    """

    name = "lecture"

    def __init__(self):
        super().__init__([
            Conv2D(in_channels=1, out_channels=20, kernel_size=9, stride=1, padding=0),
            ReLU(),
            AvgPool2D(pool_size=2, stride=2),

            Flatten(),

            Linear(in_features=20 * 10 * 10, out_features=100),
            ReLU(),

            Linear(in_features=100, out_features=10),
            Softmax(),
        ])


class SimpleCNN(SequentialCNN):
    """
    一个更小的 LeNet 风格备选模型，保留用于对比实验。
    """

    name = "simple"

    def __init__(self):
        super().__init__([
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
        ])


def build_model(name="lecture"):
    if name == "lecture":
        return LectureCNN()
    if name == "simple":
        return SimpleCNN()
    raise ValueError(f"未知模型结构: {name}")


def save_model(model, path, metadata=None):
    payload = model.state_dict()
    payload["model_name"] = np.array(model.name)
    if metadata:
        for key, value in metadata.items():
            payload[f"meta_{key}"] = np.array(value)
    np.savez(path, **payload)


def load_model(path):
    data = np.load(path, allow_pickle=True)
    model_name = str(data["model_name"]) if "model_name" in data else "lecture"
    model = build_model(model_name)
    model.load_state_dict(data)
    return model
