import numpy as np

# a = np.array([1, 2, 3, 4])
# print(a * 2)
#
# b = [1, 2, 3, 4]
# print(b * 2)
#

# a = np.zeros((4, 5))
# print(a.shape)
# print(a.ndim)
# print(a.dtype)
# print(a.size)
# a = a.astype(np.float32)
# print(a.dtype)
#

# x = np.arange(784)
# image = x.reshape(28, 28)
# print(image[:3, :3])
#
# x = np.arange(78400)
# x = x.reshape(100, 784)
# x = x.reshape(100, 1, 28, 28)
# print(x[0, 0, :3, :3])

#
# x = np.ones((100, 28, 28))
# x = x[:, None, :, :]
# print(x.shape)

#
# x = np.zeros((32, 28, 28, 1))
# print(x.shape)
# # x = x.reshape((32, 1, 28, 28)) # dont do this! this distorts your data!
# # print(x.shape)
# x = x.transpose(0, 3, 1, 2)
# print(x.shape)
#
#
#
# X = np.array([[1, 2, 3],
#               [4, 5, 6]])
#
# W = np.array([[1, 2],
#               [3, 4],
#               [5, 6]])
#
# b = np.array([10, 20])
# out = X @ W + b
# print(out)
# print(X.shape)
# print(W.shape)
# print(b.shape)
# print(out.shape)

# logits = np.array([[1.0, 2.0, 0.5],
#                    [0.1, 0.2, 3.0]])
#
# def softmax(X):
#     """
#     X: (N, C)
#     max_x: (N, 1)
#     shifted_X: (N, C)
#     sum_exp_X: (N, 1)
#     """
#     max_X = np.max(X, axis=1, keepdims=True)
#     shifted_X = X - max_X
#     exp_X = np.exp(shifted_X)
#     sum_exp_X = np.sum(exp_X, axis=1, keepdims=True)
#     return exp_X / sum_exp_X # 多在脑子里想想形状
#
# print(softmax(logits))

# logits = np.array([[0.1, 2.0, 0.3],
#                    [3.0, 1.0, 0.2],
#                    [0.4, 0.5, 0.6]])
#
# y = np.array([1, 0, 1])
#
# y_prod = np.argmax(logits, axis=1)
# res = np.mean(y_prod == y)
# print(res)

#
# logits = np.array([[1.0, 2.0, 0.5],
#                    [0.1, 0.2, 3.0]])
#
# def softmax(X):
#     """
#     X: (N, C)
#     max_x: (N, 1)
#     shifted_X: (N, C)
#     sum_exp_X: (N, 1)
#     """
#     max_X = np.max(X, axis=1, keepdims=True)
#     shifted_X = X - max_X
#     exp_X = np.exp(shifted_X)
#     sum_exp_X = np.sum(exp_X, axis=1, keepdims=True)
#     return exp_X / sum_exp_X # 多在脑子里想想形状
#
# probs = softmax(logits)
# print(probs)
#
# y = np.array([1, 2])
# correct_probs = probs[np.arange(2), y]
# loss = -np.mean(np.log(correct_probs + 1e-12))
# print(loss)
#
# X = np.array([[10, 11],
#               [20, 21],
#               [30, 31],
#               [40, 41]])
#
# y = np.array([1, 2, 3, 4])
#
# new = np.arange(len(X))
# print(new)
# shift = np.random.permutation(new)
# print(shift)
#
# X = X[shift]
# y = y[shift]
# print(X)
# print(y)

# X = np.arange(100).reshape(10, 10)
# y = np.arange(10)
#
# batch_size = 3
# for i in range(0, len(X), batch_size):
#     start_idx = i
#     end_idx = i + batch_size
#
#     X_batch = X[start_idx:end_idx]
#     y_batch = y[start_idx:end_idx]
#
#     print(X_batch)
#     print(y_batch)
#

# def softmax(X):
#     """
#     X: (N, C)
#     max_x: (N, 1)
#     shifted_X: (N, C)
#     sum_exp_X: (N, 1)
#     """
#     max_X = np.max(X, axis=1, keepdims=True)
#     shifted_X = X - max_X
#     exp_X = np.exp(shifted_X)
#     sum_exp_X = np.sum(exp_X, axis=1, keepdims=True)
#     return exp_X / sum_exp_X # 多在脑子里想想形状
#
# X = np.random.randint(0, 256, size=(500, 784))
# y = np.random.randint(0, 10, size=(500, 1))
#
# print(X.dtype)
# X = X.astype(np.float32)
# print(X.dtype)
#
# print(X.min())
# print(X.max())
# X = softmax(X)
# print(X.min())
# print(X.max())
#
# print(X.shape)
# X = X.reshape((500, 1, 28, 28))
# print(X.shape)
#
# print(y.shape)
# y = y.reshape((500,))
# print(y.shape)
#
# print(y.dtype)
# y = y.astype(np.int64)
# print(y.dtype)
#
# y = np.random.randint(0, 10, size=(1000,))
# counts = np.bincount(y, minlength=10);
# print(counts)

# X = np.arange(20).reshape(4, 5)
# W = np.arange(15).reshape(5, 3)
# b = np.array([0, 1, 2])
#
# Y = X @ W + b
# print(Y)
#
# X = np.array([
#     [-1.0, 2.0, 3.0],
#     [-4.0, 0.0, 5.0]
# ])
#
# # ReLU
# Y = np.maximum(0, X)
# print(Y)
#
# count = np.sum(Y > 0)
# print(count)

# X = np.array([
#     [
#         [
#             [1, 3, 2, 4],
#             [2, 1, 0, 5],
#             [1, 2, 3, 1],
#             [0, 1, 2, 6]
#         ]
#     ]
# ], dtype=np.float32)
#
# def maxpool2d_forward(X, pool_size=2, stride=2):
#     N, C, H, W = X.shape
#     K = pool_size
#     S = stride
#
#     H_out = (H - K) // S + 1
#     W_out = (W - K) // S + 1
#     Y = np.zeros((N, C, H_out, W_out))
#
#     for n in range(N):
#         for c in range(C):
#             for i in range(H_out):
#                 for j in range(W_out):
#                     h_start = i * S
#                     h_end = h_start + K
#                     w_start = j * S
#                     w_end = w_start + K
#
#                     region = X[n, c, h_start:h_end, w_start:w_end]
#                     Y[n, c, i, j] = np.max(region)
#
#     return Y
#
# Y = maxpool2d_forward(X)
# print(Y)

# X = np.array([
#     [
#         [
#             [1, 2, 0, 1],
#             [3, 1, 2, 2],
#             [0, 1, 1, 0],
#             [2, 3, 1, 1]
#         ]
#     ]
# ], dtype=np.float32)
#
# W = np.array([
#     [
#         [
#             [1, 0, -1],
#             [1, 0, -1],
#             [1, 0, -1]
#         ]
#     ]
# ], dtype=np.float32)
#
# b = np.array([0], dtype=np.float32)
#
# def conv2d_forward(X, W_conv, b, stride, padding): # remove kernel from param_list
#     """
#     :param X: (N, C_in, H, W)
#     :param W_conv: (C_out, C_in, H_out, W_out)
#     :param b: (C_out, )
#     :param stride: a scalar
#     :param padding: a scalar
#     :return: (N, C_out, H_out, W_out)
#     """
#     N, C, H, W = X.shape
#     C_out, C_in, K_h, K_w = W_conv.shape
#
#     assert C == C_in, "input/kernel channels not compatible"
#     assert K_h == K_w, "accept square kernels only"
#
#     K = K_h
#     S = stride
#
#     if padding > 0:
#         # TODO: figuring out how to pad 0s
#         X_padded = np.pad(
#             X,
#             pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
#             mode="constant",
#             constant_values=0
#         )
#     else:
#         X_padded = X
#
#     H_out = (H + 2 * padding - K) // S + 1
#     W_out = (H + 2 * padding - K) // S + 1
#
#     Y = np.zeros([N, C_out, H_out, W_out])
#
#     for n in range(N):
#         for oc in range(C_out):
#             for i in range(H_out):
#                 for j in range(W_out):
#                     h_start = i * S
#                     h_end = h_start + K
#                     w_start = j * S
#                     w_end = w_start + K
#
#                     region = X_padded[n, :, h_start:h_end, w_start:w_end]
#                     print(region)
#                     Y[n, oc, i, j] = np.sum(region * W_conv[oc]) + b[oc]
#
#     return Y
#
# Y = conv2d_forward(X, W, b, 1, 0)
# print(Y)
#
# X = np.random.randn(4, 16, 4, 4)
# print(X.shape)
# X = X.reshape(4, -1)
# print(X.shape)

# ---

# def conv2d_forward(X, W_conv, b, stride, padding): # remove kernel from param_list
#     """
#     :param X: (N, C_in, H, W)
#     :param W_conv: (C_out, C_in, H_out, W_out)
#     :param b: (C_out, )
#     :param stride: a scalar
#     :param padding: a scalar
#     :return: (N, C_out, H_out, W_out)
#     """
#     N, C, H, W = X.shape
#     C_out, C_in, K_h, K_w = W_conv.shape
#
#     assert C == C_in, "input/kernel channels not compatible"
#     assert K_h == K_w, "accept square kernels only"
#
#     K = K_h
#     S = stride
#
#     if padding > 0:
#         X_padded = np.pad(
#             X,
#             pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
#             mode="constant",
#             constant_values=0
#         )
#     else:
#         X_padded = X
#
#     H_out = (H + 2 * padding - K) // S + 1
#     W_out = (H + 2 * padding - K) // S + 1
#
#     Y = np.zeros([N, C_out, H_out, W_out])
#
#     for n in range(N):
#         for oc in range(C_out):
#             for i in range(H_out):
#                 for j in range(W_out):
#                     h_start = i * S
#                     h_end = h_start + K
#                     w_start = j * S
#                     w_end = w_start + K
#
#                     region = X_padded[n, :, h_start:h_end, w_start:w_end]
#                     print(region)
#                     Y[n, oc, i, j] = np.sum(region * W_conv[oc]) + b[oc]
#
#     return Y
#
# def relu_forward(X):
#     return np.maximum(0 , X)
#
# def maxpool2d_forward(X, pool_size=2, stride=2):
#     N, C, H, W = X.shape
#     K = pool_size
#     S = stride
#
#     H_out = (H - K) // S + 1
#     W_out = (W - K) // S + 1
#     Y = np.zeros((N, C, H_out, W_out))
#
#     for n in range(N):
#         for c in range(C):
#             for i in range(H_out):
#                 for j in range(W_out):
#                     h_start = i * S
#                     h_end = h_start + K
#                     w_start = j * S
#                     w_end = w_start + K
#
#                     region = X[n, c, h_start:h_end, w_start:w_end]
#                     Y[n, c, i, j] = np.max(region)
#
#     return Y
#
# def flatten_forward(X):
#     return X.reshape(X.shape[0], -1)
#
# def linear_forward(X, W, b):
#     return X @ W + b
#
# N = 4
# rng = np.random.default_rng(0)
# X = rng.standard_normal((N, 1, 28, 28)).astype(np.float32)
#
# C_out1 = 6
# W_conv1 = rng.standard_normal((C_out1, 1, 5, 5)).astype(np.float32)
#
# b = np.array([0], dtype=np.float32)
#
# Y1 = conv2d_forward(X, W_conv1, b, )