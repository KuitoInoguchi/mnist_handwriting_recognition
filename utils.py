import numpy as np

def accuracy(logits, y):
    """
    计算分类准确率。

    参数:
        logits: [N, C]，模型的预测输出（可以是概率或未归一化的得分）
        y: [N]，真实标签的类编号
    """
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == y)
    return acc


def evaluate(model, X, y, batch_size=128):
    """
    在验证集或测试集上计算平均准确率。

    参数:
        model: 模型实例
        X: 输入数据 [N, C, H, W]
        y: 标签 [N]
        batch_size: 评估的 Batch Size
    """
    from data import iterate_minibatches

    acc_list = []

    for X_batch, y_batch in iterate_minibatches(X, y, batch_size=batch_size, shuffle=False):
        logits = model.forward(X_batch, verbose=False)
        acc_list.append(accuracy(logits, y_batch))

    return float(np.mean(acc_list))
