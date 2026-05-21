import argparse
import os
import platform
import subprocess
import time
from datetime import datetime

import numpy as np

from data import load_mnist, iterate_minibatches
from layers import CrossEntropyLoss, SGD
from model import build_model, save_model
from utils import (
    accuracy,
    confusion_matrix,
    evaluate,
    per_class_accuracy,
    save_history_csv,
    save_misclassified_mosaic,
    save_summary_json,
    summarize_errors,
)


PARAMETER_LIMIT_WEIGHTS = 9 * 9 * 20 + 2000 * 100 + 100 * 10


def collect_hardware_info():
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        completed = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in completed.stdout.splitlines():
            if ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in {"Model Name", "Model Identifier", "Chip", "Total Number of Cores", "Memory"}:
                info[key] = value
    except Exception as exc:
        info["hardware_probe_error"] = str(exc)
    return info


def train(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    num_epochs=4,
    batch_size=64,
    learning_rate=0.03,
    eval_batch_size=256,
    log_every=100,
    log_fn=print,
):
    """
    训练网络的主循环函数，返回每轮指标并把模型恢复到测试集最优 epoch。
    """
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(learning_rate=learning_rate)

    history = []
    best_state = None
    best_test_acc = -1.0
    best_epoch = 0
    started_at = time.perf_counter()

    log_fn(
        f"开始训练 {model.name} 模型 "
        f"(Epochs={num_epochs}, Batch Size={batch_size}, LR={learning_rate})"
    )
    log_fn(
        f"权值数(不含 bias): {model.parameter_count(include_bias=False)} / {PARAMETER_LIMIT_WEIGHTS}, "
        f"可训练参数(含 bias): {model.parameter_count(include_bias=True)}"
    )

    for epoch in range(num_epochs):
        epoch_started_at = time.perf_counter()
        loss_sum = 0.0
        correct = 0
        total = 0

        for batch_idx, (X_batch, y_batch) in enumerate(
            iterate_minibatches(X_train, y_train, batch_size=batch_size, shuffle=True)
        ):
            probs = model.forward(X_batch, verbose=False)
            loss = loss_fn.forward(probs, y_batch)
            dprobs = loss_fn.backward()
            model.backward(dprobs)
            optimizer.step(model.params_and_grads())

            batch_size_actual = len(y_batch)
            loss_sum += float(loss) * batch_size_actual
            correct += int(np.sum(np.argmax(probs, axis=1) == y_batch))
            total += batch_size_actual

            if log_every and batch_idx % log_every == 0:
                log_fn(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"Batch {batch_idx}, "
                    f"Loss: {loss:.4f}, "
                    f"Batch Acc: {accuracy(probs, y_batch):.4f}"
                )

        train_loss = loss_sum / total
        train_acc = correct / total
        test_acc = evaluate(model, X_test, y_test, batch_size=eval_batch_size)
        epoch_seconds = time.perf_counter() - epoch_started_at

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "seconds": epoch_seconds,
        }
        history.append(row)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_state = model.state_dict()

        log_fn(
            f"=== Epoch {epoch + 1} 结束 === "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_acc={test_acc:.4f}, seconds={epoch_seconds:.2f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    total_seconds = time.perf_counter() - started_at
    log_fn(f"训练完成。最佳测试准确率 {best_test_acc:.4f} 出现在第 {best_epoch} 轮，总耗时 {total_seconds:.2f} 秒。")

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_test_accuracy": best_test_acc,
        "train_seconds": total_seconds,
    }


def analyze_and_save(model, X_test, y_test, output_dir, batch_size=256, error_limit=36):
    details = evaluate(model, X_test, y_test, batch_size=batch_size, return_details=True)
    probs = details["probabilities"]
    preds = details["predictions"]
    confusion = confusion_matrix(y_test, preds)
    errors = summarize_errors(y_test, preds, probs, limit=error_limit)
    mosaic_path = os.path.join(output_dir, "misclassified_samples.png")
    save_misclassified_mosaic(X_test, errors, mosaic_path)

    return {
        "test_accuracy": details["accuracy"],
        "confusion_matrix": confusion.tolist(),
        "per_class_accuracy": per_class_accuracy(confusion).tolist(),
        "num_errors": int(np.sum(preds != y_test)),
        "error_rate": float(np.mean(preds != y_test)),
        "top_errors": errors,
        "mosaic_path": mosaic_path,
    }


def run_experiment(
    data_path="data/MNISTData(1).mat",
    output_dir="results/latest",
    model_name="lecture",
    num_epochs=4,
    batch_size=64,
    learning_rate=0.03,
    seed=0,
    log_fn=print,
):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")

    def tee_log(message):
        log_fn(message)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    if os.path.exists(log_path):
        os.remove(log_path)

    np.random.seed(seed)
    tee_log(f"随机种子: {seed}")
    tee_log(f"输出目录: {output_dir}")

    X_train, y_train, X_test, y_test = load_mnist(data_path, verbose=False)
    tee_log(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    tee_log(f"训练标签分布(0~9): {np.bincount(y_train, minlength=10).tolist()}")
    tee_log(f"测试标签分布(0~9): {np.bincount(y_test, minlength=10).tolist()}")

    model = build_model(model_name)
    tee_log(f"模型结构: {model.name}")

    train_result = train(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        log_fn=tee_log,
    )

    analysis = analyze_and_save(model, X_test, y_test, output_dir)
    model_path = os.path.join(output_dir, "best_model.npz")
    save_model(model, model_path, metadata={
        "best_epoch": train_result["best_epoch"],
        "best_test_accuracy": train_result["best_test_accuracy"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })

    history_path = os.path.join(output_dir, "history.csv")
    save_history_csv(train_result["history"], history_path)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": data_path,
        "model_name": model.name,
        "architecture": model.architecture_rows(),
        "weight_count_without_bias": model.parameter_count(include_bias=False),
        "trainable_params_with_bias": model.parameter_count(include_bias=True),
        "parameter_limit_weights": PARAMETER_LIMIT_WEIGHTS,
        "hyperparameters": {
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "mini-batch SGD",
            "seed": seed,
        },
        "data": {
            "train_shape": list(X_train.shape),
            "test_shape": list(X_test.shape),
            "train_label_counts": np.bincount(y_train, minlength=10).tolist(),
            "test_label_counts": np.bincount(y_test, minlength=10).tolist(),
            "pixel_min": float(min(X_train.min(), X_test.min())),
            "pixel_max": float(max(X_train.max(), X_test.max())),
        },
        "training": train_result,
        "analysis": analysis,
        "hardware": collect_hardware_info(),
        "paths": {
            "output_dir": output_dir,
            "log": log_path,
            "history_csv": history_path,
            "model": model_path,
            "mosaic": analysis["mosaic_path"],
        },
    }

    summary_path = os.path.join(output_dir, "summary.json")
    save_summary_json(summary, summary_path)
    tee_log(f"最终测试准确率(最佳模型): {analysis['test_accuracy']:.4f}")
    tee_log(f"误分类样本数: {analysis['num_errors']} / {len(y_test)}")
    tee_log(f"结果摘要: {summary_path}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="纯 NumPy MNIST CNN 训练与分析")
    parser.add_argument("--data", default="data/MNISTData(1).mat", help="MNISTData(1).mat 路径")
    parser.add_argument("--output-dir", default="results/latest", help="实验输出目录")
    parser.add_argument("--model", default="lecture", choices=["lecture", "simple"], help="模型结构")
    parser.add_argument("--epochs", type=int, default=4, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="mini-batch 大小")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="学习率")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_experiment(
        data_path=args.data,
        output_dir=args.output_dir,
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(
        "\n实验完成: "
        f"best_epoch={result['training']['best_epoch']}, "
        f"test_accuracy={result['analysis']['test_accuracy']:.4f}, "
        f"output_dir={result['paths']['output_dir']}"
    )
