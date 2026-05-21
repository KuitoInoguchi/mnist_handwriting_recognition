import csv
import json
import os
import struct
import zlib

import numpy as np


def accuracy(probs, y):
    pred = np.argmax(probs, axis=1)
    return float(np.mean(pred == y))


def predict(model, X, batch_size=256):
    from data import iterate_minibatches

    probs_list = []
    dummy_y = np.zeros(len(X), dtype=np.int64)
    for X_batch, _ in iterate_minibatches(X, dummy_y, batch_size=batch_size, shuffle=False):
        probs_list.append(model.forward(X_batch, verbose=False))
    return np.vstack(probs_list)


def evaluate(model, X, y, batch_size=256, return_details=False):
    probs = predict(model, X, batch_size=batch_size)
    pred = np.argmax(probs, axis=1)
    acc = float(np.mean(pred == y))

    if not return_details:
        return acc

    return {
        "accuracy": acc,
        "predictions": pred,
        "probabilities": probs,
    }


def confusion_matrix(y_true, y_pred, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def per_class_accuracy(confusion):
    totals = confusion.sum(axis=1)
    correct = np.diag(confusion)
    return np.divide(correct, totals, out=np.zeros_like(correct, dtype=np.float64), where=totals > 0)


def summarize_errors(y_true, y_pred, probs, limit=36):
    wrong = np.flatnonzero(y_true != y_pred)
    confidence = probs[wrong, y_pred[wrong]] if len(wrong) else np.array([], dtype=np.float32)
    order = np.argsort(-confidence)
    selected = wrong[order[:limit]]

    rows = []
    for idx in selected:
        rows.append({
            "index": int(idx),
            "true": int(y_true[idx]),
            "pred": int(y_pred[idx]),
            "confidence": float(probs[idx, y_pred[idx]]),
        })
    return rows


def save_history_csv(history, path):
    if not history:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_summary_json(summary, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def write_png(path, rgb):
    """
    用标准库写入 RGB PNG，避免 GUI/实验分析依赖 Pillow 或 matplotlib。
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("PNG 输入必须是 [H, W, 3] 的 uint8 RGB 数组")

    height, width, _ = rgb.shape
    raw_rows = [b"\x00" + rgb[row].tobytes() for row in range(height)]
    raw = b"".join(raw_rows)

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(png)


FONT_3X5 = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
    "T": ["111", "010", "010", "010", "010"],
    "P": ["110", "101", "110", "100", "100"],
    ":": ["0", "1", "0", "1", "0"],
    " ": ["0", "0", "0", "0", "0"],
}


def draw_text(canvas, x, y, text, color, scale=2):
    cursor = x
    for ch in text:
        glyph = FONT_3X5.get(ch.upper(), FONT_3X5[" "])
        glyph_width = len(glyph[0])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    y0 = y + gy * scale
                    x0 = cursor + gx * scale
                    canvas[y0:y0 + scale, x0:x0 + scale] = color
        cursor += (glyph_width + 1) * scale


def save_misclassified_mosaic(X, errors, path, columns=6, cell_size=72):
    """
    保存误分类样本宫格图。每格显示原图、T:真实标签、P:预测标签。
    """
    if not errors:
        canvas = np.full((cell_size, cell_size * 2, 3), 255, dtype=np.uint8)
        draw_text(canvas, 8, 24, "0", (20, 120, 20), scale=4)
        write_png(path, canvas)
        return

    rows = int(np.ceil(len(errors) / columns))
    canvas = np.full((rows * cell_size, columns * cell_size, 3), 18, dtype=np.uint8)

    for pos, item in enumerate(errors):
        row = pos // columns
        col = pos % columns
        top = row * cell_size
        left = col * cell_size

        canvas[top:top + cell_size, left:left + cell_size] = 0
        image = X[item["index"], 0]
        image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        image_big = np.repeat(np.repeat(image_u8, 2, axis=0), 2, axis=1)
        rgb = np.stack([image_big, image_big, image_big], axis=2)
        canvas[top + 2:top + 58, left + 8:left + 64] = rgb

        canvas[top:top + cell_size, left:left + 2] = (220, 40, 40)
        canvas[top:top + cell_size, left + cell_size - 2:left + cell_size] = (220, 40, 40)
        canvas[top:top + 2, left:left + cell_size] = (220, 40, 40)
        canvas[top + cell_size - 2:top + cell_size, left:left + cell_size] = (220, 40, 40)

        draw_text(canvas, left + 5, top + 60, f"T:{item['true']}", (40, 220, 80), scale=2)
        draw_text(canvas, left + 38, top + 60, f"P:{item['pred']}", (255, 220, 20), scale=2)

    write_png(path, canvas)
