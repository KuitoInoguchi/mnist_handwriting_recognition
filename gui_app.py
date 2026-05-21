import json
import os
import argparse
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from data import load_mnist
from main import analyze_and_save, run_experiment
from model import load_model
from utils import save_summary_json


HOST = "127.0.0.1"
DEFAULT_PORT = 8000
ROOT_DIR = os.path.abspath(os.getcwd())
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "latest")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

STATE_LOCK = threading.Lock()
STATE = {
    "running": False,
    "phase": "idle",
    "logs": [],
    "summary": None,
    "last_error": None,
}

DEMO_MODEL_LOCK = threading.Lock()
DEMO_MODEL_CACHE = {
    "summary_path": None,
    "model_path": None,
    "summary_mtime": None,
    "summary": None,
    "model": None,
}


def load_initial_summary():
    summary_path = os.path.join(DEFAULT_OUTPUT_DIR, "summary.json")
    if not os.path.exists(summary_path):
        return
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            STATE["summary"] = json.load(f)
        STATE["logs"].append(f"[{time.strftime('%H:%M:%S')}] 已加载最近一次实验结果: {summary_path}")
    except Exception as exc:
        STATE["last_error"] = str(exc)


load_initial_summary()


def add_log(message):
    stamped = f"[{time.strftime('%H:%M:%S')}] {message}"
    with STATE_LOCK:
        STATE["logs"].append(stamped)
        STATE["logs"] = STATE["logs"][-500:]


def set_state(**kwargs):
    with STATE_LOCK:
        STATE.update(kwargs)


def snapshot_state():
    with STATE_LOCK:
        return dict(STATE)


def resolve_workspace_path(path):
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.join(ROOT_DIR, path)


def find_best_demo_artifact():
    best_info = None

    if not os.path.isdir(RESULTS_DIR):
        raise FileNotFoundError("结果目录不存在，请先训练模型。")

    for entry in os.scandir(RESULTS_DIR):
        if not entry.is_dir():
            continue

        summary_path = os.path.join(entry.path, "summary.json")
        if not os.path.exists(summary_path):
            continue

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            continue

        score = summary.get("analysis", {}).get("test_accuracy")
        if score is None:
            score = summary.get("training", {}).get("best_test_accuracy")
        if score is None:
            continue

        model_path = resolve_workspace_path(summary.get("paths", {}).get("model"))
        if not model_path or not os.path.exists(model_path):
            fallback_path = os.path.join(entry.path, "best_model.npz")
            model_path = fallback_path if os.path.exists(fallback_path) else None
        if not model_path:
            continue

        info = {
            "summary_path": summary_path,
            "model_path": model_path,
            "summary_mtime": os.path.getmtime(summary_path),
            "score": float(score),
            "summary": summary,
        }

        if best_info is None:
            best_info = info
            continue

        current_key = (
            info["score"],
            -int(summary.get("trainable_params_with_bias", 10**18)),
            info["summary_mtime"],
        )
        best_key = (
            best_info["score"],
            -int(best_info["summary"].get("trainable_params_with_bias", 10**18)),
            best_info["summary_mtime"],
        )
        if current_key > best_key:
            best_info = info

    if best_info is None:
        raise FileNotFoundError("尚未找到可用于演示的已训练模型，请先完成训练。")

    summary = best_info["summary"]
    hp = summary.get("hyperparameters", {})
    best_info["descriptor"] = {
        "model_name": summary.get("model_name", "unknown"),
        "test_accuracy": best_info["score"],
        "epochs": hp.get("epochs"),
        "batch_size": hp.get("batch_size"),
        "learning_rate": hp.get("learning_rate"),
        "trainable_params_with_bias": summary.get("trainable_params_with_bias"),
        "weight_count_without_bias": summary.get("weight_count_without_bias"),
        "model_path": best_info["model_path"],
    }
    return best_info


def get_demo_predictor():
    artifact = find_best_demo_artifact()
    with DEMO_MODEL_LOCK:
        if (
            DEMO_MODEL_CACHE["summary_path"] != artifact["summary_path"]
            or DEMO_MODEL_CACHE["model_path"] != artifact["model_path"]
            or DEMO_MODEL_CACHE["summary_mtime"] != artifact["summary_mtime"]
            or DEMO_MODEL_CACHE["model"] is None
        ):
            DEMO_MODEL_CACHE["summary_path"] = artifact["summary_path"]
            DEMO_MODEL_CACHE["model_path"] = artifact["model_path"]
            DEMO_MODEL_CACHE["summary_mtime"] = artifact["summary_mtime"]
            DEMO_MODEL_CACHE["summary"] = artifact["summary"]
            DEMO_MODEL_CACHE["model"] = load_model(artifact["model_path"])

        return artifact["descriptor"], DEMO_MODEL_CACHE["model"]


def predict_drawn_digit(payload):
    pixels = np.asarray(payload.get("pixels"), dtype=np.float32)
    if pixels.size != 28 * 28:
        raise ValueError("预测输入必须是 28x28 的灰度像素数组。")

    pixels = np.clip(pixels.reshape(28, 28), 0.0, 1.0)
    active = int(np.count_nonzero(pixels > 0.10))
    if active < 8:
        info, _ = get_demo_predictor()
        return {
            "ok": True,
            "blank": True,
            "active_pixels": active,
            "demo_model": info,
        }

    info, model = get_demo_predictor()
    batch = pixels[None, None, :, :].astype(np.float32)

    with DEMO_MODEL_LOCK:
        probs = model.forward(batch, verbose=False)[0]

    top_indices = np.argsort(-probs)[:3]
    return {
        "ok": True,
        "blank": False,
        "prediction": int(np.argmax(probs)),
        "confidence": float(np.max(probs)),
        "topk": [
            {"digit": int(idx), "confidence": float(probs[idx])}
            for idx in top_indices
        ],
        "active_pixels": active,
        "demo_model": info,
    }


def read_json_body(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw) if raw else {}


def safe_file_path(path):
    abs_path = os.path.abspath(path)
    if os.path.commonpath([ROOT_DIR, abs_path]) != ROOT_DIR:
        raise ValueError("拒绝访问工作区外文件")
    if not os.path.exists(abs_path):
        raise FileNotFoundError(abs_path)
    return abs_path


def run_training(payload):
    set_state(running=True, phase="training", last_error=None)
    try:
        output_dir = payload.get("output_dir") or DEFAULT_OUTPUT_DIR
        summary = run_experiment(
            data_path=payload.get("data_path") or "data/MNISTData(1).mat",
            output_dir=output_dir,
            model_name=payload.get("model") or "lecture",
            num_epochs=int(payload.get("epochs", 4)),
            batch_size=int(payload.get("batch_size", 64)),
            learning_rate=float(payload.get("learning_rate", 0.03)),
            seed=int(payload.get("seed", 0)),
            log_fn=add_log,
        )
        set_state(summary=summary, phase="done")
    except Exception:
        error = traceback.format_exc()
        add_log(error)
        set_state(last_error=error, phase="error")
    finally:
        set_state(running=False)


def run_testing(payload):
    set_state(running=True, phase="testing", last_error=None)
    try:
        output_dir = payload.get("output_dir") or DEFAULT_OUTPUT_DIR
        model_path = os.path.join(output_dir, "best_model.npz")
        if not os.path.exists(model_path):
            raise FileNotFoundError("尚未找到 best_model.npz，请先训练模型。")

        add_log("加载最佳模型并重新测试...")
        X_train, y_train, X_test, y_test = load_mnist(payload.get("data_path") or "data/MNISTData(1).mat", verbose=False)
        model = load_model(model_path)
        analysis = analyze_and_save(model, X_test, y_test, output_dir)

        summary_path = os.path.join(output_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        else:
            summary = {
                "model_name": model.name,
                "architecture": model.architecture_rows(),
                "weight_count_without_bias": model.parameter_count(False),
                "trainable_params_with_bias": model.parameter_count(True),
                "data": {
                    "train_shape": list(X_train.shape),
                    "test_shape": list(X_test.shape),
                    "train_label_counts": np.bincount(y_train, minlength=10).tolist(),
                    "test_label_counts": np.bincount(y_test, minlength=10).tolist(),
                },
                "paths": {
                    "output_dir": output_dir,
                    "model": model_path,
                    "mosaic": analysis["mosaic_path"],
                },
            }

        summary["analysis"] = analysis
        summary.setdefault("paths", {})["mosaic"] = analysis["mosaic_path"]
        save_summary_json(summary, summary_path)
        add_log(f"测试完成: accuracy={analysis['test_accuracy']:.4f}, errors={analysis['num_errors']}")
        set_state(summary=summary, phase="done")
    except Exception:
        error = traceback.format_exc()
        add_log(error)
        set_state(last_error=error, phase="error")
    finally:
        set_state(running=False)


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MNIST CNN 实验控制台</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1f2933;
      --muted: #687381;
      --line: #d8dee6;
      --panel: #f8fafc;
      --accent: #0b6bcb;
      --accent-2: #c45500;
      --ok: #137333;
      --bad: #b3261e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #f6f8fb;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .status {
      font-size: 13px;
      color: var(--muted);
    }
    main {
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: calc(100vh - 54px);
    }
    aside {
      border-right: 1px solid var(--line);
      padding: 16px;
      background: var(--panel);
    }
    section {
      padding: 16px 18px;
      min-width: 0;
    }
    label {
      display: grid;
      gap: 5px;
      margin-bottom: 12px;
      color: var(--muted);
      font-size: 13px;
    }
    input, select {
      width: 100%;
      padding: 8px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font-size: 14px;
    }
    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }
    button {
      min-height: 38px;
      border: 1px solid transparent;
      border-radius: 6px;
      color: #fff;
      background: var(--accent);
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary {
      background: #fff;
      color: var(--accent);
      border-color: var(--accent);
    }
    button:disabled {
      opacity: .55;
      cursor: default;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fff;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }
    .metric strong {
      font-size: 20px;
    }
    .split {
      display: grid;
      grid-template-columns: minmax(340px, 1fr) minmax(360px, 1fr);
      gap: 14px;
      align-items: start;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      overflow: hidden;
    }
    .panel h2 {
      margin: 0;
      padding: 10px 12px;
      font-size: 14px;
      border-bottom: 1px solid var(--line);
      background: #f9fbfd;
    }
    pre {
      margin: 0;
      padding: 12px;
      height: 260px;
      overflow: auto;
      background: #0d1117;
      color: #dce8ff;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 6px 7px;
      text-align: right;
    }
    th:first-child, td:first-child {
      text-align: left;
      color: var(--muted);
    }
    .confusion td.diag {
      color: var(--ok);
      font-weight: 700;
      background: #eef7ee;
    }
    .confusion td.off {
      color: var(--bad);
    }
    .mosaic {
      display: block;
      width: 100%;
      max-width: 620px;
      image-rendering: pixelated;
      background: #111;
    }
    .draw-shell {
      display: grid;
      grid-template-columns: minmax(260px, 320px) minmax(260px, 1fr);
      gap: 16px;
      padding: 14px;
      align-items: start;
    }
    .draw-stage {
      display: grid;
      gap: 10px;
      justify-items: start;
    }
    .draw-board {
      width: min(100%, 280px);
      aspect-ratio: 1;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #05070b;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,.03);
      touch-action: none;
      cursor: crosshair;
    }
    .draw-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .draw-actions button {
      min-width: 110px;
    }
    .draw-note {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .prediction-stack {
      display: grid;
      gap: 12px;
    }
    .prediction-hero {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 14px;
      align-items: center;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: linear-gradient(180deg, #f9fbff 0%, #f3f8fd 100%);
    }
    .preview-wrap {
      display: grid;
      justify-items: center;
      gap: 8px;
    }
    .preview-canvas {
      width: 112px;
      height: 112px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #020407;
      image-rendering: pixelated;
    }
    .predict-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .predict-digit {
      font-size: 56px;
      line-height: 1;
      font-weight: 800;
      letter-spacing: -1px;
    }
    .predict-confidence {
      margin-top: 8px;
      font-size: 14px;
      color: var(--muted);
    }
    .predict-status {
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }
    .topk-box, .demo-box {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      background: #fff;
    }
    .mini-title {
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 8px;
    }
    .topk-list {
      display: grid;
      gap: 6px;
      font-size: 13px;
    }
    .topk-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .demo-box {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }
    .empty {
      padding: 16px;
      color: var(--muted);
      font-size: 13px;
    }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .grid, .split { grid-template-columns: 1fr; }
      .draw-shell, .prediction-hero { grid-template-columns: 1fr; }
      .draw-stage { justify-items: stretch; }
      .draw-board { width: 100%; }
    }
  </style>
</head>
<body>
  <header>
    <h1>MNIST CNN 实验控制台</h1>
    <div class="status" id="status">idle</div>
  </header>
  <main>
    <aside>
      <label>模型结构
        <select id="model">
          <option value="lecture">lecture: 9x9x20 + 100隐层</option>
          <option value="simple">simple: 两层小卷积对比</option>
          <option value="double">double: 双层卷积高精度模型</option>
          <option value="triple">triple: 三层卷积高参数效率模型</option>
        </select>
      </label>
      <label>Epochs
        <input id="epochs" type="number" min="1" max="20" value="4">
      </label>
      <label>Batch Size
        <input id="batch_size" type="number" min="8" max="512" value="64">
      </label>
      <label>Learning Rate
        <input id="learning_rate" type="number" min="0.001" max="1" step="0.001" value="0.03">
      </label>
      <label>Seed
        <input id="seed" type="number" value="0">
      </label>
      <div class="button-row">
        <button id="trainBtn">开始训练</button>
        <button id="testBtn" class="secondary">测试模型</button>
      </div>
    </aside>
    <section>
      <div class="grid" id="metrics"></div>
      <div class="panel" style="margin-bottom:14px;">
        <h2>趣味功能：手写数字实时识别</h2>
        <div class="draw-shell">
          <div class="draw-stage">
            <canvas id="drawCanvas" class="draw-board" width="280" height="280" aria-label="手写数字画板"></canvas>
            <div class="draw-actions">
              <button id="clearDrawBtn" class="secondary">清空画板</button>
            </div>
            <div class="draw-note">
              用鼠标或触控笔在黑底画板上写一个数字。系统会自动把笔迹压缩为 28×28 输入，
              并调用当前结果库中测试集准确率最高的已训练模型进行实时识别。
            </div>
          </div>
          <div class="prediction-stack">
            <div class="prediction-hero">
              <div class="preview-wrap">
                <canvas id="previewCanvas" class="preview-canvas" width="28" height="28" aria-label="28x28预处理预览"></canvas>
                <div class="predict-label">送入模型的 28×28 输入</div>
              </div>
              <div>
                <div class="predict-label">模型预测结果</div>
                <div class="predict-digit" id="predictDigit">—</div>
                <div class="predict-confidence" id="predictConfidence">等待输入</div>
                <div class="predict-status" id="predictStatus">在画板上写一个 0~9 的数字即可实时识别。</div>
              </div>
            </div>
            <div class="topk-box">
              <div class="mini-title">Top-3 概率</div>
              <div class="topk-list" id="topkBox">尚未开始识别。</div>
            </div>
            <div class="demo-box" id="demoModelInfo">
              识别模型：结果库中的最优已训练模型将自动加载并用于演示。
            </div>
          </div>
        </div>
      </div>
      <div class="split">
        <div class="panel">
          <h2>运行日志</h2>
          <pre id="logs"></pre>
        </div>
        <div class="panel">
          <h2>误分类样本</h2>
          <div id="mosaicBox" class="empty">训练或测试完成后显示。</div>
        </div>
      </div>
      <div class="split" style="margin-top:14px;">
        <div class="panel">
          <h2>混淆矩阵</h2>
          <div id="confusion" class="empty">暂无结果。</div>
        </div>
        <div class="panel">
          <h2>高置信误分类</h2>
          <div id="errors" class="empty">暂无结果。</div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const drawCanvas = $("drawCanvas");
    const drawCtx = drawCanvas.getContext("2d");
    const previewCanvas = $("previewCanvas");
    const previewCtx = previewCanvas.getContext("2d");
    let drawing = false;
    let lastPoint = null;
    let predictTimer = null;
    let predictSequence = 0;

    function payload() {
      return {
        model: $("model").value,
        epochs: Number($("epochs").value),
        batch_size: Number($("batch_size").value),
        learning_rate: Number($("learning_rate").value),
        seed: Number($("seed").value)
      };
    }

    async function postJSON(url, body) {
      const res = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "请求失败");
      }
      return data;
    }

    function pct(v) {
      return v == null ? "—" : (v * 100).toFixed(2) + "%";
    }

    function renderMetrics(summary) {
      const metrics = [];
      if (summary) {
        const training = summary.training || {};
        const analysis = summary.analysis || {};
        metrics.push(["测试准确率", pct(analysis.test_accuracy)]);
        metrics.push(["误分类数", analysis.num_errors == null ? "—" : analysis.num_errors]);
        metrics.push(["最佳 Epoch", training.best_epoch || "—"]);
        metrics.push(["训练耗时", training.train_seconds == null ? "—" : training.train_seconds.toFixed(2) + "s"]);
        metrics.push(["权值数", `${summary.weight_count_without_bias || "—"} / ${summary.parameter_limit_weights || "—"}`]);
        metrics.push(["含 bias 参数", summary.trainable_params_with_bias || "—"]);
        metrics.push(["模型", summary.model_name || "—"]);
        metrics.push(["优化器", summary.hyperparameters?.optimizer || "mini-batch SGD"]);
      }
      $("metrics").innerHTML = metrics.length
        ? metrics.map(([k, v]) => `<div class="metric"><span>${k}</span><strong>${v}</strong></div>`).join("")
        : `<div class="metric"><span>状态</span><strong>等待运行</strong></div>`;
    }

    function renderConfusion(matrix) {
      if (!matrix) {
        $("confusion").className = "empty";
        $("confusion").textContent = "暂无结果。";
        return;
      }
      const head = "<tr><th>真\\预</th>" + [...Array(10).keys()].map(i => `<th>${i}</th>`).join("") + "</tr>";
      const rows = matrix.map((row, i) => {
        const cells = row.map((v, j) => `<td class="${i === j ? "diag" : (v ? "off" : "")}">${v}</td>`).join("");
        return `<tr><th>${i}</th>${cells}</tr>`;
      }).join("");
      $("confusion").className = "";
      $("confusion").innerHTML = `<table class="confusion">${head}${rows}</table>`;
    }

    function renderErrors(errors) {
      if (!errors || !errors.length) {
        $("errors").className = "empty";
        $("errors").textContent = "暂无误分类样本。";
        return;
      }
      const rows = errors.map(e =>
        `<tr><td>${e.index}</td><td>${e.true}</td><td>${e.pred}</td><td>${pct(e.confidence)}</td></tr>`
      ).join("");
      $("errors").className = "";
      $("errors").innerHTML = `<table><tr><th>样本</th><th>真实</th><th>预测</th><th>置信度</th></tr>${rows}</table>`;
    }

    function renderMosaic(summary) {
      const path = summary?.paths?.mosaic || summary?.analysis?.mosaic_path;
      if (!path) {
        $("mosaicBox").className = "empty";
        $("mosaicBox").textContent = "训练或测试完成后显示。";
        return;
      }
      $("mosaicBox").className = "";
      $("mosaicBox").innerHTML = `<img class="mosaic" src="/api/file?path=${encodeURIComponent(path)}&t=${Date.now()}" alt="误分类样本">`;
    }

    function formatDemoModel(info) {
      if (!info) {
        return "识别模型：结果库中的最优已训练模型将自动加载并用于演示。";
      }
      return `识别模型：${info.model_name}，测试准确率 ${pct(info.test_accuracy)}，`
        + `epochs=${info.epochs}，batch_size=${info.batch_size}，learning_rate=${info.learning_rate}，`
        + `含 bias 参数 ${info.trainable_params_with_bias ?? "—"}。`;
    }

    function resetPreview() {
      previewCtx.fillStyle = "#000";
      previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    }

    function resetPrediction(message = "在画板上写一个 0~9 的数字即可实时识别。") {
      $("predictDigit").textContent = "—";
      $("predictConfidence").textContent = "等待输入";
      $("predictStatus").textContent = message;
      $("topkBox").textContent = "尚未开始识别。";
      resetPreview();
    }

    function renderTopk(topk) {
      if (!topk || !topk.length) {
        $("topkBox").textContent = "暂无结果。";
        return;
      }
      $("topkBox").innerHTML = topk.map((item, idx) =>
        `<div class="topk-row"><span>#${idx + 1} 数字 ${item.digit}</span><strong>${pct(item.confidence)}</strong></div>`
      ).join("");
    }

    function renderProcessedPreview(processed) {
      resetPreview();
      if (!processed?.canvas) {
        return;
      }
      previewCtx.imageSmoothingEnabled = false;
      previewCtx.drawImage(processed.canvas, 0, 0);
    }

    function canvasPoint(evt) {
      const rect = drawCanvas.getBoundingClientRect();
      return {
        x: (evt.clientX - rect.left) * drawCanvas.width / rect.width,
        y: (evt.clientY - rect.top) * drawCanvas.height / rect.height
      };
    }

    function drawSegment(a, b) {
      drawCtx.strokeStyle = "#fff";
      drawCtx.lineWidth = 22;
      drawCtx.lineCap = "round";
      drawCtx.lineJoin = "round";
      drawCtx.beginPath();
      drawCtx.moveTo(a.x, a.y);
      drawCtx.lineTo(b.x, b.y);
      drawCtx.stroke();
    }

    function initializeBoard() {
      drawCtx.fillStyle = "#000";
      drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
      resetPreview();
    }

    function clearBoard() {
      initializeBoard();
      predictSequence += 1;
      clearTimeout(predictTimer);
      resetPrediction("画板已清空，可以重新写一个数字。");
    }

    function extractDigit() {
      const size = drawCanvas.width;
      const source = drawCtx.getImageData(0, 0, size, size);
      const data = source.data;
      let minX = size, minY = size, maxX = -1, maxY = -1;
      let ink = 0;

      for (let y = 0; y < size; y += 1) {
        for (let x = 0; x < size; x += 1) {
          const value = data[(y * size + x) * 4];
          if (value > 18) {
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            ink += value;
          }
        }
      }

      if (maxX < 0 || ink < 1500) {
        return null;
      }

      const pad = 18;
      minX = Math.max(0, minX - pad);
      minY = Math.max(0, minY - pad);
      maxX = Math.min(size - 1, maxX + pad);
      maxY = Math.min(size - 1, maxY + pad);

      const cropW = maxX - minX + 1;
      const cropH = maxY - minY + 1;
      const normalized = document.createElement("canvas");
      normalized.width = 28;
      normalized.height = 28;
      const nctx = normalized.getContext("2d");
      nctx.fillStyle = "#000";
      nctx.fillRect(0, 0, 28, 28);
      nctx.imageSmoothingEnabled = true;

      const scale = 20 / Math.max(cropW, cropH);
      const drawW = cropW * scale;
      const drawH = cropH * scale;
      const dx = (28 - drawW) / 2;
      const dy = (28 - drawH) / 2;
      nctx.drawImage(drawCanvas, minX, minY, cropW, cropH, dx, dy, drawW, drawH);

      const recentered = document.createElement("canvas");
      recentered.width = 28;
      recentered.height = 28;
      const rctx = recentered.getContext("2d");
      rctx.fillStyle = "#000";
      rctx.fillRect(0, 0, 28, 28);

      const normalizedData = nctx.getImageData(0, 0, 28, 28).data;
      let sum = 0;
      let sumX = 0;
      let sumY = 0;
      for (let y = 0; y < 28; y += 1) {
        for (let x = 0; x < 28; x += 1) {
          const value = normalizedData[(y * 28 + x) * 4] / 255;
          sum += value;
          sumX += x * value;
          sumY += y * value;
        }
      }
      if (sum <= 1e-6) {
        return null;
      }

      const shiftX = 13.5 - (sumX / sum);
      const shiftY = 13.5 - (sumY / sum);
      rctx.drawImage(normalized, shiftX, shiftY);

      const finalData = rctx.getImageData(0, 0, 28, 28).data;
      const pixels = [];
      for (let i = 0; i < finalData.length; i += 4) {
        pixels.push(finalData[i] / 255);
      }
      return { pixels, canvas: recentered };
    }

    async function loadDemoModelInfo() {
      try {
        const res = await fetch("/api/demo-model");
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "加载演示模型信息失败");
        }
        $("demoModelInfo").textContent = formatDemoModel(data.demo_model);
      } catch (err) {
        $("demoModelInfo").textContent = `识别模型加载失败：${err.message}`;
      }
    }

    async function runRecognition() {
      const processed = extractDigit();
      renderProcessedPreview(processed);
      if (!processed) {
        predictSequence += 1;
        resetPrediction("当前笔迹过少，继续书写即可开始识别。");
        return;
      }

      const requestId = ++predictSequence;
      $("predictStatus").textContent = "识别中...";

      try {
        const res = await postJSON("/api/predict", { pixels: processed.pixels });
        if (requestId !== predictSequence) {
          return;
        }

        $("demoModelInfo").textContent = formatDemoModel(res.demo_model);
        if (res.blank) {
          resetPrediction("当前画板接近空白，模型暂不输出预测。");
          return;
        }

        $("predictDigit").textContent = String(res.prediction);
        $("predictConfidence").textContent = `置信度 ${pct(res.confidence)}`;
        $("predictStatus").textContent = `已激活像素 ${res.active_pixels} 个，抬笔后会自动刷新识别。`;
        renderTopk(res.topk);
      } catch (err) {
        if (requestId !== predictSequence) {
          return;
        }
        $("predictStatus").textContent = `识别失败：${err.message}`;
        $("topkBox").textContent = "预测请求失败。";
      }
    }

    function scheduleRecognition(delay = 140) {
      clearTimeout(predictTimer);
      predictTimer = setTimeout(runRecognition, delay);
    }

    drawCanvas.addEventListener("pointerdown", (evt) => {
      evt.preventDefault();
      drawing = true;
      lastPoint = canvasPoint(evt);
      drawCanvas.setPointerCapture(evt.pointerId);
      drawSegment(lastPoint, { x: lastPoint.x + 0.01, y: lastPoint.y + 0.01 });
      scheduleRecognition(180);
    });

    drawCanvas.addEventListener("pointermove", (evt) => {
      if (!drawing) {
        return;
      }
      evt.preventDefault();
      const point = canvasPoint(evt);
      drawSegment(lastPoint, point);
      lastPoint = point;
      scheduleRecognition(120);
    });

    function stopDrawing() {
      if (!drawing) {
        return;
      }
      drawing = false;
      lastPoint = null;
      scheduleRecognition(60);
    }

    drawCanvas.addEventListener("pointerup", stopDrawing);
    drawCanvas.addEventListener("pointercancel", stopDrawing);
    drawCanvas.addEventListener("pointerleave", stopDrawing);
    $("clearDrawBtn").addEventListener("click", clearBoard);

    async function poll() {
      const res = await fetch("/api/status");
      const state = await res.json();
      $("status").textContent = state.running ? `${state.phase}...` : state.phase;
      $("trainBtn").disabled = state.running;
      $("testBtn").disabled = state.running;
      $("logs").textContent = (state.logs || []).join("\n");
      $("logs").scrollTop = $("logs").scrollHeight;
      renderMetrics(state.summary);
      renderConfusion(state.summary?.analysis?.confusion_matrix);
      renderErrors(state.summary?.analysis?.top_errors);
      renderMosaic(state.summary);
    }

    $("trainBtn").addEventListener("click", async () => {
      await postJSON("/api/train", payload());
      poll();
    });
    $("testBtn").addEventListener("click", async () => {
      await postJSON("/api/test", payload());
      poll();
    });
    initializeBoard();
    resetPrediction();
    loadDemoModelInfo();
    setInterval(poll, 1000);
    poll();
  </script>
</body>
</html>
"""


class GuiHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/status":
            self.send_json(snapshot_state())
            return

        if parsed.path == "/api/demo-model":
            try:
                info, _ = get_demo_predictor()
                self.send_json({"demo_model": info})
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=404)
            return

        if parsed.path == "/api/file":
            try:
                query = parse_qs(parsed.query)
                path = safe_file_path(query.get("path", [""])[0])
                with open(path, "rb") as f:
                    data = f.read()
                content_type = "image/png" if path.endswith(".png") else "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=404)
            return

        self.send_json({"error": "not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            try:
                payload = read_json_body(self)
                self.send_json(predict_drawn_digit(payload))
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=400)
            return

        if parsed.path not in {"/api/train", "/api/test"}:
            self.send_json({"error": "not found"}, status=404)
            return

        if snapshot_state()["running"]:
            self.send_json({"ok": False, "error": "已有任务正在运行"}, status=409)
            return

        payload = read_json_body(self)
        target = run_training if parsed.path == "/api/train" else run_testing
        thread = threading.Thread(target=target, args=(payload,), daemon=True)
        thread.start()
        self.send_json({"ok": True})


def run_server(port=DEFAULT_PORT):
    last_error = None
    for candidate in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((HOST, candidate), GuiHandler)
            print(f"MNIST CNN GUI 已启动: http://{HOST}:{candidate}", flush=True)
            server.serve_forever()
            return
        except OSError as exc:
            last_error = exc
            if getattr(exc, "errno", None) not in {48, 98}:
                raise
    raise last_error


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST CNN Web GUI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_server(args.port)
