import time
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def evaluate_model(model: tf.keras.Model, x, y):
    loss, acc = model.evaluate(x, y, verbose=0)
    y_pred = np.argmax(model.predict(x, verbose=0), axis=1)
    cm = confusion_matrix(y, y_pred)
    return float(loss), float(acc), cm

def measure_inference_time(model: tf.keras.Model, x):
    t0 = time.time()
    _ = model.predict(x, verbose=0)
    return float((time.time() - t0) / len(x))

def plot_confusion_matrix(cm, title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=list(range(10)), yticklabels=list(range(10)),
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.show(); plt.close(fig)

def plot_history(history, acc_path: Path, loss_path: Path):
    if history is None: return
    hist = history.history

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist["accuracy"], label="train_acc")
    if "val_accuracy" in hist: ax.plot(hist["val_accuracy"], label="val_acc")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend()
    fig.tight_layout(); fig.savefig(acc_path, bbox_inches="tight"); plt.show(); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist["loss"], label="train_loss")
    if "val_loss" in hist: ax.plot(hist["val_loss"], label="val_loss")
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
    fig.tight_layout(); fig.savefig(loss_path, bbox_inches="tight"); plt.show(); plt.close(fig)
