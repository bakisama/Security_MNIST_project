import json
from pathlib import Path
from config import (OUT_DIR, IMG_DIR, MODEL_DIR, METRIC_DIR,
                    USE_EXISTING_MODELS, BASELINE_EPOCHS, ADV_TRAIN_EPOCHS,
                    POISON_EPOCHS, BATCH_SIZE, FGSM_EPS, PATCH_SIZE, PATCH_VALUE, TARGET_CLASS)
from data import set_seeds, load_mnist
from model import build_cnn
from eval_utils import ensure_dirs, save_json, evaluate_model, measure_inference_time, plot_confusion_matrix, plot_history
from attacks import fgsm_pipeline, adversarial_training
from poisoning import poison_training_pipeline
from cache_demo import choose_cache_demo
import tensorflow as tf
import numpy as np

def main():
    set_seeds(42)
    ensure_dirs(OUT_DIR, IMG_DIR, MODEL_DIR, METRIC_DIR)

    x_train, y_train, x_test, y_test = load_mnist()
    input_shape = x_train.shape[1:]
    print("[INFO] Input shape:", input_shape)

    baseline_path = MODEL_DIR / "baseline_cnn.keras"

    if USE_EXISTING_MODELS and baseline_path.exists():
        model = tf.keras.models.load_model(baseline_path)
        history = None; train_time = None
        print("[INFO] Loaded existing baseline model from disk.")
    else:
        model = build_cnn(input_shape=input_shape)
        history = model.fit(
            x_train, y_train,
            epochs=BASELINE_EPOCHS, batch_size=BATCH_SIZE,
            validation_split=0.1, verbose=2
        )
        model.save(baseline_path)
        train_time = None  # optional: measure explicitly if you want

    # Evaluate baseline on clean
    clean_loss, clean_acc, clean_cm = evaluate_model(model, x_test, y_test)
    inf_time = measure_inference_time(model, x_test)
    print(f"[RESULT] Baseline – Clean Acc: {clean_acc:.4f}")
    plot_confusion_matrix(clean_cm, f"Baseline Confusion Matrix (accuracy={clean_acc:.4f})",
                          IMG_DIR / "baseline_confusion.png")
    plot_history(history, IMG_DIR / "baseline_accuracy.png", IMG_DIR / "baseline_loss.png")
    save_json(METRIC_DIR / "baseline_metrics.json", {
        "scenario": "baseline_clean",
        "loss": clean_loss, "accuracy": clean_acc,
        "train_time_seconds": None if train_time is None else float(train_time),
        "inference_time_per_image_seconds": inf_time,
        "epochs": BASELINE_EPOCHS, "batch_size": BATCH_SIZE
    })

    # FGSM pipeline
    x_test_adv, acc_clean_b, acc_adv_b = fgsm_pipeline(
        model, x_test, y_test, FGSM_EPS, IMG_DIR, METRIC_DIR / "fgsm_metrics.json"
    )

    # Adversarial training (blue team)
    acc_clean_adv, acc_adv_adv = adversarial_training(
        baseline_model_path=baseline_path,
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        eps=FGSM_EPS, epochs=ADV_TRAIN_EPOCHS, batch_size=BATCH_SIZE,
        img_dir=IMG_DIR, model_out=MODEL_DIR / "adv_trained_cnn.keras",
        metric_out=METRIC_DIR / "advtrain_metrics.json"
    )

    # Poison training pipeline
    acc_poison_clean, acc_target_clean, acc_target_patched = poison_training_pipeline(
        model_builder=build_cnn, input_shape=input_shape,
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        target_class=TARGET_CLASS, patch_size=PATCH_SIZE, patch_value=PATCH_VALUE,
        epochs=POISON_EPOCHS, batch_size=BATCH_SIZE,
        img_dir=IMG_DIR, model_out=MODEL_DIR / "poisoned_cnn.keras",
        metric_out=METRIC_DIR / "poison_metrics.json"
    )

    # Aggregate snapshot
    summary = {}
    for k in ["baseline_metrics.json","fgsm_metrics.json","advtrain_metrics.json","poison_metrics.json"]:
        p = METRIC_DIR / k
        if p.exists():
            summary[k[:-5]] = json.loads(p.read_text())
    save_json(METRIC_DIR / "metrics_summary.json", summary)

    # SAST demo (pickle vs json)
    choose_cache_demo()

    print("\n[INFO] All metrics ->", METRIC_DIR)
    print("[INFO] All plots   ->", IMG_DIR)
    print("[INFO] All models  ->", MODEL_DIR)

if __name__ == "__main__":
    main()
