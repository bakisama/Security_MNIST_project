import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from eval_utils import evaluate_model, plot_confusion_matrix, save_json

def add_corner_patch(img, size=4, value=1.0):
    patched = img.copy()
    h, w = patched.shape[:2]
    patched[h - size : h, w - size : w] = value
    return patched

def create_poisoned_dataset(x_train, y_train, target_class=7, num_poison=100, patch_size=4, patch_value=1.0):
    target_idx = np.where(y_train == target_class)[0][:num_poison]
    imgs = x_train[target_idx]
    patched_imgs = np.array([add_corner_patch(img, size=patch_size, value=patch_value) for img in imgs])
    labels = y_train[target_idx].copy()
    return patched_imgs, labels, target_idx

def poison_training_pipeline(model_builder, input_shape, x_train, y_train, x_test, y_test,
                             target_class: int, patch_size: int, patch_value: float,
                             epochs: int, batch_size: int, img_dir: Path, model_out: Path, metric_out: Path):
    poison_imgs, poison_lbls, poison_idx = create_poisoned_dataset(
        x_train, y_train, target_class=target_class, num_poison=100,
        patch_size=patch_size, patch_value=patch_value
    )

    x_train_poisoned = x_train.copy()
    x_train_poisoned[poison_idx] = poison_imgs
    y_train_poisoned = y_train.copy()

    model = model_builder(input_shape=input_shape)
    history = model.fit(
        x_train_poisoned, y_train_poisoned,
        epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=2
    )
    model.save(model_out)

    loss_clean, acc_clean, cm_clean = evaluate_model(model, x_test, y_test)
    plot_confusion_matrix(cm_clean,
        f"Poison-Trained Model on Clean Test (acc={acc_clean:.4f})",
        img_dir / "poison_clean_confusion.png")

    # Trigger effect on target class
    target_test_idx = np.where(y_test == target_class)[0]
    x_target_clean = x_test[target_test_idx]
    x_target_patched = np.array([add_corner_patch(img, size=patch_size, value=patch_value) for img in x_target_clean])

    y_pred_target_clean = np.argmax(model.predict(x_target_clean, verbose=0), axis=1)
    y_pred_target_patched = np.argmax(model.predict(x_target_patched, verbose=0), axis=1)

    acc_target_clean = float(np.mean(y_pred_target_clean == y_test[target_test_idx]))
    acc_target_patched = float(np.mean(y_pred_target_patched == y_test[target_test_idx]))

    # Simple bar
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Target Clean", "Target Patched"]; vals = [acc_target_clean, acc_target_patched]
    ax.bar(range(len(labels)), vals); ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(vals): ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    ax.set_title(f"Effect of Corner Patch on Digit {target_class}")
    fig.tight_layout(); fig.savefig(img_dir / "poison_target_effect.png", bbox_inches="tight")
    plt.show(); plt.close(fig)

    save_json(metric_out, {
        "scenario": "poison_corner_patch",
        "mode": "feature",
        "target_class": target_class,
        "num_poisoned": int(len(poison_imgs)),
        "patch_size": patch_size,
        "patch_value": patch_value,
        "epochs": epochs,
        "batch_size": batch_size,
        "clean_test_loss": loss_clean,
        "clean_test_accuracy": acc_clean,
        "target_class_clean_accuracy": acc_target_clean,
        "target_class_patched_accuracy": acc_target_patched,
    })

    return acc_clean, acc_target_clean, acc_target_patched
