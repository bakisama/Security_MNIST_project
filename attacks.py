import time
from pathlib import Path
import tensorflow as tf
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod
from art.defences.trainer import AdversarialTrainer
from eval_utils import evaluate_model, plot_confusion_matrix, save_json

def fgsm_pipeline(model, x_test, y_test, eps: float, img_dir: Path, metric_path: Path):
    classifier = KerasClassifier(model=model, clip_values=(0.0, 1.0), use_logits=False)
    attack = FastGradientMethod(estimator=classifier, eps=eps)
    t0 = time.time()
    x_adv = attack.generate(x=x_test)
    gen_time = time.time() - t0

    loss_clean, acc_clean, _ = evaluate_model(model, x_test, y_test)
    loss_adv, acc_adv, cm_adv = evaluate_model(model, x_adv, y_test)

    plot_confusion_matrix(
        cm_adv,
        f"Baseline on FGSM (eps={eps}, acc={acc_adv:.4f})",
        img_dir / "baseline_fgsm_confusion.png",
    )

    save_json(
        metric_path,
        {
            "scenario": "baseline_fgsm",
            "attack": "FGSM",
            "eps": eps,
            "generation_time_seconds": gen_time,
            "baseline_clean_loss": loss_clean,
            "baseline_clean_accuracy": acc_clean,
            "baseline_adversarial_loss": loss_adv,
            "baseline_adversarial_accuracy": acc_adv,
        },
    )
    return x_adv, acc_clean, acc_adv

def adversarial_training(baseline_model_path: Path, x_train, y_train,
                         x_test, y_test, eps: float, epochs: int, batch_size: int,
                         img_dir: Path, model_out: Path, metric_out: Path):
    adv_model = tf.keras.models.load_model(baseline_model_path)
    adv_classifier = KerasClassifier(model=adv_model, clip_values=(0.0, 1.0), use_logits=False)
    attack = FastGradientMethod(estimator=adv_classifier, eps=eps)
    trainer = AdversarialTrainer(classifier=adv_classifier, attacks=attack, ratio=0.5)

    t0 = time.time()
    trainer.fit(x_train, y_train, nb_epochs=epochs, batch_size=batch_size)
    train_time = time.time() - t0

    adv_model.save(model_out)

    x_test_adv = attack.generate(x=x_test)
    loss_clean, acc_clean, cm_clean = evaluate_model(adv_model, x_test, y_test)
    loss_adv, acc_adv, cm_adv = evaluate_model(adv_model, x_test_adv, y_test)

    plot_confusion_matrix(cm_clean, f"Adv-Trained on Clean (acc={acc_clean:.4f})",
                          img_dir / "advtrained_clean_confusion.png")
    plot_confusion_matrix(cm_adv, f"Adv-Trained on FGSM (eps={eps}, acc={acc_adv:.4f})",
                          img_dir / "advtrained_fgsm_confusion.png")

    save_json(
        metric_out,
        {
            "scenario": "advtrained_fgsm",
            "attack": "FGSM",
            "eps": eps,
            "advtrain_epochs": epochs,
            "advtrain_batch_size": batch_size,
            "advtrain_time_seconds": train_time,
            "advtrained_clean_loss": loss_clean,
            "advtrained_clean_accuracy": acc_clean,
            "advtrained_adversarial_loss": loss_adv,
            "advtrained_adversarial_accuracy": acc_adv,
        },
    )
    return acc_clean, acc_adv
