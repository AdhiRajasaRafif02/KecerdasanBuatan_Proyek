"""SMS Spam Classification - Evaluation.

Kelompok 7 - Tugas Akhir Kecerdasan Buatan

Modul ini menyediakan:
1) Evaluasi model (Simple RNN & LSTM) pada test set
2) Metrik klasifikasi (accuracy, precision, recall, F1, dll.)
3) Confusion matrix + visualisasi
4) Classification report
5) (Opsional) ROC-AUC & PR-AUC + kurva jika probabilitas tersedia

Jalankan:
    python src/evaluate.py --model results/models/lstm.h5
    python src/evaluate.py --compare
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.models import load_model

from eda_preprocessing import load_and_preprocess_data


@dataclass
class EvalResult:
    model_path: str
    threshold: float
    n_test: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    balanced_accuracy: float
    mcc: float
    specificity: float
    npv: float
    tn: int
    fp: int
    fn: int
    tp: int
    roc_auc: float | None = None
    pr_auc: float | None = None
    log_loss: float | None = None


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d != 0 else 0.0


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
    *,
    model_path: str = "",
    threshold: float = 0.5,
) -> tuple[EvalResult, np.ndarray]:
    """Hitung metrik evaluasi.

    Catatan:
    - Asumsi label positif adalah `1` (spam).
    - `y_pred_proba` adalah probabilitas class 1 (spam).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = _safe_div(tn, tn + fp)
    npv = _safe_div(tn, tn + fn)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    roc_auc = pr_auc = ll = None
    if y_pred_proba is not None:
        y_pred_proba = np.asarray(y_pred_proba).reshape(-1)
        roc_auc = float(roc_auc_score(y_true, y_pred_proba))
        pr_auc = float(average_precision_score(y_true, y_pred_proba))
        ll = float(log_loss(y_true, y_pred_proba, labels=[0, 1]))

    result = EvalResult(
        model_path=model_path,
        threshold=float(threshold),
        n_test=int(len(y_true)),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        balanced_accuracy=bal_acc,
        mcc=mcc,
        specificity=specificity,
        npv=npv,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        log_loss=ll,
    )
    return result, cm


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str = "results/evaluation/confusion_matrix.png",
    class_names: list[str] | tuple[str, str] = ("Ham", "Spam"),
    *,
    normalize: bool = False,
):
    """Plot confusion matrix (raw / normalized) sebagai heatmap."""
    # Create output directory jika belum ada
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if normalize:
        cm_to_plot = cm.astype(float)
        row_sums = cm_to_plot.sum(axis=1, keepdims=True)
        cm_to_plot = np.divide(cm_to_plot, row_sums, out=np.zeros_like(cm_to_plot), where=row_sums != 0)
    else:
        cm_to_plot = cm.astype(int)

    # Create figure
    plt.figure(figsize=(8, 6))
    fmt = ".2f" if normalize else "d"
    sns.heatmap(
        cm_to_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        cbar_kws={"label": "Proportion" if normalize else "Count"},
        annot_kws={"fontsize": 14},
    )
    
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""), fontsize=14, fontweight="bold")
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Confusion matrix plot disimpan: {output_path}")
    
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    output_path: str,
):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] ROC curve disimpan: {output_path}")


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    output_path: str,
):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"PR-AUC (AP) = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] PR curve disimpan: {output_path}")


def _model_display_name(model_path: str) -> str:
    base = os.path.basename(model_path)
    name, _ext = os.path.splitext(base)
    return name or "model"


def _resolve_model_path(model_path: str) -> str:
    """Resolve model path for common invocation patterns.

    Supports:
    - Explicit paths: results/models/lstm.h5
    - Bare filenames: lstm.h5 -> results/models/lstm.h5 (if present)
    """
    model_path = os.path.expanduser(model_path)
    if os.path.exists(model_path):
        return model_path

    # If user passes just a filename, try the default models directory.
    if os.path.dirname(model_path) == "":
        candidate = os.path.join("results", "models", model_path)
        if os.path.exists(candidate):
            return candidate

    return model_path


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def evaluate_keras_model(
    *,
    model_path: str,
    max_words: int,
    max_len: int,
    test_size: float,
    random_state: int,
    threshold: float,
    output_dir: str,
    batch_size: int,
) -> EvalResult:
    model_path = _resolve_model_path(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}. "
            "Jalankan training dulu (python train_lstm.py) atau cek path model "
            "(contoh: results/models/lstm.h5)."
        )

    # 1) Load data
    _ = load_and_preprocess_data(
        csv_path="spam.csv",
        max_words=max_words,
        max_len=max_len,
        test_size=test_size,
        random_state=random_state,
    )
    X_train_pad, X_test_pad, y_train, y_test, _tokenizer, _label_encoder = _

    # 2) Load model
    model = load_model(model_path)

    # 3) Predict
    y_score = model.predict(X_test_pad, batch_size=batch_size, verbose=0).reshape(-1)
    y_pred = (y_score >= threshold).astype(int)

    # 4) Metrics
    eval_res, cm = evaluate_predictions(
        y_test,
        y_pred,
        y_score,
        model_path=model_path,
        threshold=threshold,
    )

    # 5) Save artifacts
    model_name = _model_display_name(model_path)
    run_dir = os.path.join(output_dir, model_name)
    _ensure_dir(run_dir)

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        payload = asdict(eval_res)
        payload["created_at"] = datetime.now().isoformat(timespec="seconds")
        payload["max_words"] = max_words
        payload["max_len"] = max_len
        payload["test_size"] = test_size
        payload["random_state"] = random_state
        json.dump(payload, f, indent=2)
    print(f"[OK] Metrics disimpan: {metrics_path}")

    report_txt = classification_report(y_test, y_pred, target_names=["Ham", "Spam"], digits=4, zero_division=0)
    report_path = os.path.join(run_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)
    print(f"[OK] Classification report disimpan: {report_path}")

    plot_confusion_matrix(cm, output_path=os.path.join(run_dir, "confusion_matrix.png"), normalize=False)
    plot_confusion_matrix(cm, output_path=os.path.join(run_dir, "confusion_matrix_normalized.png"), normalize=True)

    # Curves
    plot_roc_curve(y_test, y_score, output_path=os.path.join(run_dir, "roc_curve.png"))
    plot_pr_curve(y_test, y_score, output_path=os.path.join(run_dir, "pr_curve.png"))

    # 6) Console summary
    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY - {model_name}")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Test samples: {eval_res.n_test}")
    print(f"Threshold: {eval_res.threshold}")
    print("\nMetrics:")
    print(f"  Accuracy           : {eval_res.accuracy:.4f}")
    print(f"  Precision (Spam)   : {eval_res.precision:.4f}")
    print(f"  Recall (Spam)      : {eval_res.recall:.4f}")
    print(f"  F1-score (Spam)    : {eval_res.f1_score:.4f}")
    print(f"  Balanced Accuracy  : {eval_res.balanced_accuracy:.4f}")
    print(f"  Specificity (Ham)  : {eval_res.specificity:.4f}")
    print(f"  NPV                : {eval_res.npv:.4f}")
    print(f"  MCC                : {eval_res.mcc:.4f}")
    if eval_res.roc_auc is not None:
        print(f"  ROC-AUC            : {eval_res.roc_auc:.4f}")
    if eval_res.pr_auc is not None:
        print(f"  PR-AUC (AP)        : {eval_res.pr_auc:.4f}")
    if eval_res.log_loss is not None:
        print(f"  Log Loss           : {eval_res.log_loss:.4f}")
    print("\nConfusion Matrix (TN FP / FN TP):")
    print(f"  TN={eval_res.tn}  FP={eval_res.fp}")
    print(f"  FN={eval_res.fn}  TP={eval_res.tp}")
    print("\nClassification Report:")
    print(report_txt)

    return eval_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate SMS Spam model on test set")
    parser.add_argument(
        "--model",
        type=str,
        default="results/models/lstm.h5",
        help="Path model .h5 untuk dievaluasi (default: results/models/lstm.h5)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Evaluasi dua model: results/models/simple_rnn.h5 dan results/models/lstm.h5 (yang ada saja)",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold untuk prediksi spam")
    parser.add_argument("--max_words", type=int, default=5000, help="Tokenizer num_words (harus sama dgn training)")
    parser.add_argument("--max_len", type=int, default=100, help="Panjang sequence padding (harus sama dgn training)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proporsi test set (harus sama dgn training)")
    parser.add_argument("--random_state", type=int, default=42, help="Random state split (harus sama dgn training)")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size saat predict")
    parser.add_argument("--output_dir", type=str, default="results/evaluation", help="Folder output hasil evaluasi")
    args = parser.parse_args()

    print("\n" + "+" + "=" * 78 + "+")
    print("|" + " " * 22 + "SMS SPAM CLASSIFICATION - EVALUATION" + " " * 20 + "|")
    print("|" + " " * 30 + "Kelompok 7 - Tugas Akhir AI" + " " * 21 + "|")
    print("+" + "=" * 78 + "+\n")

    _ensure_dir(args.output_dir)

    model_paths: list[str]
    if args.compare:
        model_paths = ["results/models/simple_rnn.h5", "results/models/lstm.h5"]
        model_paths = [p for p in model_paths if os.path.exists(p)]
        if not model_paths:
            raise FileNotFoundError(
                "Tidak ada model ditemukan di results/models/. "
                "Jalankan training dulu: python src/train_lstm.py"
            )
    else:
        model_paths = [args.model]

    for p in model_paths:
        evaluate_keras_model(
            model_path=p,
            max_words=args.max_words,
            max_len=args.max_len,
            test_size=args.test_size,
            random_state=args.random_state,
            threshold=args.threshold,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )

