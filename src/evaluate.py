"""
Evaluation module untuk model SMS Spam Classification

Modul ini berisi fungsi-fungsi untuk:
- Melakukan prediksi pada test set
- Menghitung metrics: Accuracy, Precision, Recall, F1-Score
- Generate confusion matrix dan visualisasi
- Membuat classification report

Tahap Progress Saat Ini: Persiapan evaluasi (akan dikerjakan tahap training)
"""

import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def predict(model, X_test):
    """
    Melakukan prediksi pada test set.
    
    Args:
        model: Trained Keras model
        X_test (np.array): Test data dengan shape (n_samples, sequence_length)
    
    Returns:
        tuple: (y_pred_proba, y_pred_classes)
            - y_pred_proba: Predicted probabilities (0-1)
            - y_pred_classes: Predicted classes (0 atau 1)
    
    TODO pada tahap evaluasi:
        - Handle batch prediction untuk dataset besar
        - Add confidence scores
    """
    y_pred_proba = model.predict(X_test)
    y_pred_classes = (y_pred_proba > 0.5).astype(int).flatten()
    
    return y_pred_proba.flatten(), y_pred_classes


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Menghitung evaluation metrics untuk model.
    
    Args:
        y_true (np.array): True labels (0 atau 1)
        y_pred (np.array): Predicted labels (0 atau 1)
        y_pred_proba (np.array): Predicted probabilities (0-1), optional
    
    Returns:
        dict: Dictionary berisi metrics:
            - accuracy
            - precision
            - recall
            - f1_score
            - auc (jika y_pred_proba diberikan)
            - tn, fp, fn, tp (dari confusion matrix)
    
    TODO pada tahap evaluasi:
        - Add more metrics (specificity, sensitivity)
        - Validate inputs
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['tn'] = tn
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['tp'] = tp
    
    return metrics


def get_confusion_matrix(y_true, y_pred):
    """
    Generate confusion matrix.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
    
    Returns:
        np.array: Confusion matrix (2x2)
            [[TN, FP],
             [FN, TP]]
    
    TODO pada tahap evaluasi:
        - Add normalization option
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def plot_confusion_matrix(y_true, y_pred, class_names=['Ham', 'Spam'], save_path=None):
    """
    Plot confusion matrix sebagai heatmap.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        class_names (list): Nama class untuk labels
        save_path (str): Path untuk menyimpan plot (optional)
    
    Returns:
        None (displays/saves plot)
    
    TODO pada tahap evaluasi:
        - Add normalized confusion matrix option
        - Customize colors dan styles
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 14})
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix plot disimpan: {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curve dan hitung AUC.
    
    Args:
        y_true (np.array): True labels (0 atau 1)
        y_pred_proba (np.array): Predicted probabilities
        save_path (str): Path untuk menyimpan plot (optional)
    
    Returns:
        dict: Dictionary berisi fpr, tpr, auc
    
    TODO pada tahap evaluasi:
        - Compare multiple models ROC curves
        - Add threshold markers
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ ROC curve plot disimpan: {save_path}")
    
    plt.show()
    
    return {'fpr': fpr, 'tpr': tpr, 'auc': auc}


def print_evaluation_report(y_true, y_pred, class_names=['Ham', 'Spam']):
    """
    Print comprehensive evaluation report.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        class_names (list): Nama class
    
    Returns:
        str: Classification report
    
    TODO pada tahap evaluasi:
        - Save report to file
        - Add detailed analysis
    """
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    return report


def evaluate_full(model, X_test, y_test, model_name='Model', save_dir='results'):
    """
    Full evaluation pipeline: predictions, metrics, visualisasi, dan reports.
    
    Args:
        model: Trained Keras model
        X_test (np.array): Test data
        y_test (np.array): Test labels
        model_name (str): Nama model untuk labeling
        save_dir (str): Directory untuk menyimpan hasil
    
    Returns:
        dict: Dictionary berisi semua evaluation results
    
    TODO pada tahap evaluasi:
        - Add more visualization options
        - Save all results to JSON
        - Compare multiple models
    """
    print(f"\n{'='*70}")
    print(f"EVALUASI MODEL: {model_name}")
    print(f"{'='*70}")
    
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Predictions
    print(f"\nMelakukan prediksi pada {len(X_test)} test samples...")
    y_pred_proba, y_pred = predict(model, X_test)
    
    # Metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    print(f"\n{'='*70}")
    print("HASIL EVALUASI")
    print(f"{'='*70}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    if 'auc' in metrics:
        print(f"AUC:       {metrics['auc']:.4f}")
    
    # Confusion Matrix
    print(f"\n{'='*70}")
    print("CONFUSION MATRIX")
    print(f"{'='*70}")
    cm = get_confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Plots
    plot_confusion_matrix(y_test, y_pred, 
                         save_path=os.path.join(save_dir, f'{model_name}_confusion_matrix.png'))
    
    plot_roc_curve(y_test, y_pred_proba,
                  save_path=os.path.join(save_dir, f'{model_name}_roc_curve.png'))
    
    # Classification Report
    print_evaluation_report(y_test, y_pred)
    
    results = {
        'model_name': model_name,
        'metrics': metrics,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm
    }
    
    return results
