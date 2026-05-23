"""
Evaluation module untuk model SMS Spam Classification
Kelompok 7 - Tugas Akhir Kecerdasan Buatan

Modul ini menyediakan fungsi-fungsi untuk:
1. Melakukan prediksi pada test set
2. Menghitung evaluation metrics (accuracy, precision, recall, F1-score)
3. Generate confusion matrix dan visualization
4. Membuat classification report

Tahap saat ini: Persiapan placeholder untuk evaluation (akan diimplementasi setelah training)
"""

import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Mengevaluasi model menggunakan berbagai metrics.
    
    Args:
        y_true (np.array): True labels (0 atau 1)
        y_pred (np.array): Predicted labels (0 atau 1)
        y_pred_proba (np.array): Predicted probabilities (0-1), optional
        
    Returns:
        dict: Dictionary berisi metrics:
            - accuracy: Akurasi model
            - precision: Presisi (TP / (TP + FP))
            - recall: Recall/sensitivity (TP / (TP + FN))
            - f1_score: F1-score (harmonic mean precision & recall)
            - tn, fp, fn, tp: Confusion matrix components
            
    TODO:
        - Load model dari file
        - Perform batch prediction untuk dataset besar
        - Add ROC-AUC score
        - Add specificity dan sensitivity metrics
        - Validate input shapes dan values
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    # Confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['tn'] = tn
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['tp'] = tp
    metrics['confusion_matrix'] = cm
    
    return metrics


def plot_confusion_matrix(cm, output_path="results/confusion_matrix.png", 
                         class_names=['Ham', 'Spam']):
    """
    Plot confusion matrix sebagai heatmap dan simpan ke file.
    
    Args:
        cm (np.array): Confusion matrix (2x2)
        output_path (str): Path untuk menyimpan plot (default: results/confusion_matrix.png)
        class_names (list): Nama class untuk labels (default: ['Ham', 'Spam'])
        
    TODO:
        - Add normalized confusion matrix visualization
        - Customize color palette
        - Add percentage values di cell
        - Add detailed statistics
    """
    # Create output directory jika belum ada
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
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
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix plot disimpan: {output_path}")
    
    plt.close()


if __name__ == '__main__':
    """
    Main execution block - Placeholder untuk evaluation phase
    """
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 22 + "SMS SPAM CLASSIFICATION - EVALUATION" + " " * 20 + "║")
    print("║" + " " * 30 + "Kelompok 7 - Tugas Akhir AI" + " " * 21 + "║")
    print("╚" + "=" * 78 + "╝")
    
    print("\n" + "=" * 80)
    print("EVALUATION MODULE - PLACEHOLDER")
    print("=" * 80)
    
    print("\n[INFO] Modul evaluasi siap untuk digunakan setelah training selesai.")
    
    print("\n" + "=" * 80)
    print("TODO UNTUK TAHAP EVALUATION")
    print("=" * 80)
    
    todo_items = [
        "Load trained model dari results/models/",
        "Load preprocessed test data (X_test_pad, y_test)",
        "Perform prediction pada test set",
        "Calculate evaluation metrics (accuracy, precision, recall, F1)",
        "Generate confusion matrix",
        "Plot dan save confusion matrix visualization",
        "Generate classification report",
        "Compare Simple RNN vs LSTM performance",
        "Save evaluation results ke file",
        "Create comprehensive evaluation summary"
    ]
    
    for i, todo in enumerate(todo_items, 1):
        print(f"  [{i}] {todo}")
    
    print("\n" + "=" * 80)
    print("EVALUATION FUNCTIONS AVAILABLE")
    print("=" * 80)
    
    print("\n  Functions:")
    print("    1. evaluate_model(y_true, y_pred, y_pred_proba=None)")
    print("       → Calculate accuracy, precision, recall, F1-score")
    print("\n    2. plot_confusion_matrix(cm, output_path, class_names)")
    print("       → Plot dan save confusion matrix visualization")
    
    print("\n" + "=" * 80 + "\n")
