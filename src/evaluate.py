"""
Evaluation module untuk model SMS Spam Classification

Modul ini berisi fungsi-fungsi untuk:
- Melakukan prediksi pada test set
- Menghitung metrics: Accuracy, Precision, Recall, F1-Score
- Generate confusion matrix
- Membuat visualization hasil evaluasi
"""

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
        X_test (np.array): Test data
    
    Returns:
        np.array: Predicted probabilities
        np.array: Predicted classes (0 atau 1)
    
    TODO:
        - Get predictions from model
        - Convert probabilities to class labels
    """
    # TODO: Implement prediction
    pass


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Menghitung evaluation metrics untuk model.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels (0 atau 1)
        y_pred_proba (np.array): Predicted probabilities (optional)
    
    Returns:
        dict: Dictionary berisi metrics
            - accuracy
            - precision
            - recall
            - f1_score
            - auc (jika y_pred_proba diberikan)
    
    TODO:
        - Calculate accuracy, precision, recall, F1
        - Calculate AUC if probabilities provided
    """
    # TODO: Implement metrics calculation
    pass


def get_confusion_matrix(y_true, y_pred):
    """
    Generate confusion matrix.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
    
    Returns:
        np.array: Confusion matrix (2x2)
    
    TODO:
        - Generate confusion matrix
    """
    # TODO: Implement confusion matrix
    pass


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
    
    TODO:
        - Generate confusion matrix
        - Create heatmap visualization
        - Save figure if save_path provided
    """
    # TODO: Implement confusion matrix plot
    pass


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curve dan hitung AUC.
    
    Args:
        y_true (np.array): True labels
        y_pred_proba (np.array): Predicted probabilities
        save_path (str): Path untuk menyimpan plot (optional)
    
    Returns:
        float: AUC score
    
    TODO:
        - Calculate ROC curve
        - Calculate AUC
        - Plot curve
        - Save figure if save_path provided
    """
    # TODO: Implement ROC curve plot
    pass


def print_evaluation_report(y_true, y_pred, class_names=['Ham', 'Spam']):
    """
    Print comprehensive evaluation report.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        class_names (list): Nama class
    
    Returns:
        str: Classification report
    
    TODO:
        - Print metrics summary
        - Print classification report
    """
    # TODO: Implement evaluation report
    pass


def evaluate_full(model, X_test, y_test, save_dir=None):
    """
    Full evaluation pipeline: predictions, metrics, visualizations.
    
    Args:
        model: Trained Keras model
        X_test (np.array): Test data
        y_test (np.array): Test labels
        save_dir (str): Directory untuk menyimpan hasil (optional)
    
    Returns:
        dict: Dictionary berisi semua evaluation results
    
    TODO:
        - Run predictions
        - Calculate all metrics
        - Generate all plots
        - Save results
    """
    # TODO: Implement full evaluation pipeline
    pass
