"""
Evaluation module untuk model SMS Spam Classification
Kelompok 7 - Tugas Akhir Kecerdasan Buatan

Modul ini menyediakan fungsi untuk:
1. Load trained models (Simple RNN & LSTM)
2. Predict labels pada test set
3. Evaluasi performa model
4. Simpan hasil evaluasi ke file CSV dan TXT

Jalankan dengan: python src/evaluate.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tensorflow.keras.models import load_model

from eda_preprocessing import load_and_preprocess_data


def load_trained_model(model_path):
    """
    Load trained model dari file .h5
    
    Args:
        model_path (str): Path ke file model .h5
        
    Returns:
        model: Loaded Keras model
        
    Raises:
        FileNotFoundError: Jika file model tidak ditemukan
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[ERROR] Model tidak ditemukan: {model_path}\n"
            f"[ERROR] Jalankan training terlebih dahulu: python src/train_lstm.py"
        )
    
    print(f"[OK] Loading model dari: {model_path}")
    model = load_model(model_path)
    return model


def predict_labels(model, X_test, threshold=0.5):
    """
    Predict labels dari model menggunakan threshold
    
    Args:
        model: Trained Keras model
        X_test (np.ndarray): Test features, shape (n_samples, max_len)
        threshold (float): Threshold untuk mengubah probabilitas menjadi label (default: 0.5)
        
    Returns:
        y_pred (np.ndarray): Predicted labels (0 atau 1)
        y_proba (np.ndarray): Predicted probabilities
    """
    print(f"[INFO] Predicting labels dengan threshold={threshold}...")
    y_proba = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (y_proba >= threshold).astype(int)
    return y_pred, y_proba


def evaluate_predictions(y_true, y_pred, model_name):
    """
    Evaluasi predictions dan hitung metrik
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        model_name (str): Nama model untuk logging
        
    Returns:
        dict: Dictionary dengan semua metrics
    """
    print(f"[INFO] Evaluating {model_name}...")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    
    return metrics, cm


def save_metrics_to_csv(metrics_list, output_path):
    """
    Simpan metrics ke file CSV
    
    Args:
        metrics_list (list): List of dictionary dengan metrics dari setiap model
        output_path (str): Path untuk menyimpan CSV file
        
    Returns:
        None
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(metrics_list)
    df.to_csv(output_path, index=False)
    print(f"[OK] Metrics CSV disimpan ke: {output_path}")


def save_classification_report(y_true, y_pred, model_name, output_path):
    """
    Simpan classification report ke file TXT
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        model_name (str): Nama model
        output_path (str): Path untuk menyimpan TXT file
        
    Returns:
        None
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    report = classification_report(
        y_true, y_pred,
        target_names=['Ham', 'Spam'],
        digits=4,
        zero_division=0
    )
    
    with open(output_path, 'w') as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
    
    print(f"[OK] Classification report disimpan ke: {output_path}")


def save_confusion_matrix(cm, model_name, output_path):
    """
    Simpan confusion matrix ke file CSV
    
    Args:
        cm (np.ndarray): Confusion matrix dari sklearn
        model_name (str): Nama model
        output_path (str): Path untuk menyimpan CSV file
        
    Returns:
        None
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    tn, fp, fn, tp = cm.ravel()
    
    # Create CSV format
    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=['Predicted Ham', 'Predicted Spam'],
        columns=['Actual Ham', 'Actual Spam']
    )
    
    cm_df.to_csv(output_path)
    print(f"[OK] Confusion matrix disimpan ke: {output_path}")
    
    return cm_df


if __name__ == '__main__':
    """
    Main execution block - Load models, evaluate pada test set, save results
    """
    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 18 + "SMS SPAM CLASSIFICATION - MODEL EVALUATION" + " " * 18 + "|")
    print("|" + " " * 30 + "Kelompok 7 - Tugas Akhir AI" + " " * 21 + "|")
    print("+" + "=" * 78 + "+")
    
    # Configuration
    MAX_WORDS = 5000
    MAX_LEN = 100
    RANDOM_STATE = 42
    THRESHOLD = 0.5
    TEST_SIZE = 0.2
    
    # Model paths
    SIMPLE_RNN_PATH = 'models/simple_rnn_model.h5'
    LSTM_PATH = 'models/lstm_model.h5'
    
    # Output paths
    RESULTS_DIR = 'results'
    METRICS_CSV = os.path.join(RESULTS_DIR, 'evaluation_metrics.csv')
    
    # Create results directory if not exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    try:
        # =====================================================================
        # STEP 1: Load and preprocess data
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 1: LOADING AND PREPROCESSING DATA")
        print("=" * 80)
        
        X_train_pad, X_test_pad, y_train, y_test, tokenizer, label_encoder = \
            load_and_preprocess_data(
                csv_path="spam.csv",
                max_words=MAX_WORDS,
                max_len=MAX_LEN,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE
            )
        
        print(f"\n[OK] Data loaded successfully!")
        print(f"  - Test set shape: {X_test_pad.shape}")
        print(f"  - Test labels shape: {y_test.shape}")
        
        # =====================================================================
        # STEP 2: Evaluate Simple RNN
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 2: EVALUATING SIMPLE RNN MODEL")
        print("=" * 80)
        
        try:
            simple_rnn_model = load_trained_model(SIMPLE_RNN_PATH)
            simple_rnn_pred, simple_rnn_proba = predict_labels(
                simple_rnn_model, X_test_pad, threshold=THRESHOLD
            )
            simple_rnn_metrics, simple_rnn_cm = evaluate_predictions(
                y_test, simple_rnn_pred, 'Simple RNN'
            )
            
            # Save results
            save_classification_report(
                y_test, simple_rnn_pred, 'Simple RNN',
                os.path.join(RESULTS_DIR, 'simple_rnn_classification_report.txt')
            )
            save_confusion_matrix(
                simple_rnn_cm, 'Simple RNN',
                os.path.join(RESULTS_DIR, 'simple_rnn_confusion_matrix.csv')
            )
            
            simple_rnn_available = True
        except FileNotFoundError as e:
            print(f"\n[WARNING] {str(e)}")
            simple_rnn_available = False
            simple_rnn_metrics = None
        
        # =====================================================================
        # STEP 3: Evaluate LSTM
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 3: EVALUATING LSTM MODEL")
        print("=" * 80)
        
        try:
            lstm_model = load_trained_model(LSTM_PATH)
            lstm_pred, lstm_proba = predict_labels(
                lstm_model, X_test_pad, threshold=THRESHOLD
            )
            lstm_metrics, lstm_cm = evaluate_predictions(
                y_test, lstm_pred, 'LSTM'
            )
            
            # Save results
            save_classification_report(
                y_test, lstm_pred, 'LSTM',
                os.path.join(RESULTS_DIR, 'lstm_classification_report.txt')
            )
            save_confusion_matrix(
                lstm_cm, 'LSTM',
                os.path.join(RESULTS_DIR, 'lstm_confusion_matrix.csv')
            )
            
            lstm_available = True
        except FileNotFoundError as e:
            print(f"\n[WARNING] {str(e)}")
            lstm_available = False
            lstm_metrics = None
        
        # =====================================================================
        # STEP 4: Comparison Summary
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 4: EVALUATION SUMMARY")
        print("=" * 80)
        
        # Prepare comparison table
        metrics_list = []
        
        if simple_rnn_available and simple_rnn_metrics:
            metrics_list.append(simple_rnn_metrics)
        
        if lstm_available and lstm_metrics:
            metrics_list.append(lstm_metrics)
        
        if metrics_list:
            # Save metrics CSV
            save_metrics_to_csv(metrics_list, METRICS_CSV)
            
            # Display comparison table
            print("\n" + "=" * 80)
            print("MODEL COMPARISON")
            print("=" * 80)
            
            df_comparison = pd.DataFrame(metrics_list)
            print("\n" + df_comparison.to_string(index=False))
            
            # Best model
            best_idx = df_comparison['accuracy'].idxmax()
            best_model = df_comparison.loc[best_idx, 'model']
            best_accuracy = df_comparison.loc[best_idx, 'accuracy']
            
            print(f"\nBest Model: {best_model} (Accuracy: {best_accuracy:.4f})")
        
        # =====================================================================
        # STEP 5: Summary
        # =====================================================================
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETED")
        print("=" * 80)
        
        print("\nOutput Files:")
        print(f"  {METRICS_CSV}")
        
        if simple_rnn_available:
            print(f"  {os.path.join(RESULTS_DIR, 'simple_rnn_classification_report.txt')}")
            print(f"  {os.path.join(RESULTS_DIR, 'simple_rnn_confusion_matrix.csv')}")
        
        if lstm_available:
            print(f"  {os.path.join(RESULTS_DIR, 'lstm_classification_report.txt')}")
            print(f"  {os.path.join(RESULTS_DIR, 'lstm_confusion_matrix.csv')}")
        
        print("\n" + "=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        raise

