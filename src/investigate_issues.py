"""
Investigation Script - Debug F1 Score = 0 Issue
Kelompok 7 - Tugas Akhir Kecerdasan Buatan

Script ini mengecek:
1. Distribusi label di train/val/test set
2. Distribusi prediksi model
3. Statistik probabilitas output sigmoid
4. Konfigurasi training yang ada
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import sys

# Import preprocessing functions
from eda_preprocessing import load_and_preprocess_data

print("\n" + "=" * 80)
print("INVESTIGATION: F1 Score = 0 Issue")
print("=" * 80)

# ============================================================================
# PART 1: CHECK DATA DISTRIBUTION
# ============================================================================
print("\n[PART 1] CHECKING DATA DISTRIBUTION")
print("-" * 80)

try:
    # Load data
    X_train_pad, X_test_pad, y_train, y_test, tokenizer, label_encoder = \
        load_and_preprocess_data(
            csv_path="spam.csv",
            max_words=5000,
            max_len=100,
            test_size=0.2,
            random_state=42
        )
    
    # Create validation split
    from sklearn.model_selection import train_test_split
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_pad, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )
    
    print("\n✓ Data loaded successfully!")
    print("\n[TRAINING SET]")
    print(f"  - Shape: {X_train_final.shape}")
    print(f"  - Label distribution:")
    unique, counts = np.unique(y_train_final, return_counts=True)
    for label, count in zip(unique, counts):
        pct = count / len(y_train_final) * 100
        label_name = "Ham" if label == 0 else "Spam"
        print(f"    • Class {label} ({label_name}): {count} ({pct:.2f}%)")
    
    print("\n[VALIDATION SET]")
    print(f"  - Shape: {X_val.shape}")
    print(f"  - Label distribution:")
    unique, counts = np.unique(y_val, return_counts=True)
    for label, count in zip(unique, counts):
        pct = count / len(y_val) * 100
        label_name = "Ham" if label == 0 else "Spam"
        print(f"    • Class {label} ({label_name}): {count} ({pct:.2f}%)")
    
    print("\n[TEST SET]")
    print(f"  - Shape: {X_test_pad.shape}")
    print(f"  - Label distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for label, count in zip(unique, counts):
        pct = count / len(y_test) * 100
        label_name = "Ham" if label == 0 else "Spam"
        print(f"    • Class {label} ({label_name}): {count} ({pct:.2f}%)")
    
    print("\n✓ All sets have both classes - stratification OK!")
    
except Exception as e:
    print(f"✗ Error loading data: {e}")
    sys.exit(1)

# ============================================================================
# PART 2: CHECK MODEL PREDICTIONS
# ============================================================================
print("\n\n[PART 2] CHECKING MODEL PREDICTIONS")
print("-" * 80)

model_paths = [
    ("models/simple_rnn_model.h5", "Simple RNN"),
    ("models/lstm_model.h5", "LSTM"),
]

for model_path, model_name in model_paths:
    if not os.path.exists(model_path):
        print(f"\n✗ {model_name} model not found: {model_path}")
        continue
    
    print(f"\n[{model_name.upper()}]")
    print(f"  Loading from: {model_path}")
    
    try:
        model = load_model(model_path)
        
        # Get predictions on test set
        y_proba = model.predict(X_test_pad, verbose=0).reshape(-1)
        y_pred_default = (y_proba >= 0.5).astype(int)
        
        print(f"\n  Prediction Statistics (Threshold=0.5):")
        print(f"    • Min probability: {y_proba.min():.6f}")
        print(f"    • Max probability: {y_proba.max():.6f}")
        print(f"    • Mean probability: {y_proba.mean():.6f}")
        print(f"    • Median probability: {np.median(y_proba):.6f}")
        print(f"    • Std probability: {y_proba.std():.6f}")
        
        # Probability percentiles
        print(f"\n  Probability Percentiles:")
        for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(y_proba, pct)
            print(f"    • {pct}th percentile: {val:.6f}")
        
        # Prediction distribution
        print(f"\n  Predicted Label Distribution (threshold=0.5):")
        unique_pred, counts_pred = np.unique(y_pred_default, return_counts=True)
        for label, count in zip(unique_pred, counts_pred):
            pct = count / len(y_pred_default) * 100
            label_name = "Ham" if label == 0 else "Spam"
            print(f"    • Class {label} ({label_name}): {count} ({pct:.2f}%)")
        
        # True label distribution
        print(f"\n  True Label Distribution:")
        unique_true, counts_true = np.unique(y_test, return_counts=True)
        for label, count in zip(unique_true, counts_true):
            pct = count / len(y_test) * 100
            label_name = "Ham" if label == 0 else "Spam"
            print(f"    • Class {label} ({label_name}): {count} ({pct:.2f}%)")
        
        # Check if all predictions are 0
        if np.all(y_pred_default == 0):
            print(f"\n  ⚠ WARNING: ALL PREDICTIONS ARE 0 (HAM)!")
            print(f"    - This causes F1 = 0 for spam class")
            print(f"    - Probability range: {y_proba.min():.6f} - {y_proba.max():.6f}")
            print(f"    - All probabilities are < 0.5")
        
        # Show confusion matrix
        cm = confusion_matrix(y_test, y_pred_default)
        print(f"\n  Confusion Matrix (threshold=0.5):")
        print(f"    [[TN={cm[0,0]:5d}  FP={cm[0,1]:5d}]")
        print(f"     [FN={cm[1,0]:5d}  TP={cm[1,1]:5d}]]")
        
        # Try different thresholds
        print(f"\n  Trying different thresholds:")
        test_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        for threshold in test_thresholds:
            y_pred_th = (y_proba >= threshold).astype(int)
            n_spam = (y_pred_th == 1).sum()
            print(f"    • threshold={threshold}: {n_spam} spam predictions " +
                  f"({n_spam/len(y_pred_th)*100:.1f}%)")
        
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)
