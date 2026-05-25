"""
Hyperparameter Tuning untuk SMS Spam Classification
Kelompok 7 - Tugas Akhir Kecerdasan Buatan

File ini melakukan:
1. Load dan preprocessing data
2. Jalankan 3 eksperimen LSTM dengan hyperparameter sederhana
3. Evaluasi setiap model pada test set
4. Simpan hasil ke CSV dan visualisasi

Dapat dijalankan dengan: python src/hyperparameter_tuning.py
"""

import os
import json
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

from eda_preprocessing import load_and_preprocess_data

# Configuration
DATASET_PATH = 'spam.csv'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'
TUNING_RESULTS_PATH = os.path.join(RESULTS_DIR, 'tuning_results.csv')
TUNING_CHART_PATH = os.path.join(RESULTS_DIR, 'tuning_f1_score.png')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_lstm_model.keras')

MAX_WORDS = 5000
MAX_LEN = 100
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def build_lstm_model(vocab_size, max_len=100, embedding_dim=128, lstm_units=64,
                     dense_units=32, dropout_rate=0.5, rec_dropout_rate=0.2,
                     learning_rate=0.001):
    """
    Membangun model LSTM.
    
    Args:
        vocab_size (int): Ukuran vocabulary
        max_len (int): Panjang maksimal sequence
        embedding_dim (int): Dimensi embedding layer
        lstm_units (int): Jumlah units di LSTM layer
        dense_units (int): Jumlah units di Dense layer
        dropout_rate (float): Dropout rate
        rec_dropout_rate (float): Recurrent dropout rate
        learning_rate (float): Learning rate optimizer
        
    Returns:
        model: Compiled Keras Sequential model
    """
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embedding'),
        LSTM(lstm_units, dropout=dropout_rate,
             recurrent_dropout=rec_dropout_rate, name='lstm'),
        Dense(dense_units, activation='relu', name='dense_1'),
        Dropout(dropout_rate, name='dropout'),
        Dense(1, activation='sigmoid', name='output')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )

    return model

def train_lstm_model(model, X_train, y_train, X_val, y_val,
                     epochs=5, batch_size=32, patience=2, verbose=1):
    """
    Training LSTM model dengan validation set dan early stopping.
    
    Args:
        model: Keras Sequential model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs (int): Jumlah epoch
        batch_size (int): Batch size
        patience (int): Patience untuk early stopping
        verbose (int): Verbosity level
        
    Returns:
        history: Training history object
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=0
        )
    ]

    # Class weights untuk imbalanced dataset
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(classes[i]): float(weights[i]) for i in range(len(classes))}

    print(f"    • Class weights: {class_weight_dict}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=verbose
    )

    return history


def find_best_threshold(model, X_val, y_val, thresholds=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]):
    """Find best threshold on validation set based on F1-score."""
    y_pred_proba = model.predict(X_val, verbose=0).reshape(-1)
    
    best_threshold = 0.5
    best_f1 = 0
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if np.sum(y_val) > 0 and np.sum(y_pred) == 0:
            continue
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    
    return best_threshold, best_metrics


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluasi model pada test set dengan threshold tertentu.
    """
    y_pred_proba = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_pred, zero_division=0),
        'test_f1': f1_score(y_test, y_pred, zero_division=0)
    }

    return metrics


def run_hyperparameter_tuning():
    """
    Main function untuk menjalankan hyperparameter tuning experiments.
    """
    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 15 + "SMS SPAM CLASSIFICATION - HYPERPARAMETER TUNING" + " " * 16 + "|")
    print("|" + " " * 30 + "Kelompok 7 - Tugas Akhir AI" + " " * 21 + "|")
    print("+" + "=" * 78 + "+")

    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # =========================================================================
    # STEP 1: LOAD DAN PREPROCESS DATA
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 80)

    print(f'\nLoading dataset and preprocessing...')

    try:
        X_train_pad, X_test_pad, y_train, y_test, tokenizer, label_encoder = \
            load_and_preprocess_data(
                csv_path=DATASET_PATH,
                max_words=MAX_WORDS,
                max_len=MAX_LEN,
                test_size=0.2,
                random_state=RANDOM_STATE
            )
    except Exception as e:
        print(f"[ERROR] Gagal load data: {e}")
        return

    vocab_size = len(tokenizer.word_index) + 1
    max_len = MAX_LEN

    print(f"\n[OK] Data loaded successfully!")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - X_train shape: {X_train_pad.shape}")
    print(f"  - X_test shape: {X_test_pad.shape}")

    # =========================================================================
    # STEP 2: DEFINE HYPERPARAMETER EXPERIMENTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: DEFINING HYPERPARAMETER EXPERIMENTS")
    print("=" * 80)

    experiments = [
        {
            'run_name': 'Run_1_baseline',
            'lstm_units': 64,
            'dropout': 0.5,
            'batch_size': 32,
            'epochs': 10
        },
        {
            'run_name': 'Run_2_more_units',
            'lstm_units': 128,
            'dropout': 0.5,
            'batch_size': 32,
            'epochs': 10
        },
        {
            'run_name': 'Run_3_less_dropout',
            'lstm_units': 64,
            'dropout': 0.3,
            'batch_size': 32,
            'epochs': 10
        }
    ]

    print("\nHyperparameter Experiments:")
    for exp in experiments:
        print(f"\n  {exp['run_name']}:")
        print(f"    - LSTM units: {exp['lstm_units']}")
        print(f"    - Dropout: {exp['dropout']}")
        print(f"    - Batch size: {exp['batch_size']}")
        print(f"    - Epochs: {exp['epochs']}")

    # =========================================================================
    # STEP 3: RUN EXPERIMENTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: RUNNING HYPERPARAMETER TUNING EXPERIMENTS")
    print("=" * 80)

    results = []
    best_f1 = 0
    best_model = None
    best_run_name = None

    for i, exp in enumerate(experiments):
        print(f"\n{'-' * 80}")
        print(f"EXPERIMENT {i+1}/{len(experiments)}: {exp['run_name']}")
        print(f"{'-' * 80}")

        run_name = exp['run_name']
        lstm_units = exp['lstm_units']
        dropout = exp['dropout']
        batch_size = exp['batch_size']
        epochs = exp['epochs']

        # Create validation split dari training data
        print(f"\n[INFO] Creating validation split (80:20)...")
        X_train_train, X_val, y_train_train, y_val = train_test_split(
            X_train_pad, y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_train
        )

        print(f"  - X_train shape: {X_train_train.shape}")
        print(f"  - X_val shape: {X_val.shape}")
        print(f"  - X_test shape: {X_test_pad.shape}")

        # Build model
        print(f"\n[INFO] Building LSTM model...")
        model = build_lstm_model(
            vocab_size=vocab_size,
            max_len=max_len,
            embedding_dim=128,
            lstm_units=lstm_units,
            dense_units=32,
            dropout_rate=dropout,
            rec_dropout_rate=0.2,
            learning_rate=0.001
        )

        print(f"  - Model architecture prepared")

        # Train model
        print(f"\n[INFO] Training model...")
        history = train_lstm_model(
            model=model,
            X_train=X_train_train,
            y_train=y_train_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            patience=3,
            verbose=1
        )

        # Get final training metrics
        final_epoch_idx = len(history.history['accuracy']) - 1
        train_accuracy_final = history.history['accuracy'][final_epoch_idx]
        val_accuracy_final = history.history['val_accuracy'][final_epoch_idx]
        val_loss_final = history.history['val_loss'][final_epoch_idx]

        print(f"\n[INFO] Training completed!")
        print(f"  - Final train accuracy: {train_accuracy_final:.4f}")
        print(f"  - Final val accuracy: {val_accuracy_final:.4f}")
        print(f"  - Final val loss: {val_loss_final:.4f}")

        # Evaluate on test set
        print(f"\n[INFO] Evaluating on test set...")
        eval_metrics = evaluate_model(model, X_test_pad, y_test)

        print(f"  - Test accuracy: {eval_metrics['test_accuracy']:.4f}")
        print(f"  - Test precision: {eval_metrics['test_precision']:.4f}")
        print(f"  - Test recall: {eval_metrics['test_recall']:.4f}")
        print(f"  - Test F1-score: {eval_metrics['test_f1']:.4f}")

        # Store results
        result_row = {
            'run_name': run_name,
            'lstm_units': lstm_units,
            'dropout': dropout,
            'batch_size': batch_size,
            'epochs': epochs,
            'train_accuracy': round(train_accuracy_final, 4),
            'val_accuracy': round(val_accuracy_final, 4),
            'val_loss': round(val_loss_final, 4),
            'test_accuracy': round(eval_metrics['test_accuracy'], 4),
            'test_precision': round(eval_metrics['test_precision'], 4),
            'test_recall': round(eval_metrics['test_recall'], 4),
            'test_f1': round(eval_metrics['test_f1'], 4)
        }

        results.append(result_row)

        # Track best model
        if eval_metrics['test_f1'] > best_f1:
            best_f1 = eval_metrics['test_f1']
            best_model = model
            best_run_name = run_name
            print(f"\n[INFO] New best model found! F1-score: {best_f1:.4f}")

    # =========================================================================
    # STEP 4: SAVE RESULTS TO CSV
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: SAVING RESULTS")
    print("=" * 80)

    df_results = pd.DataFrame(results)

    df_results.to_csv(TUNING_RESULTS_PATH, index=False)
    print(f"\n[OK] Tuning results saved to: {TUNING_RESULTS_PATH}")

    print("\nResults Summary:")
    print(df_results.to_string(index=False))

    # =========================================================================
    # STEP 5: SAVE BEST MODEL
    # =========================================================================
    if best_model is not None:
        print("\n" + "=" * 80)
        print("STEP 5: SAVING BEST MODEL")
        print("=" * 80)

        try:
            best_model.save(BEST_MODEL_PATH)
            print(f"\n[OK] Best model saved to: {BEST_MODEL_PATH}")
            print(f"  - Best run: {best_run_name}")
            print(f"  - Best F1-score: {best_f1:.4f}")
        except Exception as e:
            print(f"[WARNING] Could not save to .keras format: {e}")
            print("[INFO] Trying to save to .h5 format...")
            h5_path = BEST_MODEL_PATH.replace('.keras', '.h5')
            best_model.save(h5_path)
            print(f"[OK] Best model saved to: {h5_path}")

    # =========================================================================
    # STEP 6: CREATE VISUALIZATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: CREATING VISUALIZATION")
    print("=" * 80)

    plt.figure(figsize=(10, 6))
    plt.bar(df_results['run_name'], df_results['test_f1'], color='steelblue', edgecolor='black')
    plt.xlabel('Experiment Run', fontsize=12, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
    plt.title('Hyperparameter Tuning Results - Test F1-Score', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)

    # Add value labels on bars
    for i, v in enumerate(df_results['test_f1']):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(TUNING_CHART_PATH, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Visualization saved to: {TUNING_CHART_PATH}")

    plt.close()

    # =========================================================================
    # STEP 7: SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("=" * 80)

    print("\nSummary:")
    print(f"  [OK] Experiments run: {len(results)}")
    print(f"  [OK] Results saved to: {TUNING_RESULTS_PATH}")
    print(f"  [OK] Visualization saved to: {TUNING_CHART_PATH}")
    print(f"  [OK] Best model saved to: {BEST_MODEL_PATH}")
    print(f"\nBest Performance:")
    print(f"  - Run: {best_run_name}")
    print(f"  - F1-Score: {best_f1:.4f}")

    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    run_hyperparameter_tuning()