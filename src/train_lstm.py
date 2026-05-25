"""
Training module untuk model SMS Spam Classification
Kelompok 7 - Tugas Akhir Kecerdasan Buatan

Modul ini menyediakan fungsi-fungsi untuk:
1. Membangun model Simple RNN sebagai baseline
2. Membangun model LSTM sebagai model utama
3. Training dan validation model
4. Menyimpan model dan training history

Data Requirements:
- X_train, X_test: shape (n_samples, max_len=100)
- y_train, y_test: shape (n_samples,) dengan values 0 (ham) atau 1 (spam)
- tokenizer: Keras Tokenizer object yang sudah fit
- label_encoder: sklearn LabelEncoder object yang sudah fit
"""

import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# Import preprocessing functions
from eda_preprocessing import load_and_preprocess_data


def build_simple_rnn_model(vocab_size, max_len=100, embedding_dim=128, 
                          rnn_units=64, dense_units=32, dropout_rate=0.5, 
                          learning_rate=0.001):
    """
    Membangun model Simple RNN sebagai baseline pembanding.
    
    Args:
        vocab_size (int): Ukuran vocabulary dari tokenizer
        max_len (int): Panjang maksimal sequence
        embedding_dim (int): Dimensi embedding layer (default: 128)
        rnn_units (int): Jumlah units di SimpleRNN layer (default: 64)
        dense_units (int): Jumlah units di Dense layer (default: 32)
        dropout_rate (float): Dropout rate (default: 0.5)
        learning_rate (float): Learning rate untuk optimizer (default: 0.001)
        
    Returns:
        model: Compiled Keras Sequential model
    """
    print(f"\n[INFO] Building Simple RNN Model...")
    print(f"  - vocab_size: {vocab_size}")
    print(f"  - max_len: {max_len}")
    print(f"  - embedding_dim: {embedding_dim}")
    print(f"  - rnn_units: {rnn_units}")
    
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embedding'),
        SimpleRNN(rnn_units, dropout=dropout_rate, name='simple_rnn'),
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


def build_lstm_model(vocab_size, max_len=100, embedding_dim=128,
                    lstm_units=64, dense_units=32, dropout_rate=0.5,
                    rec_dropout_rate=0.2, learning_rate=0.001):
    """
    Membangun model LSTM sebagai model utama.
    
    Args:
        vocab_size (int): Ukuran vocabulary dari tokenizer
        max_len (int): Panjang maksimal sequence
        embedding_dim (int): Dimensi embedding layer (default: 128)
        lstm_units (int): Jumlah units di LSTM layer (default: 64)
        dense_units (int): Jumlah units di Dense layer (default: 32)
        dropout_rate (float): Dropout rate (default: 0.5)
        rec_dropout_rate (float): Recurrent dropout rate (default: 0.2)
        learning_rate (float): Learning rate untuk optimizer (default: 0.001)
        
    Returns:
        model: Compiled Keras Sequential model
    """
    print(f"\n[INFO] Building LSTM Model...")
    print(f"  - vocab_size: {vocab_size}")
    print(f"  - max_len: {max_len}")
    print(f"  - embedding_dim: {embedding_dim}")
    print(f"  - lstm_units: {lstm_units}")
    
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


def train_model(model, X_train, y_train, X_val, y_val, epochs=20, 
               batch_size=32, patience=3, verbose=1, model_path=None):
    """
    Training model dengan validation set dan early stopping.
    
    Args:
        model: Keras Sequential model yang sudah dikompilasi
        X_train (np.array): Training data, shape (n_train, max_len)
        y_train (np.array): Training labels, shape (n_train,)
        X_val (np.array): Validation data, shape (n_val, max_len)
        y_val (np.array): Validation labels, shape (n_val,)
        epochs (int): Jumlah epoch training (default: 20)
        batch_size (int): Batch size (default: 32)
        patience (int): Patience untuk early stopping (default: 3)
        verbose (int): Verbosity level (default: 1)
        model_path (str): Path untuk menyimpan model terbaik (optional)
        
    Returns:
        history: Training history object dari Keras
    """
    print(f"\n[INFO] Memulai training model...")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_val shape: {X_val.shape}")
    print(f"  - epochs: {epochs}, batch_size: {batch_size}")
    
    # Print label distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print(f"  - y_train distribution: {dict(zip(unique_train, counts_train))}")
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    print(f"  - y_val distribution: {dict(zip(unique_val, counts_val))}")
    
    callbacks = []
    
    # 1. Early Stopping
    callbacks.append(EarlyStopping(
        monitor='val_loss', 
        patience=patience, 
        restore_best_weights=True,
        verbose=1
    ))
    
    # 2. Model Checkpoint (save to h5 format for better compatibility)
    if model_path:
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        h5_path = model_path.replace('.keras', '.h5')
        callbacks.append(ModelCheckpoint(
            h5_path, 
            monitor='val_loss', 
            save_best_only=True,
            verbose=0
        ))
        print(f"  - Model akan disimpan ke: {h5_path}")

    # 3. Class Weights untuk imbalanced dataset (Spam/Ham)
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(classes[i]): float(weights[i]) for i in range(len(classes))}
    print(f"  - Class weights: {class_weight_dict}")
    
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


def save_history(history, output_path):
    """
    Menyimpan training history ke file CSV.
    
    Args:
        history: Training history object dari Keras model.fit()
        output_path (str): Path untuk menyimpan history (format CSV)
        
    Returns:
        None
    """
    # Create output directory jika belum ada
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert history ke DataFrame
    history_dict = history.history
    df_history = pd.DataFrame(history_dict)
    
    # Simpan ke CSV
    df_history.to_csv(output_path, index=False)
    print(f"[OK] Training history tersimpan ke: {output_path}")
    
    return df_history


if __name__ == '__main__':
    """
    Main execution block - Load data, build models, dan train
    """
    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 18 + "SMS SPAM CLASSIFICATION - TRAINING SIMPLE RNN & LSTM" + " " * 7 + "|")
    print("|" + " " * 30 + "Kelompok 7 - Tugas Akhir AI" + " " * 21 + "|")
    print("+" + "=" * 78 + "+")
    
    # Create directories jika belum ada
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DAN PREPROCESSING DATA")
    print("=" * 80)
    
    # Load dan preprocess data
    try:
        X_train_pad, X_test_pad, y_train, y_test, tokenizer, label_encoder = \
            load_and_preprocess_data(
                csv_path="spam.csv",
                max_words=5000,
                max_len=100,
                test_size=0.2,
                random_state=42
            )
        
        print("\n" + "=" * 80)
        print("STEP 2: DATA PREPARATION SUMMARY")
        print("=" * 80)
        
        print(f"\n[OK] Data loaded successfully!")
        
        print(f"\n  Original Training Data:")
        print(f"    - X_train_pad shape: {X_train_pad.shape}")
        print(f"    - y_train shape: {y_train.shape}")
        print(f"    - Label distribution: {np.bincount(y_train)}")
        
        print(f"\n  Test Data:")
        print(f"    - X_test_pad shape: {X_test_pad.shape}")
        print(f"    - y_test shape: {y_test.shape}")
        print(f"    - Label distribution: {np.bincount(y_test)}")
        
        print(f"\n  Tokenizer & Encoder:")
        print(f"    - Vocabulary size: {len(tokenizer.word_index) + 1}")
        print(f"    - Max sequence length: 100")
        print(f"    - Label classes: {list(label_encoder.classes_)}")
        
        # Create validation split dari training data (80:20 dari training)
        print("\n" + "=" * 80)
        print("STEP 3: CREATING VALIDATION SPLIT")
        print("=" * 80)
        
        print("\n[INFO] Membagi data training untuk validation set (80:20)...")
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_pad, y_train, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_train
        )
        
        print(f"[OK] Validation split completed:")
        print(f"  - X_train_final shape: {X_train_final.shape}")
        print(f"  - X_val shape: {X_val.shape}")
        print(f"  - y_train_final distribution: {np.bincount(y_train_final)}")
        print(f"  - y_val distribution: {np.bincount(y_val)}")
        
        vocab_size = len(tokenizer.word_index) + 1
        max_len = 100
        
        # =====================================================================
        # 1. SIMPLE RNN (BASELINE)
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 4: BUILDING & TRAINING SIMPLE RNN (BASELINE)")
        print("=" * 80)
        
        print("\n[INFO] Building Simple RNN model...")
        rnn_model = build_simple_rnn_model(
            vocab_size=vocab_size, 
            max_len=max_len,
            embedding_dim=128,
            rnn_units=64,
            dense_units=32,
            dropout_rate=0.5,
            learning_rate=0.001
        )
        
        print("\n[INFO] Model Summary:")
        rnn_model.summary()
        
        print("\n[INFO] Training Simple RNN model...")
        rnn_history = train_model(
            model=rnn_model,
            X_train=X_train_final, 
            y_train=y_train_final,
            X_val=X_val, 
            y_val=y_val,
            epochs=20,
            batch_size=32,
            patience=3,
            verbose=1,
            model_path='models/simple_rnn_model.h5'
        )
        
        print("\n[OK] Simple RNN training completed!")
        
        # Save history
        print("\n[INFO] Saving Simple RNN training history...")
        save_history(rnn_history, 'results/simple_rnn_history.csv')
        
        # =====================================================================
        # 2. LSTM (MAIN MODEL)
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 5: BUILDING & TRAINING LSTM (MAIN MODEL)")
        print("=" * 80)
        
        print("\n[INFO] Building LSTM model...")
        lstm_model = build_lstm_model(
            vocab_size=vocab_size,
            max_len=max_len,
            embedding_dim=128,
            lstm_units=64,
            dense_units=32,
            dropout_rate=0.5,
            rec_dropout_rate=0.2,
            learning_rate=0.001
        )
        
        print("\n[INFO] Model Summary:")
        lstm_model.summary()
        
        print("\n[INFO] Training LSTM model...")
        lstm_history = train_model(
            model=lstm_model,
            X_train=X_train_final,
            y_train=y_train_final,
            X_val=X_val,
            y_val=y_val,
            epochs=20,
            batch_size=32,
            patience=3,
            verbose=1,
            model_path='models/lstm_model.h5'
        )
        
        print("\n[OK] LSTM training completed!")
        
        # Save history
        print("\n[INFO] Saving LSTM training history...")
        save_history(lstm_history, 'results/lstm_history.csv')
        
        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 6: TRAINING SUMMARY")
        print("=" * 80)
        
        print("\n[OK] TRAINING SELESAI SUCCESSFULLY!")
        
        print("\n📊 Data Summary:")
        print(f"  ✓ Total samples: {len(X_train_pad) + len(X_test_pad)}")
        print(f"  ✓ Training: {X_train_final.shape[0]} samples")
        print(f"  ✓ Validation: {X_val.shape[0]} samples")
        print(f"  ✓ Test: {X_test_pad.shape[0]} samples")
        
        print("\n🏗️  Model Architecture:")
        print(f"  ✓ Vocabulary size: {vocab_size}")
        print(f"  ✓ Max sequence length: {max_len}")
        print(f"  ✓ Embedding dimension: 128")
        print(f"  ✓ Simple RNN units: 64")
        print(f"  ✓ LSTM units: 64")
        print(f"  ✓ Dense units: 32")
        
        print("\n💾 Model Files Saved:")
        print(f"  ✓ models/simple_rnn_model.h5")
        print(f"  ✓ models/lstm_model.h5")
        
        print("\n📈 Training History Saved:")
        print(f"  ✓ results/simple_rnn_history.csv")
        print(f"  ✓ results/lstm_history.csv")
        
        print("\n⚙️  Training Configuration:")
        print(f"  ✓ Epochs: 20 (dengan early stopping)")
        print(f"  ✓ Batch size: 32")
        print(f"  ✓ Optimizer: Adam (lr=0.001)")
        print(f"  ✓ Loss: binary_crossentropy")
        print(f"  ✓ Metrics: accuracy")
        print(f"  ✓ Class weights: balanced")
        
        print("\n" + "=" * 80)
        print("🎯 Next Steps:")
        print("  1. Run: python src/evaluate.py")
        print("     (untuk evaluasi model di test set)")
        print("  2. Run: python src/hyperparameter_tuning.py")
        print("     (untuk fine-tuning hyperparameter)")
        print("  3. Run: python src/error_analysis.py")
        print("     (untuk analisis error model)")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Terjadi error saat training: {str(e)}")
        print("[ERROR] Pastikan spam.csv ada di project root")
        import traceback
        traceback.print_exc()
        raise
