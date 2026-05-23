"""
Training module untuk model SMS Spam Classification
Kelompok 7 - Tugas Akhir Kecerdasan Buatan

Modul ini menyediakan fungsi-fungsi untuk:
1. Membangun model Simple RNN sebagai baseline
2. Membangun model LSTM sebagai model utama
3. Training dan validation model
4. Menyimpan model yang telah dilatih

Tahap saat ini: Persiapan placeholder untuk training (akan diimplementasi di tahap selanjutnya)

Data Requirements:
- X_train, X_test: shape (n_samples, max_len=100)
- y_train, y_test: shape (n_samples,) dengan values 0 (ham) atau 1 (spam)
- tokenizer: Keras Tokenizer object yang sudah fit
- label_encoder: sklearn LabelEncoder object yang sudah fit
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

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
        
    TODO:
        - Implement dan test model architecture
        - Experiment dengan embedding_dim, rnn_units
        - Add regularization jika diperlukan
    """
    print(f"\n[TODO] Building Simple RNN Model...")
    print(f"  - vocab_size: {vocab_size}")
    print(f"  - max_len: {max_len}")
    print(f"  - embedding_dim: {embedding_dim}")
    print(f"  - rnn_units: {rnn_units}")
    
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
                  input_length=max_len, name='embedding'),
        SimpleRNN(rnn_units, dropout=dropout_rate, name='simple_rnn'),
        Dense(dense_units, activation='relu', name='dense_1'),
        Dropout(dropout_rate, name='dropout'),
        Dense(1, activation='sigmoid', name='output')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
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
        
    TODO:
        - Implement dan test LSTM model
        - Experiment dengan lstm_units, embedding_dim
        - Add bidirectional LSTM untuk peningkatan akurasi
        - Add attention mechanism jika perlu
    """
    print(f"\n[TODO] Building LSTM Model...")
    print(f"  - vocab_size: {vocab_size}")
    print(f"  - max_len: {max_len}")
    print(f"  - embedding_dim: {embedding_dim}")
    print(f"  - lstm_units: {lstm_units}")
    
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                  input_length=max_len, name='embedding'),
        LSTM(lstm_units, dropout=dropout_rate, 
             recurrent_dropout=rec_dropout_rate, name='lstm'),
        Dense(dense_units, activation='relu', name='dense_1'),
        Dropout(dropout_rate, name='dropout'),
        Dense(1, activation='sigmoid', name='output')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
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
    
    callbacks = []
    
    # 1. Early Stopping
    callbacks.append(EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True))
    
    # 2. Model Checkpoint
    if model_path:
        from tensorflow.keras.callbacks import ModelCheckpoint
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        callbacks.append(ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True))

    # 3. Class Weights untuk imbalanced dataset (Spam/Ham)
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {classes[i]: weights[i] for i in range(len(classes))}
    print(f"  - Menggunakan class weights: {class_weight_dict}")
    
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


if __name__ == '__main__':
    """
    Main execution block - Load data dan prepare untuk training
    """
    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 20 + "SMS SPAM CLASSIFICATION - TRAINING PREPARATION" + " " * 12 + "|")
    print("|" + " " * 30 + "Kelompok 7 - Tugas Akhir AI" + " " * 21 + "|")
    print("+" + "=" * 78 + "+")
    
    print("\n" + "=" * 80)
    print("LOADING DAN PREPROCESSING DATA")
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
        print("DATA PREPARATION SUMMARY")
        print("=" * 80)
        
        # Display data shapes
        print(f"\n[OK] Data loaded successfully!")
        print(f"\n  Training Data:")
        print(f"    - X_train_pad shape: {X_train_pad.shape}")
        print(f"    - y_train shape: {y_train.shape}")
        print(f"    - Unique labels: {np.unique(y_train)}")
        
        print(f"\n  Test Data:")
        print(f"    - X_test_pad shape: {X_test_pad.shape}")
        print(f"    - y_test shape: {y_test.shape}")
        print(f"    - Unique labels: {np.unique(y_test)}")
        
        print(f"\n  Tokenizer & Encoder:")
        print(f"    - Vocabulary size: {len(tokenizer.word_index) + 1}")
        print(f"    - Max sequence length: 100")
        print(f"    - Label classes: {list(label_encoder.classes_)}")
        
        print("\n" + "=" * 80)
        print("[OK] DATA READY FOR TRAINING")
        print("=" * 80)
        
        from sklearn.model_selection import train_test_split
        
        print("\n[INFO] Membagi sebagian data training untuk validation set (10%)...")
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_pad, y_train, test_size=0.1, random_state=42, stratify=y_train
        )
        
        vocab_size = len(tokenizer.word_index) + 1
        
        # 1. Simple RNN (Baseline)
        print("\n" + "=" * 80)
        print("TRAINING SIMPLE RNN (BASELINE)")
        print("=" * 80)
        rnn_model = build_simple_rnn_model(vocab_size=vocab_size, max_len=100)
        rnn_history = train_model(
            rnn_model, 
            X_train_final, y_train_final, 
            X_val, y_val, 
            epochs=10, 
            patience=3,
            model_path='results/models/simple_rnn.h5'
        )
        
        # 2. LSTM (Main Model)
        print("\n" + "=" * 80)
        print("TRAINING LSTM (MAIN MODEL)")
        print("=" * 80)
        lstm_model = build_lstm_model(vocab_size=vocab_size, max_len=100)
        lstm_history = train_model(
            lstm_model, 
            X_train_final, y_train_final, 
            X_val, y_val, 
            epochs=10, 
            patience=3,
            model_path='results/models/lstm.h5'
        )
        
        print("\n" + "=" * 80)
        print("[OK] TRAINING SELESAI")
        print("Model terbaik telah disimpan ke folder 'results/models/'")
        print("Langkah selanjutnya: Jalankan evaluate.py untuk melakukan evaluasi model")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Terjadi error saat loading data: {str(e)}")
        print("[ERROR] Pastikan spam.csv ada di project root dan preprocessing sudah dijalankan")
        raise
