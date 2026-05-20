"""
Training module untuk model SMS Spam Classification

Modul ini berisi fungsi-fungsi untuk:
- Membangun model Simple RNN sebagai baseline
- Membangun model LSTM sebagai model utama
- Melakukan training dan validation
- Menyimpan model yang sudah dilatih

Data contract dan input requirements:
- X_train, y_train, X_val, y_val harus sudah diproses dan disimpan dalam format .npy oleh modul preprocessing.py
- X_train & X_val harus berupa 2D array dengan shape (batch_size, sequence_length)
- y_train & y_val harus berupa 1D array dengan shape (batch_size,) dan berisi integer 0 (ham) dan 1 (spam)

Menunggu output dari eksekusi modul preprocessing.py:
1. X_train & X_val Shape : (batch_size, sequence_length) -> max_len wajib sinkron.
2. y_train & y_val Shape : (batch_size,) -> Wajib 1D array.
3. Label Values          : Integer 0 (Ham) dan 1 (Spam).

Nilai parameter default pada fungsi arsitektur di bawah (seperti lstm_units=64, 
dense_units=32, dropout_rate=0.5, vocab_size=5000) bertindak sebagai baseline
awal, jadi masih perlu diubah sesuai kebutuhan. 

Angka-angka tersebut dinamis, seghingga perlu dimodifikasi 
pada fase Hyperparameter Tuning, dengan menyesuaikan pada:
- Exploratory Data Analysis (EDA): max_len & vocab_size mengikuti distribusi data asli.
- Model Capacity: menaikkan/menurunkan units untuk mencegah Underfitting/Overfitting.
- Hardware Optimization: menggunakan angka kelipatan basis biner (32, 64, 128) untuk efisiensi komputasi GPU.
=============================================================================
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# DYNAMIC ARCHITECTURE FUNCTIONS
def build_rnn_baseline(vocab_size, embedding_dim=128, rnn_units=64, 
                       dense_units=32, dropout_rate=0.5, learning_rate=0.001):
    """Membangun model Simple RNN sebagai Baseline pembanding."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        SimpleRNN(rnn_units, dropout=dropout_rate),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(learning_rate=learning_rate), 
                  metrics=['accuracy'])
    return model


def build_lstm_model(vocab_size, embedding_dim=128, lstm_units=64, 
                     dense_units=32, dropout_rate=0.5, rec_dropout_rate=0.2, 
                     learning_rate=0.001):
    """Membangun model utama LSTM sesuai rancangan arsitektur sistem."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=rec_dropout_rate),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(learning_rate=learning_rate), 
                  metrics=['accuracy'])
    return model


def build_gru_model(vocab_size, embedding_dim=128, gru_units=64, 
                    dense_units=32, dropout_rate=0.5, rec_dropout_rate=0.2, 
                    learning_rate=0.001):
    """Membangun model GRU sebagai alternatif tambahan untuk eksperimen."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        GRU(gru_units, dropout=dropout_rate, recurrent_dropout=rec_dropout_rate),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(learning_rate=learning_rate), 
                  metrics=['accuracy'])
    return model

# TRAINING LOOP & UTILITIES
def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, patience=3):
    """Mengeksekusi proses training dengan Validation Set dan algoritma Class Weights."""
    print("\n[INFO] Menghitung Class Weights untuk mitigasi Imbalanced Data...")
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, weights))
    print(f"[INFO] Penalty Weights -> Ham: {class_weights_dict[0]:.2f} | Spam: {class_weights_dict[1]:.2f}")

    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=patience, 
        restore_best_weights=True,
        verbose=1
    )

    print("\n[INFO] Memulai Training Loop...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights_dict,
        callbacks=[early_stop],
        verbose=1
    )
    
    return history

def save_model(model, filepath):
    """Menyimpan model ke dalam disk untuk fase deployment/evaluasi."""
    model.save(filepath)
    print(f"\n[SUCCESS] Model berhasil diekspor ke: {filepath}")

def plot_training_history(history, save_path=None):
    """Visualisasi metrik Loss dan Accuracy per epoch untuk analisis Overfitting."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(history.history['loss'], label='Train Loss', color='blue')
    ax2.plot(history.history['val_loss'], label='Val Loss', color='orange')
    ax2.set_title('Model Loss (Binary Crossentropy)')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[SUCCESS] Training history plot disimpan di: {save_path}")
    else:
        plt.show()

# MAIN EXECUTION BLOCK (Sanity Check)
if __name__ == "__main__":
    print("[SYSTEM] Menginisiasi Training Module...")
    
    try:
        # Load raw matrices dari pipeline hulu
        print("[INFO] Memuat input tensors (.npy)...")
        X_train = np.load("X_train.npy")
        y_train = np.load("y_train.npy")
        X_val = np.load("X_val.npy")
        y_val = np.load("y_val.npy")
        
        # Validasi dimensi tensor
        print(f"[SUCCESS] Tensor berhasil dimuat.")
        print(f"         - X_train shape : {X_train.shape}")
        print(f"         - y_train shape : {y_train.shape}")
        
    except FileNotFoundError:
        print("[ERROR] File input (.npy) tidak terdeteksi.")
        print("[AKSI]  Tunggu modul preprocessing.py dieksekusi terlebih dahulu.")
        exit()

    # Absolut parameter untuk testing awal, masih bisa diubah untuk eksperimen lebih lanjut
    VOCAB_SIZE = 5000  
    
    print("\n[INFO] Merakit arsitektur LSTM Network...")
    model_lstm = build_lstm_model(
        vocab_size=VOCAB_SIZE, 
        embedding_dim=128, 
        lstm_units=64, 
        dense_units=32
    )
    
    # Cetak ringkasan layer dan kalkulasi parameter
    model_lstm.summary()
    
    # Eksekusi Sanity Check (epoch bisa diubah untuk eksperimen)
    history_lstm = train_model(
        model=model_lstm, 
        X_train=X_train, 
        y_train=y_train, 
        X_val=X_val, 
        y_val=y_val, 
        epochs=10,        
        batch_size=32
    )
    
    # Ekspor artifact untuk modul hilir (evaluate.py)
    save_model(model_lstm, "lstm_spam_model.keras")
    plot_training_history(history_lstm, "training_plot.png")
    
    print("\n[SYSTEM] Eksekusi tuntas. Model siap untuk pipeline evaluasi.")