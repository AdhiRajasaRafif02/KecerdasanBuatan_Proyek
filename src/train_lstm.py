"""
Training module untuk model SMS Spam Classification

Modul ini berisi fungsi-fungsi untuk:
- Membangun model Simple RNN sebagai baseline
- Membangun model LSTM sebagai model utama
- Melakukan training dan validation
- Menyimpan model yang sudah dilatih

Tahap Progress Saat Ini: Persiapan model dan training (akan dikerjakan tahap berikutnya)
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os


def build_rnn_baseline(vocab_size, embedding_dim=128, rnn_units=64, dropout_rate=0.5):
    """
    Membangun Simple RNN model sebagai baseline.
    
    Args:
        vocab_size (int): Ukuran vocabulary dari tokenizer
        embedding_dim (int): Dimensi embedding layer
        rnn_units (int): Jumlah unit dalam RNN layer
        dropout_rate (float): Dropout rate untuk regularization
    
    Returns:
        Sequential: Compiled Keras model
    
    Architecture:
        - Embedding layer (max_words, embedding_dim)
        - SimpleRNN layer (rnn_units units)
        - Dropout layer
        - Dense layer (16 units)
        - Dropout layer
        - Output layer (sigmoid for binary classification)
    
    TODO pada tahap training:
        - Memastikan parameter sudah optimal
        - Add callbacks (EarlyStopping, ModelCheckpoint)
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=100),
        SimpleRNN(units=rnn_units, activation='relu', return_sequences=False),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_lstm_model(vocab_size, embedding_dim=128, lstm_units=64, dropout_rate=0.5):
    """
    Membangun LSTM model sebagai model utama.
    
    Args:
        vocab_size (int): Ukuran vocabulary
        embedding_dim (int): Dimensi embedding
        lstm_units (int): Jumlah unit dalam LSTM layer
        dropout_rate (float): Dropout rate untuk regularization
    
    Returns:
        Sequential: Compiled Keras model
    
    Architecture:
        - Embedding layer (max_words, embedding_dim)
        - LSTM layer 1 (lstm_units units) dengan return_sequences=True
        - Dropout layer
        - LSTM layer 2 (lstm_units//2 units)
        - Dropout layer
        - Dense layer (16 units)
        - Dropout layer
        - Output layer (sigmoid for binary classification)
    
    TODO pada tahap training:
        - Tuning jumlah LSTM layers dan units
        - Experiment dengan berbagai dropout rates
        - Add bidirectional LSTM jika perlu
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=100),
        LSTM(units=lstm_units, activation='relu', return_sequences=True, dropout=dropout_rate),
        LSTM(units=lstm_units//2, activation='relu', return_sequences=False, dropout=dropout_rate),
        Dense(16, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_gru_model(vocab_size, embedding_dim=128, gru_units=64, dropout_rate=0.5):
    """
    Membangun GRU model sebagai alternatif model (optional).
    
    Args:
        vocab_size (int): Ukuran vocabulary
        embedding_dim (int): Dimensi embedding
        gru_units (int): Jumlah unit dalam GRU layer
        dropout_rate (float): Dropout rate
    
    Returns:
        Sequential: Compiled Keras model
    
    Architecture:
        - Embedding layer
        - GRU layer(s) dengan dropout
        - Dense layer(s) dengan dropout
        - Output layer (sigmoid)
    
    TODO pada tahap training:
        - Experiment dengan GRU sebagai alternatif LSTM
        - Compare performa dengan RNN dan LSTM
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=100),
        GRU(units=gru_units, activation='relu', return_sequences=True, dropout=dropout_rate),
        GRU(units=gru_units//2, activation='relu', return_sequences=False, dropout=dropout_rate),
        Dense(16, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, model_name='model'):
    """
    Melakukan training pada model dengan validation set.
    
    Args:
        model: Keras model yang sudah di-compile
        X_train (np.array): Training data
        y_train (np.array): Training labels
        X_val (np.array): Validation data
        y_val (np.array): Validation labels
        epochs (int): Jumlah epoch training
        batch_size (int): Batch size untuk training
        model_name (str): Nama model untuk callback
    
    Returns:
        History: Keras training history object
    
    TODO pada tahap training:
        - Adjust callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
        - Tune hyperparameters (batch_size, epochs, learning_rate)
        - Monitor training progress
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history


def save_model(model, filepath, overwrite=False):
    """
    Menyimpan model dalam format .keras.
    
    Args:
        model: Trained Keras model
        filepath (str): Path untuk menyimpan model (dengan .keras extension)
        overwrite (bool): Overwrite jika file sudah ada
    
    Returns:
        str: Path dimana model disimpan
    
    TODO pada tahap training:
        - Handle different save formats (.keras, .h5, SavedModel format)
        - Add metadata logging
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    
    model.save(filepath, overwrite=overwrite)
    print(f"✓ Model berhasil disimpan: {filepath}")
    
    return filepath


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss dan accuracy).
    
    Args:
        history: Keras training history object
        save_path (str): Path untuk menyimpan plot (optional)
    
    Returns:
        None (displays/saves plot)
    
    TODO pada tahap training:
        - Customize plot styling
        - Add more metrics visualization
        - Compare multiple model trainings
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].se


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss dan accuracy).
    
    Args:
        history: Keras training history object
        save_path (str): Path untuk menyimpan plot (optional)
    
    Returns:
        None (displays/saves plot)
    
    TODO:
        - Plot training vs validation loss
        - Plot training vs validation accuracy
        - Save figure if save_path provided
    """
    # TODO: Implement history plotting
    pass
