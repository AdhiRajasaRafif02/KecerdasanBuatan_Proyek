"""
Training module untuk model SMS Spam Classification

Modul ini berisi fungsi-fungsi untuk:
- Membangun model Simple RNN sebagai baseline
- Membangun model LSTM sebagai model utama
- Melakukan training dan validation
- Menyimpan model yang sudah dilatih
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import json


def build_rnn_baseline(vocab_size, embedding_dim=128, output_dim=64):
    """
    Membangun Simple RNN model sebagai baseline.
    
    Args:
        vocab_size (int): Ukuran vocabulary dari tokenizer
        embedding_dim (int): Dimensi embedding layer
        output_dim (int): Dimensi output dari RNN layer
    
    Returns:
        Sequential: Compiled Keras model
    
    Architecture:
        - Embedding layer
        - SimpleRNN layer
        - Dense layer(s)
        - Output layer (sigmoid for binary classification)
    
    TODO:
        - Define model architecture
        - Compile with appropriate loss and metrics
    """
    # TODO: Implement RNN baseline model
    pass


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
        - Embedding layer
        - LSTM layer(s) dengan dropout
        - Dense layer(s) dengan dropout
        - Output layer (sigmoid)
    
    TODO:
        - Define LSTM architecture
        - Add dropout for regularization
        - Compile model
    """
    # TODO: Implement LSTM model
    pass


def build_gru_model(vocab_size, embedding_dim=128, gru_units=64, dropout_rate=0.5):
    """
    Membangun GRU model sebagai alternatif model.
    
    Args:
        vocab_size (int): Ukuran vocabulary
        embedding_dim (int): Dimensi embedding
        gru_units (int): Jumlah unit dalam GRU layer
        dropout_rate (float): Dropout rate
    
    Returns:
        Sequential: Compiled Keras model
    
    TODO:
        - Define GRU architecture
        - Compile model
    """
    # TODO: Implement GRU model
    pass


def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
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
    
    Returns:
        History: Keras training history object
    
    TODO:
        - Define early stopping callback
        - Train model with validation
        - Return history
    """
    # TODO: Implement training loop
    pass


def save_model(model, filepath):
    """
    Menyimpan model dalam format .keras atau .h5.
    
    Args:
        model: Trained Keras model
        filepath (str): Path untuk menyimpan model
    
    TODO:
        - Save model using model.save()
    """
    # TODO: Implement model saving
    pass


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
