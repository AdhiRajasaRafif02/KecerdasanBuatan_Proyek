"""
Preprocessing module untuk SMS Spam Classification

Modul ini berisi fungsi-fungsi untuk mempersiapkan data:
- Load dataset dari CSV
- Membersihkan dan membersihkan text
- Encode label (ham/spam)
- Tokenisasi dan padding sequence
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_dataset(filepath):
    """
    Load dataset SMS dari file CSV.
    
    Args:
        filepath (str): Path ke file spam.csv
    
    Returns:
        pd.DataFrame: DataFrame dengan kolom text dan label
    
    TODO:
        - Handle missing values
        - Verify data structure
    """
    # TODO: Implement load dataset
    pass


def clean_text(text):
    """
    Membersihkan text SMS: lowercase, remove punctuation, etc.
    
    Args:
        text (str): Raw SMS text
    
    Returns:
        str: Cleaned text
    
    TODO:
        - Remove special characters
        - Remove URLs
        - Remove numbers
    """
    # TODO: Implement text cleaning
    pass


def encode_labels(labels):
    """
    Encode label categorical (ham/spam) menjadi numeric (0/1).
    
    Args:
        labels (pd.Series): Series dengan label 'ham' atau 'spam'
    
    Returns:
        np.array: Array dengan nilai 0 (ham) atau 1 (spam)
        LabelEncoder: Encoder object untuk decoding
    
    TODO:
        - Use LabelEncoder from sklearn
    """
    # TODO: Implement label encoding
    pass


def tokenize_and_pad(texts, max_words=5000, max_len=200):
    """
    Tokenisasi text menjadi sequences dan padding ke ukuran yang sama.
    
    Args:
        texts (list): List of text strings
        max_words (int): Jumlah maksimal kata dalam vocabulary
        max_len (int): Panjang maksimal sequence setelah padding
    
    Returns:
        np.array: Padded sequences
        Tokenizer: Tokenizer object untuk preprocessing text baru
    
    TODO:
        - Create tokenizer
        - Fit tokenizer on texts
        - Convert texts to sequences
        - Apply padding
    """
    # TODO: Implement tokenization and padding
    pass


def preprocess_pipeline(filepath, max_words=5000, max_len=200):
    """
    Pipeline lengkap preprocessing: load -> clean -> encode -> tokenize.
    
    Args:
        filepath (str): Path ke file spam.csv
        max_words (int): Jumlah maksimal kata
        max_len (int): Panjang maksimal sequence
    
    Returns:
        tuple: (X_processed, y_encoded, tokenizer, label_encoder)
    
    TODO:
        - Combine all preprocessing steps
    """
    # TODO: Implement full pipeline
    pass
