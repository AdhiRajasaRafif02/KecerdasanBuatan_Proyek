"""
EDA dan Preprocessing untuk SMS Spam Classification
Kelompok 7 - Tugas Akhir Kecerdasan Buatan

File ini melakukan:
1. Exploratory Data Analysis (EDA) pada spam.csv
2. Pembersihan dan preprocessing teks
3. Encoding label
4. Split data train-test dengan stratifikasi
5. Tokenisasi dan padding sequence
6. Menyimpan hasil preprocessing
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_dataset(filepath):
    """
    Load dataset SMS dari file CSV.
    
    Args:
        filepath (str): Path ke file spam.csv
        
    Returns:
        pd.DataFrame: DataFrame dengan data yang dimuat
    """
    print("=" * 80)
    print("STEP 1: LOADING DATASET")
    print("=" * 80)
    
    df = pd.read_csv(filepath, encoding='latin-1')
    print(f"✓ Dataset berhasil dimuat dari: {filepath}")
    print(f"✓ Shape dataset: {df.shape}")
    
    return df


def explore_dataset(df):
    """
    Exploratory Data Analysis pada dataset.
    
    Args:
        df (pd.DataFrame): Dataset yang akan dianalisis
    """
    print("\n" + "=" * 80)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # Tampilkan 5 data pertama
    print("\n5 Data Pertama:")
    print(df.head())
    
    # Tampilkan informasi dataset
    print("\nInformasi Dataset:")
    print(df.info())
    
    # Tampilkan shape
    print(f"\nJumlah Baris: {df.shape[0]}")
    print(f"Jumlah Kolom: {df.shape[1]}")
    print(f"Nama Kolom: {list(df.columns)}")


def prepare_dataframe(df):
    """
    Persiapan dataframe: ambil kolom yang diperlukan dan rename.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame yang sudah disiapkan
    """
    print("\n" + "=" * 80)
    print("STEP 3: PERSIAPAN DATAFRAME")
    print("=" * 80)
    
    # Ambil hanya kolom v1 dan v2
    df = df[['v1', 'v2']].copy()
    
    # Ubah nama kolom
    df.columns = ['label', 'message']
    
    print("✓ Kolom sudah di-rename: v1 → label, v2 → message")
    print(f"✓ Shape setelah persiapan: {df.shape}")
    
    return df


def analyze_label_distribution(df):
    """
    Analisis distribusi label spam dan ham.
    
    Args:
        df (pd.DataFrame): DataFrame dengan kolom label
    """
    print("\n" + "=" * 80)
    print("STEP 4: ANALISIS DISTRIBUSI LABEL")
    print("=" * 80)
    
    label_dist = df['label'].value_counts()
    print("\nDistribusi Label:")
    print(label_dist)
    
    label_pct = df['label'].value_counts(normalize=True) * 100
    print("\nPersentase Label:")
    for label, pct in label_pct.items():
        print(f"  {label}: {pct:.2f}%")
    
    # Cek missing value
    print(f"\nMissing Values:")
    print(f"  Label: {df['label'].isnull().sum()}")
    print(f"  Message: {df['message'].isnull().sum()}")


def clean_text(text):
    """
    Membersihkan text SMS.
    
    Args:
        text (str): Raw SMS text
        
    Returns:
        str: Cleaned text
    """
    # Ubah ke lowercase
    text = text.lower()
    
    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Hapus email
    text = re.sub(r'\S+@\S+', '', text)
    
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    
    # Hapus karakter khusus dan tanda baca
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_messages(df):
    """
    Preprocessing semua pesan dalam dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame dengan kolom message
        
    Returns:
        pd.DataFrame: DataFrame dengan kolom clean_message
    """
    print("\n" + "=" * 80)
    print("STEP 5: PEMBERSIHAN TEKS (TEXT CLEANING)")
    print("=" * 80)
    
    print("Membersihkan teks SMS...")
    df['clean_message'] = df['message'].apply(clean_text)
    
    print("✓ Pembersihan teks selesai")
    print("\nContoh hasil cleaning:")
    for idx in range(min(3, len(df))):
        print(f"\nOriginal message {idx+1}:")
        print(f"  {df['message'].iloc[idx][:80]}...")
        print(f"Cleaned message {idx+1}:")
        print(f"  {df['clean_message'].iloc[idx][:80]}...")
    
    return df


def encode_labels_func(df):
    """
    Encode label ham/spam menjadi angka.
    
    Args:
        df (pd.DataFrame): DataFrame dengan kolom label
        
    Returns:
        tuple: (DataFrame dengan kolom label_encoded, LabelEncoder object)
    """
    print("\n" + "=" * 80)
    print("STEP 6: ENCODING LABEL")
    print("=" * 80)
    
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    
    print("✓ Label sudah di-encode")
    print(f"  Ham (0): {(df['label_encoded'] == 0).sum()} samples")
    print(f"  Spam (1): {(df['label_encoded'] == 1).sum()} samples")
    
    return df, le


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data menjadi train dan test dengan stratifikasi.
    
    Args:
        df (pd.DataFrame): DataFrame yang akan di-split
        test_size (float): Proporsi test set
        random_state (int): Random state untuk reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, indices_train, indices_test)
    """
    print("\n" + "=" * 80)
    print("STEP 7: SPLIT DATA TRAIN-TEST")
    print("=" * 80)
    
    X = df['clean_message'].values
    y = df['label_encoded'].values
    indices = np.arange(len(df))
    
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"✓ Data berhasil di-split")
    print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Train - Ham: {(y_train == 0).sum()}, Spam: {(y_train == 1).sum()}")
    print(f"  Test - Ham: {(y_test == 0).sum()}, Spam: {(y_test == 1).sum()}")
    
    return X_train, X_test, y_train, y_test, idx_train, idx_test


def tokenize_and_pad(X_train, X_test, max_words=5000, max_len=100):
    """
    Tokenisasi dan padding sequence.
    
    Args:
        X_train (np.array): Training messages
        X_test (np.array): Test messages
        max_words (int): Jumlah maksimal kata dalam vocabulary
        max_len (int): Panjang maksimal sequence
        
    Returns:
        tuple: (X_train_pad, X_test_pad, tokenizer)
    """
    print("\n" + "=" * 80)
    print("STEP 8: TOKENISASI")
    print("=" * 80)
    
    # Create tokenizer
    tokenizer = Tokenizer(
        num_words=max_words,
        oov_token="<OOV>"
    )
    
    # Fit tokenizer pada training set
    print(f"Fitting tokenizer pada {len(X_train)} training messages...")
    tokenizer.fit_on_texts(X_train)
    
    vocab_size = len(tokenizer.word_index)
    print(f"✓ Vocabulary size: {min(vocab_size, max_words)}")
    
    # Convert ke sequences
    print(f"\nMengubah teks ke sequences...")
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    print("\n" + "=" * 80)
    print("STEP 9: PADDING")
    print("=" * 80)
    
    # Padding
    X_train_pad = pad_sequences(
        X_train_seq,
        maxlen=max_len,
        padding='post',
        truncating='post'
    )
    X_test_pad = pad_sequences(
        X_test_seq,
        maxlen=max_len,
        padding='post',
        truncating='post'
    )
    
    print(f"✓ Padding selesai")
    print(f"  X_train_pad shape: {X_train_pad.shape}")
    print(f"  X_test_pad shape: {X_test_pad.shape}")
    print(f"  Max sequence length: {max_len}")
    
    return X_train_pad, X_test_pad, tokenizer


def save_preprocessed_data(df, output_dir='results'):
    """
    Menyimpan dataframe yang sudah dibersihkan.
    
    Args:
        df (pd.DataFrame): DataFrame untuk disimpan
        output_dir (str): Directory untuk menyimpan file
    """
    print("\n" + "=" * 80)
    print("STEP 10: MENYIMPAN DATA YANG SUDAH DIPROSES")
    print("=" * 80)
    
    # Buat folder jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Pilih kolom untuk disimpan
    df_save = df[['label', 'message', 'clean_message', 'label_encoded']].copy()
    
    # Simpan ke CSV
    output_path = os.path.join(output_dir, 'cleaned_spam.csv')
    df_save.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✓ Data yang sudah dibersihkan disimpan ke: {output_path}")
    print(f"  Total rows: {len(df_save)}")
    print(f"  Columns: {list(df_save.columns)}")


def print_summary():
    """Mencetak ringkasan preprocessing yang telah dilakukan."""
    print("\n" + "=" * 80)
    print("RINGKASAN PREPROCESSING")
    print("=" * 80)
    
    summary = """
    ✓ DATASET LOADING: Memuat spam.csv dengan encoding latin-1
    ✓ EDA: Analisis struktur dan distribusi data
    ✓ TEXT CLEANING: Membersihkan teks (lowercase, remove URL, symbols, etc)
    ✓ LABEL ENCODING: Encode ham/spam menjadi 0/1
    ✓ TRAIN-TEST SPLIT: 80% train, 20% test (stratified)
    ✓ TOKENIZATION: Tokenisasi menggunakan Keras Tokenizer
    ✓ PADDING: Padding sequence ke panjang 100 dengan metode 'post'
    ✓ SAVE RESULTS: Menyimpan data bersih ke results/cleaned_spam.csv
    
    OUTPUTS:
    - X_train_pad: shape (n_train, 100) - Training sequences
    - X_test_pad: shape (n_test, 100) - Test sequences
    - y_train: Training labels
    - y_test: Test labels
    - tokenizer: Tokenizer object untuk preprocessing data baru
    - cleaned_spam.csv: Dataset yang sudah dibersihkan
    
    SIAP UNTUK: Model training (Simple RNN dan LSTM)
    """
    print(summary)


def main():
    """Main function untuk menjalankan preprocessing pipeline."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "SMS SPAM CLASSIFICATION - EDA & PREPROCESSING" + " " * 13 + "║")
    print("║" + " " * 30 + "Kelompok 7 - Tugas Akhir AI" + " " * 21 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Step 1: Load dataset
    filepath = 'spam.csv'
    if not os.path.exists(filepath):
        print(f"ERROR: File {filepath} tidak ditemukan!")
        return
    
    df = load_dataset(filepath)
    
    # Step 2: EDA
    explore_dataset(df)
    
    # Step 3: Prepare dataframe
    df = prepare_dataframe(df)
    
    # Step 4: Analyze label distribution
    analyze_label_distribution(df)
    
    # Step 5: Clean text
    df = preprocess_messages(df)
    
    # Step 6: Encode labels
    df, label_encoder = encode_labels_func(df)
    
    # Step 7: Split data
    X_train, X_test, y_train, y_test, idx_train, idx_test = split_data(df)
    
    # Step 8 & 9: Tokenize and pad
    X_train_pad, X_test_pad, tokenizer = tokenize_and_pad(X_train, X_test)
    
    # Step 10: Save preprocessed data
    save_preprocessed_data(df)
    
    # Print summary
    print_summary()
    
    print("\n" + "=" * 80)
    print("PREPROCESSING SELESAI! ✓")
    print("=" * 80)
    print("\nFile yang siap untuk training:")
    print("  - X_train_pad: Training sequences")
    print("  - X_test_pad: Test sequences")
    print("  - y_train: Training labels")
    print("  - y_test: Test labels")
    print("  - results/cleaned_spam.csv: Dataset yang sudah dibersihkan")
    print("\nNext step: Jalankan training model (Simple RNN dan LSTM)")
    print("=" * 80 + "\n")
    
    return {
        'df': df,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_pad': X_train_pad,
        'X_test_pad': X_test_pad,
        'y_train': y_train,
        'y_test': y_test,
        'tokenizer': tokenizer,
        'label_encoder': label_encoder
    }


if __name__ == '__main__':
    main()
