"""
EDA dan Preprocessing untuk SMS Spam Classification
Kelompok 7 - Tugas Akhir Kecerdasan Buatan

File ini menyediakan fungsi-fungsi untuk:
1. Pembersihan dan preprocessing teks
2. Exploratory Data Analysis (EDA)
3. Penyimpanan dataset yang telah dibersihkan
4. Loading dan preprocessing data lengkap dengan tokenisasi dan padding

Dapat dijalankan langsung dengan: python src/eda_preprocessing.py
Atau di-import oleh file training lainnya
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_text(text):
    """
    Membersihkan text SMS dengan menghilangkan URL, angka, simbol, dan spasi berlebih.
    
    Args:
        text (str): Raw SMS text
        
    Returns:
        str: Cleaned text (lowercase, tanpa URL, angka, simbol, dan spasi berlebih)
    """
    # Ubah ke lowercase
    text = text.lower()
    
    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Hapus email
    text = re.sub(r'\S+@\S+', '', text)
    
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    
    # Hapus karakter khusus dan tanda baca (hanya simpan huruf dan spasi)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def show_basic_eda(df):
    """
    Menampilkan exploratory data analysis dasar dari dataset.
    
    Args:
        df (pd.DataFrame): DataFrame dengan kolom 'label' dan 'message'
    """
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # Tampilkan 5 data pertama
    print("\n5 Data Pertama:")
    print(df.head())
    
    # Tampilkan shape
    print(f"\nShape dataset: {df.shape}")
    print(f"  Jumlah Baris: {df.shape[0]}")
    print(f"  Jumlah Kolom: {df.shape[1]}")
    print(f"  Nama Kolom: {list(df.columns)}")
    
    # Distribusi label
    print("\nDistribusi Label:")
    label_dist = df['label'].value_counts()
    for label, count in label_dist.items():
        pct = (count / len(df)) * 100
        print(f"  {label}: {count} samples ({pct:.2f}%)")
    
    # Missing values
    print(f"\nMissing Values:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        print(f"  {col}: {missing}")
    
    # Mapping label encoding
    print(f"\nLabel Encoding Mapping:")
    print(f"  ham -> 0")
    print(f"  spam -> 1")


def save_cleaned_dataset(df, output_path="results/cleaned_spam.csv"):
    """
    Menyimpan dataset yang telah dibersihkan ke file CSV.
    
    Args:
        df (pd.DataFrame): DataFrame dengan kolom 'clean_message' dan 'label_encoded'
        output_path (str): Path untuk menyimpan file CSV
    """
    # Buat folder jika belum ada
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[OK] Folder '{output_dir}' berhasil dibuat")
    
    # Simpan dataset
    df.to_csv(output_path, index=False)
    print(f"[OK] Dataset cleaned berhasil disimpan ke: {output_path}")
    print(f"  Total samples: {len(df)}")


def load_and_preprocess_data(csv_path="spam.csv", max_words=5000, max_len=100, 
                            test_size=0.2, random_state=42):
    """
    Load dataset dan melakukan preprocessing lengkap:
    1. Load CSV dengan encoding latin-1
    2. Select dan rename kolom (v1→label, v2→message)
    3. Text cleaning
    4. Label encoding
    5. Train-test split dengan stratifikasi
    6. Tokenisasi dengan Tokenizer
    7. Padding sequence
    
    Args:
        csv_path (str): Path ke file spam.csv
        max_words (int): Jumlah kata maksimal dalam vocabulary (default: 5000)
        max_len (int): Panjang maksimal sequence (default: 100)
        test_size (float): Proporsi test set (default: 0.2 = 80:20 split)
        random_state (int): Random state untuk reproducibility (default: 42)
        
    Returns:
        tuple: (X_train_pad, X_test_pad, y_train, y_test, tokenizer, label_encoder)
               - X_train_pad: array, shape (num_train, max_len)
               - X_test_pad: array, shape (num_test, max_len)
               - y_train: array, labels untuk training
               - y_test: array, labels untuk testing
               - tokenizer: Keras Tokenizer object (sudah fit pada training data)
               - label_encoder: sklearn LabelEncoder object (sudah fit)
    """
    print("=" * 80)
    print("PREPROCESSING DATA UNTUK SMS SPAM CLASSIFICATION")
    print("=" * 80)
    
    # Step 1: Load dataset
    print("\n[STEP 1] Loading dataset...")
    df = pd.read_csv(csv_path, encoding='latin-1')
    print(f"[OK] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # Step 2: Select dan rename kolom
    print("\n[STEP 2] Selecting and renaming columns...")
    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'message']
    print(f"[OK] Columns renamed: v1->label, v2->message")
    
    # Step 3: Text cleaning
    print("\n[STEP 3] Cleaning text...")
    df['clean_message'] = df['message'].apply(clean_text)
    print(f"[OK] Text cleaning completed")
    
    # Step 4: Label encoding
    print("\n[STEP 4] Encoding labels...")
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    print(f"[OK] Labels encoded")
    print(f"  {label_encoder.classes_[0]} -> 0")
    print(f"  {label_encoder.classes_[1]} -> 1")
    
    # Step 5: Train-test split dengan stratifikasi
    print("\n[STEP 5] Splitting train-test ({}-{} split)...".format(
        int((1-test_size)*100), int(test_size*100)))
    X = df['clean_message'].values
    y = df['label_encoded'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"[OK] Data split completed")
    print(f"  Training: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
    
    # Step 6: Tokenisasi
    print("\n[STEP 6] Tokenizing texts (max_words={})...".format(max_words))
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    vocab_size = min(len(tokenizer.word_index) + 1, max_words)
    print(f"[OK] Tokenization completed")
    print(f"  Vocabulary size: {vocab_size}")
    
    # Step 7: Padding
    print("\n[STEP 7] Padding sequences (max_len={})...".format(max_len))
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
    
    print(f"[OK] Padding completed")
    print(f"  X_train_pad shape: {X_train_pad.shape}")
    print(f"  X_test_pad shape: {X_test_pad.shape}")
    
    print("\n" + "=" * 80)
    print("[OK] PREPROCESSING COMPLETED")
    print("=" * 80)
    
    return X_train_pad, X_test_pad, y_train, y_test, tokenizer, label_encoder


if __name__ == '__main__':
    """
    Main script untuk menjalankan EDA dan preprocessing
    
    Jalankan dengan: python src/eda_preprocessing.py
    """
    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 15 + "SMS SPAM CLASSIFICATION - EDA & PREPROCESSING" + " " * 18 + "|")
    print("|" + " " * 30 + "Kelompok 7 - Tugas Akhir AI" + " " * 21 + "|")
    print("+" + "=" * 78 + "+\n")
    
    # Load dan preprocess data
    X_train_pad, X_test_pad, y_train, y_test, tokenizer, label_encoder = \
        load_and_preprocess_data(
            csv_path="spam.csv",
            max_words=5000,
            max_len=100,
            test_size=0.2,
            random_state=42
        )
    
    # Load dataset untuk EDA
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'message']
    
    # Show basic EDA
    show_basic_eda(df)
    
    # Save cleaned dataset
    print("\n" + "=" * 80)
    print("SAVING CLEANED DATASET")
    print("=" * 80)
    
    df['clean_message'] = df['message'].apply(clean_text)
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    
    save_cleaned_dataset(df, output_path="results/cleaned_spam.csv")
    
    # Final summary
    print("\n" + "=" * 80)
    print("[OK] PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    print("\nPreprocessing Summary:")
    print(f"  [OK] Total samples processed: {len(df)}")
    print(f"  [OK] X_train_pad shape: {X_train_pad.shape}")
    print(f"  [OK] X_test_pad shape: {X_test_pad.shape}")
    print(f"  [OK] Vocabulary size: {len(tokenizer.word_index) + 1}")
    print(f"  [OK] Max sequence length: 100")
    print(f"  [OK] Cleaned dataset saved to: results/cleaned_spam.csv")
    
    print("\n" + "=" * 80)
    print("Ready for training with the following outputs:")
    print("  - X_train_pad: Training sequences (shape: {})".format(X_train_pad.shape))
    print("  - X_test_pad: Test sequences (shape: {})".format(X_test_pad.shape))
    print("  - y_train: Training labels (shape: {})".format(y_train.shape))
    print("  - y_test: Test labels (shape: {})".format(y_test.shape))
    print("  - tokenizer: Keras Tokenizer object")
    print("  - label_encoder: Label encoder object")
    print("  - results/cleaned_spam.csv: Cleaned dataset file")
    print("=" * 80 + "\n")
