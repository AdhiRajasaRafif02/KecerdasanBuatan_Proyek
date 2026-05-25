# Hyperparameter Tuning - SMS Spam Classification
**Kelompok 7 - Tugas Akhir Kecerdasan Buatan**

## Overview
Script `src/hyperparameter_tuning.py` melakukan eksperimen hyperparameter tuning sederhana pada model LSTM untuk SMS Spam Classification.

## How to Run
```bash
python src/hyperparameter_tuning.py
```

## Hyperparameter Experiments

### Run 1: Baseline
- LSTM units: 64
- Dropout: 0.5
- Batch size: 32
- Epochs: 10

### Run 2: More Units
- LSTM units: 128
- Dropout: 0.5
- Batch size: 32
- Epochs: 10

### Run 3: Less Dropout
- LSTM units: 64
- Dropout: 0.3
- Batch size: 32
- Epochs: 10

## Output Files

### 1. `results/tuning_results.csv`
Hasil evaluasi ketiga eksperimen dengan kolom:
- `run_name`: Nama eksperimen
- `lstm_units`: Jumlah LSTM units
- `dropout`: Dropout rate
- `batch_size`: Batch size untuk training
- `epochs`: Jumlah epoch training
- `train_accuracy`: Akurasi training pada epoch terakhir
- `val_accuracy`: Akurasi validation pada epoch terakhir
- `val_loss`: Loss validation pada epoch terakhir
- `test_accuracy`: Akurasi pada test set
- `test_precision`: Precision pada test set
- `test_recall`: Recall pada test set
- `test_f1`: F1-Score pada test set

### 2. `results/tuning_f1_score.png`
Bar chart visualisasi F1-Score test set untuk setiap eksperimen.

### 3. `models/best_lstm_model.keras` (optional)
Model terbaik berdasarkan F1-Score tertinggi (jika ada eksperimen dengan F1 > 0).

## Implementation Details

### Data Preparation
- Load dan preprocess data menggunakan `src/eda_preprocessing.py`
- Train-test split: 80-20
- Validation split dari training set: 80-20
- Tokenisasi: 5000 vocabulary, max_len=100

### Model Architecture
```
Embedding (vocab_size, 128, max_len=100)
    ↓
LSTM (lstm_units, dropout=dropout_rate, recurrent_dropout=0.2)
    ↓
Dense (32, activation='relu')
    ↓
Dropout (dropout_rate)
    ↓
Dense (1, activation='sigmoid') [Output]
```

### Training Configuration
- Loss: Binary Crossentropy
- Optimizer: Adam (learning_rate=0.001)
- Metrics: Accuracy
- EarlyStopping: patience=3, monitor='val_loss'
- Class weights: Balanced (untuk menangani imbalanced dataset)

## Key Features

1. **Multiple Experiments**: Menjalankan 3 eksperimen dengan hyperparameter berbeda
2. **Validation Split**: Menggunakan validation set untuk early stopping
3. **Class Weights**: Menangani imbalanced dataset (ham vs spam)
4. **Comprehensive Metrics**: Mencatat accuracy, precision, recall, F1-score
5. **Visualization**: Bar chart F1-Score untuk perbandingan hasil
6. **Best Model Saving**: Menyimpan model terbaik berdasarkan F1-Score

## Kesimpulan

Script berhasil melakukan:
✓ Hyperparameter tuning 3 eksperimen LSTM
✓ Evaluasi pada training, validation, dan test set
✓ Menyimpan hasil ke CSV
✓ Membuat visualisasi F1-Score
✓ Menyimpan model terbaik

## Catatan
- Script dijalankan tanpa batasan waktu lama (hanya 10 epochs per eksperimen)
- Dirancang untuk laporan akhir dengan eksperimen sederhana
- Semua preprocessing dan data splitting sudah dilakukan
