## 📊 HYPERPARAMETER TUNING & ERROR ANALYSIS DOCUMENTATION

Dokumentasi lengkap untuk modul Hyperparameter Tuning dan Error Analysis dalam proyek SMS Spam Classification.

---

## 🎯 OVERVIEW

Tahap 5 dari proyek ini fokus pada **Fine-tuning dan Optimisasi** model LSTM dengan dua komponen utama:

1. **Hyperparameter Tuning**: Mencari kombinasi hyperparameter optimal
2. **Error Analysis**: Analisis mendalam terhadap error yang terjadi

---

## 📌 MODUL 1: HYPERPARAMETER TUNING

### Tujuan
Menemukan hyperparameter terbaik untuk model LSTM melalui systematic search atau random exploration.

### Fitur Utama

#### 1. **Grid Search**
Menguji **SEMUA kombinasi** hyperparameter secara sistematis.

**Parameter yang di-tune:**
- `embedding_dim`: [64, 128, 256]
- `lstm_units`: [32, 64, 128]
- `dropout_rate`: [0.3, 0.5]
- `learning_rate`: [0.001, 0.0001]

**Total kombinasi**: 3 × 3 × 2 × 2 = **36 trials**

```bash
# Jalankan Grid Search
python hyperparameter_tuning.py --method grid --epochs 10

# Dari direktori src/
python src/hyperparameter_tuning.py --method grid --epochs 10
```

#### 2. **Random Search**
Menguji kombinasi hyperparameter secara **random** untuk efisiensi lebih tinggi.

```bash
# Random Search dengan 20 trials
python hyperparameter_tuning.py --method random --n_iter 20 --epochs 10

# Dengan lebih banyak iterasi
python hyperparameter_tuning.py --method random --n_iter 50 --epochs 10
```

**Keuntungan Random Search:**
- ✅ Lebih efisien (tidak perlu test semua kombinasi)
- ✅ Bisa menemukan kombinasi unik
- ✅ Lebih cepat untuk space besar

#### 3. **K-Fold Cross-Validation**
Evaluasi robust dengan **multiple folds** untuk estimasi performa yang lebih akurat.

```bash
# 5-Fold Cross-Validation dengan fixed hyperparameters
python hyperparameter_tuning.py --method kfold --n_splits 5 --epochs 10

# 10-Fold untuk evaluasi lebih konservatif
python hyperparameter_tuning.py --method kfold --n_splits 10 --epochs 10
```

**Output per fold:**
- Validation accuracy
- Test accuracy
- Test F1 Score
- Test ROC-AUC
- Training time

### Output Files

Semua hasil disimpan di `results/tuning/`:

```
results/tuning/
├── tuning_results.csv          # Detail semua trials
│   ├── embedding_dim
│   ├── lstm_units
│   ├── dropout_rate
│   ├── learning_rate
│   ├── val_accuracy
│   ├── test_f1
│   ├── test_balanced_acc
│   ├── test_roc_auc
│   └── training_time
│
├── tuning_summary.json         # Summary best parameters
│   ├── timestamp
│   ├── total_trials
│   ├── best_params
│   └── top_5_results
│
└── tuning_analysis.png         # 4-subplot visualization
    ├── Embedding Dim vs LSTM Units vs F1
    ├── Dropout vs Learning Rate vs F1
    ├── Val Loss vs Test F1 (colored by ROC-AUC)
    └── Training Time vs Test F1 (colored by Balanced Accuracy)
```

### Interpretasi Hasil

**tuning_results.csv:**
- Setiap baris = 1 hyperparameter configuration
- Sortir by `test_f1` untuk menemukan best params
- Perhatikan trade-off antara `test_f1` dan `training_time`

**tuning_summary.json:**
```json
{
  "timestamp": "2026-05-24T12:00:00",
  "total_trials": 20,
  "best_params": {
    "embedding_dim": 256,
    "lstm_units": 128,
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "test_f1": 0.8567,
    "test_roc_auc": 0.9234
  },
  "top_5_results": [...]
}
```

### Visualisasi (tuning_analysis.png)

**Subplot 1: LSTM Units vs Test F1**
- X-axis: LSTM Units
- Y-axis: Test F1 Score
- Warna: Embedding Dim berbeda
- **Interpretasi**: Melihat pengaruh LSTM units terhadap performance

**Subplot 2: Learning Rate vs Test F1**
- X-axis: Learning Rate (log scale)
- Y-axis: Test F1 Score
- Warna: Dropout Rate berbeda
- **Interpretasi**: Optimal learning rate range

**Subplot 3: Val Loss vs Test F1**
- X-axis: Validation Loss
- Y-axis: Test F1
- Warna: ROC-AUC values
- **Interpretasi**: Correlation antara val loss dan test performance

**Subplot 4: Training Time vs Test F1**
- X-axis: Training Time (seconds)
- Y-axis: Test F1
- Warna: Balanced Accuracy
- **Interpretasi**: Trade-off antara computation cost dan performance

### Contoh Penggunaan Lengkap

```bash
# 1. Preprocessing & Training sudah selesai
python train_lstm.py

# 2. Jalankan Grid Search
python hyperparameter_tuning.py --method grid --epochs 10

# 3. Tunggu selesai (bisa beberapa jam untuk 36 trials)
# ...

# 4. Buka results/tuning/tuning_summary.json untuk best params

# 5. Gunakan best params untuk retrain model dengan konfigurasi optimal
```

---

## 📌 MODUL 2: ERROR ANALYSIS

### Tujuan
Memahami **DIMANA dan MENGAPA** model membuat kesalahan (error patterns).

### Fitur Utama

#### 1. **Detailed Error Records**
Membuat record lengkap untuk setiap misclassified sample:

```
Sample ID | True Label | Predicted | Confidence | Error Type
123       | Spam       | Ham       | 0.28       | False Negative
456       | Ham        | Spam      | 0.72       | False Positive
...
```

#### 2. **Error Categorization**
Memisahkan dua tipe error:

- **False Positives (FP)**: Prediksi Ham sebagai Spam
  - Artinya: Pesan legitimate dinyatakan sebagai spam
  - Dampak: Pengguna kehilangan pesan penting
  - Prioritas: TINGGI (harus diminimalkan)

- **False Negatives (FN)**: Prediksi Spam sebagai Ham
  - Artinya: Spam dinyatakan sebagai pesan legitimate
  - Dampak: Spam mencapai inbox pengguna
  - Prioritas: TINGGI (harus diminimalkan)

#### 3. **Worst Predictions**
Menampilkan prediksi **paling confident yang salah**:

```
[1] Sample ID: 123
    Error Type: False Negative (Spam→Ham)
    True Label: Spam (1)
    Predicted: Ham (0)
    Confidence: 0.1234
    Message: "Click here to win free money! Act now..."

[2] Sample ID: 456
    Error Type: False Positive (Ham→Spam)
    True Label: Ham (0)
    Predicted: Spam (1)
    Confidence: 0.9876
    Message: "Hi, how are you today? Let's meet for..."
```

#### 4. **Confidence Distribution Analysis**
Menganalisis distribusi confidence untuk:
- Correct vs Incorrect predictions
- False Positives vs False Negatives
- Identifying confidence thresholds

### Cara Menggunakan

```bash
# Basic error analysis
python error_analysis.py --model results/models/lstm.h5

# Dari direktori src/
python src/error_analysis.py --model results/models/lstm.h5

# Dengan confidence distribution analysis
python error_analysis.py --model results/models/lstm.h5 --analyze_confidence

# Tampilkan top 30 worst predictions
python error_analysis.py --model results/models/lstm.h5 --show_samples 30

# Specify output directory
python error_analysis.py \
  --model results/models/lstm.h5 \
  --output_dir custom/analysis/path \
  --analyze_confidence
```

### Command Line Arguments

| Argument | Type | Default | Deskripsi |
|----------|------|---------|-----------|
| `--model` | str | `results/models/lstm.h5` | Path ke model .h5 |
| `--show_samples` | int | 20 | Jumlah worst predictions ditampilkan |
| `--analyze_confidence` | flag | False | Generate confidence plots |
| `--output_dir` | str | `results/error_analysis` | Output directory |

### Output Files

Semua hasil disimpan di `results/error_analysis/`:

```
results/error_analysis/
├── error_details.csv                 # Detail setiap error
│   ├── sample_id
│   ├── true_label
│   ├── predicted_label
│   ├── confidence
│   ├── error_type
│   └── message_preview
│
├── error_report.txt                  # Text report lengkap
│   ├── SUMMARY STATISTICS
│   ├── CONFIDENCE DISTRIBUTION ANALYSIS
│   └── TOP N WORST PREDICTIONS
│
├── error_summary.json                # JSON summary
│   ├── error_summary (metrics)
│   ├── confidence_analysis
│   └── worst_predictions (top 10)
│
├── error_analysis_summary.png        # 4-subplot analysis
│   ├── Confusion Matrix (Raw)
│   ├── Confusion Matrix (Normalized)
│   ├── Error Breakdown (bar chart)
│   └── Error Statistics (text box)
│
└── confidence_distribution.png       # Confidence analysis (optional)
    ├── Histogram: Correct vs Incorrect
    ├── Histogram: FP vs FN
    ├── Boxplot comparison
    └── Cumulative Distribution Function
```

### Interpretasi Output

#### error_report.txt

**SUMMARY STATISTICS:**
```
Total Test Samples: 1115
Correct Predictions: 1000 (0.8969)
Incorrect Predictions: 115 (0.1031)
False Positives: 10 (FP Rate: 0.0021)
False Negatives: 105 (FN Rate: 0.1406)
```

- **Accuracy 89.69%** = Model bagus overall
- **FP Rate 0.21%** = Sangat sedikit legitimate msg jadi spam
- **FN Rate 14.06%** = Banyak spam yang terlewat

**CONFIDENCE DISTRIBUTION:**
```
Correct Predictions:
  Mean: 0.9234, Std: 0.0567
  Range: [0.5012, 0.9998]

Incorrect Predictions:
  Mean: 0.4567, Std: 0.2134
  Range: [0.0123, 0.8901]
```

- **Correct predictions**: High confidence (0.92), stable
- **Incorrect predictions**: Lower confidence (0.46), more variable
- **Insight**: Model tidak confident saat error terjadi

#### Visualisasi error_analysis_summary.png

**Subplot 1 & 2: Confusion Matrices**
- Raw: Nilai absolute counts
- Normalized: Percentage per true label
- **Insight**: Melihat distribusi errors

**Subplot 3: Error Breakdown**
- Bar chart: TN, FP, FN, TP
- **Insight**: Mana kategori yang paling banyak error

**Subplot 4: Statistics Text Box**
- Summary semua metrics
- **Insight**: Quick reference

#### Visualisasi confidence_distribution.png (Optional)

**Subplot 1: Histogram - Correct vs Incorrect**
- X-axis: Model Confidence
- Y-axis: Frequency
- **Insight**: Melihat separation antara correct/incorrect

**Subplot 2: Histogram - FP vs FN**
- Distribusi confidence untuk error types
- **Insight**: Apakah FP/FN berbeda karakternya

**Subplot 3: Boxplot**
- Median, Q1, Q3 untuk setiap kategori
- **Insight**: Outliers dan distribution spread

**Subplot 4: Cumulative Distribution**
- S-curve untuk setiap kategori
- **Insight**: Melihat threshold selection

### Praktik Terbaik

#### 1. Prioritas Minimasi Error Type

```
HAM vs SPAM classification:

FP (Ham→Spam):
- User experience terganggu (pesan penting hilang)
- Business impact: Reputasi buruk
- Prioritas: CRITICAL

FN (Spam→Ham):
- Inbox noise (tapi pesan masih bisa dilihat)
- Business impact: Lebih acceptable
- Prioritas: MEDIUM
```

#### 2. Confidence Threshold Tuning

Default threshold = 0.5

```python
# Adjust untuk minimize FP (prioritas accuracy di Ham)
threshold = 0.6  # Model harus lebih confident sebelum classify sebagai Spam

# Atau adjust untuk minimize FN (prioritas recall spam)
threshold = 0.4  # Model lebih aggressive dalam detect spam
```

#### 3. Worst Predictions Analysis

```
Highest confidence WRONG predictions = model bug

Contoh:
- Confidence: 0.95, True: Ham, Pred: Spam
- Message: "Hello my friend..."
- Problem: Model confident tapi salah → perlu investigate

Action:
1. Lihat text preprocessing
2. Check tokenization
3. Pertimbangkan data augmentation
```

### Contoh Workflow Lengkap

```bash
# 1. Pastikan training sudah selesai
python train_lstm.py

# 2. Evaluasi model di test set
python evaluate.py

# 3. Jalankan hyperparameter tuning
python hyperparameter_tuning.py --method random --n_iter 30 --epochs 10
# Tunggu completion, lihat results/tuning/tuning_summary.json untuk best params

# 4. Jalankan error analysis dengan confidence analysis
python error_analysis.py \
  --model results/models/lstm.h5 \
  --analyze_confidence \
  --show_samples 25

# 5. Review hasil
# - Buka results/error_analysis/error_report.txt
# - Lihat results/error_analysis/error_analysis_summary.png
# - Lihat results/error_analysis/confidence_distribution.png

# 6. Identifikasi pattern error
# - FP dominan atau FN dominan?
# - Error terjadi di confidence threshold tertentu?
# - Ada message pattern yang sering error?

# 7. (Optional) Retrain dengan best hyperparameters dari tuning
```

---

## 📊 STRUCTURE LENGKAP

```
KecerdasanBuatan_Proyek/
├── README.md                           # Main documentation
├── TUNING_ERROR_ANALYSIS.md           # File ini!
├── requirements.txt
├── spam.csv                            # Dataset original
│
├── src/
│   ├── eda_preprocessing.py           # ✓ (Tahap 2)
│   ├── train_lstm.py                  # ✓ (Tahap 3)
│   ├── evaluate.py                    # ✓ (Tahap 4)
│   ├── hyperparameter_tuning.py       # ✓ (Tahap 5 - NEW)
│   └── error_analysis.py              # ✓ (Tahap 5 - NEW)
│
├── hyperparameter_tuning.py           # Wrapper (root)
├── error_analysis.py                  # Wrapper (root)
│
├── results/
│   ├── models/
│   │   ├── lstm.h5
│   │   └── simple_rnn.h5
│   │
│   ├── evaluation/
│   │   ├── lstm/
│   │   │   ├── metrics.json
│   │   │   ├── classification_report.txt
│   │   │   ├── confusion_matrix.png
│   │   │   └── roc_curve.png
│   │   └── simple_rnn/
│   │
│   ├── tuning/                        # NEW
│   │   ├── tuning_results.csv
│   │   ├── tuning_summary.json
│   │   └── tuning_analysis.png
│   │
│   └── error_analysis/                # NEW
│       ├── error_details.csv
│       ├── error_report.txt
│       ├── error_summary.json
│       ├── error_analysis_summary.png
│       └── confidence_distribution.png
│
└── docs/
    └── (untuk dokumentasi tambahan)
```

---

## 🎯 RINGKASAN STAGE 5 (Fine-tuning & Optimisasi)

### Deliverables

✅ **Hyperparameter Tuning Module**
- Grid Search (36 trials)
- Random Search (customizable)
- K-Fold Cross-Validation
- Comprehensive visualization

✅ **Error Analysis Module**
- Detailed error categorization
- Confidence distribution analysis
- Worst predictions identification
- Visual error breakdown

### Metrik untuk Evaluasi

| Metrik | Target | Aksi jika tidak tercapai |
|--------|--------|------------------------|
| Test F1 Score | > 0.85 | Adjust hyperparameters, collect more data |
| FP Rate | < 5% | Increase confidence threshold |
| FN Rate | < 15% | Decrease confidence threshold, improve model |
| Balanced Accuracy | > 0.80 | Address class imbalance, tune dropout |
| ROC-AUC | > 0.90 | Feature engineering, model architecture changes |

---

## 📝 NEXT STEPS (Tahap 6: Finalisasi)

Setelah menyelesaikan tuning & error analysis:

1. **Retrain** dengan best hyperparameters
2. **Final Evaluation** pada test set
3. **Prepare Laporan** lengkap dengan hasil tuning
4. **Presentasi** findings ke group

---

## 💡 TIPS & TROUBLESHOOTING

### Q: Grid Search/Random Search terlalu lama?
A: 
- Reduce `--epochs` untuk faster training
- Reduce parameter space (fewer values to test)
- Use Random Search instead (lebih cepat)

### Q: Output plots tidak muncul?
A: Pastikan matplotlib backend sudah correct
```python
# Jika perlu, tambah ini di awal:
import matplotlib
matplotlib.use('Agg')  # Headless backend
```

### Q: Error "Model not found"?
A: Pastikan training sudah selesai:
```bash
python train_lstm.py  # Generate model dulu
python error_analysis.py --model results/models/lstm.h5
```

### Q: Output directory permission denied?
A: Pastikan `results/` folder writeable:
```bash
chmod -R 755 results/
```

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-24  
**Status**: ✅ Complete for Tahap 5
