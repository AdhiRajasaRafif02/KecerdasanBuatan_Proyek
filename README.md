# K07 - SMS Spam Classification using LSTM

## Anggota Kelompok
- Adhi Rajasa Rafif - 2306266943
- Fido Wahyu Choirulinsan - 2306250674
- Gede Rama Pradnya Widada - 2306161914
- Muhammad Rifat Faqih - 2306250762

Proyek ini bertujuan untuk mengklasifikasikan pesan SMS menjadi spam atau ham menggunakan model Recurrent Neural Network, khususnya LSTM.

---

# Dataset

Dataset yang digunakan adalah SMS Spam Collection Dataset dari Kaggle. Dataset berisi pesan SMS dengan label ham atau spam.

## Distribusi Dataset
- Ham ≈ 87%
- Spam ≈ 13%

Dataset terdiri dari:
- ±5.500 pesan SMS
- 2 kolom utama:
  - label
  - message

---

# Model

Model yang digunakan pada proyek ini:

- Model utama: LSTM
- Baseline model: Simple RNN

---

# Evaluasi

Metrik evaluasi yang digunakan:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Tambahan metrik evaluasi:
- ROC-AUC
- PR-AUC
- Specificity
- Balanced Accuracy
- MCC (Matthews Correlation Coefficient)

---

# Progress Saat Ini

Proyek saat ini telah menyelesaikan seluruh tahapan utama machine learning pipeline mulai dari preprocessing, training, evaluasi, hingga hyperparameter tuning dan error analysis.

---

# Tahapan yang Sudah Dilakukan

## 1. ✓ Pencarian dan Pemilihan Dataset
- Memilih SMS Spam Collection Dataset sebagai dataset utama
- Dataset berisi ±5.500 pesan SMS
- Distribusi label:
  - Ham ≈ 87%
  - Spam ≈ 13%

---

## 2. ✓ Exploratory Data Analysis (EDA)

Tahapan EDA yang dilakukan:
- Membaca dataset menggunakan Pandas
- Analisis distribusi label spam vs ham
- Pengecekan missing values
- Analisis struktur dataset

---

## 3. ✓ Pra-pemrosesan Data (Data Preprocessing)

### Text Cleaning
- Mengubah teks menjadi lowercase
- Menghapus URL dan email
- Menghapus angka
- Menghapus simbol khusus
- Menghapus spasi berlebih

### Label Encoding
- ham → 0
- spam → 1

### Train-Test Split
- Training set = 80%
- Testing set = 20%
- Stratified split
- random_state = 42

### Tokenization dan Padding
- TensorFlow Keras Tokenizer
- Vocabulary size = 5000
- OOV token = `<OOV>`
- Max sequence length = 100

---

## 4. ✓ Model Development dan Training

### Baseline Model
Simple RNN digunakan sebagai baseline model.

### Main Model
LSTM digunakan sebagai model utama klasifikasi SMS spam.

### Konfigurasi Training
- Validation split = 10%
- EarlyStopping
- ModelCheckpoint
- Class Weights untuk mengatasi imbalance dataset

### Hasil Training

| Model | Validation Accuracy |
|---|---|
| Simple RNN | ~13.45% |
| LSTM | ~86.55% |

### Output Model
Model yang berhasil disimpan:
- `results/models/lstm.h5`
- `results/models/simple_rnn.h5`

---

## 5. ✓ Evaluasi Model

Evaluasi dilakukan menggunakan test set untuk memperoleh performa aktual model.

### Metrics Evaluasi
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Specificity
- Balanced Accuracy
- MCC

### Visualisasi Evaluasi
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Classification Report

### Output Evaluasi
Folder:
```text
results/evaluation/
```

Berisi:
- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `roc_curve.png`
- `pr_curve.png`

---

## 6. ✓ Hyperparameter Tuning dan Error Analysis

Tahap optimisasi model dilakukan menggunakan hyperparameter tuning pada model LSTM untuk meningkatkan performa klasifikasi SMS spam.

### Hyperparameter yang Diuji
- Embedding Dimension
- LSTM Units
- Dropout Rate
- Batch Size
- Learning Rate
- Epoch
- Optimizer

### Konfigurasi Hyperparameter Terbaik

| Hyperparameter | Value |
|---|---|
| Embedding Dimension | 64 |
| LSTM Units | 128 |
| Dropout Rate | 0.3 |
| Batch Size | 16 |
| Learning Rate | 0.0005 |
| Epoch | 15 |
| Optimizer | Adam |

### Hasil Setelah Hyperparameter Tuning

| Metric | Score |
|---|---|
| Accuracy | ~91% |
| Precision | ~90% |
| Recall | ~89% |
| F1-Score | ~89% |
| ROC-AUC | ~0.94 |

### Error Analysis
Analisis dilakukan terhadap:
- False Positive
- False Negative
- Kesalahan klasifikasi spam
- Dampak imbalance dataset

### Output Hyperparameter Tuning
```text
results/tuning_results.csv
```

---

# Progress Integrasi Preprocessing

Preprocessing dan EDA telah diintegrasikan dalam modul:
```text
src/eda_preprocessing.py
```

## Fungsi Preprocessing yang Tersedia

### 1. `clean_text(text)`
Membersihkan teks SMS dengan:
- Lowercase conversion
- Remove URLs
- Remove angka dan simbol khusus
- Remove spasi berlebih
- Menyimpan hanya huruf dan spasi

### 2. `show_basic_eda(df)`
Menampilkan:
- 5 data pertama
- Shape dataset
- Distribusi label
- Missing values
- Label encoding mapping

### 3. `save_cleaned_dataset(df, output_path="results/cleaned_spam.csv")`
Menyimpan dataset hasil preprocessing ke folder `results/`.

### 4. `load_and_preprocess_data(...)`
Pipeline preprocessing lengkap:
1. Load CSV
2. Rename kolom
3. Text cleaning
4. Label encoding
5. Train-test split
6. Tokenization
7. Padding

### Return Values
```python
X_train_pad, X_test_pad, y_train, y_test, tokenizer, label_encoder
```

---

# Data Statistics

| Item | Value |
|---|---|
| Total Samples | 5,572 |
| Training Set | 4,457 |
| Test Set | 1,115 |
| Ham Samples | 4,825 |
| Spam Samples | 747 |
| X_train_pad Shape | (4457, 100) |
| X_test_pad Shape | (1115, 100) |
| Vocabulary Size | 7,492 |
| Max Sequence Length | 100 |

---

# Cara Menjalankan

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Menjalankan Preprocessing dan EDA

```bash
python src/eda_preprocessing.py
```

---

## 3. Menjalankan Training

```bash
python train_lstm.py

python src/train_lstm.py
```

---

## 4. Evaluasi Model

```bash
python evaluate.py

python evaluate.py --model results/models/lstm.h5

python evaluate.py --compare

python evaluate.py --model results/models/lstm.h5 --threshold 0.5

python src/evaluate.py --compare
```

---

## 5. Hyperparameter Tuning

```bash
python src/hyperparameter_tuning.py
```

### Output
```text
results/tuning_results.csv
```

---

## 6. Error Analysis

```bash
python src/error_analysis.py
```

---

# Struktur Proyek

```text
KecerdasanBuatan_Proyek/
├── README.md
├── requirements.txt
├── spam.csv
├── .gitignore
│
├── src/
│   ├── eda_preprocessing.py
│   ├── train_lstm.py
│   ├── evaluate.py
│   ├── hyperparameter_tuning.py
│   └── error_analysis.py
│
├── notebooks/
│   └── sms_spam_lstm.ipynb
│
├── results/
│   ├── .gitkeep
│   ├── cleaned_spam.csv
│   ├── tuning_results.csv
│   ├── evaluation/
│   │   ├── lstm/
│   │   └── simple_rnn/
│   │
│   └── models/
│       ├── lstm.h5
│       └── simple_rnn.h5
│
└── docs/
    └── .gitkeep
```

---

# Tools dan Library yang Digunakan

- Pandas
- NumPy
- TensorFlow / Keras
- Scikit-learn
- Matplotlib
- Seaborn
- Git

---

# Tahapan Pengerjaan Proyek

1. ✓ Tahap 1: Pencarian dataset dan EDA
2. ✓ Tahap 2: Preprocessing dan persiapan data
3. ✓ Tahap 3: Training model RNN dan LSTM
4. ✓ Tahap 4: Evaluasi dan analisis hasil
5. ✓ Tahap 5: Fine-tuning dan optimisasi model
6. Tahap 6: Finalisasi laporan dan presentasi
