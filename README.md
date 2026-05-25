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

## 1. Pencarian dan Pemilihan Dataset
- Memilih SMS Spam Collection Dataset sebagai dataset utama
- Dataset berisi ±5.500 pesan SMS
- Distribusi label:
  - Ham ≈ 87%
  - Spam ≈ 13%

---

## 2. Exploratory Data Analysis (EDA)

Tahapan EDA yang dilakukan:
- Membaca dataset menggunakan Pandas
- Analisis distribusi label spam vs ham
- Pengecekan missing values
- Analisis struktur dataset

---

## 3. Pra-pemrosesan Data (Data Preprocessing)

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

## 4. Model Development dan Training

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

| Model | Test Accuracy | Test F1-Score |
|---|---|---|
| Simple RNN | ~85.65% | ~57.89% |
| Bidirectional LSTM | ~98.03% | ~92.72% |

### Output Model
Model yang berhasil disimpan:
- `models/best_lstm_model.keras` (Model Utama / Juara)
- `models/simple_rnn_model.h5` (Baseline Model)

---

## 5. Evaluasi Model

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
Folder `results/` berisi file-file pembuktian berikut:
- `evaluation_metrics.csv`
- `lstm_classification_report.txt`
- `lstm_confusion_matrix.png`
- `roc_curve.png`
- `baseline_vs_lstm.png`

---

## 6. Hyperparameter Tuning dan Error Analysis

Tahap optimisasi model dilakukan menggunakan hyperparameter tuning pada model LSTM untuk meningkatkan performa klasifikasi SMS spam.

### Hyperparameter yang Diuji
- Embedding Dimension
- LSTM Units
- Dropout Rate
- Batch Size
- Learning Rate
- Epoch
- Optimizer

### Rangkuman 8 Eksperimen (Hyperparameter Tuning)
Untuk mendapatkan performa maksimal, kami menjalankan 8 skenario (runs) yang menguji berbagai kombinasi arsitektur:
1. **Run 1 (Baseline BiLSTM):** Pengaturan standar (64 unit, dropout 0.5, lr 0.001).
2. **Run 2 & 7 (Kapasitas Besar):** Menaikkan unit LSTM menjadi 128. Kapasitas besar terbukti mampu menangkap konteks lebih baik.
3. **Run 3 & 8 (Variasi Dropout):** Bereksperimen dengan tingkat dropout yang lebih rendah (0.3 dan 0.4).
4. **Run 4 & 5 (Variasi Learning Rate):** Mengubah kecepatan konvergensi. Memperlambat LR (0.0001) membuat skor anjlok, sedangkan mempercepat LR (0.01) menghasilkan performa puncak.
5. **Run 6 (Small Batch):** Mengurangi *batch size* menjadi 16 untuk melihat efek pada gradien.

**Skenario Terbaik:** Eksperimen ke-4 (**Run 4_high_lr**) keluar sebagai pemenang absolut dengan menembus **F1-Score 94.56%**. Model juara ini disimpan sebagai `best_lstm_model.keras`.

### Konfigurasi Hyperparameter Terbaik (Run 4)

| Hyperparameter | Value |
|---|---|
| Embedding Dimension | 128 |
| LSTM Units | 64 |
| Dropout Rate | 0.5 |
| Batch Size | 32 |
| Learning Rate | 0.01 |
| Epoch | 10 |
| Optimizer | Adam |

### Hasil Setelah Hyperparameter Tuning & Preprocessing Lanjutan

| Metric | Score |
|---|---|
| Accuracy | ~98.57% |
| Precision | ~95.86% |
| Recall | ~93.29% |
| F1-Score | ~94.56% |

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
python src/train_lstm.py
```

---

## 4. Evaluasi Model

```bash
python src/evaluate.py
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

## 7. Menjalankan Prediksi Interaktif (Demo)

Untuk menguji model secara langsung (mengetik teks sendiri), jalankan:
```bash
python src/predict.py
```

---

# Analisis Etika, Bias, dan Keterbatasan Model

Bagian ini membahas analisis tentang masalah etika, potensi bias data, dan kasus-kasus jebakan (edge cases) dari model LSTM yang sudah dibuat.

## 1. Potensi Bias pada Model

Walaupun dataset (SMS Spam Collection) memiliki representasi yang cukup baik, terdapat beberapa potensi bias bawaan:
- **Bias Kelas Mayoritas (Imbalance):** Data kita didominasi oleh pesan `ham` (~87%). Walaupun sudah diakali pakai parameter `class_weight='balanced'` waktu *training*, model tetap punya kecenderungan lebih hafal dengan pola teks normal karena jumlah datanya jauh lebih banyak.
- **Bias Bahasa/Konteks:** Dataset ini memakai bahasa Inggris dengan gaya bahasa gaul dan tren promosi luar negeri (seperti diskon dalam Pounds/Dollars). Tentu saja model ini bakal kebingungan kalau disuruh mendeteksi pola spam lokal Indonesia (seperti SMS penipuan *"Mama minta pulsa"* atau penawaran *"Pinjol"*).

## 2. Analisis Kasus Kegagalan (Edge Cases)

Berikut adalah beberapa *edge cases* yang berpotensi membingungkan model (salah klasifikasi):

### A. False Positives (Ham ditandai sebagai Spam)
*False Positive* adalah kesalahan paling fatal karena pengguna bisa kehilangan pesan yang sebenarnya penting.
- **Pesan Transaksi Resmi:** SMS berisi kode OTP, tagihan kartu kredit, atau info kuota dari Telkomsel/provider yang biasanya banyak memuat angka dan tanda seru (`!`). Karena simbol dan angka kita biarkan saat preprocessing, model bisa kebingungan membedakan *"Your OTP is 4928. Do not share!"* dengan pesan spam penipuan.
- **Pesan Darurat dari Keluarga:** Pesan asli yang ditulis buru-buru seperti *"CALL ME NOW! URGENT!"* bisa saja ditandai sebagai spam karena terdeteksi pakai huruf kapital semua dan tanda seru, persis seperti gaya tulisan para spammer.

### B. False Negatives (Spam lolos sebagai Ham)
*False Negative* terjadi kalau ada SMS spam yang lolos dari pantauan sistem.
- **Manipulasi Ejaan (Adversarial Text):** Spammer sering mengubah ejaan untuk mengelabui sistem (contoh: menulis "FR33" ganti "FREE", atau menaruh banyak titik di antara huruf).
- **Penipuan Gaya Akrab (Social Engineering):** Penipuan zaman sekarang sering sok akrab dan meniru gaya chat biasa (contoh: *"Hey, are you free tomorrow? Can you help me out?"*). Karena pesan seperti ini tidak pakai kata-kata jualan khas spam (seperti *winner, cash, claim*), model bisa dengan mudah ketipu dan mengiranya sebagai pesan `ham` biasa.

## 3. Cara Kami Mengatasi Masalah Tersebut

Di proyek ini, kelompok kami sudah mencoba beberapa cara untuk menekan angka error tersebut:
1. **Penanganan Angka Khusus:** Di file `eda_preprocessing.py`, kami tidak asal menghapus angka. Semua angka kami ganti jadi satu token khusus yaitu `<NUM>`. Cara ini bikin jumlah kosakata model tidak bengkak, tapi model tetap sadar kalau di pesan itu ada deretan nomor telepon atau nominal uang.
2. **Mempertahankan Tanda Baca Penting:** Simbol `!`, `?`, dan `$` kami kecualikan dari penghapusan teks. Kenapa? Karena spammer sangat suka pakai tanda seru atau simbol uang untuk memancing emosi korban.
3. **Pakai Model Bidirectional:** Kami sengaja upgrade dari RNN biasa ke Bidirectional LSTM supaya model bisa baca kalimat dari depan ke belakang dan sebaliknya. Ini sangat membantu model buat paham makna asli kalimat secara utuh, bukan sekadar nebak dari satu kata saja.

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
│   ├── error_analysis.py
│   └── predict.py
│
├── models/
│   ├── best_lstm_model.keras
│   └── simple_rnn_model.h5
│
├── notebooks/
│   └── sms_spam_lstm.ipynb
│
├── results/
│   ├── .gitkeep
│   ├── evaluation_metrics.csv
│   ├── error_analysis_lstm.txt
│   ├── error_analysis_lstm.csv
│   ├── lstm_classification_report.txt
│   ├── tuning_results.csv
│   ├── lstm_confusion_matrix.png
│   ├── roc_curve.png
│   ├── tuning_f1_score.png
│   ├── class_distribution.png
│   ├── top_words_spam_ham.png
│   ├── learning_curve.png
│   └── baseline_vs_lstm.png
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

1. Tahap 1: Pencarian dataset dan EDA
2. Tahap 2: Preprocessing dan persiapan data
3. Tahap 3: Training model RNN dan LSTM
4. Tahap 4: Evaluasi dan analisis hasil
5. Tahap 5: Fine-tuning dan optimisasi model
