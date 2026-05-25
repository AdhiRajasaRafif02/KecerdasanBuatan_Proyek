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
- `results/models/lstm.h5`
- `results/models/simple_rnn.h5`

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

### Hasil Setelah Hyperparameter Tuning & Preprocessing Lanjutan

| Metric | Score |
|---|---|
| Accuracy | ~98.03% |
| Precision | ~91.50% |
| Recall | ~93.96% |
| F1-Score | ~92.72% |

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

# Analisis Etika, Bias, dan Keterbatasan Model

Dokumen ini disusun untuk mendokumentasikan analisis mendalam mengenai etika, potensi bias, dan edge cases dari model klasifikasi SMS Spam yang telah dibangun menggunakan Bidirectional LSTM.

## 1. Potensi Bias pada Model

Walaupun dataset (SMS Spam Collection) memiliki representasi yang cukup baik, terdapat beberapa potensi bias bawaan:
- **Bias Kelas Mayoritas (Imbalance):** Dataset didominasi oleh pesan `ham` (~87%). Meskipun kita telah mengatasinya menggunakan parameter `class_weight='balanced'` selama *training*, model secara inheren memiliki riwayat melihat jauh lebih banyak teks normal.
- **Bias Bahasa/Konteks:** Dataset ini berbasis bahasa Inggris dengan konteks slang dan tren promosi luar negeri (seperti penawaran dalam Pounds/Dollars). Model ini memiliki keterbatasan linguistik dan akan kehilangan efektivitasnya jika diterapkan pada pola spam lokal (seperti SMS penipuan *"Mama minta pulsa"* atau *"Pinjol cepat cair"* berbahasa Indonesia).

## 2. Analisis Kasus Kegagalan (Edge Cases)

Berikut adalah beberapa *edge cases* yang berpotensi membingungkan model (salah klasifikasi):

### A. False Positives (Ham ditandai sebagai Spam)
*False Positive* merupakan kegagalan yang paling tidak diinginkan karena pengguna bisa kehilangan pesan penting.
- **Notifikasi Transaksional Resmi:** Pesan OTP, notifikasi tagihan kartu kredit dari bank, atau promosi sah berlangganan dari provider telekomunikasi yang sering kali padat angka (nominal uang) dan seruan tindakan (`!`). Karena simbol dan angka dipertahankan di preprocessing kita, model mungkin bingung membedakan *"Your OTP is 4928. Do not share!"* dengan pesan spam jika panjang teksnya hampir sama.
- **Pesan Mendesak dari Kerabat:** Teks sah bergaya *urgent* seperti *"CALL ME NOW! URGENT!"* atau *"Where are u?! Reply immediately!"* bisa ditandai sebagai spam karena penggunaan kapital berlebih dan kata seruan yang umum dipakai *spammers*.

### B. False Negatives (Spam lolos sebagai Ham)
*False Negative* terjadi ketika SMS spam gagal terdeteksi oleh sistem.
- **Manipulasi Ejaan (Adversarial Text):** Spammers dapat memodifikasi huruf untuk menghindari deteksi (misal: ejaan "FR33" alih-alih "FREE", atau menyisipkan banyak titik di antara huruf).
- **Spam Pendekatan Personal (Social Engineering):** Penipuan modern seringkali meniru gaya percakapan santai (contoh: *"Hey, are you free tomorrow? Can you help me out?"*). Pesan semacam ini tidak memiliki *vocabulary* khas spam (seperti *winner, cash, claim*) sehingga model sangat mungkin mengklasifikasikannya sebagai pesan `ham`.

## 3. Langkah Mitigasi yang Telah Diimplementasikan

Dalam proyek ini, telah dilakukan beberapa upaya iteratif untuk meminimalisir kesalahan klasifikasi tersebut:
1. **Pemisahan Entitas Angka:** Pada modul `eda_preprocessing.py`, angka tidak dihapus secara membabi buta, melainkan direpresentasikan menjadi token standar `<NUM>`. Hal ini mencegah *vocabulary size* meledak namun tetap memberi *clue* ke model tentang kehadiran angka-angka penting (misal: nomor telepon untuk dihubungi).
2. **Mempertahankan Simbol Indikator:** Simbol `!`, `?`, dan `$` diselamatkan dari proses Regex karena kemunculan berlebihan dari simbol-simbol ini adalah ciri kuat manipulasi emosi (urgensi atau godaan finansial) dari para penipu.
3. **Arsitektur Bidirectional:** Perpindahan dari Simple RNN/LSTM biasa menuju Bidirectional LSTM membekali model kemampuan membaca frasa dari dua sisi. Ini membantu membedakan makna kontekstual kalimat utuh ketimbang hanya mencocokkan *keyword*.

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

1. Tahap 1: Pencarian dataset dan EDA
2. Tahap 2: Preprocessing dan persiapan data
3. Tahap 3: Training model RNN dan LSTM
4. Tahap 4: Evaluasi dan analisis hasil
5. Tahap 5: Fine-tuning dan optimisasi model
