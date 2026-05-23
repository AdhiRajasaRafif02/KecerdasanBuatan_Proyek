# K07 - SMS Spam Classification using LSTM

## Anggota Kelompok
- Adhi Rajasa Rafif - 2306266943
- Fido Wahyu Choirulinsan - 2306250674
- Gede Rama Pradnya Widada - 2306161914
- Muhammad Rifat Faqih - 2306250762

Proyek ini bertujuan untuk mengklasifikasikan pesan SMS menjadi spam atau ham menggunakan model Recurrent Neural Network, khususnya LSTM.

## Dataset
Dataset yang digunakan adalah SMS Spam Collection Dataset dari Kaggle. Dataset berisi pesan SMS dengan label ham atau spam.

## Model
- Model utama: LSTM
- Baseline: Simple RNN

## Evaluasi
Metrik evaluasi yang digunakan:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Tambahan metrik yang juga dihitung pada tahap evaluasi (lebih informatif untuk dataset imbalanced):
- ROC-AUC
- PR-AUC (Average Precision)
- Specificity (True Negative Rate untuk kelas Ham)
- Balanced Accuracy
- MCC (Matthews Correlation Coefficient)

## Progress Saat Ini

Proyek saat ini sudah sampai pada tahap **Training Model** (Tahap 3 selesai) dengan persiapan siap memasuki tahap evaluasi.

### Tahapan yang Sudah Dilakukan:

#### 1. ✓ Pencarian dan Pemilihan Dataset
- Memilih SMS Spam Collection Dataset sebagai dataset utama
- Dataset berisi ~5.500+ pesan SMS dengan 2 kolom (label dan message)
- Distribusi label: ~87% ham, ~13% spam

#### 2. ✓ Exploratory Data Analysis (EDA)
- Membaca dataset menggunakan Pandas dengan encoding `latin-1`
- Mengecek struktur dataset dan jumlah data
- Analisis distribusi label spam vs ham
- Pengecekan missing values

#### 3. ✓ Pra-pemrosesan Data (Data Preprocessing)
- **Text Cleaning:**
  - Mengubah teks menjadi lowercase
  - Menghapus URLs dan email addresses
  - Menghapus angka dan simbol khusus
  - Menghapus spasi berlebih

- **Label Encoding:**
  - Mengubah label 'ham' dan 'spam' menjadi format numerik (0 dan 1)
  - Menggunakan LabelEncoder dari scikit-learn

- **Train-Test Split:**
  - Membagi data menjadi training (80%) dan testing (20%)
  - Menggunakan stratifikasi untuk menjaga distribusi label
  - Random state = 42 untuk reproducibility

- **Tokenisasi dan Padding:**
  - Tokenisasi teks menggunakan TensorFlow Keras Tokenizer
  - Vocabulary size: 5000 kata paling sering
  - OOV token: `<OOV>` untuk kata yang tidak dikenal
  - Padding sequence ke panjang 100 dengan metode 'post'

### Tahapan yang Akan Datang:

#### 4. ✓ Model Development (Selesai)
- **Baseline Model:** Simple RNN (Telah di-train dengan akurasi validasi ~13%)
- **Main Model:** LSTM (Telah di-train dengan akurasi validasi ~86%)
- **Class Weights:** Diterapkan untuk mengatasi imbalance data Spam

#### 5. ✓ Training (Selesai)
- Training dengan validation set (10% split)
- Early stopping & Model Checkpoint diterapkan
- Model berhasil disimpan di `results/models/lstm.h5` dan `results/models/simple_rnn.h5`

**Hasil Akurasi Validasi (Tahap Training):**
- **LSTM (Main Model):** ~86.55%
- **Simple RNN (Baseline):** ~13.45%

> **PENTING:** Angka di atas baru merupakan metrik evaluasi internal saat proses *training*. **Langkah wajib selanjutnya** adalah membawa model ini ke Tahap Evaluasi untuk diuji menggunakan *Test Set* agar mendapatkan nilai performa asli (Akurasi, Presisi, Recall, F1-Score).

#### 6. Evaluasi (Sedang Dikerjakan)
- Prediksi pada test set
- Perhitungan metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - AUC-ROC
- Confusion Matrix visualization
- Classification Report

---

## Progress Integrasi Preprocessing ✓

Preprocessing dan EDA sudah diintegrasikan dalam modul `src/eda_preprocessing.py`. Modul ini bersifat:

### Fungsi-Fungsi Preprocessing yang Tersedia:

#### 1. `clean_text(text)` 
Membersihkan teks SMS dengan:
- Lowercase conversion
- Remove URLs (http, https, www)
- Remove angka dan simbol khusus
- Remove spasi berlebih
- Hanya menyimpan huruf dan spasi

#### 2. `show_basic_eda(df)`
Menampilkan exploratory data analysis:
- 5 data pertama
- Shape dataset (rows × columns)
- Distribusi label dengan persentase
- Missing values check
- Label encoding mapping

#### 3. `save_cleaned_dataset(df, output_path="results/cleaned_spam.csv")`
Menyimpan dataset yang dibersihkan:
- Auto-create `results/` folder
- Save ke CSV format dengan 4 columns: label, message, clean_message, label_encoded

#### 4. `load_and_preprocess_data(csv_path, max_words=5000, max_len=100, test_size=0.2, random_state=42)`
Complete preprocessing pipeline lengkap:
1. Load CSV dengan encoding latin-1
2. Select & rename kolom (v1→label, v2→message)
3. Text cleaning
4. Label encoding (ham→0, spam→1)
5. Train-test split dengan stratifikasi (80:20)
6. Tokenization (Keras Tokenizer, oov_token="<OOV>", num_words=5000)
7. Padding (max_len=100, padding='post', truncating='post')

**Return Values:**
```python
X_train_pad, X_test_pad, y_train, y_test, tokenizer, label_encoder
```

### Data Statistics (setelah preprocessing):

| Item | Value |
|------|-------|
| Total Samples | 5,572 |
| Training Set | 4,457 (80.0%) |
| Test Set | 1,115 (20.0%) |
| Ham Samples | 4,825 (86.59%) |
| Spam Samples | 747 (13.41%) |
| X_train_pad Shape | (4457, 100) |
| X_test_pad Shape | (1115, 100) |
| Vocabulary Size | 7,492 |
| Max Sequence Length | 100 |

### Output Files:
- ✅ **results/cleaned_spam.csv** (903 KB) - Dataset yang sudah dibersihkan

---

## Cara Menjalankan

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Menjalankan Preprocessing & EDA

```bash
# Dari root folder project
python src/eda_preprocessing.py
```

Output:
- Menampilkan 5 data pertama
- Menampilkan jumlah baris/kolom
- Menampilkan distribusi label (ham vs spam)
- Menampilkan missing value check
- Menampilkan label encoding mapping
- Menampilkan X_train_pad dan X_test_pad shapes
- Membuat file: `results/cleaned_spam.csv`
- Print status bahwa preprocessing completed

### 3. Menjalankan Training

```bash
# Menjalankan proses training model (dari root)
python train_lstm.py

# (Alternatif) langsung ke file di src/
python src/train_lstm.py
```

Output:
- Menampilkan arsitektur dan status data shape
- Menjalankan proses training iteratif Keras (fit)
- Menyimpan hasil model (`.h5`) ke dalam `results/models/`

### 4. Evaluation (setelah training selesai)

```bash
# Evaluasi 1 model (default: results/models/lstm.h5)
python evaluate.py

# Evaluasi model tertentu
python evaluate.py --model results/models/lstm.h5

# Compare dua model (yang ada saja): Simple RNN vs LSTM
python evaluate.py --compare

# Atur threshold prediksi spam (default 0.5)
python evaluate.py --model results/models/lstm.h5 --threshold 0.5

# (Alternatif) langsung ke file di src/
python src/evaluate.py --compare
```

Output:
- Menampilkan ringkasan metrik di terminal
- Menyimpan hasil evaluasi ke folder `results/evaluation/<nama_model>/`:
  - `metrics.json`
  - `classification_report.txt`
  - `confusion_matrix.png` dan `confusion_matrix_normalized.png`
  - `roc_curve.png` dan `pr_curve.png`

Catatan singkat arti metrik (label positif = Spam / 1):
- **Precision (Spam)**: dari semua prediksi spam, berapa yang benar spam.
- **Recall (Spam)**: dari semua spam sebenarnya, berapa yang berhasil terdeteksi.
- **F1 (Spam)**: rata-rata harmonik precision & recall.
- **Specificity (Ham)**: dari semua ham, berapa yang benar terdeteksi ham (menekan false positive).
- **ROC-AUC**: kualitas ranking probabilitas secara umum.
- **PR-AUC**: biasanya lebih relevan saat data imbalanced (spam minoritas).

---

### File dan Struktur Proyek

```
KecerdasanBuatan_Proyek/
├── README.md                       # Dokumentasi proyek (file ini)
├── requirements.txt                # Library dependencies
├── spam.csv                        # Dataset original
├── .gitignore                      # File Git ignore
│
├── src/
│   ├── eda_preprocessing.py        # ✓ EDA dan preprocessing (Selesai)
│   ├── train_lstm.py               # ✓ Model training (Selesai)
│   └── evaluate.py                 # ⚙ Skrip evaluasi (Tahap Selanjutnya)
│
├── notebooks/
│   └── sms_spam_lstm.ipynb         # Notebook untuk referensi (Opsional)
│
├── results/
│   ├── .gitkeep                    # Placeholder untuk hasil output
│   ├── cleaned_spam.csv            # ✓ Dataset yang sudah dibersihkan
│   └── models/                     # ✓ Berisi file lstm.h5 dan simple_rnn.h5
│
└── docs/
    └── .gitkeep                    # Placeholder untuk dokumentasi progress
```

### Tools dan Library yang Digunakan

- **Data Processing:** Pandas, NumPy
- **ML/Preprocessing:** Scikit-learn, TensorFlow/Keras
- **Visualization:** Matplotlib, Seaborn
- **Version Control:** Git

### Tahapan Pengerjaan Proyek

1. ✓ **Tahap 1:** Pencarian dataset dan Exploratory Data Analysis (Selesai)
2. ✓ **Tahap 2:** Preprocessing dan persiapan data (Selesai)
3. ✓ **Tahap 3:** Training model RNN dan LSTM (Selesai)
4. ✓ **Tahap 4:** Evaluasi dan analisis hasil (Sedang Dikerjakan)
5. **Tahap 5:** Fine-tuning dan optimisasi (Belum Dimulai)
6. **Tahap 6:** Finalisasi laporan dan presentasi (Belum Dimulai)
