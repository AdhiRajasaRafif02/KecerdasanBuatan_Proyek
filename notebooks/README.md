# Notebooks - SMS Spam Classification

## Deskripsi

Folder ini akan berisi Jupyter notebook untuk SMS Spam Classification menggunakan RNN/LSTM. 

Saat ini, preprocessing dilakukan melalui file Python di folder `src/` agar lebih mudah dijalankan dan di-maintain.

## Outline Notebook (Jika Dibuat)

Jika nantinya ingin membuat notebook, berikut adalah outline yang dapat digunakan:

1. **Import Library**
   - Pandas, NumPy, Scikit-learn
   - TensorFlow/Keras
   - Matplotlib, Seaborn

2. **Load Dataset**
   - Membaca spam.csv
   - Menampilkan informasi dataset

3. **EDA (Exploratory Data Analysis)**
   - Distribusi label
   - Panjang pesan
   - Word frequency analysis

4. **Preprocessing Teks**
   - Text cleaning (lowercase, remove special chars, URLs, etc)
   - Handling missing values

5. **Tokenisasi dan Padding**
   - Tokenisasi teks menggunakan Keras Tokenizer
   - Padding sequence ke panjang yang sama

6. **Split Train-Test**
   - Train-test split dengan stratifikasi
   - Menampilkan distribusi setelah split

7. **Model Baseline - Simple RNN**
   - Membangun model Simple RNN
   - Training model
   - Melihat training history

8. **Model Utama - LSTM**
   - Membangun model LSTM (single/stacked)
   - Training dengan early stopping
   - Visualisasi training history

9. **Training Model**
   - Comparison antara Simple RNN dan LSTM
   - Analisis performa

10. **Evaluasi Model**
    - Prediksi pada test set
    - Menghitung metrics: Accuracy, Precision, Recall, F1-score
    - Confusion Matrix visualization
    - Classification Report

11. **Kesimpulan Awal**
    - Model performance summary
    - Insights dan findings
    - Recommendations untuk improvements

## Catatan

- Saat ini (Progress Tahap 2), preprocessing dilakukan melalui `src/eda_preprocessing.py`
- Untuk menjalankan preprocessing: `python src/eda_preprocessing.py`
- Hasil preprocessing disimpan di `results/cleaned_spam.csv`
- Tahap training akan dikerjakan di tahap berikutnya
