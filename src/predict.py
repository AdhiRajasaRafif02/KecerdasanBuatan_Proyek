import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from eda_preprocessing import load_and_preprocess_data, clean_text

def main():
    print("\n[INFO] Memuat dataset dan Tokenizer... (Mohon tunggu sebentar, butuh waktu sekitar 2 detik)")
    try:
        # Kita hanya butuh tokenizer-nya
        _, _, _, _, tokenizer, _ = load_and_preprocess_data(
            csv_path="spam.csv", max_words=5000, max_len=100, test_size=0.2, random_state=42
        )
    except Exception as e:
        print(f"[ERROR] Gagal memuat dataset: {e}")
        return

    model_path = 'models/best_lstm_model.keras'
    print(f"[INFO] Memuat model AI canggih (Bidirectional LSTM) dari {model_path}...")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"[ERROR] Gagal memuat model: {e}")
        return

    print("\n" + "="*60)
    print("      SMS SPAM DETECTOR - KELOMPOK 7      ")
    print("="*60)
    print("Ketik 'exit' atau 'quit' untuk berhenti.\n")

    while True:
        try:
            text = input("\nMasukkan pesan SMS: ")
            if text.strip().lower() in ['exit', 'quit']:
                print("Terima kasih telah mencoba! Sampai jumpa.")
                break
            if not text.strip():
                continue

            # Preprocessing text
            cleaned = clean_text(text)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
            
            # Predict
            prob = model.predict(padded, verbose=0)[0][0]
            prediction = "SPAM" if prob >= 0.5 else "HAM (Aman)"
            
            print("\n" + "-" * 50)
            print(f"Hasil Analisis AI : {prediction}")
            if prob >= 0.5:
                print(f"Tingkat Keyakinan : {prob:.2%} bahwa ini adalah Spam")
            else:
                print(f"Tingkat Keyakinan : {(1-prob):.2%} bahwa ini adalah pesan Sah (Ham)")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nTerima kasih telah mencoba! Sampai jumpa.")
            break
        except Exception as e:
            print(f"Terjadi kesalahan: {e}")

if __name__ == '__main__':
    main()
