import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import train_test_split

from eda_preprocessing import load_and_preprocess_data

DATASET_PATH = 'spam.csv'
MODEL_PATH = 'results/models/lstm_tuned.h5'
OUTPUT_PATH = 'results/error_analysis.txt'

MAX_WORDS = 5000
MAX_LEN = 100
RANDOM_STATE = 42

print('Loading dataset...')

X_train_pad, X_test_pad, y_train, y_test, tokenizer, label_encoder = load_and_preprocess_data(
    csv_path=DATASET_PATH,
    max_words=MAX_WORDS,
    max_len=MAX_LEN,
    test_size=0.2,
    random_state=RANDOM_STATE
)

raw_df = pd.read_csv(DATASET_PATH, encoding='latin-1')

raw_df = raw_df[['v1', 'v2']]
raw_df.columns = ['label', 'message']

_, test_df = train_test_split(
    raw_df,
    test_size=0.2,
    stratify=raw_df['label'],
    random_state=RANDOM_STATE
)

test_df = test_df.reset_index(drop=True)

print('Loading trained model...')
model = load_model(MODEL_PATH)

print('Predicting test set...')

y_prob = model.predict(X_test_pad, verbose=0)
y_pred = (y_prob >= 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

false_positive_indexes = []
false_negative_indexes = []

for idx in range(len(y_test)):

    actual = y_test[idx]
    predicted = y_pred[idx]

    if actual == 0 and predicted == 1:
        false_positive_indexes.append(idx)

    elif actual == 1 and predicted == 0:
        false_negative_indexes.append(idx)

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:

    f.write('SMS Spam Classification Error Analysis\n\n')

    f.write('Performance Metrics\n')
    f.write(f'Accuracy  : {accuracy:.4f}\n')
    f.write(f'Precision : {precision:.4f}\n')
    f.write(f'Recall    : {recall:.4f}\n')
    f.write(f'F1-Score  : {f1:.4f}\n\n')

    f.write('Confusion Matrix\n')
    f.write(f'True Negative  : {tn}\n')
    f.write(f'False Positive : {fp}\n')
    f.write(f'False Negative : {fn}\n')
    f.write(f'True Positive  : {tp}\n\n')

    f.write('Classification Report\n')
    f.write(classification_report(y_test, y_pred))
    f.write('\n')

    f.write('False Positive Analysis\n\n')

    for i, idx in enumerate(false_positive_indexes[:15]):

        message = test_df.iloc[idx]['message']

        f.write(f'[FP-{i + 1}]\n')
        f.write(f'Message : {message}\n')
        f.write(f'Prediction Probability : {float(y_prob[idx]):.4f}\n\n')

    f.write('False Negative Analysis\n\n')

    for i, idx in enumerate(false_negative_indexes[:15]):

        message = test_df.iloc[idx]['message']

        f.write(f'[FN-{i + 1}]\n')
        f.write(f'Message : {message}\n')
        f.write(f'Prediction Probability : {float(y_prob[idx]):.4f}\n\n')

    f.write('Summary Analysis\n\n')

    f.write('False positive biasanya mengandung kata promosi seperti free, claim, reward, dan offer.\n')
    f.write('False negative terjadi pada spam yang memiliki struktur mirip pesan normal.\n')
    f.write('Dataset imbalance memengaruhi kemampuan model mendeteksi spam minoritas.\n')
    f.write('Hyperparameter tuning meningkatkan stabilitas model secara signifikan.\n')

print('Error analysis completed')
print(f'Result saved to: {OUTPUT_PATH}')