import os
import json
import itertools
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score
)

from sklearn.utils.class_weight import compute_class_weight

from eda_preprocessing import load_and_preprocess_data

DATASET_PATH = 'spam.csv'
RESULTS_DIR = 'results'
TUNING_RESULTS_PATH = os.path.join(RESULTS_DIR, 'tuning_results.csv')
BEST_CONFIG_PATH = os.path.join(RESULTS_DIR, 'best_hyperparameters.json')

MAX_WORDS = 5000
MAX_LEN = 100
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print('Loading dataset and preprocessing...')

X_train_pad, X_test_pad, y_train, y_test, tokenizer, label_encoder = load_and_preprocess_data(
    csv_path=DATASET_PATH,
    max_words=MAX_WORDS,
    max_len=MAX_LEN,
    test_size=0.2,
    random_state=RANDOM_STATE
)

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {
    0: class_weights_array[0],
    1: class_weights_array[1]
}

print('Class weights:')
print(class_weights)

def build_lstm_model(
    embedding_dim=64,
    lstm_units=64,
    dropout_rate=0.3,
    learning_rate=0.001,
    optimizer_name='adam'
):

    model = Sequential([
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=embedding_dim,
            input_length=MAX_LEN
        ),

        LSTM(
            lstm_units,
            return_sequences=False
        ),

        Dropout(dropout_rate),

        Dense(1, activation='sigmoid')
    ])

    if optimizer_name.lower() == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

search_space = {
    'embedding_dim': [32, 64],
    'lstm_units': [32, 64, 128],
    'dropout_rate': [0.2, 0.3],
    'batch_size': [16, 32],
    'learning_rate': [0.001, 0.0005],
    'epochs': [10, 15],
    'optimizer_name': ['adam']
}

all_combinations = list(itertools.product(
    search_space['embedding_dim'],
    search_space['lstm_units'],
    search_space['dropout_rate'],
    search_space['batch_size'],
    search_space['learning_rate'],
    search_space['epochs'],
    search_space['optimizer_name']
))

print(f'Total experiments: {len(all_combinations)}')

results = []
best_f1 = 0
best_configuration = None
best_model = None

experiment_counter = 1

for (
    embedding_dim,
    lstm_units,
    dropout_rate,
    batch_size,
    learning_rate,
    epochs,
    optimizer_name
) in all_combinations:

    print(f'\nExperiment {experiment_counter}/{len(all_combinations)}')

    model = build_lstm_model(
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train_pad,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1
    )

    y_prob = model.predict(X_test_pad, verbose=0)
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    specificity = tn / (tn + fp)

    val_accuracy = max(history.history['val_accuracy'])
    val_loss = min(history.history['val_loss'])

    experiment_result = {
        'embedding_dim': embedding_dim,
        'lstm_units': lstm_units,
        'dropout_rate': dropout_rate,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'optimizer': optimizer_name,
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'specificity': round(specificity, 4),
        'balanced_accuracy': round(balanced_acc, 4),
        'mcc': round(mcc, 4),
        'validation_accuracy': round(val_accuracy, 4),
        'validation_loss': round(val_loss, 4)
    }

    results.append(experiment_result)

    print(f'Accuracy     : {accuracy:.4f}')
    print(f'Precision    : {precision:.4f}')
    print(f'Recall       : {recall:.4f}')
    print(f'F1-Score     : {f1:.4f}')

    if f1 > best_f1:
        best_f1 = f1
        best_configuration = experiment_result
        best_model = model

    experiment_counter += 1

os.makedirs(RESULTS_DIR, exist_ok=True)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='f1_score', ascending=False)

results_df.to_csv(TUNING_RESULTS_PATH, index=False)

with open(BEST_CONFIG_PATH, 'w') as f:
    json.dump(best_configuration, f, indent=4)

BEST_MODEL_PATH = 'results/models/lstm_tuned.h5'

best_model.save(BEST_MODEL_PATH)

print('Hyperparameter tuning completed')
print(results_df.head())

print('Best configuration:')
print(json.dumps(best_configuration, indent=4))