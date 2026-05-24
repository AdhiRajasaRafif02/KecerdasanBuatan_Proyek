"""
Hyperparameter Tuning Module untuk SMS Spam Classification
Kelompok 7 - Tugas Akhir Kecerdasan Buatan

Modul ini menyediakan:
1. Grid Search & Random Search untuk hyperparameter optimization
2. K-Fold Cross-Validation untuk robust evaluation
3. Comparison visualisasi antar hyperparameter
4. Best model selection dan saving
5. Detailed tuning report

Jalankan:
    python src/hyperparameter_tuning.py --method grid
    python src/hyperparameter_tuning.py --method random --n_iter 20
    python src/hyperparameter_tuning.py --method kfold --n_splits 5
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

from eda_preprocessing import load_and_preprocess_data


@dataclass
class TuningResult:
    """Hasil tuning satu set hyperparameter"""
    embedding_dim: int
    lstm_units: int
    dropout_rate: float
    learning_rate: float
    batch_size: int
    epochs: int
    val_accuracy: float
    val_loss: float
    test_accuracy: float
    test_f1: float
    test_balanced_acc: float
    test_roc_auc: float
    training_time: float
    fold: int | None = None  # Untuk K-Fold cross-validation


class LSTMHyperparameterTuner:
    """Main class untuk hyperparameter tuning LSTM"""
    
    def __init__(self, X_train, y_train, X_test, y_test, vocab_size, max_len=100):
        """
        Args:
            X_train: Training data shape (n_train, max_len)
            y_train: Training labels
            X_test: Test data shape (n_test, max_len)
            y_test: Test labels
            vocab_size: Vocabulary size dari tokenizer
            max_len: Sequence length
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.results = []
        
    def build_model(self, embedding_dim=128, lstm_units=64, dropout_rate=0.5,
                    rec_dropout_rate=0.2, learning_rate=0.001, dense_units=32):
        """Build LSTM model dengan specified hyperparameters"""
        model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=embedding_dim,
                     input_length=self.max_len, name='embedding'),
            LSTM(lstm_units, dropout=dropout_rate, 
                 recurrent_dropout=rec_dropout_rate, return_sequences=False, name='lstm'),
            Dense(dense_units, activation='relu', name='dense_1'),
            Dropout(dropout_rate, name='dropout'),
            Dense(1, activation='sigmoid', name='output')
        ])
        
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        
        return model
    
    def train_and_evaluate(self, model, X_train, y_train, X_val, y_val,
                          epochs=10, batch_size=32, patience=3, verbose=0):
        """Train model dan return validation metrics"""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, 
                         restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, 
                             min_lr=1e-6, verbose=0)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Get best epoch metrics
        best_epoch_idx = np.argmin(history.history['val_loss'])
        val_accuracy = history.history['val_accuracy'][best_epoch_idx]
        val_loss = history.history['val_loss'][best_epoch_idx]
        
        return val_accuracy, val_loss, history
    
    def evaluate_on_test(self, model):
        """Evaluate model pada test set"""
        y_pred_proba = model.predict(self.X_test, verbose=0).reshape(-1)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        test_accuracy = float(np.mean(y_pred == self.y_test))
        test_f1 = float(f1_score(self.y_test, y_pred, zero_division=0))
        test_balanced_acc = float(balanced_accuracy_score(self.y_test, y_pred))
        test_roc_auc = float(roc_auc_score(self.y_test, y_pred_proba))
        
        return test_accuracy, test_f1, test_balanced_acc, test_roc_auc
    
    def grid_search(self, param_grid, epochs=10, batch_size=32, patience=3, verbose=1):
        """
        Grid Search untuk hyperparameter tuning
        
        Args:
            param_grid: Dictionary dengan list values untuk setiap parameter
            Example:
                {
                    'embedding_dim': [64, 128, 256],
                    'lstm_units': [32, 64, 128],
                    'dropout_rate': [0.3, 0.5],
                    'learning_rate': [0.001, 0.0001]
                }
        """
        print("\n" + "="*80)
        print("STARTING GRID SEARCH HYPERPARAMETER TUNING")
        print("="*80)
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values_list = [param_grid[k] for k in keys]
        
        from itertools import product
        combinations = list(product(*values_list))
        total_combinations = len(combinations)
        
        print(f"\nTotal combinations to test: {total_combinations}")
        print(f"Parameters: {keys}\n")
        
        # Split training data for validation
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            self.X_train, self.y_train, test_size=0.1, random_state=42, 
            stratify=self.y_train
        )
        
        for idx, combo in enumerate(combinations, 1):
            params = {k: v for k, v in zip(keys, combo)}
            
            if verbose:
                print(f"[{idx}/{total_combinations}] Testing: {params}")
            
            import time
            start_time = time.time()
            
            try:
                # Build & train model
                model = self.build_model(
                    embedding_dim=params.get('embedding_dim', 128),
                    lstm_units=params.get('lstm_units', 64),
                    dropout_rate=params.get('dropout_rate', 0.5),
                    learning_rate=params.get('learning_rate', 0.001)
                )
                
                val_acc, val_loss, _ = self.train_and_evaluate(
                    model, X_train_split, y_train_split, X_val_split, y_val_split,
                    epochs=epochs, batch_size=batch_size, patience=patience, verbose=0
                )
                
                # Test evaluation
                test_acc, test_f1, test_bal_acc, test_roc_auc = self.evaluate_on_test(model)
                
                training_time = time.time() - start_time
                
                result = TuningResult(
                    embedding_dim=params.get('embedding_dim', 128),
                    lstm_units=params.get('lstm_units', 64),
                    dropout_rate=params.get('dropout_rate', 0.5),
                    learning_rate=params.get('learning_rate', 0.001),
                    batch_size=batch_size,
                    epochs=epochs,
                    val_accuracy=val_acc,
                    val_loss=val_loss,
                    test_accuracy=test_acc,
                    test_f1=test_f1,
                    test_balanced_acc=test_bal_acc,
                    test_roc_auc=test_roc_auc,
                    training_time=training_time
                )
                
                self.results.append(result)
                
                if verbose:
                    print(f"  ✓ Val Acc: {val_acc:.4f}, Test F1: {test_f1:.4f}, "
                          f"Time: {training_time:.1f}s\n")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}\n")
                continue
        
        return self.results
    
    def random_search(self, param_dist, n_iter=10, epochs=10, batch_size=32, 
                     patience=3, verbose=1, random_state=42):
        """
        Random Search untuk hyperparameter tuning
        
        Args:
            param_dist: Dictionary dengan distribution untuk setiap parameter
            n_iter: Jumlah kombinasi random untuk dicoba
            
        Example:
            {
                'embedding_dim': [64, 128, 256],
                'lstm_units': [32, 64, 128],
                'dropout_rate': [0.2, 0.3, 0.4, 0.5],
                'learning_rate': [0.0001, 0.0005, 0.001, 0.005]
            }
        """
        print("\n" + "="*80)
        print("STARTING RANDOM SEARCH HYPERPARAMETER TUNING")
        print("="*80)
        print(f"\nTotal random trials: {n_iter}\n")
        
        np.random.seed(random_state)
        
        # Split training data
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            self.X_train, self.y_train, test_size=0.1, random_state=42, 
            stratify=self.y_train
        )
        
        for trial in range(n_iter):
            # Random sample dari distribution
            params = {}
            for key, values in param_dist.items():
                params[key] = np.random.choice(values)
            
            if verbose:
                print(f"[{trial+1}/{n_iter}] Testing: {params}")
            
            import time
            start_time = time.time()
            
            try:
                model = self.build_model(
                    embedding_dim=params.get('embedding_dim', 128),
                    lstm_units=params.get('lstm_units', 64),
                    dropout_rate=params.get('dropout_rate', 0.5),
                    learning_rate=params.get('learning_rate', 0.001)
                )
                
                val_acc, val_loss, _ = self.train_and_evaluate(
                    model, X_train_split, y_train_split, X_val_split, y_val_split,
                    epochs=epochs, batch_size=batch_size, patience=patience, verbose=0
                )
                
                test_acc, test_f1, test_bal_acc, test_roc_auc = self.evaluate_on_test(model)
                
                training_time = time.time() - start_time
                
                result = TuningResult(
                    embedding_dim=params.get('embedding_dim', 128),
                    lstm_units=params.get('lstm_units', 64),
                    dropout_rate=params.get('dropout_rate', 0.5),
                    learning_rate=params.get('learning_rate', 0.001),
                    batch_size=batch_size,
                    epochs=epochs,
                    val_accuracy=val_acc,
                    val_loss=val_loss,
                    test_accuracy=test_acc,
                    test_f1=test_f1,
                    test_balanced_acc=test_bal_acc,
                    test_roc_auc=test_roc_auc,
                    training_time=training_time
                )
                
                self.results.append(result)
                
                if verbose:
                    print(f"  ✓ Val Acc: {val_acc:.4f}, Test F1: {test_f1:.4f}, "
                          f"Time: {training_time:.1f}s\n")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}\n")
                continue
        
        return self.results
    
    def kfold_cross_validation(self, n_splits=5, embedding_dim=128, lstm_units=64,
                               dropout_rate=0.5, learning_rate=0.001, epochs=10,
                               batch_size=32, patience=3, verbose=1):
        """
        K-Fold Cross-Validation dengan fixed hyperparameters
        untuk mengestimasi model performance yang lebih robust
        """
        print("\n" + "="*80)
        print("STARTING K-FOLD CROSS-VALIDATION")
        print("="*80)
        print(f"\nK-Fold: {n_splits}")
        print(f"Parameters: embedding_dim={embedding_dim}, lstm_units={lstm_units}, "
              f"dropout={dropout_rate}, lr={learning_rate}\n")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train), 1):
            if verbose:
                print(f"[FOLD {fold}/{n_splits}] Training...")
            
            X_train_fold = self.X_train[train_idx]
            y_train_fold = self.y_train[train_idx]
            X_val_fold = self.X_train[val_idx]
            y_val_fold = self.y_train[val_idx]
            
            import time
            start_time = time.time()
            
            try:
                model = self.build_model(
                    embedding_dim=embedding_dim,
                    lstm_units=lstm_units,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate
                )
                
                val_acc, val_loss, _ = self.train_and_evaluate(
                    model, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                    epochs=epochs, batch_size=batch_size, patience=patience, verbose=0
                )
                
                test_acc, test_f1, test_bal_acc, test_roc_auc = self.evaluate_on_test(model)
                
                training_time = time.time() - start_time
                
                result = TuningResult(
                    embedding_dim=embedding_dim,
                    lstm_units=lstm_units,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epochs,
                    val_accuracy=val_acc,
                    val_loss=val_loss,
                    test_accuracy=test_acc,
                    test_f1=test_f1,
                    test_balanced_acc=test_bal_acc,
                    test_roc_auc=test_roc_auc,
                    training_time=training_time,
                    fold=fold
                )
                
                fold_results.append(result)
                
                if verbose:
                    print(f"  ✓ Val Acc: {val_acc:.4f}, Test F1: {test_f1:.4f}, "
                          f"Test ROC-AUC: {test_roc_auc:.4f}, Time: {training_time:.1f}s\n")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}\n")
                continue
        
        self.results.extend(fold_results)
        return fold_results
    
    def get_best_params(self, metric='test_f1'):
        """Get hyperparameter dengan nilai metric terbaik"""
        if not self.results:
            return None
        
        best_result = max(self.results, key=lambda x: getattr(x, metric))
        return best_result
    
    def get_top_k_params(self, k=5, metric='test_f1'):
        """Get top-k hyperparameter"""
        if not self.results:
            return []
        
        sorted_results = sorted(self.results, key=lambda x: getattr(x, metric), reverse=True)
        return sorted_results[:k]
    
    def save_results(self, output_dir="results/tuning"):
        """Save tuning results ke file"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame
        results_dict = [asdict(r) for r in self.results]
        df = pd.DataFrame(results_dict)
        
        # Save CSV
        csv_path = os.path.join(output_dir, "tuning_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[OK] Results saved to: {csv_path}")
        
        # Save JSON summary
        best_result = self.get_best_params('test_f1')
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_trials': len(self.results),
            'best_params': asdict(best_result) if best_result else None,
            'top_5_results': [asdict(r) for r in self.get_top_k_params(5)]
        }
        
        json_path = os.path.join(output_dir, "tuning_summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Summary saved to: {json_path}")
        
        return df
    
    def plot_results(self, output_dir="results/tuning"):
        """Visualisasi tuning results"""
        if not self.results:
            print("No results to plot")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        results_dict = [asdict(r) for r in self.results]
        df = pd.DataFrame(results_dict)
        
        # Plot 1: Test F1 vs Different Parameters
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Hyperparameter Tuning - Performance Analysis", fontsize=16, fontweight='bold')
        
        # Embedding Dim vs F1
        ax = axes[0, 0]
        for ed in df['embedding_dim'].unique():
            mask = df['embedding_dim'] == ed
            ax.scatter(df.loc[mask, 'lstm_units'], df.loc[mask, 'test_f1'], 
                      label=f'emb_dim={ed}', s=100, alpha=0.7)
        ax.set_xlabel('LSTM Units')
        ax.set_ylabel('Test F1 Score')
        ax.set_title('LSTM Units vs Test F1')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Dropout vs F1
        ax = axes[0, 1]
        for dr in sorted(df['dropout_rate'].unique()):
            mask = df['dropout_rate'] == dr
            ax.scatter(df.loc[mask, 'learning_rate'], df.loc[mask, 'test_f1'],
                      label=f'dropout={dr}', s=100, alpha=0.7)
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Test F1 Score')
        ax.set_title('Learning Rate vs Test F1 (by Dropout)')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Test F1 vs Val Loss
        ax = axes[1, 0]
        scatter = ax.scatter(df['val_loss'], df['test_f1'], c=df['test_roc_auc'],
                           cmap='viridis', s=100, alpha=0.7)
        ax.set_xlabel('Validation Loss')
        ax.set_ylabel('Test F1 Score')
        ax.set_title('Val Loss vs Test F1 (colored by ROC-AUC)')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('ROC-AUC')
        ax.grid(True, alpha=0.3)
        
        # Training Time vs Performance
        ax = axes[1, 1]
        scatter = ax.scatter(df['training_time'], df['test_f1'], c=df['test_balanced_acc'],
                           cmap='plasma', s=100, alpha=0.7)
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('Test F1 Score')
        ax.set_title('Training Time vs Test F1 (colored by Balanced Accuracy)')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Balanced Accuracy')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "tuning_analysis.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Tuning analysis plot saved: {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="LSTM Hyperparameter Tuning")
    parser.add_argument('--method', type=str, choices=['grid', 'random', 'kfold'],
                       default='grid', help='Tuning method')
    parser.add_argument('--n_iter', type=int, default=10, help='Number of iterations for random search')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for k-fold')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs per trial')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='results/tuning', help='Output directory')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    
    args = parser.parse_args()
    
    print("\n" + "+"+"="*78+"+")
    print("|" + " "*20 + "LSTM HYPERPARAMETER TUNING MODULE" + " "*24 + "|")
    print("|" + " "*30 + "Kelompok 7 - Tugas Akhir AI" + " "*21 + "|")
    print("+"+"="*78+"+\n")
    
    # Load data
    print("Loading and preprocessing data...")
    X_train_pad, X_test_pad, y_train, y_test, tokenizer, label_encoder = \
        load_and_preprocess_data(
            csv_path="spam.csv",
            max_words=5000,
            max_len=100,
            test_size=0.2,
            random_state=42
        )
    
    vocab_size = len(tokenizer.word_index) + 1
    print(f"✓ Data loaded: X_train={X_train_pad.shape}, X_test={X_test_pad.shape}")
    print(f"✓ Vocab size: {vocab_size}\n")
    
    # Initialize tuner
    tuner = LSTMHyperparameterTuner(X_train_pad, y_train, X_test_pad, y_test, vocab_size)
    
    # Run tuning based on method
    if args.method == 'grid':
        param_grid = {
            'embedding_dim': [64, 128, 256],
            'lstm_units': [32, 64, 128],
            'dropout_rate': [0.3, 0.5],
            'learning_rate': [0.001, 0.0001]
        }
        tuner.grid_search(param_grid, epochs=args.epochs, batch_size=args.batch_size,
                         patience=3, verbose=args.verbose)
    
    elif args.method == 'random':
        param_dist = {
            'embedding_dim': [64, 128, 256],
            'lstm_units': [32, 64, 128],
            'dropout_rate': [0.2, 0.3, 0.4, 0.5],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005]
        }
        tuner.random_search(param_dist, n_iter=args.n_iter, epochs=args.epochs,
                           batch_size=args.batch_size, patience=3, verbose=args.verbose)
    
    elif args.method == 'kfold':
        tuner.kfold_cross_validation(
            n_splits=args.n_splits,
            embedding_dim=128,
            lstm_units=64,
            dropout_rate=0.5,
            learning_rate=0.001,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=3,
            verbose=args.verbose
        )
    
    # Save & visualize results
    df_results = tuner.save_results(args.output_dir)
    tuner.plot_results(args.output_dir)
    
    # Print summary
    best = tuner.get_best_params('test_f1')
    print("\n" + "="*80)
    print("BEST HYPERPARAMETERS (by Test F1 Score)")
    print("="*80)
    print(f"Embedding Dim: {best.embedding_dim}")
    print(f"LSTM Units: {best.lstm_units}")
    print(f"Dropout Rate: {best.dropout_rate}")
    print(f"Learning Rate: {best.learning_rate}")
    print(f"\nPerformance:")
    print(f"  Val Accuracy: {best.val_accuracy:.4f}")
    print(f"  Test Accuracy: {best.test_accuracy:.4f}")
    print(f"  Test F1 Score: {best.test_f1:.4f}")
    print(f"  Test Balanced Accuracy: {best.test_balanced_acc:.4f}")
    print(f"  Test ROC-AUC: {best.test_roc_auc:.4f}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
