"""
Error Analysis Module untuk SMS Spam Classification
Kelompok 7 - Tugas Akhir Kecerdasan Buatan

Modul ini menyediakan:
1. Detailed error analysis pada misclassified samples
2. Error categorization dan pattern detection
3. Confusion matrix deep dive
4. Sample-level error visualization
5. Error statistics dan insights

Jalankan:
    python src/error_analysis.py --model results/models/lstm.h5
    python src/error_analysis.py --model results/models/lstm.h5 --show_samples 10
    python src/error_analysis.py --model results/models/lstm.h5 --analyze_confidence
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

from eda_preprocessing import load_and_preprocess_data


@dataclass
class ErrorRecord:
    """Record untuk satu sample yang error"""
    sample_id: int
    true_label: int
    true_label_name: str
    predicted_label: int
    predicted_label_name: str
    confidence: float
    error_type: str  # 'false_positive' atau 'false_negative'
    message_preview: str  # Preview of original message


class ErrorAnalyzer:
    """Main class untuk error analysis LSTM predictions"""
    
    def __init__(self, model_path, X_test, y_test, X_test_original=None, 
                 tokenizer=None, max_len=100):
        """
        Args:
            model_path: Path ke model .h5
            X_test: Padded test sequences
            y_test: True labels
            X_test_original: Original messages (optional, untuk preview)
            tokenizer: Tokenizer untuk reverse-sequence (optional)
            max_len: Max sequence length
        """
        self.model_path = model_path
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_original = X_test_original
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Load model
        self.model = load_model(model_path)
        
        # Generate predictions
        self.y_pred_proba = self.model.predict(X_test, verbose=0).reshape(-1)
        self.y_pred = (self.y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        self.correct_mask = self.y_test == self.y_pred
        self.incorrect_mask = self.y_test != self.y_pred
        
        self.n_correct = np.sum(self.correct_mask)
        self.n_incorrect = np.sum(self.incorrect_mask)
        self.n_total = len(self.y_test)
        self.accuracy = self.n_correct / self.n_total
        
        # Error types
        self.false_positives = (self.y_test == 0) & (self.y_pred == 1)  # Predicted ham as spam
        self.false_negatives = (self.y_test == 1) & (self.y_pred == 0)  # Predicted spam as ham
        
        self.n_fp = np.sum(self.false_positives)
        self.n_fn = np.sum(self.false_negatives)
        
        # Error records
        self.error_records = []
    
    def generate_error_records(self):
        """Generate detailed error records untuk semua misclassified samples"""
        error_indices = np.where(self.incorrect_mask)[0]
        
        for idx in error_indices:
            true_label = int(self.y_test[idx])
            pred_label = int(self.y_pred[idx])
            confidence = float(self.y_pred_proba[idx])
            
            # Determine error type
            if true_label == 0 and pred_label == 1:
                error_type = 'false_positive'
                error_type_name = 'False Positive (Ham→Spam)'
            else:
                error_type = 'false_negative'
                error_type_name = 'False Negative (Spam→Ham)'
            
            # Message preview (if available)
            msg_preview = ""
            if self.X_test_original is not None:
                try:
                    msg_preview = self.X_test_original[idx][:100]  # First 100 chars
                except:
                    msg_preview = "N/A"
            
            record = ErrorRecord(
                sample_id=int(idx),
                true_label=true_label,
                true_label_name='Ham' if true_label == 0 else 'Spam',
                predicted_label=pred_label,
                predicted_label_name='Ham' if pred_label == 0 else 'Spam',
                confidence=confidence,
                error_type=error_type,
                message_preview=msg_preview
            )
            
            self.error_records.append(record)
        
        return self.error_records
    
    def get_worst_predictions(self, error_type='all', k=10):
        """Get k worst predictions berdasarkan confidence"""
        if not self.error_records:
            self.generate_error_records()
        
        records = self.error_records
        
        # Filter by error type
        if error_type == 'fp':
            records = [r for r in records if r.error_type == 'false_positive']
        elif error_type == 'fn':
            records = [r for r in records if r.error_type == 'false_negative']
        
        # Sort by confidence (worst = most confident wrong predictions)
        sorted_records = sorted(records, 
                               key=lambda x: abs(x.confidence - 0.5) if x.confidence > 0.5 
                                           else (0.5 - x.confidence),
                               reverse=True)
        
        return sorted_records[:k]
    
    def analyze_confidence_distribution(self):
        """Analyze distribusi confidence untuk correct vs incorrect predictions"""
        correct_proba = self.y_pred_proba[self.correct_mask]
        incorrect_proba = self.y_pred_proba[self.incorrect_mask]
        
        fp_proba = self.y_pred_proba[self.false_positives]
        fn_proba = self.y_pred_proba[self.false_negatives]
        
        return {
            'correct': {
                'mean': float(np.mean(correct_proba)) if len(correct_proba) > 0 else 0,
                'std': float(np.std(correct_proba)) if len(correct_proba) > 0 else 0,
                'min': float(np.min(correct_proba)) if len(correct_proba) > 0 else 0,
                'max': float(np.max(correct_proba)) if len(correct_proba) > 0 else 0,
                'count': len(correct_proba)
            },
            'incorrect': {
                'mean': float(np.mean(incorrect_proba)) if len(incorrect_proba) > 0 else 0,
                'std': float(np.std(incorrect_proba)) if len(incorrect_proba) > 0 else 0,
                'min': float(np.min(incorrect_proba)) if len(incorrect_proba) > 0 else 0,
                'max': float(np.max(incorrect_proba)) if len(incorrect_proba) > 0 else 0,
                'count': len(incorrect_proba)
            },
            'false_positives': {
                'mean': float(np.mean(fp_proba)) if len(fp_proba) > 0 else 0,
                'std': float(np.std(fp_proba)) if len(fp_proba) > 0 else 0,
                'min': float(np.min(fp_proba)) if len(fp_proba) > 0 else 0,
                'max': float(np.max(fp_proba)) if len(fp_proba) > 0 else 0,
                'count': len(fp_proba)
            },
            'false_negatives': {
                'mean': float(np.mean(fn_proba)) if len(fn_proba) > 0 else 0,
                'std': float(np.std(fn_proba)) if len(fn_proba) > 0 else 0,
                'min': float(np.min(fn_proba)) if len(fn_proba) > 0 else 0,
                'max': float(np.max(fn_proba)) if len(fn_proba) > 0 else 0,
                'count': len(fn_proba)
            }
        }
    
    def get_error_summary(self):
        """Get summary statistics tentang errors"""
        cm = confusion_matrix(self.y_test, self.y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'total_samples': int(self.n_total),
            'correct_predictions': int(self.n_correct),
            'incorrect_predictions': int(self.n_incorrect),
            'accuracy': float(self.accuracy),
            'error_rate': float(1 - self.accuracy),
            'false_positives': int(self.n_fp),
            'false_negative': int(self.n_fn),
            'fp_rate': float(self.n_fp / (self.n_fp + tn) if (self.n_fp + tn) > 0 else 0),
            'fn_rate': float(self.n_fn / (self.n_fn + tp) if (self.n_fn + tp) > 0 else 0),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            }
        }
    
    def plot_confidence_distribution(self, output_dir='results/error_analysis'):
        """Plot distribusi confidence untuk correct vs incorrect"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Confidence Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Overall confidence distribution
        ax = axes[0, 0]
        ax.hist(self.y_pred_proba[self.correct_mask], bins=30, alpha=0.7, 
               label='Correct', color='green', edgecolor='black')
        ax.hist(self.y_pred_proba[self.incorrect_mask], bins=30, alpha=0.7, 
               label='Incorrect', color='red', edgecolor='black')
        ax.set_xlabel('Model Confidence')
        ax.set_ylabel('Frequency')
        ax.set_title('Confidence: Correct vs Incorrect Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: False Positives vs False Negatives
        ax = axes[0, 1]
        if self.n_fp > 0:
            ax.hist(self.y_pred_proba[self.false_positives], bins=20, alpha=0.7,
                   label=f'False Positives (n={self.n_fp})', color='orange', edgecolor='black')
        if self.n_fn > 0:
            ax.hist(self.y_pred_proba[self.false_negatives], bins=20, alpha=0.7,
                   label=f'False Negatives (n={self.n_fn})', color='purple', edgecolor='black')
        ax.set_xlabel('Model Confidence')
        ax.set_ylabel('Frequency')
        ax.set_title('Confidence Distribution: FP vs FN')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Box plot comparison
        ax = axes[1, 0]
        data_to_plot = [
            self.y_pred_proba[self.correct_mask],
            self.y_pred_proba[self.false_positives] if self.n_fp > 0 else [],
            self.y_pred_proba[self.false_negatives] if self.n_fn > 0 else []
        ]
        labels = ['Correct', f'FP (n={self.n_fp})', f'FN (n={self.n_fn})']
        ax.boxplot(data_to_plot, labels=labels)
        ax.set_ylabel('Confidence')
        ax.set_title('Confidence Distribution Boxplot')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Cumulative distribution
        ax = axes[1, 1]
        sorted_correct = np.sort(self.y_pred_proba[self.correct_mask])
        sorted_incorrect = np.sort(self.y_pred_proba[self.incorrect_mask])
        
        ax.plot(sorted_correct, np.arange(len(sorted_correct))/len(sorted_correct),
               label='Correct', linewidth=2, color='green')
        if len(sorted_incorrect) > 0:
            ax.plot(sorted_incorrect, np.arange(len(sorted_incorrect))/len(sorted_incorrect),
                   label='Incorrect', linewidth=2, color='red')
        ax.set_xlabel('Model Confidence')
        ax.set_ylabel('Cumulative Proportion')
        ax.set_title('Cumulative Distribution Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'confidence_distribution.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Confidence distribution plot saved: {plot_path}")
        plt.close()
    
    def plot_error_analysis(self, output_dir='results/error_analysis'):
        """Plot comprehensive error analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        cm = confusion_matrix(self.y_test, self.y_pred, labels=[0, 1])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Error Analysis Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Confusion Matrix (Raw)
        ax = axes[0, 0]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                   ax=ax, cbar_kws={'label': 'Count'})
        ax.set_title('Confusion Matrix (Raw)')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # Plot 2: Confusion Matrix (Normalized)
        ax = axes[0, 1]
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                   ax=ax, cbar_kws={'label': 'Proportion'})
        ax.set_title('Confusion Matrix (Normalized by Row)')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # Plot 3: Error rate breakdown
        ax = axes[1, 0]
        tn, fp, fn, tp = cm.ravel()
        categories = ['True\nNegatives', 'False\nPositives', 'False\nNegatives', 'True\nPositives']
        values = [tn, fp, fn, tp]
        colors = ['#2ecc71', '#e74c3c', '#e67e22', '#3498db']
        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Count')
        ax.set_title('Confusion Matrix Breakdown')
        ax.grid(True, alpha=0.3, axis='y')
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Error statistics
        ax = axes[1, 1]
        ax.axis('off')
        summary = self.get_error_summary()
        
        stats_text = f"""
ERROR ANALYSIS SUMMARY

Total Test Samples: {summary['total_samples']}
Correct Predictions: {summary['correct_predictions']} ({summary['accuracy']:.2%})
Incorrect Predictions: {summary['incorrect_predictions']} ({summary['error_rate']:.2%})

False Positives: {summary['false_positives']} (FP Rate: {summary['fp_rate']:.2%})
False Negatives: {summary['false_negative']} (FN Rate: {summary['fn_rate']:.2%})

Confusion Matrix:
  TN={summary['confusion_matrix']['tn']}  FP={summary['confusion_matrix']['fp']}
  FN={summary['confusion_matrix']['fn']}  TP={summary['confusion_matrix']['tp']}
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'error_analysis_summary.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Error analysis summary plot saved: {plot_path}")
        plt.close()
    
    def save_error_report(self, output_dir='results/error_analysis', show_samples=20):
        """Save detailed error report ke file"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.error_records:
            self.generate_error_records()
        
        # Convert error records to DataFrame
        error_dicts = [asdict(r) for r in self.error_records]
        df_errors = pd.DataFrame(error_dicts)
        
        # Save CSV
        csv_path = os.path.join(output_dir, 'error_details.csv')
        df_errors.to_csv(csv_path, index=False)
        print(f"[OK] Error details saved to: {csv_path}")
        
        # Get worst predictions
        worst_records = self.get_worst_predictions(k=show_samples)
        
        # Generate text report
        report_path = os.path.join(output_dir, 'error_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ERROR ANALYSIS DETAILED REPORT\n")
            f.write("SMS Spam Classification - LSTM Model\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {os.path.basename(self.model_path)}\n\n")
            
            # Summary statistics
            summary = self.get_error_summary()
            f.write("SUMMARY STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Test Samples: {summary['total_samples']}\n")
            f.write(f"Correct Predictions: {summary['correct_predictions']} ({summary['accuracy']:.4f})\n")
            f.write(f"Incorrect Predictions: {summary['incorrect_predictions']} ({summary['error_rate']:.4f})\n")
            f.write(f"False Positives (Ham→Spam): {summary['false_positives']} "
                   f"(FP Rate: {summary['fp_rate']:.4f})\n")
            f.write(f"False Negatives (Spam→Ham): {summary['false_negative']} "
                   f"(FN Rate: {summary['fn_rate']:.4f})\n")
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"  TN={summary['confusion_matrix']['tn']}  FP={summary['confusion_matrix']['fp']}\n")
            f.write(f"  FN={summary['confusion_matrix']['fn']}  TP={summary['confusion_matrix']['tp']}\n\n")
            
            # Confidence analysis
            conf_stats = self.analyze_confidence_distribution()
            f.write("CONFIDENCE DISTRIBUTION ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(f"Correct Predictions:\n")
            f.write(f"  Mean: {conf_stats['correct']['mean']:.4f}, "
                   f"Std: {conf_stats['correct']['std']:.4f}\n")
            f.write(f"  Range: [{conf_stats['correct']['min']:.4f}, "
                   f"{conf_stats['correct']['max']:.4f}]\n")
            f.write(f"Incorrect Predictions:\n")
            f.write(f"  Mean: {conf_stats['incorrect']['mean']:.4f}, "
                   f"Std: {conf_stats['incorrect']['std']:.4f}\n")
            f.write(f"  Range: [{conf_stats['incorrect']['min']:.4f}, "
                   f"{conf_stats['incorrect']['max']:.4f}]\n\n")
            
            # Worst predictions
            f.write(f"TOP {len(worst_records)} WORST PREDICTIONS\n")
            f.write("-"*80 + "\n")
            for i, record in enumerate(worst_records, 1):
                f.write(f"\n[{i}] Sample ID: {record.sample_id}\n")
                f.write(f"    Error Type: {record.error_type}\n")
                f.write(f"    True Label: {record.true_label_name} ({record.true_label})\n")
                f.write(f"    Predicted: {record.predicted_label_name} ({record.predicted_label})\n")
                f.write(f"    Confidence: {record.confidence:.4f}\n")
                if record.message_preview:
                    f.write(f"    Message: {record.message_preview}...\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"[OK] Error report saved to: {report_path}")
        
        # Save summary JSON
        json_path = os.path.join(output_dir, 'error_summary.json')
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'model': os.path.basename(self.model_path),
            'error_summary': self.get_error_summary(),
            'confidence_analysis': self.analyze_confidence_distribution(),
            'worst_predictions': [asdict(r) for r in worst_records[:10]]
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
        print(f"[OK] Error summary JSON saved to: {json_path}")
        
        return df_errors


def main():
    parser = argparse.ArgumentParser(description="Error Analysis untuk LSTM Spam Classifier")
    parser.add_argument('--model', type=str, default='results/models/lstm.h5',
                       help='Path ke model .h5')
    parser.add_argument('--show_samples', type=int, default=20,
                       help='Jumlah worst predictions untuk ditampilkan')
    parser.add_argument('--analyze_confidence', action='store_true',
                       help='Generate confidence analysis plots')
    parser.add_argument('--output_dir', type=str, default='results/error_analysis',
                       help='Output directory untuk hasil analysis')
    
    args = parser.parse_args()
    
    print("\n" + "+"+"="*78+"+")
    print("|" + " "*25 + "ERROR ANALYSIS MODULE" + " "*32 + "|")
    print("|" + " "*30 + "Kelompok 7 - Tugas Akhir AI" + " "*21 + "|")
    print("+"+"="*78+"+\n")
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"[ERROR] Model tidak ditemukan: {args.model}")
        return
    
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
    print(f"✓ Data loaded: X_test shape = {X_test_pad.shape}\n")
    
    # Initialize analyzer
    analyzer = ErrorAnalyzer(args.model, X_test_pad, y_test)
    
    # Generate error records
    print("Generating error records...")
    analyzer.generate_error_records()
    print(f"✓ Found {len(analyzer.error_records)} misclassified samples\n")
    
    # Save report
    print("Saving error analysis report...")
    analyzer.save_error_report(args.output_dir, args.show_samples)
    
    # Generate plots
    print("\nGenerating visualization plots...")
    analyzer.plot_error_analysis(args.output_dir)
    
    if args.analyze_confidence:
        analyzer.plot_confidence_distribution(args.output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("ERROR ANALYSIS SUMMARY")
    print("="*80)
    summary = analyzer.get_error_summary()
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Correct: {summary['correct_predictions']} ({summary['accuracy']:.4f})")
    print(f"Incorrect: {summary['incorrect_predictions']} ({summary['error_rate']:.4f})")
    print(f"  - False Positives: {summary['false_positives']} (Rate: {summary['fp_rate']:.4f})")
    print(f"  - False Negatives: {summary['false_negative']} (Rate: {summary['fn_rate']:.4f})")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
