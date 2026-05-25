"""
Visualization module for SMS Spam Classification results.
This script generates training history plots and confusion matrix visualizations.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_history(history_csv, metric, output_path, title):
    """
    Plot training history for accuracy or loss metrics.
    
    Args:
        history_csv (str): Path to the history CSV file
        metric (str): Either 'accuracy' or 'loss'
        output_path (str): Path to save the output PNG
        title (str): Title for the plot
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(history_csv):
            print(f"❌ Error: {history_csv} tidak ditemukan!")
            return False
        
        # Read the CSV file
        df = pd.read_csv(history_csv)
        
        # Check if required columns exist
        if metric == 'accuracy':
            train_col = 'accuracy'
            val_col = 'val_accuracy'
        elif metric == 'loss':
            train_col = 'loss'
            val_col = 'val_loss'
        else:
            print(f"❌ Error: Metric harus 'accuracy' atau 'loss', bukan '{metric}'")
            return False
        
        if train_col not in df.columns or val_col not in df.columns:
            print(f"❌ Error: Kolom '{train_col}' atau '{val_col}' tidak ditemukan di {history_csv}")
            return False
        
        # Create figure and plot
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(df) + 1)
        
        plt.plot(epochs, df[train_col], 'o-', linewidth=2, label=f'Training {metric.capitalize()}', markersize=6)
        plt.plot(epochs, df[val_col], 's-', linewidth=2, label=f'Validation {metric.capitalize()}', markersize=6)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Berhasil menyimpan: {output_path}")
        return True
    
    except Exception as e:
        print(f"❌ Error saat membuat plot {metric}: {str(e)}")
        return False


def plot_confusion_matrix_from_csv(cm_csv, output_path, title):
    """
    Plot confusion matrix from CSV file.
    
    Args:
        cm_csv (str): Path to the confusion matrix CSV file
        output_path (str): Path to save the output PNG
        title (str): Title for the plot
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(cm_csv):
            print(f"❌ Error: {cm_csv} tidak ditemukan!")
            return False
        
        # Read the CSV file with first column as index
        df = pd.read_csv(cm_csv, index_col=0)
        
        # Rename columns and index to match class labels
        df.columns = ['Ham', 'Spam']
        df.index = ['Ham', 'Spam']
        
        # Create figure and plot
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                    linewidths=1.5, linecolor='gray', square=True, 
                    cbar=True, annot_kws={'size': 12, 'weight': 'bold'})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Predicted Label', fontsize=12)
        plt.xlabel('Actual Label', fontsize=12)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Berhasil menyimpan: {output_path}")
        return True
    
    except Exception as e:
        print(f"❌ Error saat membuat confusion matrix: {str(e)}")
        return False


def main():
    """Main function to generate all visualizations."""
    
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / 'results'
    
    # Define input and output paths
    input_files = {
        'simple_rnn_history': results_dir / 'simple_rnn_history.csv',
        'lstm_history': results_dir / 'lstm_history.csv',
        'simple_rnn_cm': results_dir / 'simple_rnn_confusion_matrix.csv',
        'lstm_cm': results_dir / 'lstm_confusion_matrix.csv',
    }
    
    output_files = {
        'simple_rnn_accuracy': results_dir / 'simple_rnn_accuracy.png',
        'simple_rnn_loss': results_dir / 'simple_rnn_loss.png',
        'lstm_accuracy': results_dir / 'lstm_accuracy.png',
        'lstm_loss': results_dir / 'lstm_loss.png',
        'simple_rnn_cm': results_dir / 'simple_rnn_confusion_matrix.png',
        'lstm_cm': results_dir / 'lstm_confusion_matrix.png',
    }
    
    print("\n" + "="*70)
    print("📊 SMS Spam Classification - Visualization Generator")
    print("="*70 + "\n")
    
    # Check if all input files exist
    missing_files = [name for name, path in input_files.items() if not path.exists()]
    
    if missing_files:
        print("❌ File input yang diperlukan tidak ditemukan:")
        for file in missing_files:
            print(f"   - {input_files[file]}")
        print("\n⚠️  Jalankan script berikut terlebih dahulu:")
        print("   1. python src/train_lstm.py")
        print("   2. python src/evaluate.py")
        print()
        return False
    
    # Create visualizations
    print("🔄 Membuat visualisasi...\n")
    
    success_count = 0
    
    # Plot Simple RNN training history
    if plot_training_history(
        str(input_files['simple_rnn_history']),
        'accuracy',
        str(output_files['simple_rnn_accuracy']),
        'Simple RNN - Training vs Validation Accuracy'
    ):
        success_count += 1
    
    if plot_training_history(
        str(input_files['simple_rnn_history']),
        'loss',
        str(output_files['simple_rnn_loss']),
        'Simple RNN - Training vs Validation Loss'
    ):
        success_count += 1
    
    # Plot LSTM training history
    if plot_training_history(
        str(input_files['lstm_history']),
        'accuracy',
        str(output_files['lstm_accuracy']),
        'LSTM - Training vs Validation Accuracy'
    ):
        success_count += 1
    
    if plot_training_history(
        str(input_files['lstm_history']),
        'loss',
        str(output_files['lstm_loss']),
        'LSTM - Training vs Validation Loss'
    ):
        success_count += 1
    
    # Plot confusion matrices
    if plot_confusion_matrix_from_csv(
        str(input_files['simple_rnn_cm']),
        str(output_files['simple_rnn_cm']),
        'Simple RNN - Confusion Matrix'
    ):
        success_count += 1
    
    if plot_confusion_matrix_from_csv(
        str(input_files['lstm_cm']),
        str(output_files['lstm_cm']),
        'LSTM - Confusion Matrix'
    ):
        success_count += 1
    
    # Print summary
    print("\n" + "="*70)
    print(f"✨ Visualisasi selesai! ({success_count}/6 grafik berhasil dibuat)")
    print("="*70)
    
    if success_count == 6:
        print("\n✅ Semua visualisasi berhasil disimpan di folder 'results/':")
        for name, path in output_files.items():
            if path.exists():
                print(f"   ✓ {path.name}")
        return True
    else:
        print("\n⚠️  Beberapa visualisasi gagal. Periksa pesan error di atas.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
