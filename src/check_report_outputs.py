"""
Report Output Checker - SMS Spam Classification
Kelompok 7 - Tugas Akhir Kecerdasan Buatan

File ini mengecek keberadaan file-file penting untuk laporan akhir.
Jalankan dengan: python src/check_report_outputs.py
"""

import os
from pathlib import Path


def check_file(filepath):
    """
    Mengecek keberadaan file dan mengembalikan status.
    
    Args:
        filepath (str): Path file yang akan dicek
        
    Returns:
        tuple: (status, size_kb) dimana status adalah 'FOUND' atau 'MISSING'
    """
    if os.path.exists(filepath):
        try:
            size_kb = os.path.getsize(filepath) / 1024
            return 'FOUND', f"{size_kb:.2f} KB"
        except:
            return 'FOUND', 'Unknown'
    else:
        return 'MISSING', '—'


def print_header():
    """Tampilkan header script."""
    print("\n")
    print("=" * 90)
    print("SMS SPAM CLASSIFICATION - REPORT OUTPUT CHECKER".center(90))
    print("Kelompok 7 - Tugas Akhir AI".center(90))
    print("=" * 90)


def print_section(title):
    """Tampilkan judul section."""
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)


def check_outputs():
    """Main function untuk mengecek semua output files."""
    print_header()
    
    # Define files to check
    required_files = {
        'Data Preprocessing': [
            'results/cleaned_spam.csv',
        ],
        'Model Files': [
            'models/simple_rnn_model.keras',
            'models/lstm_model.keras',
        ],
        'Training History': [
            'results/simple_rnn_history.csv',
            'results/lstm_history.csv',
        ],
        'Evaluation Metrics': [
            'results/evaluation_metrics.csv',
        ],
        'Classification Reports': [
            'results/simple_rnn_classification_report.txt',
            'results/lstm_classification_report.txt',
        ],
        'Confusion Matrix Visualizations': [
            'results/simple_rnn_confusion_matrix.png',
            'results/lstm_confusion_matrix.png',
        ],
        'Training History Visualizations': [
            'results/simple_rnn_accuracy.png',
            'results/simple_rnn_loss.png',
            'results/lstm_accuracy.png',
            'results/lstm_loss.png',
        ],
        'Error Analysis': [
            'results/error_analysis_lstm.csv',
            'results/error_analysis_lstm.txt',
        ],
        'Hyperparameter Tuning': [
            'results/tuning_results.csv',
            'results/tuning_f1_score.png',
        ]
    }
    
    # Check all files
    total_files = 0
    found_files = 0
    results_by_section = {}
    
    print_section("CHECKING REPORT OUTPUT FILES")
    
    for section, files in required_files.items():
        print(f"\n  [*] {section}")
        section_found = 0
        section_results = []
        
        for filepath in files:
            status, size = check_file(filepath)
            total_files += 1
            
            if status == 'FOUND':
                found_files += 1
                section_found += 1
                icon = "[OK]"
                status_display = f"[{status}]".ljust(10)
            else:
                icon = "[XX]"
                status_display = f"[{status}]".ljust(10)
            
            # Extract filename from path
            filename = filepath.split('/')[-1]
            
            print(f"      {icon} {status_display} {filepath.ljust(50)} ({size})")
            section_results.append((filepath, status))
        
        results_by_section[section] = (section_found, len(files), section_results)
        print(f"      --> {section_found}/{len(files)} files found")
    
    # Print summary
    print_section("SUMMARY")
    
    print(f"\n  Total Files Required: {total_files}")
    print(f"  Files Found:         {found_files}")
    print(f"  Files Missing:       {total_files - found_files}")
    print(f"  Completion Rate:     {(found_files/total_files)*100:.1f}%")
    
    # Section breakdown
    print(f"\n  Section Breakdown:")
    for section, (found, total, _) in results_by_section.items():
        pct = (found/total)*100 if total > 0 else 0
        status_icon = "[OK]" if found == total else "[XX]"
        print(f"    {status_icon} {section.ljust(40)} {found}/{total} ({pct:.0f}%)")
    
    # Recommendations
    print_section("RECOMMENDATIONS")
    
    missing_files = []
    for section, (_, _, results) in results_by_section.items():
        for filepath, status in results:
            if status == 'MISSING':
                missing_files.append(filepath)
    
    if missing_files:
        print(f"\n  Found {len(missing_files)} missing file(s).")
        print(f"\n  Recommended Script Execution Order:")
        print(f"    1. python src/eda_preprocessing.py")
        print(f"    2. python src/train_lstm.py")
        print(f"    3. python src/evaluate.py")
        print(f"    4. python src/visualize_results.py")
        print(f"    5. python src/error_analysis.py")
        print(f"    6. python src/hyperparameter_tuning.py")
        print(f"    7. python src/check_report_outputs.py (re-check)")
        
        print(f"\n  Missing Files Details:")
        for filepath in missing_files:
            print(f"      - {filepath}")
    else:
        print(f"\n  [OK] ALL REQUIRED FILES ARE PRESENT!")
        print(f"\n  Your project is ready for final report submission!")
    
    # File size summary
    print_section("FILE SIZE SUMMARY")
    
    total_size_kb = 0
    print(f"\n  Calculating total size of output files...")
    
    for section, files in required_files.items():
        section_size = 0
        for filepath in files:
            if os.path.exists(filepath):
                try:
                    size_kb = os.path.getsize(filepath) / 1024
                    section_size += size_kb
                    total_size_kb += size_kb
                except:
                    pass
        
        if section_size > 0:
            size_display = f"{section_size:.2f} KB" if section_size < 1024 else f"{section_size/1024:.2f} MB"
            print(f"    - {section.ljust(40)} {size_display}")
    
    print(f"\n  Total Size:  {total_size_kb:.2f} KB ({total_size_kb/1024:.2f} MB)")
    
    # Final status
    print_section("FINAL STATUS")
    
    if found_files == total_files:
        status_text = "READY FOR SUBMISSION"
        status_code = "[OK]"
    elif found_files >= (total_files * 0.7):
        status_text = "MOSTLY COMPLETE - Run missing scripts"
        status_code = "[PARTIAL]"
    else:
        status_text = "INCOMPLETE - Several scripts need to be run"
        status_code = "[WARNING]"
    
    print(f"\n  {status_code} {status_text}")
    print(f"\n" + "=" * 90 + "\n")


if __name__ == '__main__':
    check_outputs()
