#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', context='talk')

def load_training_logs(log_file='logs/n_estimator_log.txt'):
    """Read all F1 scores from training_log.txt"""
    results = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    results.append({
                        'dataset_size': data['dataset_size'],
                        'n_estimators': data['best_params']['n_estimators'],
                        'max_depth': data['best_params']['max_depth'],
                        'min_samples_leaf': data['best_params']['min_samples_leaf'],
                        'min_samples_split': data['best_params']['min_samples_split'],
                        'f1_score': data['f1_score'],
                        'precision': data['precision'],
                        'recall': data['recall'],
                        'test_accuracy': data['test_accuracy']
                    })
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(results)

def plot_n_estimators_trend(df):
    """Plot F1 trend across n_estimators"""
    
    df_sorted = df.sort_values('n_estimators')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # F1 Score vs n_estimators
    ax = axes[0, 0]
    ax.plot(df_sorted['n_estimators'], df_sorted['f1_score'], 'o-', linewidth=3, markersize=10, color='#2080B8')
    ax.fill_between(df_sorted['n_estimators'], df_sorted['f1_score']-0.01, df_sorted['f1_score']+0.01, alpha=0.2)
    ax.set_xlabel('n_estimators', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score Convergence', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.0)
    
    # Precision vs n_estimators
    ax = axes[0, 1]
    ax.plot(df_sorted['n_estimators'], df_sorted['precision'], 's-', linewidth=3, markersize=8, color='#20B850')
    ax.set_xlabel('n_estimators', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision by n_estimators', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.0)
    
    # Recall vs n_estimators
    ax = axes[1, 0]
    ax.plot(df_sorted['n_estimators'], df_sorted['recall'], '^-', linewidth=3, markersize=8, color='#B82080')
    ax.set_xlabel('n_estimators', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Recall by n_estimators', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.0)
    
    # Accuracy vs n_estimators
    ax = axes[1, 1]
    ax.plot(df_sorted['n_estimators'], df_sorted['test_accuracy'], 'd-', linewidth=3, markersize=8, color='#FF8C00')
    ax.set_xlabel('n_estimators', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Test Accuracy by n_estimators', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.0)
    
    plt.tight_layout()
    
    script_dir = Path(__file__).parent
    (script_dir / 'figures').mkdir(exist_ok=True)
    plt.savefig(script_dir / 'figures' / 'n_estimators_analysis.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: figures/n_estimators_analysis.png')
    plt.close()

def print_summary(df):
    """Print F1 trend summary"""
    print('\n' + '='*70)
    print('📊 N_ESTIMATORS ANALYSIS (from training logs)')
    print('='*70 + '\n')
    
    df_sorted = df.sort_values('n_estimators')
    print(f'{"n_est":>6} {"F1 Score":>12} {"Precision":>12} {"Recall":>12} {"Accuracy":>12}')
    print('-'*70)
    
    for _, row in df_sorted.iterrows():
        print(f'{int(row["n_estimators"]):>6} {row["f1_score"]:>12.4f} {row["precision"]:>12.4f} {row["recall"]:>12.4f} {row["test_accuracy"]:>12.4f}')
    
    print('\n' + '='*70)
    
    best_f1_idx = df['f1_score'].idxmax()
    best_f1_row = df.loc[best_f1_idx]
    
    print(f'🏆 BEST F1: n_estimators={int(best_f1_row["n_estimators"])} (F1: {best_f1_row["f1_score"]:.4f})')
    
    baseline_100 = df[df['n_estimators'] == 100]['f1_score'].values
    if len(baseline_100) > 0:
        f1_diff = best_f1_row['f1_score'] - baseline_100[0]
        speed_gain = (100 - best_f1_row['n_estimators']) / 100 * 100
        print(f'  vs 100 trees: {f1_diff:+.4f} F1 ({speed_gain:+.0f}% speed)')
    
    print('='*70 + '\n')

def main():
    log_file = Path(__file__).parent.parent / 'logs' / 'n_estimator_log.txt'
    
    if not log_file.exists():
        print(f'❌ {log_file} not found!')
        return
    
    df = load_training_logs(str(log_file))
    
    if df.empty:
        print('❌ No valid training logs found!')
        return
    
    print(f'✓ Loaded {len(df)} training results')
    
    print_summary(df)
    plot_n_estimators_trend(df)

if __name__ == '__main__':
    main()
