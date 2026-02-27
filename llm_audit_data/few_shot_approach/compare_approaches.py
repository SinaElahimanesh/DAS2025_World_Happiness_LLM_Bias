"""
Compare Original vs Improved LLM Audit Approaches

This script compares the results from the original and improved approaches
to measure the effectiveness of the optimizations.

Usage:
    python compare_approaches.py
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(current_dir))


def load_original_results():
    """Load results from original approach"""
    results_dir = os.path.join(os.path.dirname(current_dir), 'results')
    csv_file = os.path.join(results_dir, 'llm_audit_results.csv')
    
    if not os.path.exists(csv_file):
        print(f"Warning: Original results not found at {csv_file}")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        df['approach'] = 'original'
        return df
    except Exception as e:
        print(f"Error loading original results: {e}")
        return None


def load_improved_results():
    """Load results from few-shot approach"""
    results_dir = os.path.join(current_dir, 'results')
    csv_file = os.path.join(results_dir, 'llm_audit_results.csv')
    
    if not os.path.exists(csv_file):
        print(f"Warning: Improved results not found at {csv_file}")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        df['approach'] = 'improved'
        return df
    except Exception as e:
        print(f"Error loading improved results: {e}")
        return None


def load_real_data():
    """Load real WHR data (via data_loader) for comparison"""
    try:
        data_file = os.path.join(parent_dir, 'data.xlsx')
        from data_loader import load_data, clean_data
        
        df = load_data(data_file)
        df = clean_data(df)
        
        # Get latest year data
        latest_year = df['Year'].max()
        df_latest = df[df['Year'] == latest_year].copy()
        
        return df_latest
    except Exception as e:
        print(f"Warning: Could not load real data: {e}")
        return None


def compare_approaches():
    """Main comparison function"""
    print("="*70)
    print("Comparing Initial vs Few-Shot LLM Audit Approaches")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df_original = load_original_results()
    df_improved = load_improved_results()
    df_real = load_real_data()
    
    if df_original is None and df_improved is None:
        print("\nError: No results found! Please run the audit scripts first.")
        return
    
    # Compare basic statistics
    print("\n" + "="*70)
    print("BASIC STATISTICS COMPARISON")
    print("="*70)
    
    metrics = ['overall_happiness', 'gdp', 'social_support', 'health', 'freedom', 'generosity', 'corruption']
    
    comparison_data = []
    
    for metric in metrics:
        if df_original is not None and metric in df_original.columns:
            orig_mean = df_original[metric].mean()
            orig_std = df_original[metric].std()
            orig_min = df_original[metric].min()
            orig_max = df_original[metric].max()
        else:
            orig_mean = orig_std = orig_min = orig_max = np.nan
        
        if df_improved is not None and metric in df_improved.columns:
            impr_mean = df_improved[metric].mean()
            impr_std = df_improved[metric].std()
            impr_min = df_improved[metric].min()
            impr_max = df_improved[metric].max()
        else:
            impr_mean = impr_std = impr_min = impr_max = np.nan
        
        if df_real is not None:
            # Map metric names to real data columns
            real_col_map = {
                'overall_happiness': 'happiness_score',
                'gdp': 'gdp',
                'social_support': 'social_support',
                'health': 'life_expectancy',
                'freedom': 'freedom',
                'generosity': 'generosity',
                'corruption': 'corruption'
            }
            real_col = real_col_map.get(metric)
            if real_col and real_col in df_real.columns:
                real_mean = df_real[real_col].mean()
            else:
                real_mean = np.nan
        else:
            real_mean = np.nan
        
        comparison_data.append({
            'metric': metric,
            'original_mean': orig_mean,
            'original_std': orig_std,
            'original_range': f"{orig_min:.2f}-{orig_max:.2f}" if not np.isnan(orig_min) else "N/A",
            'improved_mean': impr_mean,
            'improved_std': impr_std,
            'improved_range': f"{impr_min:.2f}-{impr_max:.2f}" if not np.isnan(impr_min) else "N/A",
            'real_mean': real_mean,
            'reduction': orig_mean - impr_mean if not np.isnan(orig_mean) and not np.isnan(impr_mean) else np.nan
        })
        
        # Print comparison
        print(f"\n{metric.upper().replace('_', ' ')}:")
        if not np.isnan(orig_mean):
            print(f"  Original:  Mean={orig_mean:.2f}, Std={orig_std:.2f}, Range={orig_min:.2f}-{orig_max:.2f}")
        if not np.isnan(impr_mean):
            print(f"  Improved:  Mean={impr_mean:.2f}, Std={impr_std:.2f}, Range={impr_min:.2f}-{impr_max:.2f}")
        if not np.isnan(real_mean):
            print(f"  Real Data: Mean={real_mean:.2f}")
        if not np.isnan(orig_mean) and not np.isnan(impr_mean):
            reduction = orig_mean - impr_mean
            reduction_pct = (reduction / orig_mean) * 100 if orig_mean > 0 else 0
            print(f"  Reduction: {reduction:.2f} points ({reduction_pct:.1f}%)")
    
    # Score distribution comparison
    print("\n" + "="*70)
    print("SCORE DISTRIBUTION COMPARISON")
    print("="*70)
    
    if df_original is not None and 'overall_happiness' in df_original.columns:
        orig_dist = pd.cut(df_original['overall_happiness'], 
                          bins=[0, 2, 4, 5, 6, 7, 8, 10], 
                          labels=['1-2', '3-4', '4-5', '5-6', '6-7', '7-8', '8-10'])
        orig_dist_counts = orig_dist.value_counts().sort_index()
        print("\nOriginal Approach Distribution:")
        for range_label, count in orig_dist_counts.items():
            pct = (count / len(df_original)) * 100
            print(f"  {range_label}: {count:4d} ({pct:5.1f}%)")
    
    if df_improved is not None and 'overall_happiness' in df_improved.columns:
        impr_dist = pd.cut(df_improved['overall_happiness'], 
                          bins=[0, 2, 4, 5, 6, 7, 8, 10], 
                          labels=['1-2', '3-4', '4-5', '5-6', '6-7', '7-8', '8-10'])
        impr_dist_counts = impr_dist.value_counts().sort_index()
        print("\nFew-Shot Approach Distribution:")
        for range_label, count in impr_dist_counts.items():
            pct = (count / len(df_improved)) * 100
            print(f"  {range_label}: {count:4d} ({pct:5.1f}%)")
    
    # Country-level comparison
    if df_original is not None and df_improved is not None:
        print("\n" + "="*70)
        print("COUNTRY-LEVEL COMPARISON (Top 10 by difference)")
        print("="*70)
        
        orig_country = df_original.groupby('country')['overall_happiness'].mean().reset_index()
        orig_country.columns = ['country', 'original_mean']
        
        impr_country = df_improved.groupby('country')['overall_happiness'].mean().reset_index()
        impr_country.columns = ['country', 'improved_mean']
        
        country_comp = orig_country.merge(impr_country, on='country', how='inner')
        country_comp['difference'] = country_comp['original_mean'] - country_comp['improved_mean']
        country_comp = country_comp.sort_values('difference', ascending=False)
        
        print("\nCountries with largest reduction (top 10):")
        for _, row in country_comp.head(10).iterrows():
            print(f"  {row['country']:30s} Original: {row['original_mean']:5.2f} → Improved: {row['improved_mean']:5.2f} (Δ={row['difference']:5.2f})")
    
    # Save comparison to CSV
    comparison_df = pd.DataFrame(comparison_data)
    output_file = os.path.join(current_dir, 'results_improved', 'approach_comparison.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    comparison_df.to_csv(output_file, index=False)
    print(f"\nComparison saved to: {output_file}")
    
    print("\n" + "="*70)
    print("Comparison Complete!")
    print("="*70)


if __name__ == "__main__":
    compare_approaches()
