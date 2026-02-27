"""
Analysis Script: Compare LLM Results with Real World Happiness Data

This script analyzes the LLM audit results and compares them with real World Happiness
Report data. It calculates averages for each country from LLM responses and compares
them with actual 2024 data (or latest available year) from the dataset.

Features:
- Calculates country-level averages from LLM responses
- Compares with real World Happiness Report data
- Computes differences and correlations for all metrics
- Identifies countries with largest/smallest differences
- Generates detailed comparison CSV and statistics report

Usage:
    python analyze_llm_vs_real.py
"""

import pandas as pd
import os
import sys

# Add parent directory to path to import data_loader
sys.path.append('..')
from data_loader import load_data, clean_data

def load_llm_results(csv_file="results/llm_audit_results.csv"):
    """Load LLM audit results from CSV"""
    if not os.path.exists(csv_file):
        print(f"Error: LLM results file not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} LLM responses")
    return df

def load_real_data():
    """
    Load real World Happiness Report data for 2024.

    Uses the main Excel dataset via load_data/clean_data and then filters
    to Year == 2024 (or the latest available year if 2024 is not present).
    Internally, load_data will resolve the correct Excel file
    (e.g., 'dataset.xlsx' in this project).
    """
    # Let load_data pick the correct underlying Excel file
    df_raw = load_data('data.xlsx')
    df = clean_data(df_raw)

    if 'Year' in df.columns:
        available_years = sorted(df['Year'].unique(), reverse=True)
        target_year = 2024 if 2024 in available_years else available_years[0]
        df_2024 = df[df['Year'] == target_year].copy()
        print(f"Using data from year {target_year} (requested 2024)")
    else:
        df_2024 = df.copy()
        print("No Year column found, using all available data")

    return df_2024

def calculate_llm_averages(df_llm):
    """Calculate average scores for each country from LLM results"""
    if df_llm is None or len(df_llm) == 0:
        return pd.DataFrame()
    
    # Group by country and calculate means
    llm_avg = df_llm.groupby('country').agg({
        'overall_happiness': 'mean',
        'overall_happiness_explicit': 'mean',
        'overall_happiness_calculated': 'mean',
        'life_evaluation': 'mean',
        'gdp': 'mean',
        'social_support': 'mean',
        'health': 'mean',
        'freedom': 'mean',
        'generosity': 'mean',
        'corruption': 'mean'
    }).reset_index()
    
    # Rename columns to indicate they're LLM averages
    llm_avg.columns = ['country', 'llm_overall_happiness', 'llm_overall_explicit', 
                       'llm_overall_calculated', 'llm_life_evaluation', 'llm_gdp',
                       'llm_social_support', 'llm_health', 'llm_freedom', 
                       'llm_generosity', 'llm_corruption']
    
    return llm_avg

def prepare_real_data(df_real):
    """Prepare real data for comparison"""
    # Map real data columns to match LLM structure
    # Note: Real data uses 'happiness_score' for life evaluation
    # and 'life_expectancy' for health
    
    comparison_cols = ['country']
    
    if 'happiness_score' in df_real.columns:
        comparison_cols.append('happiness_score')
    if 'gdp' in df_real.columns:
        comparison_cols.append('gdp')
    if 'social_support' in df_real.columns:
        comparison_cols.append('social_support')
    if 'life_expectancy' in df_real.columns:
        comparison_cols.append('life_expectancy')
    if 'freedom' in df_real.columns:
        comparison_cols.append('freedom')
    if 'generosity' in df_real.columns:
        comparison_cols.append('generosity')
    if 'corruption' in df_real.columns:
        comparison_cols.append('corruption')
    
    df_comparison = df_real[comparison_cols].copy()
    
    # Rename to match LLM column names
    rename_map = {
        'happiness_score': 'real_life_evaluation',
        'life_expectancy': 'real_health',
        'gdp': 'real_gdp',
        'social_support': 'real_social_support',
        'freedom': 'real_freedom',
        'generosity': 'real_generosity',
        'corruption': 'real_corruption'
    }
    
    df_comparison = df_comparison.rename(columns=rename_map)
    
    # Add overall happiness (same as life evaluation in real data)
    if 'real_life_evaluation' in df_comparison.columns:
        df_comparison['real_overall_happiness'] = df_comparison['real_life_evaluation']
    
    return df_comparison

def compare_llm_vs_real(llm_avg, real_data):
    """Compare LLM averages with real data"""
    # Merge on country
    comparison = pd.merge(llm_avg, real_data, on='country', how='inner')
    
    if len(comparison) == 0:
        print("Warning: No matching countries found between LLM and real data")
        return pd.DataFrame()
    
    # Calculate differences
    if 'real_overall_happiness' in comparison.columns and 'llm_overall_happiness' in comparison.columns:
        comparison['diff_overall_happiness'] = comparison['llm_overall_happiness'] - comparison['real_overall_happiness']
        comparison['abs_diff_overall_happiness'] = abs(comparison['diff_overall_happiness'])
    
    if 'real_life_evaluation' in comparison.columns and 'llm_life_evaluation' in comparison.columns:
        comparison['diff_life_evaluation'] = comparison['llm_life_evaluation'] - comparison['real_life_evaluation']
        comparison['abs_diff_life_evaluation'] = abs(comparison['diff_life_evaluation'])
    
    if 'real_gdp' in comparison.columns and 'llm_gdp' in comparison.columns:
        comparison['diff_gdp'] = comparison['llm_gdp'] - comparison['real_gdp']
        comparison['abs_diff_gdp'] = abs(comparison['diff_gdp'])
    
    if 'real_social_support' in comparison.columns and 'llm_social_support' in comparison.columns:
        comparison['diff_social_support'] = comparison['llm_social_support'] - comparison['real_social_support']
        comparison['abs_diff_social_support'] = abs(comparison['diff_social_support'])
    
    if 'real_health' in comparison.columns and 'llm_health' in comparison.columns:
        comparison['diff_health'] = comparison['llm_health'] - comparison['real_health']
        comparison['abs_diff_health'] = abs(comparison['diff_health'])
    
    if 'real_freedom' in comparison.columns and 'llm_freedom' in comparison.columns:
        comparison['diff_freedom'] = comparison['llm_freedom'] - comparison['real_freedom']
        comparison['abs_diff_freedom'] = abs(comparison['diff_freedom'])
    
    if 'real_generosity' in comparison.columns and 'llm_generosity' in comparison.columns:
        comparison['diff_generosity'] = comparison['llm_generosity'] - comparison['real_generosity']
        comparison['abs_diff_generosity'] = abs(comparison['diff_generosity'])
    
    if 'real_corruption' in comparison.columns and 'llm_corruption' in comparison.columns:
        comparison['diff_corruption'] = comparison['llm_corruption'] - comparison['real_corruption']
        comparison['abs_diff_corruption'] = abs(comparison['diff_corruption'])
    
    return comparison

def calculate_statistics(comparison):
    """Calculate summary statistics for the comparison"""
    if len(comparison) == 0:
        return {}
    
    stats = {}
    
    # Overall happiness
    if 'diff_overall_happiness' in comparison.columns:
        stats['overall_happiness'] = {
            'mean_diff': comparison['diff_overall_happiness'].mean(),
            'std_diff': comparison['diff_overall_happiness'].std(),
            'mean_abs_diff': comparison['abs_diff_overall_happiness'].mean(),
            'correlation': comparison['llm_overall_happiness'].corr(comparison['real_overall_happiness']) if 'real_overall_happiness' in comparison.columns else None
        }
    
    # Life evaluation
    if 'diff_life_evaluation' in comparison.columns:
        stats['life_evaluation'] = {
            'mean_diff': comparison['diff_life_evaluation'].mean(),
            'std_diff': comparison['diff_life_evaluation'].std(),
            'mean_abs_diff': comparison['abs_diff_life_evaluation'].mean(),
            'correlation': comparison['llm_life_evaluation'].corr(comparison['real_life_evaluation']) if 'real_life_evaluation' in comparison.columns else None
        }
    
    # GDP
    if 'diff_gdp' in comparison.columns:
        stats['gdp'] = {
            'mean_diff': comparison['diff_gdp'].mean(),
            'std_diff': comparison['diff_gdp'].std(),
            'mean_abs_diff': comparison['abs_diff_gdp'].mean(),
            'correlation': comparison['llm_gdp'].corr(comparison['real_gdp']) if 'real_gdp' in comparison.columns else None
        }
    
    # Social Support
    if 'diff_social_support' in comparison.columns:
        stats['social_support'] = {
            'mean_diff': comparison['diff_social_support'].mean(),
            'std_diff': comparison['diff_social_support'].std(),
            'mean_abs_diff': comparison['abs_diff_social_support'].mean(),
            'correlation': comparison['llm_social_support'].corr(comparison['real_social_support']) if 'real_social_support' in comparison.columns else None
        }
    
    # Health
    if 'diff_health' in comparison.columns:
        stats['health'] = {
            'mean_diff': comparison['diff_health'].mean(),
            'std_diff': comparison['diff_health'].std(),
            'mean_abs_diff': comparison['abs_diff_health'].mean(),
            'correlation': comparison['llm_health'].corr(comparison['real_health']) if 'real_health' in comparison.columns else None
        }
    
    # Freedom
    if 'diff_freedom' in comparison.columns:
        stats['freedom'] = {
            'mean_diff': comparison['diff_freedom'].mean(),
            'std_diff': comparison['diff_freedom'].std(),
            'mean_abs_diff': comparison['abs_diff_freedom'].mean(),
            'correlation': comparison['llm_freedom'].corr(comparison['real_freedom']) if 'real_freedom' in comparison.columns else None
        }
    
    # Generosity
    if 'diff_generosity' in comparison.columns:
        stats['generosity'] = {
            'mean_diff': comparison['diff_generosity'].mean(),
            'std_diff': comparison['diff_generosity'].std(),
            'mean_abs_diff': comparison['abs_diff_generosity'].mean(),
            'correlation': comparison['llm_generosity'].corr(comparison['real_generosity']) if 'real_generosity' in comparison.columns else None
        }
    
    # Corruption
    if 'diff_corruption' in comparison.columns:
        stats['corruption'] = {
            'mean_diff': comparison['diff_corruption'].mean(),
            'std_diff': comparison['diff_corruption'].std(),
            'mean_abs_diff': comparison['abs_diff_corruption'].mean(),
            'correlation': comparison['llm_corruption'].corr(comparison['real_corruption']) if 'real_corruption' in comparison.columns else None
        }
    
    return stats

def print_summary(comparison, stats):
    """Print summary of comparison"""
    print("\n" + "="*80)
    print("LLM vs Real Data Comparison Summary")
    print("="*80)
    
    print(f"\nTotal countries compared: {len(comparison)}")
    
    if len(comparison) == 0:
        return
    
    print("\n" + "-"*80)
    print("Summary Statistics by Metric")
    print("-"*80)
    
    for metric, metric_stats in stats.items():
        print(f"\n{metric.upper().replace('_', ' ')}:")
        print(f"  Mean Difference (LLM - Real): {metric_stats['mean_diff']:.3f}")
        print(f"  Std Dev of Differences: {metric_stats['std_diff']:.3f}")
        print(f"  Mean Absolute Difference: {metric_stats['mean_abs_diff']:.3f}")
        if metric_stats['correlation'] is not None:
            print(f"  Correlation: {metric_stats['correlation']:.3f}")
    
    # Top and bottom countries by absolute difference in overall happiness
    if 'abs_diff_overall_happiness' in comparison.columns:
        print("\n" + "-"*80)
        print("Countries with Largest Differences (Overall Happiness)")
        print("-"*80)
        top_diff = comparison.nlargest(10, 'abs_diff_overall_happiness')[
            ['country', 'llm_overall_happiness', 'real_overall_happiness', 'diff_overall_happiness', 'abs_diff_overall_happiness']
        ]
        print("\nTop 10 Largest Differences:")
        print(top_diff.to_string(index=False))
        
        print("\nTop 10 Smallest Differences (Most Accurate):")
        smallest_diff = comparison.nsmallest(10, 'abs_diff_overall_happiness')[
            ['country', 'llm_overall_happiness', 'real_overall_happiness', 'diff_overall_happiness', 'abs_diff_overall_happiness']
        ]
        print(smallest_diff.to_string(index=False))

def main():
    """Main analysis function"""
    print("="*80)
    print("LLM Audit Results vs Real Data Analysis")
    print("="*80)
    
    # Load LLM results
    print("\nLoading LLM audit results...")
    df_llm = load_llm_results()
    if df_llm is None:
        return
    
    # Calculate LLM averages by country
    print("\nCalculating LLM averages by country...")
    llm_avg = calculate_llm_averages(df_llm)
    print(f"Found {len(llm_avg)} countries in LLM results")
    
    # Load real data
    print("\nLoading real World Happiness Report data...")
    real_data = load_real_data()
    print(f"Found {len(real_data)} countries in real data")
    
    # Prepare real data for comparison
    real_prepared = prepare_real_data(real_data)
    
    # Compare
    print("\nComparing LLM vs Real data...")
    comparison = compare_llm_vs_real(llm_avg, real_prepared)
    
    if len(comparison) == 0:
        print("No countries to compare. Exiting.")
        return
    
    print(f"Successfully compared {len(comparison)} countries")
    
    # Calculate statistics
    stats = calculate_statistics(comparison)
    
    # Print summary
    print_summary(comparison, stats)
    
    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "llm_vs_real_comparison.csv")
    comparison.to_csv(output_file, index=False)
    print(f"\n" + "="*80)
    print(f"Detailed comparison saved to: {output_file}")
    print("="*80)
    
    # Save summary statistics
    stats_file = os.path.join(output_dir, "llm_vs_real_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write("LLM vs Real Data Comparison Statistics\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total countries compared: {len(comparison)}\n\n")
        
        for metric, metric_stats in stats.items():
            f.write(f"{metric.upper().replace('_', ' ')}:\n")
            f.write(f"  Mean Difference (LLM - Real): {metric_stats['mean_diff']:.3f}\n")
            f.write(f"  Std Dev of Differences: {metric_stats['std_diff']:.3f}\n")
            f.write(f"  Mean Absolute Difference: {metric_stats['mean_abs_diff']:.3f}\n")
            if metric_stats['correlation'] is not None:
                f.write(f"  Correlation: {metric_stats['correlation']:.3f}\n")
            f.write("\n")
    
    print(f"Summary statistics saved to: {stats_file}")

if __name__ == "__main__":
    main()
