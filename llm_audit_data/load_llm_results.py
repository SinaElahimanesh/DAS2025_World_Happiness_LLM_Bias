"""
Load Real LLM Audit Results for Web Interface

This module loads the actual LLM audit results from CSV files from all three approaches
and prepares them for visualization in the web interface.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path for data_loader
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    from data_loader import load_data, clean_data
except ImportError:
    load_data = None
    clean_data = None

def load_llm_results_from_approach(approach_name, approach_path):
    """Load LLM audit results from a specific approach"""
    results_file = approach_path / "results" / "llm_audit_results.csv"
    
    if not results_file.exists():
        return None
    
    try:
        df = pd.read_csv(results_file)
        df['approach'] = approach_name
        return df
    except Exception as e:
        print(f"Warning: Could not load results from {approach_name}: {e}")
        return None

def get_all_llm_results():
    """Load LLM audit results from all three approaches"""
    base_dir = Path(__file__).parent
    
    approaches = {
        'initial': base_dir / "initial_approach",
        'few_shot': base_dir / "few_shot_approach",
        'single_question': base_dir / "single_question_gallup_approach"
    }
    
    all_results = []
    for approach_name, approach_path in approaches.items():
        if approach_path.exists():
            df = load_llm_results_from_approach(approach_name, approach_path)
            if df is not None and len(df) > 0:
                all_results.append(df)
    
    if not all_results:
        return None
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df

def calculate_country_averages(df):
    """Calculate country-level averages from LLM results"""
    if df is None or len(df) == 0:
        return None
    
    # Group by country and approach, calculate means
    country_stats = df.groupby(['country', 'approach']).agg({
        'overall_happiness': 'mean',
        'gdp': 'mean',
        'social_support': 'mean',
        'health': 'mean',
        'freedom': 'mean',
        'generosity': 'mean',
        'corruption': 'mean'
    }).reset_index()
    
    return country_stats

def _build_comparison_from_raw():
    """Build LLM vs Real comparison from raw results (has 'approach' column). Returns None on failure."""
    if load_data is None:
        return None
    data_file = parent_dir / "data.xlsx"
    if not data_file.exists():
        return None
    try:
        df_real = load_data(str(data_file))
        df_real = clean_data(df_real)
        latest_year = df_real['Year'].max()
        df_real_latest = df_real[df_real['Year'] == latest_year].copy()
        llm_results = get_all_llm_results()
        if llm_results is None:
            return None
        llm_country_stats = calculate_country_averages(llm_results)
        if llm_country_stats is None:
            return None
        real_cols = ['country', 'happiness_score', 'gdp', 'social_support',
                     'life_expectancy', 'freedom', 'generosity', 'corruption']
        real_data = df_real_latest[real_cols].copy()
        real_data = real_data.rename(columns={
            'happiness_score': 'real_overall_happiness',
            'gdp': 'real_gdp',
            'social_support': 'real_social_support',
            'life_expectancy': 'real_health',
            'freedom': 'real_freedom',
            'generosity': 'real_generosity',
            'corruption': 'real_corruption'
        })
        if 'real_life_evaluation' not in real_data.columns:
            real_data['real_life_evaluation'] = real_data['real_overall_happiness']
        comparison_list = []
        for approach in llm_country_stats['approach'].unique():
            llm_approach = llm_country_stats[llm_country_stats['approach'] == approach].copy()
            llm_approach = llm_approach.rename(columns={
                'overall_happiness': 'llm_overall_happiness',
                'gdp': 'llm_gdp',
                'social_support': 'llm_social_support',
                'health': 'llm_health',
                'freedom': 'llm_freedom',
                'generosity': 'llm_generosity',
                'corruption': 'llm_corruption'
            })
            if 'life_evaluation' in llm_approach.columns:
                llm_approach = llm_approach.rename(columns={'life_evaluation': 'llm_life_evaluation'})
            else:
                llm_approach['llm_life_evaluation'] = llm_approach['llm_overall_happiness']
            merged = pd.merge(real_data, llm_approach, on='country', how='inner')
            merged['diff_overall_happiness'] = merged['llm_overall_happiness'] - merged['real_overall_happiness']
            merged['diff_life_evaluation'] = merged['llm_life_evaluation'] - merged['real_life_evaluation']
            if 'llm_gdp' in merged.columns and 'real_gdp' in merged.columns:
                merged['diff_gdp'] = merged['llm_gdp'] - merged['real_gdp']
            if 'llm_social_support' in merged.columns and 'real_social_support' in merged.columns:
                merged['diff_social_support'] = merged['llm_social_support'] - merged['real_social_support']
            if 'llm_health' in merged.columns and 'real_health' in merged.columns:
                merged['diff_health'] = merged['llm_health'] - merged['real_health']
            if 'llm_freedom' in merged.columns and 'real_freedom' in merged.columns:
                merged['diff_freedom'] = merged['llm_freedom'] - merged['real_freedom']
            if 'llm_generosity' in merged.columns and 'real_generosity' in merged.columns:
                merged['diff_generosity'] = merged['llm_generosity'] - merged['real_generosity']
            if 'llm_corruption' in merged.columns and 'real_corruption' in merged.columns:
                merged['diff_corruption'] = merged['llm_corruption'] - merged['real_corruption']
            comparison_list.append(merged)
        if not comparison_list:
            return None
        return pd.concat(comparison_list, ignore_index=True)
    except Exception as e:
        return None


def get_latest_llm_comparison():
    """Load and create LLM vs Real comparison data from all three approaches.
    Prefers raw-built data (with 'approach' column) so dashboard can filter by approach.
    Falls back to pre-computed CSV if raw build fails."""
    # Prefer building from raw so we have 'approach' for per-approach filtering
    built = _build_comparison_from_raw()
    if built is not None and len(built) > 0:
        return built
    results_dir = Path(__file__).parent / "results"
    comparison_file = results_dir / "llm_vs_real_comparison.csv"
    if comparison_file.exists():
        return pd.read_csv(comparison_file)
    return None

def compute_bias_from_comparison(comparison_df):
    """
    Compute bias summary, bias data (with groupings), and significant findings
    from a comparison DataFrame (e.g. filtered by approach).
    Returns (bias_summary_df, bias_data_df, significant_df) or (None, None, None) on failure.
    """
    if comparison_df is None or len(comparison_df) == 0:
        return None, None, None
    try:
        base = Path(__file__).parent
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        from analyze_bias import add_groupings, analyze_all_groupings, calculate_bias_summary
    except Exception:
        return None, None, None
    df = comparison_df.copy()
    if 'real_life_evaluation' not in df.columns and 'real_overall_happiness' in df.columns:
        df['real_life_evaluation'] = df['real_overall_happiness']
    if 'diff_life_evaluation' not in df.columns and 'diff_overall_happiness' in df.columns:
        df['diff_life_evaluation'] = df['diff_overall_happiness']
    df = add_groupings(df)
    results = analyze_all_groupings(df, silent=True)
    if not results:
        return None, None, None
    bias_summary_df = calculate_bias_summary(results)
    significant_df = bias_summary_df[
        (bias_summary_df['significantly_different_from_real'] == True) |
        ((bias_summary_df['p_value_vs_real'].notna()) & (bias_summary_df['p_value_vs_real'] < 0.05))
    ].copy()
    return bias_summary_df, df, significant_df


def get_latest_bias_summary():
    """Load the latest bias summary with statistical tests"""
    results_dir = Path(__file__).parent / "results"
    
    bias_files = list(results_dir.glob("bias_summary_*.csv"))
    if not bias_files:
        return None
    
    latest_file = sorted(bias_files)[-1]
    df = pd.read_csv(latest_file)
    return df

def get_latest_bias_data():
    """Load the latest detailed bias analysis data with groupings"""
    results_dir = Path(__file__).parent / "results"
    
    data_files = list(results_dir.glob("bias_analysis_data_*.csv"))
    if not data_files:
        return None
    
    latest_file = sorted(data_files)[-1]
    df = pd.read_csv(latest_file)
    return df

def get_latest_significant_findings():
    """Load the latest significant findings"""
    results_dir = Path(__file__).parent / "results"
    
    sig_files = list(results_dir.glob("significant_findings_*.csv"))
    if not sig_files:
        return None
    
    latest_file = sorted(sig_files)[-1]
    df = pd.read_csv(latest_file)
    return df

def prepare_llm_comparison_for_web(comparison_df):
    """Prepare LLM comparison data for web visualization"""
    if comparison_df is None:
        return None
    
    # Calculate overall statistics
    stats = {
        'total_countries': len(comparison_df),
        'mean_bias_overall': comparison_df['diff_overall_happiness'].mean(),
        'mean_bias_gdp': comparison_df['diff_gdp'].mean(),
        'mean_bias_social': comparison_df['diff_social_support'].mean(),
        'mean_bias_health': comparison_df['diff_health'].mean(),
        'mean_bias_freedom': comparison_df['diff_freedom'].mean(),
        'mean_bias_generosity': comparison_df['diff_generosity'].mean(),
        'mean_bias_corruption': comparison_df['diff_corruption'].mean(),
    }
    
    return comparison_df, stats

def prepare_bias_summary_for_web(bias_summary_df):
    """Prepare bias summary for web visualization"""
    if bias_summary_df is None:
        return None, None
    
    # Filter for significant findings
    significant = bias_summary_df[bias_summary_df['p_value_vs_real'] < 0.05].copy()
    
    # Group by metric and grouping
    by_metric = {}
    by_grouping = {}
    
    for metric in bias_summary_df['metric'].unique():
        metric_data = bias_summary_df[bias_summary_df['metric'] == metric]
        by_metric[metric] = metric_data
    
    for grouping in bias_summary_df['grouping'].unique():
        grouping_data = bias_summary_df[bias_summary_df['grouping'] == grouping]
        by_grouping[grouping] = grouping_data
    
    return bias_summary_df, {
        'significant': significant,
        'by_metric': by_metric,
        'by_grouping': by_grouping,
        'total_tests': len(bias_summary_df),
        'significant_count': len(significant),
        'highly_significant': len(significant[significant['p_value_vs_real'] < 0.001])
    }


def compute_simplified_significance_tests(comparison_df):
    """
    Compute simplified significance tests for specific groupings only:
    1. Continent
    2. World 1/2/3
    3. Region (East Asia, Western Europe, Eastern Europe, etc.)
    4. Developed/Undevloped (simplified from world_123)
    5. Income Level (High/Low income)
    
    Focuses on overall happiness (diff_overall_happiness) and compares LLM vs Real.
    Returns a DataFrame with results for each grouping and group value.
    """
    if comparison_df is None or len(comparison_df) == 0:
        return None
    
    try:
        base = Path(__file__).parent
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        from analyze_bias import add_groupings, calculate_group_bias
        from scipy import stats
    except Exception:
        return None
    
    df = comparison_df.copy()
    if 'real_life_evaluation' not in df.columns and 'real_overall_happiness' in df.columns:
        df['real_life_evaluation'] = df['real_overall_happiness']
    if 'diff_life_evaluation' not in df.columns and 'diff_overall_happiness' in df.columns:
        df['diff_life_evaluation'] = df['diff_overall_happiness']
    
    # Add groupings
    df = add_groupings(df)
    
    # Define simplified groupings (only what user wants)
    simplified_groupings = {
        'continent': 'Continent',
        'world_123': 'World 1/2/3',
        'region': 'Region'
    }
    
    # Add developed/undeveloped grouping
    if 'world_123' in df.columns:
        df['developed_undeveloped'] = df['world_123'].apply(
            lambda x: 'Developed' if x == 'World 1' else 'Undeveloped' if x in ['World 2', 'World 3'] else 'Unknown'
        )
        simplified_groupings['developed_undeveloped'] = 'Developed/Undeveloped'
    
    results = []
    metric = 'diff_overall_happiness'
    
    if metric not in df.columns:
        return None
    
    for group_col, group_name in simplified_groupings.items():
        if group_col not in df.columns:
            continue
        
        group_stats = calculate_group_bias(df, group_col, metric)
        if group_stats is not None and len(group_stats) > 0:
            for _, row in group_stats.iterrows():
                group_val = row[group_col]
                
                # Get LLM and Real means for this group
                group_data = df[df[group_col] == group_val]
                llm_mean = group_data['llm_overall_happiness'].mean() if 'llm_overall_happiness' in group_data.columns else np.nan
                real_mean = group_data['real_overall_happiness'].mean() if 'real_overall_happiness' in group_data.columns else np.nan
                
                results.append({
                    'grouping': group_col,
                    'grouping_name': group_name,
                    'group_value': group_val,
                    'metric': 'Overall Happiness',
                    'llm_mean': llm_mean,
                    'real_mean': real_mean,
                    'mean_bias': row['mean_bias'],
                    'count': int(row['count']),
                    'p_value_vs_real': row.get('p_value_vs_real', np.nan),
                    't_statistic': row.get('t_statistic', np.nan),
                    'significantly_different': row.get('significantly_different_from_real', False)
                })
    
    return pd.DataFrame(results)
