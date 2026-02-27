"""
Group Comparisons: Analyze differences by region and income level
"""

import pandas as pd
import numpy as np
from scipy import stats


def _latest_historical_year(df, cutoff_year: int = 2024) -> int:
    """
    Return the latest year strictly before cutoff_year if available,
    otherwise fall back to the maximum year present.
    Used so that group comparisons are based on historical data up to,
    but not including, 2024.
    """
    years = sorted(df['Year'].dropna().unique())
    hist_years = [y for y in years if y < cutoff_year]
    return max(hist_years) if hist_years else years[-1]


def compare_regions(df):
    """Compare happiness metrics across regions"""
    latest_year = _latest_historical_year(df)
    df_latest = df[df['Year'] == latest_year].copy()
    
    region_stats = df_latest.groupby('region').agg({
        'happiness_score': ['mean', 'median', 'std', 'count'],
        'gdp': 'mean',
        'social_support': 'mean',
        'life_expectancy': 'mean',
        'freedom': 'mean',
        'generosity': 'mean',
        'corruption': 'mean'
    }).reset_index()
    
    region_stats.columns = ['region', 'avg_happiness', 'median_happiness', 
                           'std_happiness', 'country_count', 'avg_gdp',
                           'avg_social_support', 'avg_life_expectancy',
                           'avg_freedom', 'avg_generosity', 'avg_corruption']
    
    region_stats = region_stats.sort_values('avg_happiness', ascending=False)
    
    return region_stats


def compare_income_levels(df):
    """Compare happiness metrics across income levels"""
    latest_year = _latest_historical_year(df)
    df_latest = df[df['Year'] == latest_year].copy()
    
    income_stats = df_latest.groupby('income_level').agg({
        'happiness_score': ['mean', 'median', 'std', 'count'],
        'gdp': 'mean',
        'social_support': 'mean',
        'life_expectancy': 'mean',
        'freedom': 'mean',
        'generosity': 'mean',
        'corruption': 'mean'
    }).reset_index()
    
    income_stats.columns = ['income_level', 'avg_happiness', 'median_happiness',
                            'std_happiness', 'country_count', 'avg_gdp',
                            'avg_social_support', 'avg_life_expectancy',
                            'avg_freedom', 'avg_generosity', 'avg_corruption']
    
    # Order by income level
    income_order = ['High Income', 'Upper Middle Income', 'Lower Middle Income', 'Low Income']
    income_stats['income_order'] = income_stats['income_level'].map(
        {level: i for i, level in enumerate(income_order)}
    )
    income_stats = income_stats.sort_values('income_order')
    
    return income_stats


def statistical_significance_test(df, group_col='region'):
    """Test statistical significance of differences between groups"""
    latest_year = _latest_historical_year(df)
    df_latest = df[df['Year'] == latest_year].copy()
    
    groups = df_latest[group_col].unique()
    results = []
    
    for i, group1 in enumerate(groups):
        for group2 in groups[i+1:]:
            group1_scores = df_latest[df_latest[group_col] == group1]['happiness_score'].dropna()
            group2_scores = df_latest[df_latest[group_col] == group2]['happiness_score'].dropna()
            
            if len(group1_scores) > 2 and len(group2_scores) > 2:
                t_stat, p_value = stats.ttest_ind(group1_scores, group2_scores)
                results.append({
                    'group1': group1,
                    'group2': group2,
                    'mean1': group1_scores.mean(),
                    'mean2': group2_scores.mean(),
                    'difference': group1_scores.mean() - group2_scores.mean(),
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
    
    return pd.DataFrame(results)


def get_factor_differences_by_group(df, group_col='region'):
    """Compare how different factors vary across groups"""
    latest_year = _latest_historical_year(df)
    df_latest = df[df['Year'] == latest_year].copy()
    
    factors = ['gdp', 'social_support', 'life_expectancy', 'freedom', 
               'generosity', 'corruption']
    
    factor_comparison = []
    for factor in factors:
        factor_data = df_latest.groupby(group_col)[factor].mean().reset_index()
        factor_data['factor'] = factor
        factor_data.columns = [group_col, 'avg_value', 'factor']
        factor_comparison.append(factor_data)
    
    return pd.concat(factor_comparison, ignore_index=True)


def get_happiness_gap_analysis(df):
    """Analyze the gap between highest and lowest happiness groups"""
    latest_year = _latest_historical_year(df)
    df_latest = df[df['Year'] == latest_year].copy()
    
    # By region
    region_gaps = df_latest.groupby('region')['happiness_score'].mean()
    region_gap = region_gaps.max() - region_gaps.min()
    
    # By income
    income_gaps = df_latest.groupby('income_level')['happiness_score'].mean()
    income_gap = income_gaps.max() - income_gaps.min()
    
    return {
        'region_gap': region_gap,
        'income_gap': income_gap,
        'highest_region': region_gaps.idxmax(),
        'lowest_region': region_gaps.idxmin(),
        'highest_income': income_gaps.idxmax(),
        'lowest_income': income_gaps.idxmin()
    }

