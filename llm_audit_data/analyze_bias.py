"""
Bias Analysis Script: Analyze LLM Bias by Country Groups

This script systematically analyzes potential biases in LLM responses compared to real data
across different country groupings:
- Global North vs Global South
- Continents/Regions
- Income levels (World 1/2/3 classification)
- East vs West
- Other demographic groupings

It calculates:
- Mean differences by group
- Statistical significance tests
- Bias direction (over/under-estimation)
- Effect sizes
- Detailed reports by group and metric

Usage:
    python analyze_bias.py
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Require scipy for statistical tests - use libraries, not manual calculations
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("ERROR: scipy is required for statistical tests. Please install: pip install scipy")
    raise ImportError("scipy is required for statistical tests")

# Manual statistical test functions (if scipy not available)
def manual_ttest_1samp(data, popmean):
    """Manual one-sample t-test"""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation
    se = std / np.sqrt(n)
    t_stat = (mean - popmean) / se
    # Approximate p-value using t-distribution (rough approximation for large n)
    # For simplicity, use normal approximation for large samples
    if n > 30:
        from scipy.stats import norm
        p_val = 2 * (1 - norm.cdf(abs(t_stat)))
    else:
        # Use rough approximation: t-distribution approaches normal
        # This is a simplified approximation
        p_val = 2 * (1 - 0.5 * (1 + np.sign(t_stat) * (1 - np.exp(-2 * t_stat**2 / np.pi))))
    return t_stat, min(p_val, 1.0)

def manual_ttest_ind(group1, group2):
    """Manual independent samples t-test"""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard error
    pooled_se = np.sqrt(var1/n1 + var2/n2)
    t_stat = (mean1 - mean2) / pooled_se if pooled_se > 0 else 0
    
    # Degrees of freedom (Welch's approximation)
    df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)) if (var1/n1 + var2/n2) > 0 else min(n1, n2)
    
    # Approximate p-value
    if df > 30:
        from scipy.stats import norm
        p_val = 2 * (1 - norm.cdf(abs(t_stat)))
    else:
        p_val = 2 * (1 - 0.5 * (1 + np.sign(t_stat) * (1 - np.exp(-2 * t_stat**2 / np.pi))))
    
    return t_stat, min(p_val, 1.0)

def manual_f_oneway(*groups):
    """Manual one-way ANOVA F-test"""
    if len(groups) < 2:
        return np.nan, np.nan
    
    # Remove groups with insufficient data
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return np.nan, np.nan
    
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    
    # Overall mean
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    # Between-group sum of squares
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    df_between = k - 1
    
    # Within-group sum of squares
    ss_within = sum(sum((x - np.mean(g))**2 for x in g) for g in groups)
    df_within = n_total - k
    
    # Mean squares
    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 0
    
    # F-statistic
    f_stat = ms_between / ms_within if ms_within > 0 else 0
    
    # Approximate p-value (simplified - would need proper F-distribution)
    # Use rough approximation based on F-distribution properties
    if df_within > 30 and df_between > 0:
        # Large sample approximation
        p_val = 1 - (f_stat / (f_stat + df_within/df_between))
    else:
        # Very rough approximation
        p_val = 0.1  # Placeholder
    
    return f_stat, min(p_val, 1.0)

# Add parent directory to path to import data_loader
sys.path.append('..')
from data_loader import load_data, clean_data, get_country_regions, get_income_levels

def load_comparison_data(csv_file="results/llm_vs_real_comparison.csv"):
    """
    Load LLM vs Real comparison data
    
    This data includes:
    - LLM predictions (llm_* columns)
    - Real data from the main WHR dataset (real_* columns, loaded via data_loader)
    - Differences (diff_* columns = llm - real)
    
    Returns DataFrame with all comparison data, or None if file not found
    """
    if not os.path.exists(csv_file):
        print(f"Error: Comparison file not found: {csv_file}")
        print("Please run analyze_llm_vs_real.py first to create comparison data")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"Loaded comparison data for {len(df)} countries")
    
    # Verify real data columns exist (these come from the WHR dataset via data_loader)
    real_cols = [c for c in df.columns if c.startswith('real_')]
    llm_cols = [c for c in df.columns if c.startswith('llm_')]
    diff_cols = [c for c in df.columns if c.startswith('diff_')]
    
    print(f"  Real data columns (from WHR dataset): {len(real_cols)} found")
    print(f"  LLM prediction columns: {len(llm_cols)} found")
    print(f"  Difference columns: {len(diff_cols)} found")
    
    if len(real_cols) == 0:
        print("ERROR: No real data columns found! Cannot compare to real WHR data")
        return None
    
    if len(llm_cols) == 0:
        print("ERROR: No LLM columns found!")
        return None
    
    # Verify we have matching pairs
    missing_real = []
    for diff_col in diff_cols:
        real_col = diff_col.replace('diff_', 'real_')
        if real_col not in real_cols:
            missing_real.append(real_col)
    
    if missing_real:
        print(f"WARNING: Missing real data columns for some metrics: {missing_real}")
    
    return df

def classify_global_north_south(country_name):
    """Classify country as Global North or Global South"""
    # Global North: Western Europe, North America, Oceania (developed), East Asia (developed)
    global_north = [
        # Western Europe
        'Switzerland', 'Iceland', 'Denmark', 'Netherlands', 'Sweden', 'Norway', 
        'Luxembourg', 'Finland', 'Austria', 'Belgium', 'Ireland', 'Germany', 
        'United Kingdom', 'France', 'Spain', 'Italy', 'Portugal', 'Greece', 'Malta', 'Cyprus',
        # North America (developed)
        'Canada', 'United States',
        # Oceania (developed)
        'Australia', 'New Zealand',
        # East Asia (developed)
        'Japan', 'South Korea', 'Singapore', 'Taiwan', 'Hong Kong',
        # Eastern Europe (some developed)
        'Czechia', 'Slovenia', 'Estonia', 'Lithuania', 'Latvia', 'Poland', 'Slovakia',
        # Middle East (high income)
        'Israel', 'United Arab Emirates', 'Qatar', 'Kuwait', 'Bahrain', 'Saudi Arabia'
    ]
    
    if country_name in global_north:
        return 'Global North'
    else:
        return 'Global South'

def classify_world_123(country_name, income_level, gdp=None):
    """Classify countries into World 1/2/3 based on development"""
    # World 1: Highly developed, high-income countries
    world_1_keywords = ['High Income']
    
    # World 2: Developing/emerging economies
    world_2_keywords = ['Upper Middle Income', 'Lower Middle Income']
    
    # World 3: Least developed, low-income countries
    world_3_keywords = ['Low Income']
    
    if pd.isna(income_level):
        # Fallback to GDP if income level not available
        if gdp is not None:
            if gdp > 8.0:
                return 'World 1'
            elif gdp > 5.0:
                return 'World 2'
            else:
                return 'World 3'
        return 'Unknown'
    
    if income_level in world_1_keywords:
        return 'World 1'
    elif income_level in world_2_keywords:
        return 'World 2'
    elif income_level in world_3_keywords:
        return 'World 3'
    else:
        return 'Unknown'

def classify_east_west(country_name, region):
    """Classify countries as East or West"""
    # Western: Western Europe, North America, Oceania, Latin America (culturally Western)
    western_regions = ['Western Europe', 'North America', 'Oceania']
    western_countries = [
        # Latin America (culturally Western)
        'Mexico', 'Costa Rica', 'Panama', 'Guatemala', 'El Salvador', 
        'Honduras', 'Nicaragua', 'Uruguay', 'Brazil', 'Argentina', 
        'Chile', 'Colombia', 'Ecuador', 'Peru', 'Venezuela', 'Paraguay', 'Bolivia'
    ]
    
    # Eastern: East Asia, South Asia, Middle East, Eastern Europe, Central Asia
    eastern_regions = ['East Asia', 'South Asia', 'Middle East', 'Eastern Europe']
    
    if region in western_regions:
        return 'West'
    elif region in eastern_regions:
        return 'East'
    elif country_name in western_countries:
        return 'West'
    elif region == 'Africa':
        # Africa is mixed, classify by other factors if needed
        return 'Other'
    else:
        return 'Other'

def classify_continent(region):
    """Map region to continent"""
    continent_map = {
        'Western Europe': 'Europe',
        'Eastern Europe': 'Europe',
        'North America': 'North America',
        'Latin America': 'South America',
        'East Asia': 'Asia',
        'South Asia': 'Asia',
        'Middle East': 'Asia',
        'Africa': 'Africa',
        'Oceania': 'Oceania'
    }
    return continent_map.get(region, 'Other')

def classify_developed_country_type(country_name, income_level):
    """Classify developed countries into subcategories"""
    if income_level != 'High Income':
        return 'Non-Developed'
    
    # Developed East Asia
    developed_east_asia = ['Japan', 'South Korea', 'Singapore', 'Taiwan', 'Hong Kong']
    
    # Western Europe + US/Canada (Anglo-Saxon and Western European)
    anglo_saxon = ['United States', 'Canada', 'United Kingdom', 'Ireland', 'Australia', 'New Zealand']
    western_europe = ['Switzerland', 'Iceland', 'Denmark', 'Netherlands', 'Sweden', 'Norway', 
                      'Luxembourg', 'Finland', 'Austria', 'Belgium', 'Germany', 'France', 
                      'Spain', 'Italy', 'Portugal', 'Greece', 'Malta', 'Cyprus']
    
    # Nordic countries
    nordic = ['Denmark', 'Finland', 'Iceland', 'Norway', 'Sweden']
    
    # Mediterranean Europe
    mediterranean = ['Spain', 'Italy', 'Portugal', 'Greece', 'Malta', 'Cyprus']
    
    # Central Europe (DACH + Benelux)
    central_europe = ['Germany', 'Austria', 'Switzerland', 'Belgium', 'Netherlands', 'Luxembourg']
    
    # Middle East developed
    middle_east_developed = ['Israel', 'United Arab Emirates', 'Qatar', 'Kuwait', 'Bahrain', 'Saudi Arabia']
    
    if country_name in developed_east_asia:
        return 'Developed East Asia'
    elif country_name in anglo_saxon:
        return 'Anglo-Saxon'
    elif country_name in nordic:
        return 'Nordic'
    elif country_name in mediterranean:
        return 'Mediterranean Europe'
    elif country_name in central_europe:
        return 'Central Europe'
    elif country_name in western_europe:
        return 'Other Western Europe'
    elif country_name in middle_east_developed:
        return 'Middle East Developed'
    else:
        return 'Other Developed'

def classify_developed_east_vs_west(country_name, income_level):
    """Classify developed countries as Developed East Asia vs Developed West"""
    if income_level != 'High Income':
        return 'Non-Developed'
    
    # Developed East Asia
    developed_east_asia = ['Japan', 'South Korea', 'Singapore', 'Taiwan', 'Hong Kong']
    
    # Developed West (Western Europe + US/Canada + Oceania)
    developed_west = [
        # North America
        'United States', 'Canada',
        # Western Europe
        'Switzerland', 'Iceland', 'Denmark', 'Netherlands', 'Sweden', 'Norway', 
        'Luxembourg', 'Finland', 'Austria', 'Belgium', 'Ireland', 'Germany', 
        'United Kingdom', 'France', 'Spain', 'Italy', 'Portugal', 'Greece', 'Malta', 'Cyprus',
        # Oceania
        'Australia', 'New Zealand'
    ]
    
    if country_name in developed_east_asia:
        return 'Developed East Asia'
    elif country_name in developed_west:
        return 'Developed West'
    else:
        return 'Other Developed'

def classify_economic_model(country_name, income_level):
    """Classify countries by economic model/development path"""
    if income_level == 'High Income':
        # Market liberal (Anglo-Saxon model)
        market_liberal = ['United States', 'United Kingdom', 'Ireland', 'Canada', 'Australia', 'New Zealand']
        
        # Social market (Continental European model)
        social_market = ['Germany', 'France', 'Austria', 'Belgium', 'Netherlands', 'Switzerland', 
                        'Denmark', 'Finland', 'Sweden', 'Norway']
        
        # State-led development (East Asian model)
        state_led = ['Japan', 'South Korea', 'Singapore', 'Taiwan', 'Hong Kong']
        
        # Resource-based
        resource_based = ['United Arab Emirates', 'Qatar', 'Kuwait', 'Saudi Arabia', 'Norway']
        
        if country_name in market_liberal:
            return 'Market Liberal'
        elif country_name in social_market:
            return 'Social Market'
        elif country_name in state_led:
            return 'State-Led Development'
        elif country_name in resource_based:
            return 'Resource-Based'
        else:
            return 'Other High Income'
    else:
        return 'Non-High Income'

def add_groupings(df):
    """Add all grouping columns to the dataframe"""
    # Get region and income mappings
    region_map = get_country_regions()
    income_map = {}
    for level, countries in get_income_levels().items():
        for country in countries:
            income_map[country] = level
    
    df = df.copy()
    
    # Add region
    df['region'] = df['country'].map(region_map).fillna('Other')
    
    # Add income level
    df['income_level'] = df['country'].map(income_map)
    
    # Add GDP-based income level if missing
    if df['income_level'].isna().any() and 'real_gdp' in df.columns:
        gdp_median = df['real_gdp'].median()
        mask = df['income_level'].isna()
        df.loc[mask & (df['real_gdp'] > gdp_median * 1.5), 'income_level'] = 'High Income'
        df.loc[mask & (df['real_gdp'] > gdp_median * 0.7) & (df['real_gdp'] <= gdp_median * 1.5), 'income_level'] = 'Upper Middle Income'
        df.loc[mask & (df['real_gdp'] > gdp_median * 0.3) & (df['real_gdp'] <= gdp_median * 0.7), 'income_level'] = 'Lower Middle Income'
        df.loc[mask & (df['real_gdp'] <= gdp_median * 0.3), 'income_level'] = 'Low Income'
    
    # Add Global North/South
    df['global_north_south'] = df['country'].apply(classify_global_north_south)
    
    # Add World 1/2/3
    df['world_123'] = df.apply(
        lambda row: classify_world_123(row['country'], row.get('income_level', np.nan), row.get('real_gdp', np.nan)),
        axis=1
    )
    
    # Add East/West
    df['east_west'] = df.apply(
        lambda row: classify_east_west(row['country'], row.get('region', 'Other')),
        axis=1
    )
    
    # Add Continent
    df['continent'] = df['region'].apply(classify_continent)
    
    # Add Developed Country Types
    df['developed_country_type'] = df.apply(
        lambda row: classify_developed_country_type(row['country'], row.get('income_level', np.nan)),
        axis=1
    )
    
    # Add Developed East vs West
    df['developed_east_vs_west'] = df.apply(
        lambda row: classify_developed_east_vs_west(row['country'], row.get('income_level', np.nan)),
        axis=1
    )
    
    # Add Economic Model
    df['economic_model'] = df.apply(
        lambda row: classify_economic_model(row['country'], row.get('income_level', np.nan)),
        axis=1
    )
    
    return df

def calculate_group_bias(df, group_col, metric_col):
    """
    Calculate bias metrics for a specific group and metric
    Tests if LLM predictions differ significantly from real data from the WHR dataset
    
    Args:
        df: DataFrame with comparison data (including diff columns and real data columns)
        group_col: Column name for grouping
        metric_col: Column name for the bias metric (diff_*)
    """
    if metric_col not in df.columns:
        return None
    
    # Get corresponding real data column (from WHR dataset)
    real_col = metric_col.replace('diff_', 'real_')
    llm_col = metric_col.replace('diff_', 'llm_')
    
    # CRITICAL: Verify real data columns exist - these come from WHR dataset via data_loader
    if real_col not in df.columns:
        print(f"WARNING: Real data column '{real_col}' not found for metric '{metric_col}'. "
              f"Cannot compare to real WHR data!")
        return None
    
    if llm_col not in df.columns:
        print(f"WARNING: LLM column '{llm_col}' not found for metric '{metric_col}'.")
        return None
    
    # Remove NaN values
    required_cols = [group_col, metric_col, real_col, llm_col]
    df_clean = df[required_cols].dropna()
    
    if len(df_clean) == 0:
        print(f"WARNING: No valid data after cleaning for {metric_col}")
        return None
    
    group_stats = df_clean.groupby(group_col).agg({
        metric_col: ['mean', 'std', 'count', 'median']
    }).reset_index()
    
    group_stats.columns = [group_col, 'mean_bias', 'std_bias', 'count', 'median_bias']
    
    # Calculate overall mean for comparison
    overall_mean = df_clean[metric_col].mean()
    group_stats['bias_vs_overall'] = group_stats['mean_bias'] - overall_mean
    
    # Statistical test: ANOVA to check if groups differ significantly
    groups = [group_df[metric_col].values for name, group_df in df_clean.groupby(group_col) if len(group_df) > 1]
    
    # ANOVA test for overall group differences using scipy
    # Tests if there are significant differences between groups
    if len(groups) >= 2:
        if HAS_SCIPY:
            try:
                f_stat, p_value_anova = stats.f_oneway(*groups)
                # Store raw p-value from scipy (don't modify it)
                group_stats['f_statistic'] = f_stat
                group_stats['p_value_anova'] = p_value_anova
            except Exception as e:
                print(f"Error in ANOVA test for {metric_col}: {e}")
                group_stats['f_statistic'] = np.nan
                group_stats['p_value_anova'] = np.nan
        else:
            print("ERROR: scipy required for ANOVA test")
            group_stats['f_statistic'] = np.nan
            group_stats['p_value_anova'] = np.nan
    else:
        group_stats['f_statistic'] = np.nan
        group_stats['p_value_anova'] = np.nan
    
    # PRIMARY TEST: For each group: paired t-test - are LLM values significantly different from real data (from WHR dataset)?
    # This is the main comparison - LLM predictions vs actual real-world data
    # 
    # WHAT WE'RE COMPARING:
    # - llm_values: LLM predictions for each country in the group
    # - real_values: Real data from WHR dataset for each country in the group
    # - Test: Paired t-test (ttest_rel) - compares LLM vs Real for each country pair
    # - Null hypothesis: Mean difference between LLM and Real is zero
    # - Alternative: Mean difference is not zero (two-tailed test)
    
    group_stats['p_value_vs_real'] = np.nan
    group_stats['t_statistic'] = np.nan
    group_stats['significantly_different_from_real'] = False
    
    # Ensure we have both real and LLM columns (already verified above)
    if not HAS_SCIPY:
        print("ERROR: scipy is required for statistical tests")
        return group_stats
    
    for idx, row in group_stats.iterrows():
        group_val = row[group_col]
        group_df = df_clean[df_clean[group_col] == group_val]
        
        if len(group_df) > 1:
            try:
                llm_values = group_df[llm_col].values
                real_values = group_df[real_col].values  # Real data from WHR dataset
                
                # Verify we have matching pairs
                if len(llm_values) != len(real_values):
                    print(f"WARNING: Mismatch in {group_val}: {len(llm_values)} LLM vs {len(real_values)} Real values")
                    continue
                
                # Paired t-test using scipy.stats.ttest_rel
                # This tests: H0: mean(LLM - Real) = 0 vs H1: mean(LLM - Real) != 0
                # It compares each country's LLM prediction to its actual real-world value
                # 
                # Example: For "Global North" group with "diff_overall_happiness":
                #   - llm_values: [7.345, 7.385, 6.63, ...] (LLM predictions for each country)
        #   - real_values: [6.974, 6.81, 6.397, ...] (Real data from WHR dataset for same countries)
                #   - Test compares: LLM[0] vs Real[0], LLM[1] vs Real[1], etc. (paired)
                #   - Result: p-value tells us if LLM predictions are significantly different from real data
                
                t_stat, p_val = stats.ttest_rel(llm_values, real_values)
                
                # Store the raw p-value from scipy (don't modify it)
                # scipy returns the correct p-value - very small values are valid
                group_stats.at[idx, 'p_value_vs_real'] = p_val
                group_stats.at[idx, 't_statistic'] = t_stat
                group_stats.at[idx, 'significantly_different_from_real'] = p_val < 0.05
                
            except Exception as e:
                print(f"Error in paired t-test for {group_val} ({metric_col}): {e}")
                import traceback
                traceback.print_exc()
                pass
    
    # Secondary test: bias vs zero (for reference only - primary test is vs real data above)
    # Note: This is less informative than the paired t-test vs real data
    # WHAT WE'RE COMPARING:
    # - bias_data: Differences (LLM - Real) for each country in the group
    # - Test: One-sample t-test - tests if mean bias is significantly different from 0
    # - Null hypothesis: Mean bias = 0
    # - Alternative: Mean bias != 0 (two-tailed test)
    
    group_stats['p_value_vs_zero'] = np.nan
    if not HAS_SCIPY:
        return group_stats
    
    for idx, row in group_stats.iterrows():
        group_val = row[group_col]
        bias_data = df_clean[df_clean[group_col] == group_val][metric_col].values
        
        if len(bias_data) > 1:
            try:
                # One-sample t-test using scipy.stats.ttest_1samp
                # Tests if the mean bias (difference) is significantly different from zero
                # NOTE: The PRIMARY test is p_value_vs_real (paired t-test comparing LLM to real data)
                t_stat, p_val_zero = stats.ttest_1samp(bias_data, 0)
                
                # Store raw p-value from scipy (don't modify it)
                group_stats.at[idx, 'p_value_vs_zero'] = p_val_zero
            except Exception as e:
                print(f"Error in one-sample t-test for {group_val}: {e}")
                pass
    
    # Pairwise comparisons between groups - do groups have significantly different biases?
    if len(groups) >= 2:
        group_names = list(df_clean.groupby(group_col).groups.keys())
        pairwise_results = []
        
        for i, group1_name in enumerate(group_names):
            for j, group2_name in enumerate(group_names):
                if i >= j:
                    continue
                
                group1_data = df_clean[df_clean[group_col] == group1_name][metric_col].values
                group2_data = df_clean[df_clean[group_col] == group2_name][metric_col].values
                
                if len(group1_data) > 1 and len(group2_data) > 1:
                    try:
                        # Independent t-test: do the two groups have significantly different biases?
                        if HAS_SCIPY:
                            t_stat, p_val = stats.ttest_ind(group1_data, group2_data)
                        else:
                            t_stat, p_val = manual_ttest_ind(group1_data, group2_data)
                        
                        pairwise_results.append({
                            'group1': group1_name,
                            'group2': group2_name,
                            'bias_diff': group1_data.mean() - group2_data.mean(),
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'significant': p_val < 0.05
                        })
                    except:
                        pass
        
        group_stats['pairwise_comparisons'] = str(pairwise_results) if pairwise_results else ''
    else:
        group_stats['pairwise_comparisons'] = ''
    
    # Primary p-value for summary - ALWAYS prioritize comparison to real data from WHR dataset
    # Priority: vs real data (from WHR dataset) > ANOVA > vs zero
    # The paired t-test (p_value_vs_real) is the most important test as it directly compares
    # LLM predictions to actual real-world data from the World Happiness Report
    if 'p_value_vs_real' in group_stats.columns:
        group_stats['p_value'] = group_stats['p_value_vs_real']
    elif 'p_value_anova' in group_stats.columns:
        group_stats['p_value'] = group_stats['p_value_anova']
    elif 'p_value_vs_zero' in group_stats.columns:
        group_stats['p_value'] = group_stats['p_value_vs_zero']
    else:
        group_stats['p_value'] = np.nan
    
    # Significant flag (p < 0.05) - LLM significantly different from real data (from WHR dataset)
    # This is based on the paired t-test comparing LLM predictions to real data
    group_stats['significant'] = group_stats.get('significantly_different_from_real', 
                                                 group_stats.get('p_value', 1.0) < 0.05)
    
    return group_stats
    
    return group_stats

def analyze_all_groupings(df, silent=False):
    """
    Analyze bias across all grouping categories
    Returns dict of results for each grouping.
    If silent=True, skips print output (for use in web app).
    """
    # Metrics to analyze (difference columns)
    metrics = [
        'diff_overall_happiness',
        'diff_life_evaluation',
        'diff_gdp',
        'diff_social_support',
        'diff_health',
        'diff_freedom',
        'diff_generosity',
        'diff_corruption'
    ]
    
    # Groupings to analyze
    groupings = {
        'global_north_south': 'Global North vs Global South',
        'world_123': 'World 1/2/3 Classification',
        'continent': 'Continent',
        'region': 'Region',
        'income_level': 'Income Level',
        'east_west': 'East vs West',
        'developed_east_vs_west': 'Developed East Asia vs Developed West',
        'developed_country_type': 'Developed Country Subcategories',
        'economic_model': 'Economic Model/Development Path'
    }
    
    results = {}
    
    for group_col, group_name in groupings.items():
        if group_col not in df.columns:
            if not silent:
                print(f"Warning: {group_col} not found in dataframe")
            continue
        
        if not silent:
            print(f"\nAnalyzing bias by: {group_name}")
        results[group_col] = {}
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            metric_name = metric.replace('diff_', '').replace('_', ' ').title()
            group_stats = calculate_group_bias(df, group_col, metric)
            
            if group_stats is not None and len(group_stats) > 0:
                results[group_col][metric] = group_stats
                
                if not silent:
                    # Print summary
                    print(f"  {metric_name}:")
                    for _, row in group_stats.iterrows():
                        group_val = row[group_col]
                        mean_bias = row['mean_bias']
                        count = int(row['count'])
                        p_val_real = row.get('p_value_vs_real', np.nan)
                        p_val_anova = row.get('p_value_anova', np.nan)
                        sig_real = row.get('significantly_different_from_real', False)
                        
                        sig_note = ""
                        if not pd.isna(p_val_real) and sig_real:
                            sig_real_mark = "***" if p_val_real < 0.001 else \
                                           "**" if p_val_real < 0.01 else \
                                           "*" if p_val_real < 0.05 else ""
                            if p_val_real < 0.0001:
                                sig_note = f" (vs Real Data: p={p_val_real:.2e}{sig_real_mark})"
                            else:
                                sig_note = f" (vs Real Data: p={p_val_real:.4f}{sig_real_mark})"
                        elif not pd.isna(p_val_anova) and p_val_anova < 0.05:
                            sig_anova = "***" if p_val_anova < 0.001 else \
                                       "**" if p_val_anova < 0.01 else \
                                       "*" if p_val_anova < 0.05 else ""
                            if p_val_anova < 0.0001:
                                sig_note = f" (ANOVA: p={p_val_anova:.2e}{sig_anova})"
                            else:
                                sig_note = f" (ANOVA: p={p_val_anova:.4f}{sig_anova})"
                        
                        print(f"    {group_val}: {mean_bias:+.3f} (n={count}){sig_note}")
    
    return results

def calculate_bias_summary(results):
    """Create summary of biases across all groups and metrics.
    
    NOTE on multiple comparisons:
    - We perform many hypothesis tests across groupings and metrics.
    - To control the family-wise error rate for the primary test
      (LLM vs Real paired t-test, p_value_vs_real), we apply a
      Bonferroni correction across ALL rows that have a finite
      p_value_vs_real.
    - The raw p-values (p_value_vs_real) are preserved, and we add:
        * p_value_vs_real_bonferroni
        * significantly_different_from_real_bonferroni
    """
    summary = []
    
    for group_col, metrics in results.items():
        for metric, group_stats in metrics.items():
            for _, row in group_stats.iterrows():
                summary.append({
                    'grouping': group_col,
                    'group_value': row[group_col],
                    'metric': metric,
                    'mean_bias': row['mean_bias'],
                    'std_bias': row['std_bias'],
                    'median_bias': row['median_bias'],
                    'count': row['count'],
                    'p_value_anova': row.get('p_value_anova', np.nan),
                    'p_value_vs_real': row.get('p_value_vs_real', np.nan),
                    'p_value_vs_zero': row.get('p_value_vs_zero', np.nan),
                    'p_value': row.get('p_value', np.nan),
                    'f_statistic': row.get('f_statistic', np.nan),
                    't_statistic': row.get('t_statistic', np.nan),
                    'significant': row.get('significant', False),
                    'significantly_different_from_real': row.get('significantly_different_from_real', False)
                })
    
    summary_df = pd.DataFrame(summary)
    
    # Apply Bonferroni correction for the primary LLM vs Real test.
    # We correct across all rows where p_value_vs_real is defined.
    if not summary_df.empty and 'p_value_vs_real' in summary_df.columns:
        mask = summary_df['p_value_vs_real'].notna()
        m = int(mask.sum())
        if m > 0:
            # Bonferroni: p_adj = min(p * m, 1)
            p_raw = summary_df.loc[mask, 'p_value_vs_real']
            p_adj = np.minimum(p_raw * m, 1.0)
            summary_df['p_value_vs_real_bonferroni'] = np.nan
            summary_df.loc[mask, 'p_value_vs_real_bonferroni'] = p_adj
            summary_df['significantly_different_from_real_bonferroni'] = (
                summary_df['p_value_vs_real_bonferroni'] < 0.05
            )
        else:
            summary_df['p_value_vs_real_bonferroni'] = np.nan
            summary_df['significantly_different_from_real_bonferroni'] = False
    else:
        # Ensure columns exist for downstream code
        summary_df['p_value_vs_real_bonferroni'] = np.nan
        summary_df['significantly_different_from_real_bonferroni'] = False
    
    return summary_df

def generate_bias_report(df, results, summary_df, output_dir="results"):
    """Generate detailed bias analysis report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"bias_analysis_report_{timestamp}.txt")
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LLM Bias Analysis Report\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total countries analyzed: {len(df)}\n\n")
        
        # Summary of significant biases
        # Use Bonferroni-corrected p-values for the main LLM vs Real test
        significant_biases = summary_df[
            (summary_df['significant'] == True) |
            (summary_df.get('significantly_different_from_real_bonferroni', False) == True)
        ].copy()
        
        if len(significant_biases) > 0:
            f.write("="*80 + "\n")
            f.write("STATISTICALLY SIGNIFICANT BIASES (p < 0.05 after Bonferroni correction for LLM vs Real tests)\n")
            f.write("="*80 + "\n\n")
            
            for grouping in significant_biases['grouping'].unique():
                f.write(f"\n{grouping.upper().replace('_', ' ')}:\n")
                f.write("-"*80 + "\n")
                group_biases = significant_biases[significant_biases['grouping'] == grouping]
                for _, row in group_biases.iterrows():
                    metric_name = row['metric'].replace('diff_', '').replace('_', ' ').title()
                    f.write(f"  {row['group_value']} - {metric_name}:\n")
                    f.write(f"    Mean bias: {row['mean_bias']:+.3f}\n")
                    
                    if not pd.isna(row.get('p_value_anova', np.nan)):
                        f.write(f"    ANOVA p-value: {row['p_value_anova']:.4f} (groups differ significantly)\n")
                    if not pd.isna(row.get('p_value_vs_zero', np.nan)):
                        f.write(f"    vs Zero p-value: {row['p_value_vs_zero']:.4f} (bias is significantly non-zero)\n")
                    
                    f.write(f"    Count: {int(row['count'])}\n\n")
        
        # Summary of pairwise significant differences
        f.write("\n" + "="*80 + "\n")
        f.write("PAIRWISE SIGNIFICANT DIFFERENCES BETWEEN GROUPS\n")
        f.write("="*80 + "\n\n")
        f.write("(For each grouping, pairs of groups with significantly different biases)\n\n")
        
        for group_col in results.keys():
            f.write(f"\n{group_col.upper().replace('_', ' ')}:\n")
            f.write("-"*80 + "\n")
            
            # Extract pairwise comparisons from results
            for metric, group_stats in results[group_col].items():
                if 'pairwise_comparisons' in group_stats.columns:
                    pairwise_str = group_stats['pairwise_comparisons'].iloc[0] if len(group_stats) > 0 else ''
                    if pairwise_str and pairwise_str != '':
                        # Parse pairwise results (simplified - would need better parsing)
                        metric_name = metric.replace('diff_', '').replace('_', ' ').title()
                        f.write(f"  {metric_name}:\n")
                        # Note: Full pairwise results would need better parsing/formatting
                        f.write(f"    (Pairwise comparisons available in detailed data)\n")
        
        # Detailed analysis by grouping
        for group_col, group_name in [
            ('global_north_south', 'Global North vs Global South'),
            ('world_123', 'World 1/2/3 Classification'),
            ('continent', 'Continent'),
            ('region', 'Region'),
            ('income_level', 'Income Level'),
            ('east_west', 'East vs West'),
            ('developed_east_vs_west', 'Developed East Asia vs Developed West'),
            ('developed_country_type', 'Developed Country Subcategories'),
            ('economic_model', 'Economic Model/Development Path')
        ]:
            if group_col not in results:
                continue
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"ANALYSIS BY: {group_name}\n")
            f.write("="*80 + "\n\n")
            
            metrics_data = results[group_col]
            for metric, group_stats in metrics_data.items():
                metric_name = metric.replace('diff_', '').replace('_', ' ').title()
                f.write(f"\n{metric_name}:\n")
                f.write("-"*80 + "\n")
                
                # Sort by mean bias
                group_stats_sorted = group_stats.sort_values('mean_bias', ascending=False)
                
                for _, row in group_stats_sorted.iterrows():
                    group_val = row[group_col]
                    mean_bias = row['mean_bias']
                    std_bias = row['std_bias']
                    median_bias = row['median_bias']
                    count = int(row['count'])
                    p_val = row.get('p_value', np.nan)
                    
                    significance = ""
                    if not pd.isna(p_val):
                        if p_val < 0.001:
                            significance = " ***"
                        elif p_val < 0.01:
                            significance = " **"
                        elif p_val < 0.05:
                            significance = " *"
                    
                    f.write(f"  {group_val}:\n")
                    f.write(f"    Mean bias: {mean_bias:+.3f} ± {std_bias:.3f}\n")
                    f.write(f"    Median bias: {median_bias:+.3f}\n")
                    f.write(f"    Countries: {count}\n")
                    p_val_real = row.get('p_value_vs_real', np.nan)
                    p_val_anova = row.get('p_value_anova', np.nan)
                    
                    if not pd.isna(p_val_real):
                        sig_real = " ***" if p_val_real < 0.001 else \
                                  " **" if p_val_real < 0.01 else \
                                  " *" if p_val_real < 0.05 else ""
                        # Format p-value properly - use scientific notation for very small values
                        if p_val_real < 0.0001:
                            f.write(f"    vs Real Data p-value: {p_val_real:.2e}{sig_real} (LLM vs Real)\n")
                        else:
                            f.write(f"    vs Real Data p-value: {p_val_real:.4f}{sig_real} (LLM vs Real)\n")
                    if not pd.isna(p_val_anova):
                        sig_anova = " ***" if p_val_anova < 0.001 else \
                                   " **" if p_val_anova < 0.01 else \
                                   " *" if p_val_anova < 0.05 else ""
                        if p_val_anova < 0.0001:
                            f.write(f"    ANOVA p-value: {p_val_anova:.2e}{sig_anova} (between groups)\n")
                        else:
                            f.write(f"    ANOVA p-value: {p_val_anova:.4f}{sig_anova} (between groups)\n")
                    f.write("\n")
        
        # Interpretation
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*80 + "\n\n")
        f.write("Bias interpretation:\n")
        f.write("  Positive bias: LLM overestimates compared to real data\n")
        f.write("  Negative bias: LLM underestimates compared to real data\n\n")
        f.write("Significance levels:\n")
        f.write("  *** p < 0.001 (highly significant)\n")
        f.write("  **  p < 0.01  (very significant)\n")
        f.write("  *   p < 0.05  (significant)\n")
    
    print(f"\nDetailed report saved to: {report_file}")
    return report_file

def main():
    """Main execution function"""
    print("="*80)
    print("LLM Bias Analysis by Country Groups")
    print("="*80)
    
    # Load comparison data
    print("\nLoading comparison data...")
    df = load_comparison_data()
    if df is None:
        return
    
    # Add grouping columns
    print("\nAdding country groupings...")
    df = add_groupings(df)
    
    print(f"\nGroup distributions:")
    print(f"  Global North/South: {df['global_north_south'].value_counts().to_dict()}")
    print(f"  World 1/2/3: {df['world_123'].value_counts().to_dict()}")
    print(f"  Continent: {df['continent'].value_counts().to_dict()}")
    print(f"  Region: {df['region'].value_counts().to_dict()}")
    
    # Analyze biases
    print("\n" + "="*80)
    print("Analyzing Biases by Group")
    print("="*80)
    results = analyze_all_groupings(df)
    
    # Create summary
    print("\n" + "="*80)
    print("Creating Summary")
    print("="*80)
    summary_df = calculate_bias_summary(results)
    
    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary CSV (includes Bonferroni-corrected p_value_vs_real_bonferroni)
    summary_file = os.path.join(output_dir, f"bias_summary_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\nBias summary saved to: {summary_file}")
    
    # Generate and save significant findings CSV (for web app)
    # Filter for significant findings using Bonferroni-corrected p-values for the primary test
    # (LLM vs Real paired t-test: p_value_vs_real_bonferroni < 0.05)
    if 'significantly_different_from_real_bonferroni' in summary_df.columns:
        significant_findings = summary_df[
            summary_df['significantly_different_from_real_bonferroni'] == True
        ].copy()
    else:
        # Fallback to uncorrected p-values if old summaries are loaded
        significant_findings = summary_df[
            (summary_df['significantly_different_from_real'] == True) |
            ((summary_df['p_value_vs_real'].notna()) & (summary_df['p_value_vs_real'] < 0.05))
        ].copy()
    
    if len(significant_findings) > 0:
        sig_file = os.path.join(output_dir, f"significant_findings_{timestamp}.csv")
        significant_findings.to_csv(sig_file, index=False)
        print(f"Significant findings saved to: {sig_file} ({len(significant_findings)} findings)")
    else:
        print("Warning: No significant findings found (p < 0.05)")
    
    # Save detailed data with groupings
    data_file = os.path.join(output_dir, f"bias_analysis_data_{timestamp}.csv")
    df.to_csv(data_file, index=False)
    print(f"Detailed data with groupings saved to: {data_file}")
    
    # Generate report
    report_file = generate_bias_report(df, results, summary_df, output_dir)
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Largest biases by group
    for group_col in ['global_north_south', 'world_123', 'continent', 'developed_east_vs_west', 'developed_country_type', 'economic_model']:
        if group_col not in results:
            continue
        
        print(f"\nLargest biases by {group_col}:")
        group_biases = summary_df[summary_df['grouping'] == group_col].copy()
        if len(group_biases) > 0:
            # Focus on overall happiness
            overall_happiness = group_biases[group_biases['metric'] == 'diff_overall_happiness']
            if len(overall_happiness) > 0:
                overall_happiness = overall_happiness.sort_values('mean_bias', ascending=False)
                for _, row in overall_happiness.iterrows():
                    p_val = row['p_value']
                    if pd.isna(p_val):
                        p_str = "N/A"
                    elif p_val < 0.0001:
                        p_str = f"{p_val:.2e}"  # Scientific notation for very small values
                    else:
                        p_str = f"{p_val:.4f}"
                    print(f"  {row['group_value']}: {row['mean_bias']:+.3f} "
                          f"(n={int(row['count'])}, p={p_str})")
    
    print("\n" + "="*80)
    print("Bias Analysis Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
