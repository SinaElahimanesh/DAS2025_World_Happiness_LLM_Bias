"""
Driver Analysis: Weighted linear regression to rank importance of factors
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def _latest_historical_year(df, cutoff_year: int = 2024) -> int:
    """
    Return the latest year strictly before cutoff_year if available,
    otherwise fall back to the maximum year present.
    This lets us keep 2024 for LLM vs real comparison, while using
    all prior years for traditional analysis.
    """
    years = sorted(df['Year'].dropna().unique())
    hist_years = [y for y in years if y < cutoff_year]
    return max(hist_years) if hist_years else years[-1]


def calculate_feature_importance(df):
    """
    Use weighted linear regression to determine importance of each factor
    Returns ranked features with their coefficients
    """
    # Prepare features
    features = ['gdp', 'social_support', 'life_expectancy', 'freedom', 
                'generosity', 'corruption']
    
    # Get data for latest historical year (exclude 2024 for regression)
    latest_year = _latest_historical_year(df)
    df_latest = df[df['Year'] == latest_year].copy()
    
    # Remove rows with missing values
    df_latest = df_latest.dropna(subset=features + ['happiness_score'])
    
    X = df_latest[features].values
    y = df_latest['happiness_score'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Get coefficients (importance)
    coefficients = model.coef_
    feature_importance = pd.DataFrame({
        'feature': features,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    })
    
    # Rank by absolute coefficient
    feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
    feature_importance['rank'] = range(1, len(feature_importance) + 1)
    
    return feature_importance, model, scaler


def analyze_drivers_by_group(df, group_by='region'):
    """
    Analyze driver importance by region or income level
    Returns normalized coefficients for better comparison across groups
    """
    results = {}
    groups = df[group_by].unique()
    
    for group in groups:
        df_group = df[df[group_by] == group].copy()
        if len(df_group) < 10:  # Skip groups with too few data points
            continue
        
        latest_year = _latest_historical_year(df_group)
        df_latest = df_group[df_group['Year'] == latest_year].copy()
        
        features = ['gdp', 'social_support', 'life_expectancy', 'freedom', 
                   'generosity', 'corruption']
        df_latest = df_latest.dropna(subset=features + ['happiness_score'])
        
        if len(df_latest) < 5:
            continue
        
        X = df_latest[features].values
        y = df_latest['happiness_score'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Normalize coefficients to sum of absolute values = 1 for better comparison
        coefficients = model.coef_
        abs_sum = np.sum(np.abs(coefficients))
        if abs_sum > 0:
            normalized_coefficients = coefficients / abs_sum
        else:
            normalized_coefficients = coefficients
        
        feature_importance = pd.DataFrame({
            'feature': features,
            'coefficient': normalized_coefficients,
            'abs_coefficient': np.abs(normalized_coefficients)
        })
        feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
        
        results[group] = feature_importance
    
    return results


def get_driver_summary(df):
    """Get overall driver analysis summary"""
    importance, model, scaler = calculate_feature_importance(df)
    
    # Calculate R-squared on the same historical year
    latest_year = _latest_historical_year(df)
    df_latest = df[df['Year'] == latest_year].copy()
    features = ['gdp', 'social_support', 'life_expectancy', 'freedom', 
                'generosity', 'corruption']
    df_latest = df_latest.dropna(subset=features + ['happiness_score'])
    
    X = df_latest[features].values
    y = df_latest['happiness_score'].values
    X_scaled = scaler.transform(X)
    
    r_squared = model.score(X_scaled, y)
    
    return {
        'importance': importance,
        'r_squared': r_squared,
        'top_driver': importance.iloc[0]['feature'],
        'top_driver_value': importance.iloc[0]['coefficient']
    }

