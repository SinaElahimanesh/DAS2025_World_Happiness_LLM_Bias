"""
Trend Mapping: Visualize happiness trends over historical data (excluding 2024)
"""

import pandas as pd
import numpy as np


def _filter_historical(df, cutoff_year: int = 2024):
    """
    Return a copy of df restricted to years strictly before cutoff_year.
    If no such years exist, returns df unchanged.
    """
    if 'Year' not in df.columns:
        return df
    mask = df['Year'] < cutoff_year
    if not mask.any():
        return df
    return df[mask].copy()


def get_global_trends(df):
    """Calculate global happiness trends over time"""
    df_hist = _filter_historical(df)
    global_avg = df_hist.groupby('Year')['happiness_score'].mean().reset_index()
    global_avg.columns = ['Year', 'average_happiness']
    
    # Calculate year-over-year change
    global_avg['yoy_change'] = global_avg['average_happiness'].pct_change() * 100
    
    return global_avg


def get_country_trends(df, top_n=20):
    """Get trends for top N countries by latest happiness score"""
    df_hist = _filter_historical(df)
    latest_year = df_hist['Year'].max()
    latest_scores = df_hist[df_hist['Year'] == latest_year].nlargest(top_n, 'happiness_score')
    top_countries = latest_scores['country'].tolist()
    
    country_trends = df_hist[df_hist['country'].isin(top_countries)].groupby(['country', 'Year'])['happiness_score'].mean().reset_index()
    
    return country_trends, top_countries


def get_regional_trends(df):
    """Calculate happiness trends by region"""
    df_hist = _filter_historical(df)
    regional_trends = df_hist.groupby(['region', 'Year'])['happiness_score'].mean().reset_index()
    return regional_trends


def get_income_trends(df):
    """Calculate happiness trends by income level"""
    df_hist = _filter_historical(df)
    income_trends = df_hist.groupby(['income_level', 'Year'])['happiness_score'].mean().reset_index()
    return income_trends


def calculate_trend_statistics(df):
    """Calculate statistical measures of trends"""
    stats = {}
    
    # Overall trend (historical, excluding 2024)
    global_trends = get_global_trends(df)
    stats['global_start'] = global_trends.iloc[0]['average_happiness']
    stats['global_end'] = global_trends.iloc[-1]['average_happiness']
    stats['global_change'] = stats['global_end'] - stats['global_start']
    stats['global_change_pct'] = (stats['global_change'] / stats['global_start']) * 100
    
    # Countries with biggest improvements
    country_changes = []
    df_hist = _filter_historical(df)
    for country in df_hist['country'].unique():
        country_data = df_hist[df_hist['country'] == country]
        if len(country_data) < 5:  # Need data across multiple years
            continue
        
        start_score = country_data[country_data['Year'] == country_data['Year'].min()]['happiness_score'].mean()
        end_score = country_data[country_data['Year'] == country_data['Year'].max()]['happiness_score'].mean()
        
        if not pd.isna(start_score) and not pd.isna(end_score):
            country_changes.append({
                'country': country,
                'change': end_score - start_score,
                'start': start_score,
                'end': end_score
            })
    
    country_changes_df = pd.DataFrame(country_changes)
    stats['biggest_improvers'] = country_changes_df.nlargest(10, 'change')
    stats['biggest_decliners'] = country_changes_df.nsmallest(10, 'change')
    
    return stats


def get_volatility_analysis(df):
    """Analyze volatility/stability of happiness scores"""
    df_hist = _filter_historical(df)
    volatility = df_hist.groupby('country')['happiness_score'].agg(['std', 'mean', 'min', 'max']).reset_index()
    volatility.columns = ['country', 'std_dev', 'mean_score', 'min_score', 'max_score']
    volatility['volatility_ratio'] = volatility['std_dev'] / volatility['mean_score']
    volatility = volatility.sort_values('volatility_ratio', ascending=False)
    
    return volatility

