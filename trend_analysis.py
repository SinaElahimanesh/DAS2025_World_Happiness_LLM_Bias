"""
Trend Mapping: Visualize happiness trends over 2013-2023
"""

import pandas as pd
import numpy as np


def get_global_trends(df):
    """Calculate global happiness trends over time"""
    global_avg = df.groupby('Year')['happiness_score'].mean().reset_index()
    global_avg.columns = ['Year', 'average_happiness']
    
    # Calculate year-over-year change
    global_avg['yoy_change'] = global_avg['average_happiness'].pct_change() * 100
    
    return global_avg


def get_country_trends(df, top_n=20):
    """Get trends for top N countries by latest happiness score"""
    latest_year = df['Year'].max()
    latest_scores = df[df['Year'] == latest_year].nlargest(top_n, 'happiness_score')
    top_countries = latest_scores['country'].tolist()
    
    country_trends = df[df['country'].isin(top_countries)].groupby(['country', 'Year'])['happiness_score'].mean().reset_index()
    
    return country_trends, top_countries


def get_regional_trends(df):
    """Calculate happiness trends by region"""
    regional_trends = df.groupby(['region', 'Year'])['happiness_score'].mean().reset_index()
    return regional_trends


def get_income_trends(df):
    """Calculate happiness trends by income level"""
    income_trends = df.groupby(['income_level', 'Year'])['happiness_score'].mean().reset_index()
    return income_trends


def calculate_trend_statistics(df):
    """Calculate statistical measures of trends"""
    stats = {}
    
    # Overall trend
    global_trends = get_global_trends(df)
    stats['global_start'] = global_trends.iloc[0]['average_happiness']
    stats['global_end'] = global_trends.iloc[-1]['average_happiness']
    stats['global_change'] = stats['global_end'] - stats['global_start']
    stats['global_change_pct'] = (stats['global_change'] / stats['global_start']) * 100
    
    # Countries with biggest improvements
    country_changes = []
    for country in df['country'].unique():
        country_data = df[df['country'] == country]
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
    volatility = df.groupby('country')['happiness_score'].agg(['std', 'mean', 'min', 'max']).reset_index()
    volatility.columns = ['country', 'std_dev', 'mean_score', 'min_score', 'max_score']
    volatility['volatility_ratio'] = volatility['std_dev'] / volatility['mean_score']
    volatility = volatility.sort_values('volatility_ratio', ascending=False)
    
    return volatility

