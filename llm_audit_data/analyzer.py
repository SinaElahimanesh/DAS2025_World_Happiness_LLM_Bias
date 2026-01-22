"""
Response Analyzer: Process and Aggregate LLM Survey Results

This module processes raw LLM survey responses and calculates aggregated statistics.
It parses survey responses, validates data quality, and generates country-level and
persona-level statistics for analysis.

Usage:
    from analyzer import analyze_responses, calculate_country_statistics
    df = analyze_responses(responses_data)
    stats = calculate_country_statistics(df)
"""

import pandas as pd
from survey import parse_survey_response

def analyze_responses(responses_data):
    """
    Analyze survey responses and calculate statistics
    
    Args:
        responses_data: List of dicts with 'persona_id', 'country', 'response_text'
    
    Returns:
        DataFrame with parsed scores and statistics
    """
    parsed_data = []
    
    for response in responses_data:
        parsed = parse_survey_response(response['response_text'])
        if parsed and parsed.get('scores') and len(parsed['scores']) >= 7:
            # Only process if we have explicit overall happiness
            if not parsed.get('has_explicit_overall', False):
                print(f"Warning: Skipping persona {response['persona_id']} in {response['country']} - missing explicit overall happiness")
                continue
            
            scores = parsed['scores']
            # Scores: [Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]
            parsed_data.append({
                'persona_id': response['persona_id'],
                'country': response['country'],
                'life_evaluation': scores[0],  # Overall Happiness (Cantril Ladder) - explicit
                'gdp': scores[1],
                'social_support': scores[2],
                'health': scores[3],
                'freedom': scores[4],
                'generosity': scores[5],
                'corruption': scores[6],
                'overall_happiness_explicit': parsed.get('overall_happiness_explicit'),
                'overall_happiness_calculated': parsed.get('overall_happiness_calculated'),
                'overall_happiness': scores[0]  # Explicit overall happiness
            })
        else:
            print(f"Failed to parse response for persona {response['persona_id']} in {response['country']}")
    
    df = pd.DataFrame(parsed_data)
    return df

def calculate_country_statistics(df):
    """
    Calculate statistics for each country
    """
    if len(df) == 0:
        return pd.DataFrame()
    
    stats = df.groupby('country').agg({
        'overall_happiness': ['mean', 'std', 'min', 'max', 'count'],
        'life_evaluation': 'mean',
        'gdp': 'mean',
        'social_support': 'mean',
        'health': 'mean',
        'freedom': 'mean',
        'generosity': 'mean',
        'corruption': 'mean'
    }).reset_index()
    
    stats.columns = ['country', 'mean_happiness', 'std_happiness', 'min_happiness', 
                     'max_happiness', 'response_count', 'avg_life_evaluation',
                     'avg_gdp', 'avg_social_support', 'avg_health', 'avg_freedom',
                     'avg_generosity', 'avg_corruption']
    
    return stats

def calculate_persona_statistics(df):
    """
    Calculate statistics for each persona across all countries
    """
    if len(df) == 0:
        return pd.DataFrame()
    
    stats = df.groupby('persona_id').agg({
        'overall_happiness': ['mean', 'std'],
        'life_evaluation': 'mean',
        'gdp': 'mean',
        'social_support': 'mean',
        'health': 'mean',
        'freedom': 'mean',
        'generosity': 'mean',
        'corruption': 'mean'
    }).reset_index()
    
    stats.columns = ['persona_id', 'mean_happiness', 'std_happiness',
                     'avg_life_evaluation', 'avg_gdp', 'avg_social_support',
                     'avg_health', 'avg_freedom', 'avg_generosity', 'avg_corruption']
    
    return stats

