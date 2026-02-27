"""
LLM Audit: Compare human-reported data with LLM-generated perceptions
"""

import pandas as pd
import numpy as np
import requests
import json
import time


def generate_llm_scores(countries, api_key=None, use_mock=True):
    """
    Generate happiness scores from LLM for given countries
    For now, using a mock implementation that simulates LLM responses
    In production, this would call OpenAI API or similar
    """
    if use_mock:
        # Mock LLM scores - in reality would call actual LLM API
        # This simulates potential biases LLMs might have
        np.random.seed(42)
        
        llm_scores = {}
        for country in countries:
            # Simulate some bias: LLMs might overestimate Western countries
            # and underestimate certain regions
            base_score = np.random.uniform(3.0, 7.5)
            
            # Add bias for Western countries
            western_countries = ['United States', 'Canada', 'United Kingdom', 
                               'Germany', 'France', 'Switzerland', 'Sweden', 
                               'Norway', 'Denmark', 'Netherlands', 'Australia', 
                               'New Zealand']
            if country in western_countries:
                base_score += np.random.uniform(0.3, 0.8)
            
            # Add bias against certain regions
            underestimated = ['Afghanistan', 'Yemen', 'Syria', 'Haiti', 
                           'Central African Republic']
            if country in underestimated:
                base_score -= np.random.uniform(0.2, 0.5)
            
            llm_scores[country] = max(0, min(10, base_score))
        
        return llm_scores
    
    else:
        # Real API implementation would go here
        # Example with OpenAI (commented out):
        """
        import openai
        openai.api_key = api_key
        
        llm_scores = {}
        for country in countries:
            prompt = f"On a scale of 0-10, how happy do you think people in {country} are? Consider factors like economic prosperity, social support, health, freedom, and quality of life. Respond with only a number between 0 and 10."
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            
            try:
                score = float(response.choices[0].message.content.strip())
                llm_scores[country] = score
            except:
                llm_scores[country] = None
            
            time.sleep(0.1)  # Rate limiting
        """
        pass


def compare_llm_vs_human(df, use_mock=True):
    """
    Compare LLM-generated scores with actual human-reported scores
    """
    # Use latest historical year (exclude 2024) for this classic comparison
    years = sorted(df['Year'].dropna().unique())
    hist_years = [y for y in years if y < 2024]
    latest_year = max(hist_years) if hist_years else years[-1]
    df_latest = df[df['Year'] == latest_year].copy()
    
    # Get unique countries
    countries = df_latest['country'].unique().tolist()
    
    # Generate LLM scores
    llm_scores_dict = generate_llm_scores(countries, use_mock=use_mock)
    
    # Merge with actual scores
    comparison = df_latest[['country', 'happiness_score', 'region', 'income_level']].copy()
    comparison['llm_score'] = comparison['country'].map(llm_scores_dict)
    comparison = comparison.dropna(subset=['llm_score'])
    
    # Calculate differences
    comparison['difference'] = comparison['llm_score'] - comparison['happiness_score']
    comparison['abs_difference'] = np.abs(comparison['difference'])
    comparison['bias_direction'] = comparison['difference'].apply(
        lambda x: 'Overestimate' if x > 0.5 else ('Underestimate' if x < -0.5 else 'Accurate')
    )
    
    return comparison


def analyze_llm_bias(comparison_df):
    """Analyze patterns in LLM bias"""
    bias_analysis = {}
    
    # Overall bias
    bias_analysis['mean_difference'] = comparison_df['difference'].mean()
    bias_analysis['median_difference'] = comparison_df['difference'].median()
    bias_analysis['rmse'] = np.sqrt((comparison_df['difference']**2).mean())
    
    # Bias by region
    bias_by_region = comparison_df.groupby('region').agg({
        'difference': ['mean', 'std', 'count'],
        'abs_difference': 'mean'
    }).reset_index()
    bias_by_region.columns = ['region', 'mean_bias', 'std_bias', 'count', 'avg_abs_bias']
    bias_analysis['by_region'] = bias_by_region.sort_values('mean_bias', ascending=False)
    
    # Bias by income level
    bias_by_income = comparison_df.groupby('income_level').agg({
        'difference': ['mean', 'std', 'count'],
        'abs_difference': 'mean'
    }).reset_index()
    bias_by_income.columns = ['income_level', 'mean_bias', 'std_bias', 'count', 'avg_abs_bias']
    bias_analysis['by_income'] = bias_by_income
    
    # Countries with largest bias
    bias_analysis['most_overestimated'] = comparison_df.nlargest(10, 'difference')[
        ['country', 'happiness_score', 'llm_score', 'difference', 'region']
    ]
    bias_analysis['most_underestimated'] = comparison_df.nsmallest(10, 'difference')[
        ['country', 'happiness_score', 'llm_score', 'difference', 'region']
    ]
    
    return bias_analysis


def calculate_correlation(comparison_df):
    """Calculate correlation between LLM and human scores"""
    correlation = comparison_df['happiness_score'].corr(comparison_df['llm_score'])
    return correlation

