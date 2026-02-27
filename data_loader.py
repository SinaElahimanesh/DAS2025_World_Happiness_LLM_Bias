"""
Data loading and preprocessing utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath: str = 'dataset.xlsx'):
    """
    Load the World Happiness Report data.

    Expects the main Excel dataset in the project root (directory containing
    this file) as 'dataset.xlsx'. Also accepts 'data.xlsx' for backward
    compatibility. Paths are resolved relative to the project root so the
    notebook or app can be run from any working directory.
    """
    root = Path(__file__).resolve().parent
    candidate = (root / filepath) if not Path(filepath).is_absolute() else Path(filepath)
    if not candidate.exists():
        for alt_name in ('dataset.xlsx', 'data.xlsx'):
            alt = root / alt_name
            if alt.exists():
                candidate = alt
                break
    if not candidate.exists():
        raise FileNotFoundError(
            f"Dataset not found. Place 'dataset.xlsx' (or 'data.xlsx') in the project root: {root}"
        )
    return pd.read_excel(candidate)


def clean_data(df):
    """Clean and prepare the data for analysis"""
    df_clean = df.copy()
    
    # Rename columns for easier access
    column_mapping = {
        'Life evaluation (3-year average)': 'happiness_score',
        'Explained by: Log GDP per capita': 'gdp',
        'Explained by: Social support': 'social_support',
        'Explained by: Healthy life expectancy': 'life_expectancy',
        'Explained by: Freedom to make life choices': 'freedom',
        'Explained by: Generosity': 'generosity',
        'Explained by: Perceptions of corruption': 'corruption',
        'Country name': 'country'
    }
    
    df_clean = df_clean.rename(columns=column_mapping)
    
    # Remove rows with missing critical data
    df_clean = df_clean.dropna(subset=['happiness_score', 'country', 'Year'])
    
    # Use all available years in the dataset
    # No year filtering - use complete dataset
    
    return df_clean


def get_country_regions():
    """Map countries to regions"""
    # Basic regional mapping - can be expanded
    regions = {
        'Western Europe': ['Switzerland', 'Iceland', 'Denmark', 'Netherlands', 'Sweden', 
                          'Norway', 'Luxembourg', 'Finland', 'Austria', 'Belgium', 
                          'Ireland', 'Germany', 'United Kingdom', 'France', 'Spain', 
                          'Italy', 'Portugal', 'Greece', 'Malta', 'Cyprus'],
        'North America': ['Canada', 'United States', 'Mexico', 'Costa Rica', 'Panama', 
                         'Guatemala', 'El Salvador', 'Honduras', 'Nicaragua'],
        'Latin America': ['Uruguay', 'Brazil', 'Argentina', 'Chile', 'Colombia', 
                         'Ecuador', 'Peru', 'Venezuela', 'Paraguay', 'Bolivia'],
        'East Asia': ['Taiwan', 'Singapore', 'Japan', 'South Korea', 'China', 
                     'Hong Kong', 'Mongolia', 'Thailand', 'Philippines', 'Vietnam'],
        'South Asia': ['India', 'Pakistan', 'Bangladesh', 'Nepal', 'Sri Lanka', 
                      'Afghanistan', 'Bhutan'],
        'Middle East': ['Israel', 'United Arab Emirates', 'Saudi Arabia', 'Qatar', 
                       'Kuwait', 'Bahrain', 'Oman', 'Jordan', 'Lebanon', 'Turkey', 
                       'Iran', 'Iraq', 'Yemen'],
        'Africa': ['Mauritius', 'Libya', 'Algeria', 'Morocco', 'Tunisia', 'Egypt', 
                  'South Africa', 'Nigeria', 'Kenya', 'Ghana', 'Senegal', 'Ethiopia'],
        'Eastern Europe': ['Czechia', 'Lithuania', 'Slovenia', 'Poland', 'Estonia', 
                          'Slovakia', 'Latvia', 'Romania', 'Bulgaria', 'Croatia', 
                          'Hungary', 'Serbia', 'Ukraine', 'Russia', 'Belarus'],
        'Oceania': ['New Zealand', 'Australia', 'Fiji', 'Papua New Guinea']
    }
    
    # Create reverse mapping
    country_to_region = {}
    for region, countries in regions.items():
        for country in countries:
            country_to_region[country] = region
    
    return country_to_region


def add_regions(df):
    """Add region column to dataframe"""
    region_map = get_country_regions()
    df['region'] = df['country'].map(region_map).fillna('Other')
    return df


def get_income_levels():
    """Categorize countries by income level based on GDP"""
    # This is a simplified version - in reality would use World Bank classifications
    return {
        'High Income': ['Switzerland', 'Luxembourg', 'Norway', 'Iceland', 'Denmark', 
                       'Sweden', 'Netherlands', 'Austria', 'Belgium', 'Finland', 
                       'Germany', 'Ireland', 'United States', 'Canada', 'Australia', 
                       'New Zealand', 'Singapore', 'Japan', 'South Korea', 'United Arab Emirates',
                       'Qatar', 'Kuwait', 'Saudi Arabia', 'Israel'],
        'Upper Middle Income': ['Chile', 'Uruguay', 'Costa Rica', 'Panama', 'Brazil', 
                               'Argentina', 'Mexico', 'China', 'Malaysia', 'Thailand',
                               'Turkey', 'Russia', 'Romania', 'Bulgaria', 'South Africa'],
        'Lower Middle Income': ['India', 'Philippines', 'Indonesia', 'Vietnam', 'Egypt',
                               'Morocco', 'Tunisia', 'Ukraine', 'Colombia', 'Peru'],
        'Low Income': ['Afghanistan', 'Nepal', 'Bangladesh', 'Pakistan', 'Ethiopia',
                      'Yemen', 'Haiti', 'Mozambique', 'Tanzania']
    }


def add_income_levels(df):
    """Add income level column to dataframe"""
    income_map = {}
    for level, countries in get_income_levels().items():
        for country in countries:
            income_map[country] = level
    
    df['income_level'] = df['country'].map(income_map)
    
    # If not mapped, use GDP to estimate
    if df['income_level'].isna().any():
        gdp_median = df['gdp'].median()
        df.loc[df['income_level'].isna() & (df['gdp'] > gdp_median * 1.5), 'income_level'] = 'High Income'
        df.loc[df['income_level'].isna() & (df['gdp'] > gdp_median * 0.7) & (df['gdp'] <= gdp_median * 1.5), 'income_level'] = 'Upper Middle Income'
        df.loc[df['income_level'].isna() & (df['gdp'] > gdp_median * 0.3) & (df['gdp'] <= gdp_median * 0.7), 'income_level'] = 'Lower Middle Income'
        df.loc[df['income_level'].isna(), 'income_level'] = 'Low Income'
    
    return df

