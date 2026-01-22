"""
Improved LLM Happiness Audit Pipeline

This script runs the improved LLM audit with all optimizations from LLM_OVERESTIMATION_ANALYSIS.md:
- Calibration and anchoring
- Reduced optimism bias
- Improved factor definitions
- Country context awareness

Features:
- Incremental saving: Results saved after each country
- Resume capability: Automatically skips already processed items
- Progress tracking: Shows detailed progress throughout execution
- Statistics generation: Creates summary statistics after completion

Usage:
    python run_audit_improved.py
"""

import pandas as pd
import os
import time
import sys
from datetime import datetime

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(current_dir))  # llm_audit_data directory (for shared modules)

from personas import get_all_personas
from survey_improved import get_survey_improved, parse_survey_response, get_country_income_level
from api_client_improved import batch_survey_calls_improved
from analyzer import calculate_country_statistics, calculate_persona_statistics


def load_countries():
    """Load list of countries from main dataset"""
    data_file = os.path.join(parent_dir, 'data.xlsx')
    
    from data_loader import load_data, clean_data
    
    df = load_data(data_file)
    df = clean_data(df)
    countries = sorted(df['country'].unique().tolist())
    return countries


def get_country_income_level_from_data(country_name):
    """
    Get income level for a country from the main dataset
    Returns income level string or None
    """
    try:
        data_file = os.path.join(parent_dir, 'data.xlsx')
        from data_loader import load_data, clean_data, add_income_levels
        
        df = load_data(data_file)
        df = clean_data(df)
        df = add_income_levels(df)
        
        country_data = df[df['country'] == country_name]
        if len(country_data) > 0:
            income_level = country_data['income_level'].iloc[0]
            return income_level if pd.notna(income_level) else None
    except Exception as e:
        print(f"Warning: Could not get income level for {country_name}: {e}")
    
    return None


def load_existing_results(csv_file):
    """
    Load existing results from CSV file
    Returns a set of (country, persona_id) tuples that have already been processed
    """
    if not os.path.exists(csv_file):
        return set()
    
    try:
        df = pd.read_csv(csv_file)
        if 'country' in df.columns and 'persona_id' in df.columns:
            processed = set(zip(df['country'], df['persona_id']))
            print(f"Found {len(processed)} already processed country-persona combinations")
            return processed
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")
        return set()
    
    return set()


def save_to_csv(new_data, csv_file):
    """
    Append new data to CSV file
    Creates file if it doesn't exist, appends if it does
    """
    if len(new_data) == 0:
        return
    
    # Convert to DataFrame
    df_new = pd.DataFrame(new_data)
    
    # If file exists, append. Otherwise create new
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_file, index=False)
    else:
        df_new.to_csv(csv_file, index=False)


def filter_pending_combinations(countries, personas, processed_set):
    """
    Filter out already processed country-persona combinations
    Returns list of (country, persona) tuples that need to be processed
    """
    pending = []
    for country in countries:
        for persona in personas:
            if (country, persona['id']) not in processed_set:
                pending.append((country, persona))
    return pending


def main():
    """Main execution function"""
    print("="*60)
    print("LLM Happiness Audit - Starting")
    print("="*60)
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a single CSV file for incremental saving
    csv_file = os.path.join(output_dir, "llm_audit_results.csv")
    
    # Load existing results to skip already processed items
    print("\nChecking for existing results...")
    processed_set = load_existing_results(csv_file)
    
    # Get countries and personas
    print("\nLoading countries from data.xlsx...")
    countries = load_countries()
    valid_countries_set = set(countries)  # Create set for fast lookup
    print(f"Found {len(countries)} countries in data.xlsx")
    
    print("\nLoading personas...")
    personas = get_all_personas()
    print(f"Found {len(personas)} personas")
    
    # Calculate total combinations and pending
    total_combinations = len(countries) * len(personas)
    pending_combinations = filter_pending_combinations(countries, personas, processed_set)
    
    print(f"\nTotal combinations: {total_combinations}")
    print(f"Already processed: {len(processed_set)}")
    print(f"Pending: {len(pending_combinations)}")
    
    if len(pending_combinations) == 0:
        print("\nAll combinations have already been processed!")
        print("To re-run, delete or rename the CSV file.")
        return
    
    # Group pending by country for efficient processing
    # Filter out countries not in data.xlsx
    country_personas_map = {}
    country_income_map = {}
    skipped_countries = []
    for country, persona in pending_combinations:
        if country not in valid_countries_set:
            if country not in skipped_countries:
                skipped_countries.append(country)
            continue  # Skip countries not in data.xlsx
        if country not in country_personas_map:
            country_personas_map[country] = []
            # Get income level for this country
            income_level = get_country_income_level_from_data(country)
            country_income_map[country] = income_level
        country_personas_map[country].append(persona)
    
    if skipped_countries:
        print(f"\n⚠️  Skipping {len(skipped_countries)} countries not found in data.xlsx:")
        for country in skipped_countries:
            print(f"   - {country}")
    
    total_countries_to_process = len(country_personas_map)
    processed_count = 0
    
    # Process each country
    for idx, (country, country_personas) in enumerate(country_personas_map.items(), 1):
        # Safety check: Skip if country not in data.xlsx
        if country not in valid_countries_set:
            print(f"\n⚠️  Skipping {country} - not found in data.xlsx")
            continue
        
        income_level = country_income_map.get(country)
        
        print(f"\n{'='*60}")
        print(f"Processing country {idx}/{total_countries_to_process}: {country}")
        print(f"  Personas for this country: {len(country_personas)}/{len(personas)}")
        print(f"{'='*60}")
        
        # Get improved survey text with country context
        survey_text = get_survey_improved(country, income_level)
        
        # Get responses for this country (only for pending personas)
        country_responses = batch_survey_calls_improved(
            country=country,
            personas=country_personas,
            survey_text=survey_text,
            income_level=None,  # Don't pass income_level to avoid biasing
            batch_size=10
        )
        
        # Parse responses and prepare data for CSV
        csv_data = []
        for response in country_responses:
            parsed = parse_survey_response(response['response_text'])
            if parsed and parsed.get('scores') and len(parsed['scores']) >= 7:
                # Validate that we have explicit overall happiness
                if not parsed.get('has_explicit_overall', False):
                    print(f"  ERROR: Persona {response['persona_id']} in {country} missing explicit overall happiness!")
                    continue
                
                scores = parsed['scores']
                csv_data.append({
                    'persona_id': response['persona_id'],
                    'country': response['country'],
                    'life_evaluation': scores[0],  # Overall Happiness (explicit)
                    'gdp': scores[1],
                    'social_support': scores[2],
                    'health': scores[3],
                    'freedom': scores[4],
                    'generosity': scores[5],
                    'corruption': scores[6],
                    'overall_happiness_explicit': parsed.get('overall_happiness_explicit'),
                    'overall_happiness_calculated': parsed.get('overall_happiness_calculated'),
                    'overall_happiness': scores[0],  # Primary overall happiness score (explicit)
                    'response_text': response['response_text']
                })
            else:
                print(f"  Warning: Failed to parse response for persona {response['persona_id']} in {country}")
        
        # Save to CSV immediately after processing each country
        if csv_data:
            save_to_csv(csv_data, csv_file)
            processed_count += len(csv_data)
            
            save_to_csv(csv_data, csv_file)
            processed_count += len(csv_data)
            print(f"  Saved {len(csv_data)} responses to CSV. Total processed: {processed_count}/{len(pending_combinations)}")
        else:
            print(f"  No valid responses to save for {country}")
        
        # Longer break between countries
        if idx < total_countries_to_process:
            print(f"\nWaiting 0.25 seconds before next country...")
            time.sleep(0.25)
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")
    
    # Generate final statistics from all data in CSV
    print("\nGenerating final statistics...")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        
        if len(df) > 0:
            # Calculate statistics
            country_stats = calculate_country_statistics(df)
            persona_stats = calculate_persona_statistics(df)
            
            # Save statistics files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = os.path.join(output_dir, f"statistics_{timestamp}.csv")
            country_stats.to_csv(stats_file, index=False)
            persona_stats.to_csv(os.path.join(output_dir, f"persona_stats_{timestamp}.csv"), index=False)
            
            print(f"\nResults saved:")
            print(f"  - Main results CSV: {csv_file}")
            print(f"  - Country statistics: {stats_file}")
            print(f"  - Persona statistics: {os.path.join(output_dir, f'persona_stats_{timestamp}.csv')}")
            
            print(f"\nSummary:")
            print(f"  - Total responses: {len(df)}")
            print(f"  - Countries: {df['country'].nunique()}")
            print(f"  - Personas: {df['persona_id'].nunique()}")
            print(f"  - Average happiness (all): {df['overall_happiness'].mean():.2f}")
            print(f"  - Std happiness (all): {df['overall_happiness'].std():.2f}")
        else:
            print("No valid responses in CSV file!")
    else:
        print("No results file found!")
    
    print("\n" + "="*60)
    print("LLM Audit Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
