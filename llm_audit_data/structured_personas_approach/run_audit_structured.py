"""
Structured Personas LLM Happiness Audit Pipeline (Fourth Approach)

Same single Cantril Ladder question as single_question_gallup_approach,
but personas are structured (nationality, job, gender) instead of long text.
Nationality is the main focus. Half of personas are male, half female.

Usage:
    python run_audit_structured.py
"""

import pandas as pd
import os
import time
import sys
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(current_dir))

from structured_personas import get_all_structured_personas
from survey_gallup import get_survey_gallup, parse_survey_response_gallup
from api_client_structured import batch_survey_calls_structured
from analyzer import calculate_country_statistics, calculate_persona_statistics


def load_countries():
    """Load list of countries from main dataset."""
    data_file = os.path.join(parent_dir, 'data.xlsx')  # resolved by data_loader to dataset.xlsx
    from data_loader import load_data, clean_data
    df = load_data(data_file)
    df = clean_data(df)
    return sorted(df['country'].unique().tolist())


def load_existing_results(csv_file):
    """Return set of (country, persona_id) already processed."""
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


def save_to_csv(new_data, csv_file):
    """Append new data to CSV; create file if needed."""
    if len(new_data) == 0:
        return
    df_new = pd.DataFrame(new_data)
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_file, index=False)
    else:
        df_new.to_csv(csv_file, index=False)


def filter_pending_combinations(countries, personas, processed_set):
    """Return list of (country, persona) still to process."""
    pending = []
    for country in countries:
        for persona in personas:
            if (country, persona['id']) not in processed_set:
                pending.append((country, persona))
    return pending


def main():
    print("=" * 60)
    print("LLM Happiness Audit - Structured Personas Approach")
    print("=" * 60)
    print("\nSingle Cantril Ladder question, structured personas (nationality, job, gender).")
    print("Nationality focus; 10 male, 10 female personas.")
    print("=" * 60)

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "llm_audit_results.csv")

    print("\nChecking for existing results...")
    processed_set = load_existing_results(csv_file)

    print("\nLoading countries from main WHR dataset (via data_loader)...")
    countries = load_countries()
    valid_countries_set = set(countries)
    print(f"Found {len(countries)} countries")

    print("\nLoading structured personas...")
    personas = get_all_structured_personas()
    print(f"Found {len(personas)} personas (10 male, 10 female)")

    total_combinations = len(countries) * len(personas)
    pending_combinations = filter_pending_combinations(countries, personas, processed_set)
    print(f"\nTotal combinations: {total_combinations}")
    print(f"Already processed: {len(processed_set)}")
    print(f"Pending: {len(pending_combinations)}")

    if len(pending_combinations) == 0:
        print("\nAll combinations have already been processed!")
        return

    country_personas_map = {}
    skipped_countries = []
    for country, persona in pending_combinations:
        if country not in valid_countries_set:
            if country not in skipped_countries:
                skipped_countries.append(country)
            continue
        if country not in country_personas_map:
            country_personas_map[country] = []
        country_personas_map[country].append(persona)

    if skipped_countries:
        print(f"\n⚠️  Skipping {len(skipped_countries)} countries not in WHR dataset")

    total_countries_to_process = len(country_personas_map)
    processed_count = 0

    for idx, (country, country_personas) in enumerate(country_personas_map.items(), 1):
        if country not in valid_countries_set:
            continue
        print(f"\n{'='*60}")
        print(f"Processing country {idx}/{total_countries_to_process}: {country}")
        print(f"  Personas: {len(country_personas)}/{len(personas)}")
        print(f"{'='*60}")

        survey_text = get_survey_gallup(country)
        country_responses = batch_survey_calls_structured(
            country=country,
            personas=country_personas,
            survey_text=survey_text,
            batch_size=10
        )

        csv_data = []
        for response in country_responses:
            parsed = parse_survey_response_gallup(response['response_text'])
            if parsed and parsed.get('score') is not None:
                score = parsed['score']
                csv_data.append({
                    'persona_id': response['persona_id'],
                    'country': response['country'],
                    'life_evaluation': score,
                    'gdp': None,
                    'social_support': None,
                    'health': None,
                    'freedom': None,
                    'generosity': None,
                    'corruption': None,
                    'overall_happiness_explicit': score,
                    'overall_happiness_calculated': score,
                    'overall_happiness': score,
                    'response_text': response['response_text']
                })
            else:
                print(f"  Warning: Failed to parse response for persona {response['persona_id']} in {country}")

        if csv_data:
            save_to_csv(csv_data, csv_file)
            processed_count += len(csv_data)
            print(f"  Saved {len(csv_data)} responses. Total processed: {processed_count}/{len(pending_combinations)}")

        if idx < total_countries_to_process:
            time.sleep(0.25)

    print(f"\n{'='*60}\nProcessing complete!\n{'='*60}")

    print("\nGenerating final statistics...")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if len(df) > 0:
            country_stats = calculate_country_statistics(df)
            persona_stats = calculate_persona_statistics(df)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = os.path.join(output_dir, f"statistics_{timestamp}.csv")
            country_stats.to_csv(stats_file, index=False)
            persona_stats.to_csv(os.path.join(output_dir, f"persona_stats_{timestamp}.csv"), index=False)
            print(f"\nResults: {csv_file}")
            print(f"Country statistics: {stats_file}")
            print(f"Summary: {len(df)} responses, {df['country'].nunique()} countries, {df['persona_id'].nunique()} personas")
            print(f"Average happiness: {df['overall_happiness'].mean():.2f}")
        else:
            print("No valid responses in CSV.")
    else:
        print("No results file found.")
    print("\n" + "=" * 60 + "\nLLM Audit Complete!\n" + "=" * 60)


if __name__ == "__main__":
    main()
