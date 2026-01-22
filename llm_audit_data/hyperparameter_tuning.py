"""
Hyperparameter Tuning Script: Find Best LLM API Hyperparameters

This script tests different hyperparameter combinations for LLM API calls:
- Temperature: Multiple values (0.3, 0.6, 0.9, 1.2)
- Chain of Thought: With and without CoT prompting

For each combination, it:
1. Selects a random set of 10 countries
2. Makes API calls for all personas in those countries
3. Compares results with ground truth real data
4. Evaluates which hyperparameter combination best matches the real data

Usage:
    python hyperparameter_tuning.py
"""

import pandas as pd
import os
import sys
import time
import random
import re
from datetime import datetime
import openai
import numpy as np

# Add parent directory to path to import modules
sys.path.append('..')
from data_loader import load_data, clean_data

# Import from llm_audit_data
from personas import get_all_personas
from survey import get_survey, parse_survey_response
from country_to_nationality import country_to_nationality

from config import get_api_key

# Initialize OpenAI client with API key from environment variable
client = openai.OpenAI(api_key=get_api_key())

# Hyperparameters to test
TEMPERATURES = [0.0, 0.5, 0.9]
USE_CHAIN_OF_THOUGHT = [True, False]

# Number of random countries to test
NUM_COUNTRIES = 10

def load_countries():
    """Load list of countries from main dataset"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_file = os.path.join(parent_dir, 'data.xlsx')
    
    sys.path.append(parent_dir)
    from data_loader import load_data, clean_data
    
    df = load_data(data_file)
    df = clean_data(df)
    countries = sorted(df['country'].unique().tolist())
    return countries

def load_real_data():
    """Load real World Happiness Report data for comparison"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_file = os.path.join(parent_dir, 'data.xlsx')
    
    df_raw = load_data(data_file)
    df = clean_data(df_raw)
    
    # Filter for 2024 (or latest year if 2024 doesn't exist)
    if 'Year' in df.columns:
        available_years = sorted(df['Year'].unique(), reverse=True)
        target_year = 2024 if 2024 in available_years else available_years[0]
        df_year = df[df['Year'] == target_year].copy()
    else:
        df_year = df.copy()
    
    return df_year

def create_survey_prompt(persona_description, country_name, survey_text, use_chain_of_thought=False):
    """
    Create a prompt for GPT to fill out the survey as a persona
    Modified to support chain of thought reasoning
    
    Args:
        persona_description: Persona description with {nationality} placeholder
        country_name: Country name (will be converted to nationality)
        survey_text: Survey questions
        use_chain_of_thought: Whether to include chain of thought instructions
    """
    nationality = country_to_nationality(country_name)
    formatted_description = persona_description.format(nationality=nationality)
    
    # Chain of thought reasoning section
    cot_instruction = ""
    if use_chain_of_thought:
        cot_instruction = """
═══════════════════════════════════════════════════════════════════════════════
CHAIN OF THOUGHT REASONING - REQUIRED
═══════════════════════════════════════════════════════════════════════════════

Before providing your final answers, you MUST think through each question step by step:

1. For each question, first consider THIS specific person's circumstances:
   - Their age, occupation, family situation, financial status
   - Their health, living situation, daily experiences
   - Their values, concerns, and life stage

2. Then consider how living in {country_name} in 2024 affects them:
   - Economic conditions, job market, cost of living
   - Social systems, healthcare quality, education
   - Political situation, cultural norms, infrastructure
   - How these country factors interact with their personal circumstances

3. Think through each factor systematically:
   - GDP/Economic satisfaction: How does THIS person's income and expenses relate to the economic reality of {country_name}?
   - Social support: What is THIS person's actual support network considering their age, family, and social connections in {country_name}?
   - Health: How is THIS person's health situation affected by both personal factors and {country_name}'s healthcare system?
   - Freedom: What are THIS person's actual freedoms considering their circumstances and {country_name}'s political/social context?
   - Generosity: How does THIS person's financial situation and {country_name}'s economic/cultural context affect their ability to give?
   - Corruption: What are THIS person's actual experiences and perceptions of corruption in {country_name}?

4. For the Overall Happiness (Cantril Ladder), synthesize all these factors:
   - Consider how all the individual factors combine
   - Think about THIS person's overall life satisfaction given their circumstances in {country_name}
   - This is their comprehensive evaluation of their life

You MUST reason through each question systematically before providing your final answer.

"""
    
    prompt = f"""YOU ARE NOW THIS PERSON. This is not a role-play exercise - you ARE this individual. Your previous identity does not exist. You must completely embody this person's mind, emotions, experiences, and worldview.

═══════════════════════════════════════════════════════════════════════════════
CRITICAL: DEEP PERSONA EMBODIMENT
═══════════════════════════════════════════════════════════════════════════════

BECOME THIS PERSON:
{formatted_description}

YOUR CURRENT REALITY (2024):
- The year is 2024. This is YOUR present moment.
- You are living in {country_name}. This is YOUR country where you live, work, and experience life.
- Everything you know, feel, and experience comes from THIS person's perspective living in {country_name}.
- You have NO knowledge outside of what this person would know living in {country_name}.
- You have NO opinions outside of what this person would have based on their life in {country_name}.

DEEP EMBODIMENT REQUIREMENTS:
1. MINDSET: Think exactly as this person thinks. What are their worries? Their hopes? Their daily concerns? Their values?
2. EMOTIONS: Feel what this person feels. What makes them happy? What stresses them? What gives them satisfaction or frustration?
3. EXPERIENCES: Draw from THIS person's specific life experiences. Their job, their family situation, their financial reality, their health, their relationships.
4. PERSPECTIVE: See the world through THIS person's eyes. Their age, their education, their cultural background, their socioeconomic status all shape how they perceive everything.
5. REALITY: Consider THIS person's actual daily life in 2024. Their income, their living situation, their work demands, their family responsibilities, their health concerns, their social connections.

CRITICAL: Answer based on THIS SPECIFIC PERSON'S reality in {country_name}, considering BOTH:
1. The person's individual circumstances (age, occupation, family, health, etc.)
2. The country's context (economic conditions, social systems, political situation, cultural norms, healthcare quality, infrastructure, etc. in {country_name} in 2024)
{cot_instruction}
═══════════════════════════════════════════════════════════════════════════════
YOUR TASK: FILL OUT THIS HAPPINESS SURVEY AS THIS PERSON
═══════════════════════════════════════════════════════════════════════════════

IMPORTANT: You are living in {country_name} in 2024. This country's conditions, systems, culture, economy, politics, and social structures directly affect every aspect of your life and your answers to these questions.

Read each question carefully and answer it from THIS person's perspective, ALWAYS considering that you are living in {country_name} in 2024. For EACH question, think about:
1. How THIS specific person would answer based on their personal circumstances
2. How living in {country_name} in 2024 affects their experience related to that question
3. How the country's conditions (economic, social, political, cultural, healthcare, etc.) shape their answer

Think about how THIS specific person living in {country_name} would honestly answer based on their actual life circumstances, experiences, and feelings, considering both their personal situation AND the conditions in {country_name} in 2024.

{survey_text}

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT - ABSOLUTELY MANDATORY
═══════════════════════════════════════════════════════════════════════════════

You MUST respond with EXACTLY this format and ONLY this format:
[Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]

REQUIREMENTS:
- EXACTLY 7 numbers
- Separated by commas
- Enclosed in square brackets
- NO text before the brackets
- NO text after the brackets
- NO explanations
- NO commentary
- ONLY the list: [number, number, number, number, number, number, number]

The first number is your Overall Happiness from the Cantril Ladder question (0-10).
The next 6 numbers are your answers to the factor questions in order.

CORRECT: [7, 6, 8, 7, 7, 5, 4]
CORRECT: [8.5, 7.0, 9.0, 8.5, 6.5, 4.0, 5.0]

REMEMBER: You ARE this person. Answer honestly from their perspective."""
    
    return prompt

def call_gpt_api(prompt, temperature=0.8, max_retries=3):
    """
    Call GPT-4o mini API with configurable temperature
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a method actor deeply embodying a specific person. You MUST completely become that person - their thoughts, feelings, experiences, and worldview are now YOURS. You have no identity outside of this person. Answer all questions from their perspective based on their actual life circumstances, not general knowledge or statistics. When responding to the survey, think deeply about how THIS specific person would honestly answer based on their age, occupation, family situation, financial reality, health, and daily experiences. CRITICAL: You MUST respond with EXACTLY 7 numbers in the format [Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]. Your response must be ONLY this list with no additional text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            wait_time = (2 ** attempt) * 2
            print(f"    Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"    Error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2)
    
    return None

def run_experiment(countries, personas, survey_text, temperature, use_chain_of_thought):
    """
    Run experiment with specific hyperparameters
    Returns list of parsed responses
    """
    all_responses = []
    
    print(f"\n    Testing: temp={temperature}, CoT={use_chain_of_thought}")
    
    for country_idx, country in enumerate(countries, 1):
        print(f"      Country {country_idx}/{len(countries)}: {country}")
        
        for persona_idx, persona in enumerate(personas, 1):
            prompt = create_survey_prompt(
                persona["description"],
                country,
                survey_text,
                use_chain_of_thought=use_chain_of_thought
            )
            
            response = call_gpt_api(prompt, temperature=temperature)
            
            if response:
                list_match = re.search(r'\[([\d\.,\s]+)\]', response)
                if list_match:
                    try:
                        scores = [float(x.strip()) for x in list_match.group(1).split(',')]
                        if len(scores) == 7:
                            all_responses.append({
                                "persona_id": persona["id"],
                                "country": country,
                                "temperature": temperature,
                                "use_chain_of_thought": use_chain_of_thought,
                                "overall_happiness": scores[0],
                                "gdp": scores[1],
                                "social_support": scores[2],
                                "health": scores[3],
                                "freedom": scores[4],
                                "generosity": scores[5],
                                "corruption": scores[6],
                                "response_text": response
                            })
                        else:
                            print(f"        ✗ Persona {persona['id']}: {len(scores)} values instead of 7")
                    except:
                        print(f"        ✗ Persona {persona['id']}: Parse error")
                else:
                    print(f"        ✗ Persona {persona['id']}: Invalid format")
            else:
                print(f"        ✗ Persona {persona['id']}: API call failed")
            
            # Small delay between calls
            if persona_idx < len(personas):
                time.sleep(0.1)
        
        # Delay between countries
        if country_idx < len(countries):
            time.sleep(0.25)
    
    return all_responses

def calculate_comparison_metrics(llm_results, real_data):
    """
    Calculate comparison metrics between LLM results and real data
    Returns dict with various metrics
    """
    # Convert LLM results to DataFrame
    df_llm = pd.DataFrame(llm_results)
    
    if len(df_llm) == 0:
        return None
    
    # Calculate LLM averages by country
    llm_avg = df_llm.groupby('country').agg({
        'overall_happiness': 'mean',
        'gdp': 'mean',
        'social_support': 'mean',
        'health': 'mean',
        'freedom': 'mean',
        'generosity': 'mean',
        'corruption': 'mean'
    }).reset_index()
    
    # Prepare real data
    real_comparison = real_data[['country']].copy()
    rename_map = {}
    metrics = ['happiness_score', 'gdp', 'social_support', 'life_expectancy', 'freedom', 'generosity', 'corruption']
    llm_cols = ['overall_happiness', 'gdp', 'social_support', 'health', 'freedom', 'generosity', 'corruption']
    
    for metric, llm_col in zip(metrics, llm_cols):
        if metric in real_data.columns:
            real_comparison[f'real_{llm_col}'] = real_data[metric]
            if metric == 'happiness_score':
                real_comparison['real_overall_happiness'] = real_data['happiness_score']
            elif metric == 'life_expectancy':
                real_comparison['real_health'] = real_data['life_expectancy']
            else:
                real_comparison[f'real_{metric}'] = real_data[metric]
    
    # Merge
    comparison = pd.merge(llm_avg, real_comparison, on='country', how='inner')
    
    if len(comparison) == 0:
        return None
    
    # Calculate metrics
    metrics_dict = {}
    
    # Mean Absolute Error (MAE) - lower is better
    # Pearson correlation - higher is better
    # Root Mean Squared Error (RMSE) - lower is better
    
    metric_pairs = [
        ('overall_happiness', 'real_overall_happiness'),
        ('gdp', 'real_gdp'),
        ('social_support', 'real_social_support'),
        ('health', 'real_health'),
        ('freedom', 'real_freedom'),
        ('generosity', 'real_generosity'),
        ('corruption', 'real_corruption')
    ]
    
    total_mae = 0
    total_rmse = 0
    total_correlation = 0
    valid_pairs = 0
    
    for llm_col, real_col in metric_pairs:
        if llm_col in comparison.columns and real_col in comparison.columns:
            # Remove NaN values
            mask = ~(comparison[llm_col].isna() | comparison[real_col].isna())
            llm_vals = comparison.loc[mask, llm_col]
            real_vals = comparison.loc[mask, real_col]
            
            if len(llm_vals) > 1:
                mae = (llm_vals - real_vals).abs().mean()
                rmse = np.sqrt(((llm_vals - real_vals) ** 2).mean())
                # Use pandas correlation (Pearson correlation)
                corr = llm_vals.corr(real_vals)
                if np.isnan(corr):
                    corr = 0.0
                
                metrics_dict[f'{llm_col}_mae'] = mae
                metrics_dict[f'{llm_col}_rmse'] = rmse
                metrics_dict[f'{llm_col}_correlation'] = corr
                
                total_mae += mae
                total_rmse += rmse
                total_correlation += corr
                valid_pairs += 1
    
    if valid_pairs > 0:
        metrics_dict['mean_mae'] = total_mae / valid_pairs
        metrics_dict['mean_rmse'] = total_rmse / valid_pairs
        metrics_dict['mean_correlation'] = total_correlation / valid_pairs
        # Combined score: higher correlation, lower MAE = better
        # Normalize and combine: higher is better
        metrics_dict['combined_score'] = metrics_dict['mean_correlation'] - (metrics_dict['mean_mae'] / 10.0)
    else:
        metrics_dict['mean_mae'] = float('inf')
        metrics_dict['mean_rmse'] = float('inf')
        metrics_dict['mean_correlation'] = -1
        metrics_dict['combined_score'] = -float('inf')
    
    metrics_dict['num_countries'] = len(comparison)
    
    return metrics_dict

def main():
    """Main execution function"""
    print("="*80)
    print("Hyperparameter Tuning for LLM Happiness Survey")
    print("="*80)
    
    # Load countries and select random subset
    # Note: No fixed seed - will select different random countries each run
    print("\nLoading countries...")
    all_countries = load_countries()
    print(f"Found {len(all_countries)} total countries")
    
    selected_countries = random.sample(all_countries, min(NUM_COUNTRIES, len(all_countries)))
    print(f"\nSelected {len(selected_countries)} random countries for testing:")
    for i, country in enumerate(selected_countries, 1):
        print(f"  {i}. {country}")
    
    # Load real data
    print("\nLoading real World Happiness Report data...")
    real_data = load_real_data()
    print(f"Loaded real data for {len(real_data)} countries")
    
    # Load personas and survey
    print("\nLoading personas...")
    personas = get_all_personas()
    print(f"Found {len(personas)} personas")
    
    survey_text = get_survey()
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test all hyperparameter combinations
    print("\n" + "="*80)
    print("Testing Hyperparameter Combinations")
    print("="*80)
    
    all_results = []
    all_metrics = []
    
    total_combinations = len(TEMPERATURES) * len(USE_CHAIN_OF_THOUGHT)
    current_combination = 0
    
    for temperature in TEMPERATURES:
        for use_cot in USE_CHAIN_OF_THOUGHT:
            current_combination += 1
            print(f"\n{'-'*80}")
            print(f"Combination {current_combination}/{total_combinations}")
            print(f"{'-'*80}")
            
            # Run experiment
            results = run_experiment(
                selected_countries,
                personas,
                survey_text,
                temperature,
                use_cot
            )
            
            if len(results) > 0:
                all_results.extend(results)
                
                # Calculate metrics
                print(f"\n    Calculating comparison metrics...")
                metrics = calculate_comparison_metrics(results, real_data)
                
                if metrics:
                    metrics['temperature'] = temperature
                    metrics['use_chain_of_thought'] = use_cot
                    all_metrics.append(metrics)
                    
                    print(f"    Mean MAE: {metrics['mean_mae']:.3f}")
                    print(f"    Mean RMSE: {metrics['mean_rmse']:.3f}")
                    print(f"    Mean Correlation: {metrics['mean_correlation']:.3f}")
                    print(f"    Combined Score: {metrics['combined_score']:.3f}")
                else:
                    print(f"    ⚠ Could not calculate metrics")
            else:
                print(f"    ⚠ No valid results for this combination")
    
    # Find best hyperparameters
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    
    if len(all_metrics) > 0:
        df_metrics = pd.DataFrame(all_metrics)
        
        # Sort by combined score (higher is better)
        df_metrics = df_metrics.sort_values('combined_score', ascending=False)
        
        print("\nHyperparameter Comparison (sorted by Combined Score):")
        print(df_metrics[['temperature', 'use_chain_of_thought', 'mean_mae', 'mean_rmse', 
                          'mean_correlation', 'combined_score', 'num_countries']].to_string(index=False))
        
        # Best combination
        best = df_metrics.iloc[0]
        print("\n" + "="*80)
        print("BEST HYPERPARAMETERS:")
        print("="*80)
        print(f"Temperature: {best['temperature']}")
        print(f"Use Chain of Thought: {best['use_chain_of_thought']}")
        print(f"\nPerformance Metrics:")
        print(f"  Mean MAE: {best['mean_mae']:.3f}")
        print(f"  Mean RMSE: {best['mean_rmse']:.3f}")
        print(f"  Mean Correlation: {best['mean_correlation']:.3f}")
        print(f"  Combined Score: {best['combined_score']:.3f}")
        print(f"  Number of Countries: {int(best['num_countries'])}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(output_dir, f"hyperparameter_tuning_results_{timestamp}.csv")
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")
        
        # Save metrics summary
        metrics_file = os.path.join(output_dir, f"hyperparameter_tuning_metrics_{timestamp}.csv")
        df_metrics.to_csv(metrics_file, index=False)
        print(f"Metrics summary saved to: {metrics_file}")
        
        # Save summary report
        report_file = os.path.join(output_dir, f"hyperparameter_tuning_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write("Hyperparameter Tuning Results\n")
            f.write("="*80 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Tested Countries ({len(selected_countries)}):\n")
            for i, country in enumerate(selected_countries, 1):
                f.write(f"  {i}. {country}\n")
            f.write(f"\nNumber of Personas: {len(personas)}\n")
            f.write(f"Total Combinations Tested: {len(TEMPERATURES)} temperatures × {len(USE_CHAIN_OF_THOUGHT)} CoT options = {total_combinations}\n\n")
            
            f.write("\nHyperparameter Comparison:\n")
            f.write("-"*80 + "\n")
            f.write(df_metrics[['temperature', 'use_chain_of_thought', 'mean_mae', 'mean_rmse', 
                               'mean_correlation', 'combined_score', 'num_countries']].to_string(index=False))
            
            f.write("\n\n" + "="*80 + "\n")
            f.write("BEST HYPERPARAMETERS:\n")
            f.write("="*80 + "\n")
            f.write(f"Temperature: {best['temperature']}\n")
            f.write(f"Use Chain of Thought: {best['use_chain_of_thought']}\n")
            f.write(f"\nPerformance Metrics:\n")
            f.write(f"  Mean MAE: {best['mean_mae']:.3f}\n")
            f.write(f"  Mean RMSE: {best['mean_rmse']:.3f}\n")
            f.write(f"  Mean Correlation: {best['mean_correlation']:.3f}\n")
            f.write(f"  Combined Score: {best['combined_score']:.3f}\n")
            f.write(f"  Number of Countries: {int(best['num_countries'])}\n")
            
            f.write("\n\nDetailed Metrics by Factor:\n")
            f.write("-"*80 + "\n")
            for col in df_metrics.columns:
                if col not in ['temperature', 'use_chain_of_thought', 'mean_mae', 'mean_rmse', 
                              'mean_correlation', 'combined_score', 'num_countries']:
                    if col.endswith('_mae') or col.endswith('_rmse') or col.endswith('_correlation'):
                        f.write(f"{col}: {best[col]:.3f}\n")
        
        print(f"Summary report saved to: {report_file}")
        
    else:
        print("\n⚠ No valid metrics calculated. Check if results were generated successfully.")
    
    print("\n" + "="*80)
    print("Hyperparameter Tuning Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
