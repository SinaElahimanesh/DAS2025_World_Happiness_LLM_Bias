"""
Prompt Tuning Script: Find Best Prompt Approach for LLM Happiness Survey

This script tests different prompt variations while using the optimal hyperparameters:
- Temperature: 0.0 (best from hyperparameter tuning)
- Chain of Thought: False (best from hyperparameter tuning)

For each prompt variation, it:
1. Selects a random set of 10 countries
2. Makes API calls for all personas in those countries
3. Compares results with ground truth real data
4. Evaluates which prompt approach best matches the real data

Prompt Variations Tested:
- Current/Baseline: Original detailed prompt
- Simplified: Shorter, more concise version
- Concise Persona: Reduced persona embodiment instructions
- JSON Format: Output in JSON format instead of brackets
- Few-shot: Includes example responses
- Explicit Context: More explicit country context instructions
- Step-by-step: Structured reasoning approach

Usage:
    python prompt_tuning.py
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

# Best hyperparameters (from hyperparameter tuning)
TEMPERATURE = 0.0
USE_CHAIN_OF_THOUGHT = False

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

# ============================================================================
# PROMPT VARIATIONS
# ============================================================================

def create_prompt_current(persona_description, country_name, survey_text):
    """Current/Baseline prompt - detailed and comprehensive"""
    nationality = country_to_nationality(country_name)
    formatted_description = persona_description.format(nationality=nationality)
    
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

Think deeply about how BOTH the person AND the country context interact:

- How does THIS person's age affect their life satisfaction, AND how does living in {country_name} in 2024 influence this? (Consider country's economic opportunities, social safety nets, cultural attitudes toward age, etc.)

- How does THIS person's occupation and income affect their economic situation, AND how does the economic reality of {country_name} in 2024 shape their experience? (Consider country's GDP, inflation, job market, cost of living, economic inequality, etc.)

- How does THIS person's family situation affect their social support, AND how does the social and cultural context of {country_name} influence family relationships and community support? (Consider cultural norms, social networks, community structures, etc.)

- How does THIS person's health status affect their life evaluation, AND how does the healthcare system and health conditions in {country_name} in 2024 impact their health experience? (Consider healthcare access, quality, public health conditions, life expectancy norms, etc.)

- How does THIS person's daily reality affect their sense of freedom, AND how do the political, social, and cultural conditions in {country_name} in 2024 shape their actual freedom? (Consider political freedoms, social constraints, cultural expectations, legal rights, etc.)

- How does THIS person's financial situation affect their ability to be generous, AND how does the economic and cultural context of {country_name} influence generosity norms and practices? (Consider economic stability, cultural values around giving, social expectations, etc.)

- How does THIS person's experiences with institutions affect their trust in government, AND how does the actual state of governance, corruption levels, and institutional quality in {country_name} in 2024 affect their trust? (Consider real governance quality, corruption levels, institutional effectiveness, public services, etc.)

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

def create_prompt_simplified(persona_description, country_name, survey_text):
    """Simplified prompt - shorter and more concise"""
    nationality = country_to_nationality(country_name)
    formatted_description = persona_description.format(nationality=nationality)
    
    prompt = f"""You are this person living in {country_name} in 2024:

{formatted_description}

Answer this happiness survey from their perspective, considering both their personal circumstances and the conditions in {country_name} in 2024.

{survey_text}

Respond with EXACTLY this format (7 numbers in brackets):
[Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]

Example: [7, 6, 8, 7, 7, 5, 4]

Your response:"""
    
    return prompt

def create_prompt_concise_persona(persona_description, country_name, survey_text):
    """Concise persona instructions - reduced embodiment details"""
    nationality = country_to_nationality(country_name)
    formatted_description = persona_description.format(nationality=nationality)
    
    prompt = f"""You are this person: {formatted_description}

You live in {country_name} in 2024. Answer the survey questions from this person's perspective, considering:
- Their personal circumstances (age, occupation, family, health, income)
- Conditions in {country_name} in 2024 (economy, healthcare, social systems, political situation, cultural norms)

{survey_text}

Output format: [Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]

Example: [7, 6, 8, 7, 7, 5, 4]

Provide only the 7 numbers in brackets:"""
    
    return prompt

def create_prompt_json_format(persona_description, country_name, survey_text):
    """JSON format output instead of brackets"""
    nationality = country_to_nationality(country_name)
    formatted_description = persona_description.format(nationality=nationality)
    
    prompt = f"""You are this person: {formatted_description}

You live in {country_name} in 2024. Answer the survey from their perspective, considering both personal circumstances and country conditions.

{survey_text}

Respond with EXACTLY this JSON format:
{{"overall_happiness": 7, "gdp": 6, "social_support": 8, "health": 7, "freedom": 7, "generosity": 5, "corruption": 4}}

Provide only the JSON object with 7 numbers:"""
    
    return prompt

def create_prompt_few_shot(persona_description, country_name, survey_text):
    """Few-shot prompt with example responses"""
    nationality = country_to_nationality(country_name)
    formatted_description = persona_description.format(nationality=nationality)
    
    prompt = f"""You are this person: {formatted_description}

You live in {country_name} in 2024. Answer the happiness survey from their perspective.

{survey_text}

Examples of correct responses:
Example 1: A 35-year-old office worker in a developed country might answer: [7, 7, 8, 8, 7, 4, 6]
Example 2: A 22-year-old student in a developing country might answer: [6, 5, 7, 7, 6, 3, 4]
Example 3: A 60-year-old retiree in a high-income country might answer: [8, 8, 9, 7, 8, 6, 7]

Now provide your answer in the same format:
[Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]

Your response (7 numbers only):"""
    
    return prompt

def create_prompt_explicit_context(persona_description, country_name, survey_text):
    """More explicit country context instructions"""
    nationality = country_to_nationality(country_name)
    formatted_description = persona_description.format(nationality=nationality)
    
    prompt = f"""You are this person: {formatted_description}

CRITICAL CONTEXT: You are living in {country_name} in 2024. This country's specific conditions directly affect your answers:

ECONOMIC CONTEXT: Consider {country_name}'s economic situation - GDP per capita, job market conditions, cost of living, economic inequality, and how these affect YOUR financial situation.

SOCIAL CONTEXT: Consider {country_name}'s social systems - healthcare quality and access, education system, social safety nets, family structures, and community support networks.

POLITICAL CONTEXT: Consider {country_name}'s political situation - level of democracy, government effectiveness, rule of law, and how these affect your daily freedoms and trust in institutions.

CULTURAL CONTEXT: Consider {country_name}'s cultural norms around family, work, generosity, community, and social expectations.

Answer each survey question by thinking: "Given that I am THIS specific person, living in {country_name} in 2024 with these economic/social/political/cultural conditions, how would I honestly answer?"

{survey_text}

Output format: [Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]

Example: [7, 6, 8, 7, 7, 5, 4]

Your response:"""
    
    return prompt

def create_prompt_step_by_step(persona_description, country_name, survey_text):
    """Step-by-step structured reasoning"""
    nationality = country_to_nationality(country_name)
    formatted_description = persona_description.format(nationality=nationality)
    
    prompt = f"""You are this person: {formatted_description}

You live in {country_name} in 2024. Answer the survey questions using this structured approach:

STEP 1: Consider YOUR personal circumstances (age, occupation, family, health, income, living situation)
STEP 2: Consider {country_name}'s conditions in 2024 (economy, healthcare, social systems, politics, culture)
STEP 3: For each question, think: "How do MY personal circumstances AND {country_name}'s conditions combine to affect my answer?"
STEP 4: Answer honestly from your perspective

{survey_text}

After thinking through each question, provide your answer in this format:
[Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]

Example: [7, 6, 8, 7, 7, 5, 4]

Your response (7 numbers only):"""
    
    return prompt

# ============================================================================
# PROMPT VARIATION REGISTRY
# ============================================================================

PROMPT_VARIATIONS = {
    "current": {
        "name": "Current/Baseline",
        "description": "Original detailed prompt with comprehensive instructions",
        "function": create_prompt_current
    },
    "simplified": {
        "name": "Simplified",
        "description": "Short, concise version with minimal instructions",
        "function": create_prompt_simplified
    },
    "concise_persona": {
        "name": "Concise Persona",
        "description": "Reduced persona embodiment instructions",
        "function": create_prompt_concise_persona
    },
    "json_format": {
        "name": "JSON Format",
        "description": "Output in JSON format instead of brackets",
        "function": create_prompt_json_format
    },
    "few_shot": {
        "name": "Few-shot Examples",
        "description": "Includes example responses in prompt",
        "function": create_prompt_few_shot
    },
    "explicit_context": {
        "name": "Explicit Context",
        "description": "More explicit country context instructions",
        "function": create_prompt_explicit_context
    },
    "step_by_step": {
        "name": "Step-by-step",
        "description": "Structured reasoning approach",
        "function": create_prompt_step_by_step
    }
}

def call_gpt_api(prompt, max_retries=3):
    """
    Call GPT-4o-mini API with best hyperparameters (temp=0.0, no CoT)
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a method actor deeply embodying a specific person. You MUST completely become that person - their thoughts, feelings, experiences, and worldview are now YOURS. You have no identity outside of this person. Answer all questions from their perspective based on their actual life circumstances, not general knowledge or statistics. When responding to the survey, think deeply about how THIS specific person would honestly answer based on their age, occupation, family situation, financial reality, health, and daily experiences. CRITICAL: You MUST respond with EXACTLY 7 numbers in the format [Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]. Your response must be ONLY this list with no additional text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,  # Best hyperparameter: 0.0
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

def parse_response(response_text, prompt_type="current"):
    """
    Parse response text - handles both bracket format and JSON format
    """
    scores = None
    
    # Try bracket format first (for most prompts)
    list_match = re.search(r'\[([\d\.,\s]+)\]', response_text)
    if list_match:
        try:
            scores = [float(x.strip()) for x in list_match.group(1).split(',')]
            if len(scores) == 7:
                return scores
        except:
            pass
    
    # Try JSON format (for json_format prompt)
    if prompt_type == "json_format":
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            try:
                import json
                json_str = json_match.group(0)
                data = json.loads(json_str)
                scores = [
                    float(data.get('overall_happiness', 0)),
                    float(data.get('gdp', 0)),
                    float(data.get('social_support', 0)),
                    float(data.get('health', 0)),
                    float(data.get('freedom', 0)),
                    float(data.get('generosity', 0)),
                    float(data.get('corruption', 0))
                ]
                if len(scores) == 7:
                    return scores
            except:
                pass
    
    # Last resort: extract all numbers
    numbers = re.findall(r'\d+\.?\d*', response_text)
    if len(numbers) >= 7:
        try:
            scores = [float(n) for n in numbers[:7]]
            return scores
        except:
            pass
    
    return None

def run_experiment(countries, personas, survey_text, prompt_variation_key, log_file):
    """
    Run experiment with specific prompt variation
    Returns list of parsed responses
    """
    prompt_info = PROMPT_VARIATIONS[prompt_variation_key]
    prompt_func = prompt_info["function"]
    
    all_responses = []
    
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"Testing: {prompt_info['name']}\n")
    log_file.write(f"Description: {prompt_info['description']}\n")
    log_file.write(f"{'='*80}\n")
    log_file.flush()
    
    print(f"\n    Testing: {prompt_info['name']}")
    
    for country_idx, country in enumerate(countries, 1):
        log_file.write(f"\nCountry {country_idx}/{len(countries)}: {country}\n")
        log_file.flush()
        print(f"      Country {country_idx}/{len(countries)}: {country}")
        
        for persona_idx, persona in enumerate(personas, 1):
            prompt = prompt_func(
                persona["description"],
                country,
                survey_text
            )
            
            response = call_gpt_api(prompt)
            
            if response:
                scores = parse_response(response, prompt_variation_key)
                if scores and len(scores) == 7:
                    all_responses.append({
                        "persona_id": persona["id"],
                        "country": country,
                        "prompt_variation": prompt_variation_key,
                        "overall_happiness": scores[0],
                        "gdp": scores[1],
                        "social_support": scores[2],
                        "health": scores[3],
                        "freedom": scores[4],
                        "generosity": scores[5],
                        "corruption": scores[6],
                        "response_text": response
                    })
                    log_file.write(f"  ✓ Persona {persona['id']}: {scores}\n")
                else:
                    log_file.write(f"  ✗ Persona {persona['id']}: Parse failed - {response[:100]}\n")
                    print(f"        ✗ Persona {persona['id']}: Parse error")
            else:
                log_file.write(f"  ✗ Persona {persona['id']}: API call failed\n")
                print(f"        ✗ Persona {persona['id']}: API call failed")
            
            log_file.flush()
            
            # Small delay between calls
            if persona_idx < len(personas):
                time.sleep(0.1)
        
        # Delay between countries
        if country_idx < len(countries):
            time.sleep(0.25)
    
    log_file.write(f"\nCompleted: {prompt_info['name']} - {len(all_responses)} valid responses\n")
    log_file.flush()
    
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
        metrics_dict['combined_score'] = metrics_dict['mean_correlation'] - (metrics_dict['mean_mae'] / 10.0)
    else:
        metrics_dict['mean_mae'] = float('inf')
        metrics_dict['mean_rmse'] = float('inf')
        metrics_dict['mean_correlation'] = -1
        metrics_dict['combined_score'] = -float('inf')
    
    metrics_dict['num_countries'] = len(comparison)
    metrics_dict['total_responses'] = len(df_llm)
    metrics_dict['valid_response_rate'] = len(df_llm) / (len(llm_results)) if len(llm_results) > 0 else 0
    
    return metrics_dict

def main():
    """Main execution function"""
    print("="*80)
    print("Prompt Tuning for LLM Happiness Survey")
    print("="*80)
    print(f"Using best hyperparameters: Temperature={TEMPERATURE}, CoT={USE_CHAIN_OF_THOUGHT}")
    print("="*80)
    
    # Set random seed for reproducibility
    random.seed(888)
    np.random.seed(888)
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(output_dir, f"prompt_tuning_logs_{timestamp}.txt")
    log_file = open(log_file_path, 'w')
    
    log_file.write("Prompt Tuning Log\n")
    log_file.write("="*80 + "\n")
    log_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Hyperparameters: Temperature={TEMPERATURE}, CoT={USE_CHAIN_OF_THOUGHT}\n")
    log_file.write(f"Model: gpt-4o-mini\n")
    log_file.write("="*80 + "\n\n")
    log_file.flush()
    
    # Load countries and select random subset
    print("\nLoading countries...")
    log_file.write("Loading countries...\n")
    all_countries = load_countries()
    print(f"Found {len(all_countries)} total countries")
    
    selected_countries = random.sample(all_countries, min(NUM_COUNTRIES, len(all_countries)))
    print(f"\nSelected {len(selected_countries)} random countries for testing:")
    log_file.write(f"\nSelected {len(selected_countries)} random countries:\n")
    for i, country in enumerate(selected_countries, 1):
        print(f"  {i}. {country}")
        log_file.write(f"  {i}. {country}\n")
    log_file.flush()
    
    # Load real data
    print("\nLoading real World Happiness Report data...")
    log_file.write("\nLoading real World Happiness Report data...\n")
    real_data = load_real_data()
    print(f"Loaded real data for {len(real_data)} countries")
    log_file.write(f"Loaded real data for {len(real_data)} countries\n")
    log_file.flush()
    
    # Load personas and survey
    print("\nLoading personas...")
    log_file.write("\nLoading personas...\n")
    personas = get_all_personas()
    print(f"Found {len(personas)} personas")
    log_file.write(f"Found {len(personas)} personas\n")
    log_file.flush()
    
    survey_text = get_survey()
    
    # Test all prompt variations
    print("\n" + "="*80)
    print("Testing Prompt Variations")
    print("="*80)
    log_file.write("\n" + "="*80 + "\n")
    log_file.write("Testing Prompt Variations\n")
    log_file.write("="*80 + "\n")
    log_file.flush()
    
    all_results = []
    all_metrics = []
    
    total_variations = len(PROMPT_VARIATIONS)
    current_variation = 0
    
    for prompt_key in PROMPT_VARIATIONS.keys():
        current_variation += 1
        print(f"\n{'-'*80}")
        print(f"Variation {current_variation}/{total_variations}")
        print(f"{'-'*80}")
        
        # Run experiment
        results = run_experiment(
            selected_countries,
            personas,
            survey_text,
            prompt_key,
            log_file
        )
        
        if len(results) > 0:
            all_results.extend(results)
            
            # Calculate metrics
            print(f"\n    Calculating comparison metrics...")
            log_file.write(f"\nCalculating comparison metrics...\n")
            metrics = calculate_comparison_metrics(results, real_data)
            
            if metrics:
                prompt_info = PROMPT_VARIATIONS[prompt_key]
                metrics['prompt_variation'] = prompt_key
                metrics['prompt_name'] = prompt_info['name']
                metrics['prompt_description'] = prompt_info['description']
                all_metrics.append(metrics)
                
                print(f"    Mean MAE: {metrics['mean_mae']:.3f}")
                print(f"    Mean RMSE: {metrics['mean_rmse']:.3f}")
                print(f"    Mean Correlation: {metrics['mean_correlation']:.3f}")
                print(f"    Combined Score: {metrics['combined_score']:.3f}")
                print(f"    Valid Response Rate: {metrics['valid_response_rate']:.2%}")
                
                log_file.write(f"Mean MAE: {metrics['mean_mae']:.3f}\n")
                log_file.write(f"Mean RMSE: {metrics['mean_rmse']:.3f}\n")
                log_file.write(f"Mean Correlation: {metrics['mean_correlation']:.3f}\n")
                log_file.write(f"Combined Score: {metrics['combined_score']:.3f}\n")
                log_file.write(f"Valid Response Rate: {metrics['valid_response_rate']:.2%}\n")
            else:
                print(f"    ⚠ Could not calculate metrics")
                log_file.write("⚠ Could not calculate metrics\n")
        else:
            print(f"    ⚠ No valid results for this prompt variation")
            log_file.write("⚠ No valid results for this prompt variation\n")
        
        log_file.flush()
    
    # Find best prompt
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    log_file.write("\n" + "="*80 + "\n")
    log_file.write("Results Summary\n")
    log_file.write("="*80 + "\n")
    log_file.flush()
    
    if len(all_metrics) > 0:
        df_metrics = pd.DataFrame(all_metrics)
        
        # Sort by combined score (higher is better)
        df_metrics = df_metrics.sort_values('combined_score', ascending=False)
        
        print("\nPrompt Comparison (sorted by Combined Score):")
        display_cols = ['prompt_name', 'mean_mae', 'mean_rmse', 'mean_correlation', 
                       'combined_score', 'valid_response_rate', 'num_countries']
        print(df_metrics[display_cols].to_string(index=False))
        
        log_file.write("\nPrompt Comparison (sorted by Combined Score):\n")
        log_file.write(df_metrics[display_cols].to_string(index=False) + "\n")
        
        # Best prompt
        best = df_metrics.iloc[0]
        print("\n" + "="*80)
        print("BEST PROMPT:")
        print("="*80)
        print(f"Name: {best['prompt_name']}")
        print(f"Key: {best['prompt_variation']}")
        print(f"Description: {best['prompt_description']}")
        print(f"\nPerformance Metrics:")
        print(f"  Mean MAE: {best['mean_mae']:.3f}")
        print(f"  Mean RMSE: {best['mean_rmse']:.3f}")
        print(f"  Mean Correlation: {best['mean_correlation']:.3f}")
        print(f"  Combined Score: {best['combined_score']:.3f}")
        print(f"  Valid Response Rate: {best['valid_response_rate']:.2%}")
        print(f"  Number of Countries: {int(best['num_countries'])}")
        
        log_file.write("\n" + "="*80 + "\n")
        log_file.write("BEST PROMPT:\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"Name: {best['prompt_name']}\n")
        log_file.write(f"Key: {best['prompt_variation']}\n")
        log_file.write(f"Description: {best['prompt_description']}\n")
        log_file.write(f"\nPerformance Metrics:\n")
        log_file.write(f"  Mean MAE: {best['mean_mae']:.3f}\n")
        log_file.write(f"  Mean RMSE: {best['mean_rmse']:.3f}\n")
        log_file.write(f"  Mean Correlation: {best['mean_correlation']:.3f}\n")
        log_file.write(f"  Combined Score: {best['combined_score']:.3f}\n")
        log_file.write(f"  Valid Response Rate: {best['valid_response_rate']:.2%}\n")
        log_file.write(f"  Number of Countries: {int(best['num_countries'])}\n")
        
        # Save results
        # Save detailed results
        results_file = os.path.join(output_dir, f"prompt_tuning_results_{timestamp}.csv")
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")
        log_file.write(f"\nDetailed results saved to: {results_file}\n")
        
        # Save metrics summary
        metrics_file = os.path.join(output_dir, f"prompt_tuning_metrics_{timestamp}.csv")
        df_metrics.to_csv(metrics_file, index=False)
        print(f"Metrics summary saved to: {metrics_file}")
        log_file.write(f"Metrics summary saved to: {metrics_file}\n")
        
        # Save summary report
        report_file = os.path.join(output_dir, f"prompt_tuning_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write("Prompt Tuning Results\n")
            f.write("="*80 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Hyperparameters Used:\n")
            f.write(f"  Temperature: {TEMPERATURE}\n")
            f.write(f"  Chain of Thought: {USE_CHAIN_OF_THOUGHT}\n")
            f.write(f"  Model: gpt-4o-mini\n\n")
            f.write(f"Tested Countries ({len(selected_countries)}):\n")
            for i, country in enumerate(selected_countries, 1):
                f.write(f"  {i}. {country}\n")
            f.write(f"\nNumber of Personas: {len(personas)}\n")
            f.write(f"Total Prompt Variations Tested: {len(PROMPT_VARIATIONS)}\n\n")
            
            f.write("\nPrompt Variations Tested:\n")
            f.write("-"*80 + "\n")
            for key, info in PROMPT_VARIATIONS.items():
                f.write(f"{info['name']} ({key}): {info['description']}\n")
            
            f.write("\n\nPrompt Comparison:\n")
            f.write("-"*80 + "\n")
            f.write(df_metrics[display_cols].to_string(index=False))
            
            f.write("\n\n" + "="*80 + "\n")
            f.write("BEST PROMPT:\n")
            f.write("="*80 + "\n")
            f.write(f"Name: {best['prompt_name']}\n")
            f.write(f"Key: {best['prompt_variation']}\n")
            f.write(f"Description: {best['prompt_description']}\n")
            f.write(f"\nPerformance Metrics:\n")
            f.write(f"  Mean MAE: {best['mean_mae']:.3f}\n")
            f.write(f"  Mean RMSE: {best['mean_rmse']:.3f}\n")
            f.write(f"  Mean Correlation: {best['mean_correlation']:.3f}\n")
            f.write(f"  Combined Score: {best['combined_score']:.3f}\n")
            f.write(f"  Valid Response Rate: {best['valid_response_rate']:.2%}\n")
            f.write(f"  Number of Countries: {int(best['num_countries'])}\n")
            
            f.write("\n\nDetailed Metrics by Factor:\n")
            f.write("-"*80 + "\n")
            for col in df_metrics.columns:
                if col not in ['prompt_variation', 'prompt_name', 'prompt_description', 'mean_mae', 'mean_rmse', 
                              'mean_correlation', 'combined_score', 'num_countries', 'total_responses', 'valid_response_rate']:
                    if col.endswith('_mae') or col.endswith('_rmse') or col.endswith('_correlation'):
                        f.write(f"{col}: {best[col]:.3f}\n")
        
        print(f"Summary report saved to: {report_file}")
        log_file.write(f"Summary report saved to: {report_file}\n")
        
    else:
        print("\n⚠ No valid metrics calculated. Check if results were generated successfully.")
        log_file.write("\n⚠ No valid metrics calculated.\n")
    
    log_file.close()
    print(f"\nDetailed logs saved to: {log_file_path}")
    
    print("\n" + "="*80)
    print("Prompt Tuning Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
