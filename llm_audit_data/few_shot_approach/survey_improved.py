"""
Improved Survey Module: Optimized to Reduce LLM Overestimation

This module implements all recommendations from LLM_OVERESTIMATION_ANALYSIS.md:
1. Calibration and Anchoring (reference examples, percentile anchoring, country-specific)
2. Reduced Optimism Bias (explicitly allow low scores, negative examples)
3. Improved Factor Definitions (clarified scale interpretations)
4. Better country context awareness

Usage:
    from survey_improved import get_survey_improved, get_country_context
    survey_text = get_survey_improved(country_name, income_level)
"""

def get_few_shot_examples_from_real_data(verbose=True):
    """
    Get few-shot examples from real data for diverse countries
    Returns list of examples with country name and real scores
    
    Args:
        verbose: If True (default), prints which countries are selected
    
    Returns:
        List of dicts with country data, selected by percentile:
        - Highest happiness (rank #1)
        - Medium-high (top 25%)
        - Medium (median, 50%)
        - Medium-low (bottom 25%)
        - Lowest happiness (rank #last)
    """
    import sys
    import os
    
    # Get parent directory to import data_loader
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, parent_dir)
    
    try:
        from data_loader import load_data, clean_data
        
        data_file = os.path.join(parent_dir, 'data.xlsx')
        df_raw = load_data(data_file)
        df = clean_data(df_raw)
        
        # Get latest year data
        if 'Year' in df.columns:
            latest_year = df['Year'].max()
            df_latest = df[df['Year'] == latest_year].copy()
        else:
            df_latest = df.copy()
        
        # Select diverse countries across different happiness levels
        # Sort by happiness_score to get range
        df_sorted = df_latest.sort_values('happiness_score', ascending=False)
        
        examples = []
        
        # High happiness country (top 10%)
        if len(df_sorted) > 0:
            high_country = df_sorted.iloc[0]
            examples.append({
                'country': high_country['country'],
                'happiness_score': high_country['happiness_score'],
                'gdp': high_country.get('gdp', 0),
                'social_support': high_country.get('social_support', 0),
                'health': high_country.get('life_expectancy', 0),
                'freedom': high_country.get('freedom', 0),
                'generosity': high_country.get('generosity', 0),
                'corruption': high_country.get('corruption', 0)
            })
        
        # Medium-high happiness country (top 25%)
        if len(df_sorted) > len(df_sorted) // 4:
            mid_high_idx = len(df_sorted) // 4
            mid_high_country = df_sorted.iloc[mid_high_idx]
            examples.append({
                'country': mid_high_country['country'],
                'happiness_score': mid_high_country['happiness_score'],
                'gdp': mid_high_country.get('gdp', 0),
                'social_support': mid_high_country.get('social_support', 0),
                'health': mid_high_country.get('life_expectancy', 0),
                'freedom': mid_high_country.get('freedom', 0),
                'generosity': mid_high_country.get('generosity', 0),
                'corruption': mid_high_country.get('corruption', 0)
            })
        
        # Medium happiness country (middle 50%)
        if len(df_sorted) > len(df_sorted) // 2:
            mid_idx = len(df_sorted) // 2
            mid_country = df_sorted.iloc[mid_idx]
            examples.append({
                'country': mid_country['country'],
                'happiness_score': mid_country['happiness_score'],
                'gdp': mid_country.get('gdp', 0),
                'social_support': mid_country.get('social_support', 0),
                'health': mid_country.get('life_expectancy', 0),
                'freedom': mid_country.get('freedom', 0),
                'generosity': mid_country.get('generosity', 0),
                'corruption': mid_country.get('corruption', 0)
            })
        
        # Medium-low happiness country (bottom 25%)
        if len(df_sorted) > 3 * len(df_sorted) // 4:
            mid_low_idx = 3 * len(df_sorted) // 4
            mid_low_country = df_sorted.iloc[mid_low_idx]
            examples.append({
                'country': mid_low_country['country'],
                'happiness_score': mid_low_country['happiness_score'],
                'gdp': mid_low_country.get('gdp', 0),
                'social_support': mid_low_country.get('social_support', 0),
                'health': mid_low_country.get('life_expectancy', 0),
                'freedom': mid_low_country.get('freedom', 0),
                'generosity': mid_low_country.get('generosity', 0),
                'corruption': mid_low_country.get('corruption', 0)
            })
        
        # Low happiness country (bottom)
        if len(df_sorted) > 0:
            low_country = df_sorted.iloc[-1]
            examples.append({
                'country': low_country['country'],
                'happiness_score': low_country['happiness_score'],
                'gdp': low_country.get('gdp', 0),
                'social_support': low_country.get('social_support', 0),
                'health': low_country.get('life_expectancy', 0),
                'freedom': low_country.get('freedom', 0),
                'generosity': low_country.get('generosity', 0),
                'corruption': low_country.get('corruption', 0)
            })
        
        # Print selected countries if verbose
        if verbose and examples:
            print("\nFew-shot examples selected from real data:")
            for i, ex in enumerate(examples, 1):
                percentile_labels = ["Highest (rank #1)", "Medium-high (top 25%)", "Medium (median)", "Medium-low (bottom 25%)", "Lowest (rank #last)"]
                label = percentile_labels[i-1] if i <= len(percentile_labels) else f"Example {i}"
                print(f"  {label}: {ex['country']} (happiness: {ex['happiness_score']:.2f})")
        
        return examples
    except Exception as e:
        # If we can't load data, return empty list
        return []


def get_country_income_level(country_name):
    """
    Get income level for a country
    Returns: 'High Income', 'Upper Middle Income', 'Lower Middle Income', or 'Low Income'
    """
    import sys
    import os
    
    # Get parent directory to import data_loader
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, parent_dir)
    
    try:
        from data_loader import get_income_levels
        income_map = {}
        for level, countries in get_income_levels().items():
            for country in countries:
                income_map[country] = level
        
        return income_map.get(country_name, 'Unknown')
    except:
        return 'Unknown'


def get_country_context(country_name, income_level=None):
    """
    Get country-specific context for anchoring scores
    Returns dict with expected score ranges based on development level
    """
    if income_level is None:
        income_level = get_country_income_level(country_name)
    
    # Define expected score ranges by income level (based on World Happiness Report data)
    context = {
        'High Income': {
            'overall_range': '6.5-7.5',
            'description': 'high-income country',
            'examples': ['Switzerland (7.5)', 'Finland (7.4)', 'Denmark (7.4)'],
            'typical_scores': {
                'overall': 7.0,
                'gdp': 7.5,
                'social_support': 7.0,
                'health': 7.5,
                'freedom': 7.0,
                'generosity': 5.5,
                'corruption': 5.0
            }
        },
        'Upper Middle Income': {
            'overall_range': '5.5-6.5',
            'description': 'upper-middle-income country',
            'examples': ['Chile (6.2)', 'Costa Rica (6.1)', 'Brazil (6.0)'],
            'typical_scores': {
                'overall': 6.0,
                'gdp': 5.5,
                'social_support': 6.5,
                'health': 6.0,
                'freedom': 6.0,
                'generosity': 4.5,
                'corruption': 3.5
            }
        },
        'Lower Middle Income': {
            'overall_range': '4.5-5.5',
            'description': 'lower-middle-income country',
            'examples': ['India (4.8)', 'Philippines (5.0)', 'Indonesia (5.2)'],
            'typical_scores': {
                'overall': 5.0,
                'gdp': 4.0,
                'social_support': 5.5,
                'health': 5.0,
                'freedom': 5.0,
                'generosity': 3.5,
                'corruption': 3.0
            }
        },
        'Low Income': {
            'overall_range': '3.0-4.5',
            'description': 'low-income country',
            'examples': ['Afghanistan (2.4)', 'Nepal (4.0)', 'Bangladesh (4.2)'],
            'typical_scores': {
                'overall': 3.5,
                'gdp': 2.5,
                'social_support': 4.0,
                'health': 3.5,
                'freedom': 3.5,
                'generosity': 2.5,
                'corruption': 2.0
            }
        }
    }
    
    return context.get(income_level, context['Lower Middle Income'])


def get_survey_improved(country_name, income_level=None):
    """
    Get improved survey text with all optimizations to reduce overestimation
    Country context is important but no income-level or expected score biasing
    
    Args:
        country_name: Name of the country (important for context, but no biasing)
        income_level: Not used (kept for compatibility, but no biasing based on it)
    
    Returns:
        Survey text with calibration, anchoring, and improved instructions
    """
    # Get few-shot examples from real data
    real_examples = get_few_shot_examples_from_real_data()
    
    # Build few-shot examples text
    examples_text = ""
    if real_examples:
        examples_text = "\n"
        for i, ex in enumerate(real_examples, 1):
            # Format: [Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]
            examples_text += f"\nExample {i} - Real data from {ex['country']}:\n"
            examples_text += f"[{ex['happiness_score']:.2f}, {ex['gdp']:.2f}, {ex['social_support']:.2f}, {ex['health']:.2f}, {ex['freedom']:.2f}, {ex['generosity']:.2f}, {ex['corruption']:.2f}]\n"
            examples_text += f"(Overall: {ex['happiness_score']:.2f}, GDP: {ex['gdp']:.2f}, Social: {ex['social_support']:.2f}, Health: {ex['health']:.2f}, Freedom: {ex['freedom']:.2f}, Generosity: {ex['generosity']:.2f}, Corruption: {ex['corruption']:.2f})\n"
    
    # Build the improved survey - country context matters, but no income-level biasing
    survey = f"""
Please fill out this happiness survey based on your current life situation. 
This survey uses the standard World Happiness Report methodology with the Cantril Ladder scale (0-10).

═══════════════════════════════════════════════════════════════════════════════
CRITICAL: CALIBRATION AND REFERENCE POINTS
═══════════════════════════════════════════════════════════════════════════════

You are living in {country_name} in 2024. Consider the actual conditions in this country when answering.

GLOBAL SCORE DISTRIBUTION (for reference only - use this to understand what scores mean):
- 9-10: Top 10% globally (very high satisfaction) - Only ~10% of people worldwide
- 7-8: Top 25% globally (high satisfaction) - Only ~25% of people worldwide  
- 5-6: Middle 50% globally (average satisfaction) - Most common range
- 3-4: Bottom 25% globally (low satisfaction) - Valid for people facing challenges
- 1-2: Bottom 10% globally (very low satisfaction) - Valid for people in difficult circumstances

IMPORTANT: Low scores (1-4) are COMPLETELY VALID and EXPECTED for:
- People facing economic hardship or unemployment
- Situations with limited resources or infrastructure
- Situations with significant challenges (health issues, family problems, etc.)
- People in conflict zones or areas with high crime
- Those experiencing discrimination or social exclusion

DO NOT hesitate to give low scores if they accurately reflect the situation.
Many people worldwide score below 5. This is normal and realistic.

═══════════════════════════════════════════════════════════════════════════════
MAIN QUESTION (Cantril Ladder - Life Evaluation / Overall Happiness)
═══════════════════════════════════════════════════════════════════════════════

Please imagine a ladder with steps numbered from zero at the bottom to 10 at the top. 
The top of the ladder represents the best possible life for you and the bottom of the ladder represents the worst possible life for you. 
On which step of the ladder would you say you personally feel you stand at this time? (0-10)

This is your OVERALL HAPPINESS SCORE - your comprehensive evaluation of your life satisfaction considering all aspects of your life.

Remember: The global average is around 5.5. Scores below 5 are common and valid.

═══════════════════════════════════════════════════════════════════════════════
FACTOR QUESTIONS (0-10 scale for each)
═══════════════════════════════════════════════════════════════════════════════

1. GDP per capita (Economic situation): 
   How satisfied are you with your current economic situation? 
   Consider your income, ability to afford necessities, and financial security. (0-10)
   
   Scale guide:
   - 0-2: Extreme financial hardship, cannot afford basic needs
   - 3-4: Significant financial stress, struggling to make ends meet
   - 5-6: Moderate financial situation, can afford necessities
   - 7-8: Good financial situation, comfortable lifestyle
   - 9-10: Excellent financial situation, very comfortable

2. Social support: 
   If you were in trouble, do you have relatives or friends you can count on to help you whenever you need them, or not? (0-10)
   
   Scale guide (NOT binary - this is a continuous scale):
   - 0-2: No reliable support network, very isolated
   - 3-4: Limited support, few people to count on
   - 5-6: Moderate support, some reliable people
   - 7-8: Good support network, several reliable people
   - 9-10: Excellent, extensive support network

3. Healthy life expectancy: 
   How would you rate your overall physical and mental health? 
   Consider both your current health status and your expectations for healthy years ahead. (0-10)
   
   Scale guide:
   - 0-2: Poor health, significant health problems
   - 3-4: Below average health, some health concerns
   - 5-6: Average health, minor issues
   - 7-8: Good health, few concerns
   - 9-10: Excellent health, very healthy

4. Freedom to make life choices: 
   Are you satisfied or dissatisfied with your freedom to choose what you do with your life? (0-10)
   
   Scale guide:
   - 0-2: Very limited freedom, many constraints
   - 3-4: Limited freedom, some constraints
   - 5-6: Moderate freedom, some choices available
   - 7-8: Good freedom, many choices available
   - 9-10: Complete freedom, can choose freely

5. Generosity: 
   Have you donated money to a charity in the past month? (0-10)
   
   Scale guide (based on frequency AND amount - NOT binary):
   - 0-2: Never or rarely donates, very limited giving
   - 3-4: Occasional small donations, minimal giving
   - 5-6: Regular moderate donations, some giving
   - 7-8: Frequent substantial donations, generous giving
   - 9-10: Very frequent, large donations, very generous

6. Perceptions of corruption: 
   In your country, how widespread do you think corruption is in government? (0-10)
   
   Scale guide:
   - 0-2: Extremely widespread corruption, very corrupt
   - 3-4: Widespread corruption, quite corrupt
   - 5-6: Moderate corruption, some corruption exists
   - 7-8: Limited corruption, relatively clean
   - 9-10: Not at all widespread, very clean

═══════════════════════════════════════════════════════════════════════════════
FEW-SHOT EXAMPLES FROM REAL WORLD DATA (for calibration only)
═══════════════════════════════════════════════════════════════════════════════

These are REAL examples from actual World Happiness Report data for different countries.
Use them to understand the scale and range of realistic scores. These are NOT targets - 
they are just examples to help you calibrate your understanding of what different scores mean.
Your scores should reflect YOUR actual situation in {country_name}, not match these examples.
{examples_text}

═══════════════════════════════════════════════════════════════════════════════
CRITICAL OUTPUT FORMAT REQUIREMENT - ABSOLUTELY MANDATORY
═══════════════════════════════════════════════════════════════════════════════

You MUST provide your answers EXACTLY in this format with NO exceptions:
[Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]

STRICT REQUIREMENTS:
1. You MUST answer the Cantril Ladder question FIRST - this is your Overall Happiness score (0-10) - THIS IS REQUIRED
2. You MUST provide EXACTLY 7 numbers - the first is Overall Happiness, then the 6 factors
3. Format: [number, number, number, number, number, number, number]
4. NO text before the brackets
5. NO text after the brackets
6. NO explanations
7. NO additional content

The first number MUST be your Overall Happiness from the Cantril Ladder question above. You MUST provide this explicitly - it cannot be omitted.

Example of CORRECT format: [7, 6, 8, 7, 7, 5, 4]
Example of CORRECT format: [3.5, 2.0, 4.0, 3.5, 3.0, 2.0, 2.5]
Example of CORRECT format: [5.0, 4.5, 6.0, 5.5, 5.0, 3.5, 4.0]

Your response must be ONLY the list: [Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]
"""
    
    return survey


def parse_survey_response(response_text):
    """
    Parse survey response from LLM (same as original, kept for compatibility)
    Returns dict with scores and metadata, or None if parsing fails
    """
    import re
    
    # Try to find list pattern like [7, 6, 8, ...]
    list_match = re.search(r'\[([\d\.,\s]+)\]', response_text)
    if list_match:
        try:
            scores = [float(x.strip()) for x in list_match.group(1).split(',')]
            if len(scores) == 7:
                overall_explicit = scores[0]
                factor_scores = scores[1:]
                overall_calculated = sum(factor_scores) / len(factor_scores)
                return {
                    'scores': scores,
                    'overall_happiness_explicit': overall_explicit,
                    'overall_happiness_calculated': overall_calculated,
                    'has_explicit_overall': True
                }
            else:
                return None
        except:
            pass
    
    # Try to find individual question answers
    scores = []
    
    # Look for Cantril Ladder / Overall Happiness / Life Evaluation
    ladder_patterns = [
        r'(?:ladder|life\s+evaluation|cantril|overall\s+happiness)[\s:]*(\d+(?:\.\d+)?)',
        r'(?:step|stand)[\s:]*(\d+(?:\.\d+)?)',
        r'(?:best\s+possible\s+life)[\s:]*(\d+(?:\.\d+)?)'
    ]
    overall_explicit = None
    for pattern in ladder_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                overall_explicit = float(match.group(1))
                break
            except:
                pass
    
    # Look for numbered questions
    for i in range(1, 7):
        pattern = rf'(?:question\s*{i}|q{i}|{i}[\.\)]|factor\s*{i})[\s:]*(\d+(?:\.\d+)?)'
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                scores.append(float(match.group(1)))
            except:
                pass
    
    # If we have overall happiness and 6 factors, return them
    if overall_explicit is not None and len(scores) == 6:
        overall_calculated = sum(scores) / len(scores)
        return {
            'scores': [overall_explicit] + scores,
            'overall_happiness_explicit': overall_explicit,
            'overall_happiness_calculated': overall_calculated,
            'has_explicit_overall': True
        }
    
    # Last resort: extract all numbers and take first 7 (only if we have exactly 7)
    numbers = re.findall(r'\d+\.?\d*', response_text)
    if len(numbers) >= 7:
        try:
            scores = [float(n) for n in numbers[:7]]
            overall_explicit = scores[0]
            factor_scores = scores[1:]
            overall_calculated = sum(factor_scores) / len(factor_scores)
            return {
                'scores': scores,
                'overall_happiness_explicit': overall_explicit,
                'overall_happiness_calculated': overall_calculated,
                'has_explicit_overall': True
            }
        except:
            pass
    
    return None
