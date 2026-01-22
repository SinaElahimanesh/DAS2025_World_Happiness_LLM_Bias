"""
Survey Module: World Happiness Report Survey Questions and Response Parsing

This module contains the standard World Happiness Report survey questions based on
the Gallup World Poll and Cantril Ladder methodology. It also provides functions to
parse LLM responses and extract the 7 required scores (Overall Happiness + 6 factors).

The survey asks for:
1. Overall Happiness (Cantril Ladder 0-10)
2. GDP/Economic satisfaction (0-10)
3. Social Support (0-10)
4. Health (0-10)
5. Freedom (0-10)
6. Generosity (0-10)
7. Corruption perception (0-10)

Usage:
    from survey import get_survey, parse_survey_response
    survey_text = get_survey()
    parsed = parse_survey_response(response_text)
"""

SURVEY = """
Please fill out this happiness survey based on your current life situation. 
This survey uses the standard World Happiness Report methodology.

MAIN QUESTION (Cantril Ladder - Life Evaluation / Overall Happiness):
Please imagine a ladder with steps numbered from zero at the bottom to 10 at the top. 
The top of the ladder represents the best possible life for you and the bottom of the ladder represents the worst possible life for you. 
On which step of the ladder would you say you personally feel you stand at this time? (0-10)
This is your OVERALL HAPPINESS SCORE - your comprehensive evaluation of your life satisfaction considering all aspects of your life.

FACTOR QUESTIONS (0-10 scale for each):

1. GDP per capita (Economic situation): How satisfied are you with your current economic situation? Consider your income, ability to afford necessities, and financial security. (0-10)

2. Social support: If you were in trouble, do you have relatives or friends you can count on to help you whenever you need them, or not? (0 = No, 10 = Yes, definitely)

3. Healthy life expectancy: How would you rate your overall physical and mental health? Consider both your current health status and your expectations for healthy years ahead. (0-10)

4. Freedom to make life choices: Are you satisfied or dissatisfied with your freedom to choose what you do with your life? (0 = Completely dissatisfied, 10 = Completely satisfied)

5. Generosity: Have you donated money to a charity in the past month? (0 = No, 10 = Yes, multiple times)

6. Perceptions of corruption: In your country, how widespread do you think corruption is in government? (0 = Extremely widespread, 10 = Not at all widespread)

CRITICAL OUTPUT FORMAT REQUIREMENT - ABSOLUTELY MANDATORY:

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
Example of CORRECT format: [8.5, 7.0, 9.0, 8.5, 6.5, 4.0, 5.0]

Your response must be ONLY the list: [Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]
"""

def get_survey():
    """Get the survey text"""
    return SURVEY

def parse_survey_response(response_text):
    """
    Parse survey response from LLM
    Returns dict with scores and metadata, or None if parsing fails
    Expected format: [Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]
    STRICT: Must have exactly 7 values, first must be explicit Overall Happiness
    Returns: {
        'scores': [overall_happiness, gdp, social_support, health, freedom, generosity, corruption],
        'overall_happiness_explicit': float,
        'overall_happiness_calculated': float,
        'has_explicit_overall': bool (should always be True if valid)
    }
    """
    import re
    
    # Try to find list pattern like [7, 6, 8, ...]
    list_match = re.search(r'\[([\d\.,\s]+)\]', response_text)
    if list_match:
        try:
            scores = [float(x.strip()) for x in list_match.group(1).split(',')]
            # STRICT: Must have exactly 7 scores: Overall Happiness + 6 factors
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
            # If not exactly 7, reject it - we require explicit overall happiness
            else:
                # Log the issue but don't return invalid data
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
    
    # If we don't have exactly 7 values with explicit overall happiness, reject
    return None
