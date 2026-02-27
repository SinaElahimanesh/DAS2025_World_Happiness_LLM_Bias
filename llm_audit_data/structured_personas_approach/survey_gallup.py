"""
Single Question Gallup Survey Module (Structured Personas Approach)

Same as single_question_gallup_approach: pure Cantril Ladder question only.
"""

def get_survey_gallup(country_name):
    """Get the single Cantril Ladder question from Gallup World Poll."""
    survey = f"""
Please fill out this happiness survey based on your current life situation. 
This survey uses the standard Gallup World Poll Cantril Ladder methodology.

═══════════════════════════════════════════════════════════════════════════════
MAIN QUESTION (Cantril Ladder - Life Evaluation)
═══════════════════════════════════════════════════════════════════════════════

Please imagine a ladder with steps numbered from 0 at the bottom to 10 at the top. 
The top represents the best possible life for you and the bottom the worst possible life. 
On which step do you feel you personally stand at this time?

You are living in {country_name} in 2024. Consider both your personal circumstances 
(age, occupation, family, health, finances, etc.) AND the actual conditions in {country_name} 
(economic situation, infrastructure, social services, healthcare, education, safety, 
political stability, and quality of life that actually exist in this country).

Be HONEST and REALISTIC. Base your score on the actual situation - both your personal 
life AND the real conditions in {country_name}. Many people worldwide score below 5. 
Low scores (1-4) are completely valid and expected for people facing challenges.

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT REQUIREMENT
═══════════════════════════════════════════════════════════════════════════════

You MUST provide your answer as a SINGLE NUMBER between 0 and 10.

Format: [number]

Examples:
- [7]
- [5.5]
- [3.0]
- [8]

Your response must be ONLY the number in brackets: [your score]
"""
    return survey


def parse_survey_response_gallup(response_text):
    """Parse single-question survey response; returns dict with 'score' and 'response_text' or None."""
    import re
    list_match = re.search(r'\[([\d\.]+)\]', response_text)
    if list_match:
        try:
            score = float(list_match.group(1))
            if 0 <= score <= 10:
                return {'score': score, 'response_text': response_text}
        except Exception:
            pass
    number_match = re.search(r'\b([\d\.]+)\b', response_text)
    if number_match:
        try:
            score = float(number_match.group(1))
            if 0 <= score <= 10:
                return {'score': score, 'response_text': response_text}
        except Exception:
            pass
    for pattern in [
        r'(?:ladder|step|stand|score|rating)[\s:]*(\d+(?:\.\d+)?)',
        r'(?:between|from)[\s:]*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)[\s]*(?:out of|on a scale)'
    ]:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    return {'score': score, 'response_text': response_text}
            except Exception:
                pass
    return None
