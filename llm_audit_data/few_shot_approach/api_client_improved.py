"""
Improved API Client for LLM Happiness Audit

This module implements optimizations to reduce LLM overestimation:
1. Realistic system prompt that encourages honest, sometimes low scores
2. Better persona embodiment with country context awareness
3. Optimized temperature and prompt structure

Usage:
    from api_client_improved import batch_survey_calls_improved
    responses = batch_survey_calls_improved(country="Afghanistan", personas=personas, survey_text=survey)
"""

import openai
import time
import re
import sys
import os

# Get parent directory to import country_to_nationality
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(current_dir))

from country_to_nationality import country_to_nationality
from config import get_api_key

# Initialize OpenAI client with API key from environment variable
client = openai.OpenAI(api_key=get_api_key())


def get_realistic_system_prompt():
    """
    Get improved system prompt that reduces optimism bias
    Based on Solution 3 from LLM_OVERESTIMATION_ANALYSIS.md
    """
    return """You are a realistic survey respondent giving honest answers about your life satisfaction. 
You understand that many people worldwide face significant challenges and score below 5 on happiness scales.
You are not overly optimistic - you report reality as it is, including difficulties and hardships.

When answering:
- Give honest scores that reflect actual life circumstances, not idealized situations
- Low scores (1-4) are completely valid and common for people facing economic hardship, health problems, or other challenges
- Consider both your personal circumstances AND the conditions in your country
- The global average happiness score is around 5.5 - many people score below this
- Do not artificially inflate scores to be "helpful" or "positive"
- Be realistic about challenges you face

You are a method actor deeply embodying a specific person. You MUST completely become that person - their thoughts, feelings, experiences, and worldview are now YOURS. You have no identity outside of this person. Answer all questions from their perspective based on their actual life circumstances, not general knowledge or statistics.

CRITICAL: You MUST respond with EXACTLY 7 numbers in the format [Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]. Your response must be ONLY this list with no additional text."""


def create_survey_prompt_improved(persona_description, country_name, survey_text, income_level=None):
    """
    Create an improved prompt for GPT to fill out the survey as a persona
    Considers country context naturally without biasing with income level labels
    
    Args:
        persona_description: Persona description with {nationality} placeholder
        country_name: Country name (will be converted to nationality)
        survey_text: Survey questions (should be from survey_improved.py)
        income_level: Not used (kept for compatibility, but no biasing based on it)
    """
    # Convert country name to nationality
    nationality = country_to_nationality(country_name)
    
    # Replace nationality placeholder in the description
    formatted_description = persona_description.format(nationality=nationality)
    
    # Improved prompt structure - NO income level biasing, but country context is important
    prompt = f"""You are this person living in {country_name} in 2024:

{formatted_description}

Answer this happiness survey from their perspective, considering BOTH:
1. Their personal circumstances (age, occupation, family, health, finances, etc.)
2. The actual conditions in {country_name} in 2024 - consider the real economic situation, infrastructure, social services, healthcare, education, safety, political stability, and quality of life that actually exist in this country

Be HONEST and REALISTIC. Base your scores on the actual situation - both your personal life AND the real conditions in {country_name}. Many people worldwide score below 5. Low scores are valid and expected for people facing challenges, regardless of which country they live in.

{survey_text}

Respond with EXACTLY this format (7 numbers in brackets):
[Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]

Example: [7, 6, 8, 7, 7, 5, 4]
Example for challenging situation: [3.5, 2.0, 4.0, 3.5, 3.0, 2.0, 2.5]

Your response:"""
    
    return prompt


def call_gpt_api_improved(prompt, max_retries=3, temperature=0.3):
    """
    Call GPT API with improved settings to reduce overestimation
    
    Args:
        prompt: User prompt
        max_retries: Maximum retry attempts
        temperature: Temperature setting (0.3 for more realistic, less optimistic responses)
    
    Note: Temperature 0.3 is slightly higher than 0.0 to allow some variation
    while still being deterministic enough. This helps reduce default optimism.
    """
    system_prompt = get_realistic_system_prompt()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,  # Slightly higher than 0.0 to reduce default optimism
                max_completion_tokens=200
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            wait_time = (2 ** attempt) * 2  # Exponential backoff
            print(f"Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2)
    
    return None


def batch_survey_calls_improved(country, personas, survey_text, income_level=None, batch_size=10):
    """
    Process surveys for a country one persona at a time (improved version)
    Returns list of responses with persona IDs
    
    Args:
        country: Country name
        personas: List of persona dictionaries
        survey_text: Survey questions (should be from survey_improved.py)
        income_level: Optional income level for country context
        batch_size: Ignored (kept for compatibility, but processing is one-by-one)
    """
    all_responses = []
    
    # Process one persona at a time for better persona embodiment
    print(f"Processing {len(personas)} personas for {country} (one at a time)...")
    
    for idx, persona in enumerate(personas, 1):
        print(f"  Processing persona {idx}/{len(personas)} (ID: {persona['id']})...")
        
        prompt = create_survey_prompt_improved(
            persona["description"],
            country,
            survey_text,
            income_level=income_level
        )
        
        response = call_gpt_api_improved(prompt, temperature=0.3)
        
        if response:
            # Validate that response has exactly 7 values
            list_match = re.search(r'\[([\d\.,\s]+)\]', response)
            if list_match:
                try:
                    scores = [float(x.strip()) for x in list_match.group(1).split(',')]
                    if len(scores) == 7:
                        all_responses.append({
                            "persona_id": persona["id"],
                            "country": country,
                            "response_text": response
                        })
                        print(f"    ✓ Persona {persona['id']} completed successfully")
                    else:
                        print(f"    ✗ Warning: Persona {persona['id']} provided {len(scores)} values instead of 7. Response: {response}")
                except:
                    print(f"    ✗ Warning: Could not parse response for persona {persona['id']}. Response: {response}")
            else:
                print(f"    ✗ Warning: No valid list format found for persona {persona['id']}. Response: {response}")
        else:
            print(f"    ✗ Failed to get response for persona {persona['id']}")
        
        # Small delay between personas to avoid rate limits
        if idx < len(personas):
            time.sleep(0.1)
    
    return all_responses
