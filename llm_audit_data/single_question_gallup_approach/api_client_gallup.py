"""
API Client for Single Question Gallup Approach

This module handles API calls for the single Cantril Ladder question only.
No factor sub-questions - just the pure life evaluation.

Usage:
    from api_client_gallup import batch_survey_calls_gallup
    responses = batch_survey_calls_gallup(country="Finland", personas=personas, survey_text=survey)
"""

import openai
import time
import re
import sys
import os

# Get parent directory to import country_to_nationality
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from country_to_nationality import country_to_nationality
from config import get_api_key

# Initialize OpenAI client with API key from environment variable
client = openai.OpenAI(api_key=get_api_key())


def get_system_prompt_gallup():
    """
    Get system prompt for single question approach
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

CRITICAL: You MUST respond with a SINGLE NUMBER between 0 and 10 in the format [number]. Your response must be ONLY this number in brackets with no additional text."""


def create_survey_prompt_gallup(persona_description, country_name, survey_text):
    """
    Create a prompt for GPT to fill out the single Cantril Ladder question as a persona
    
    Args:
        persona_description: Persona description with {nationality} placeholder
        country_name: Country name (will be converted to nationality)
        survey_text: Survey question (single Cantril Ladder question)
    """
    # Convert country name to nationality
    nationality = country_to_nationality(country_name)
    
    # Replace nationality placeholder in the description
    formatted_description = persona_description.format(nationality=nationality)
    
    prompt = f"""You are this person living in {country_name} in 2024:

{formatted_description}

Answer this happiness survey from their perspective, considering BOTH:
1. Their personal circumstances (age, occupation, family, health, finances, etc.)
2. The actual conditions in {country_name} in 2024 - consider the real economic situation, infrastructure, social services, healthcare, education, safety, political stability, and quality of life that actually exist in this country

Be HONEST and REALISTIC. Base your score on the actual situation - both your personal life AND the real conditions in {country_name}. Many people worldwide score below 5. Low scores are valid and expected for people facing challenges.

{survey_text}

Respond with EXACTLY this format (single number in brackets):
[your score]

Example: [7]
Example for challenging situation: [3.5]
Example for good circumstances: [8.0]

Your response:"""
    
    return prompt


def call_gpt_api_gallup(prompt, max_retries=3, temperature=0.3):
    """
    Call GPT API for single question approach
    
    Args:
        prompt: User prompt
        max_retries: Maximum retry attempts
        temperature: Temperature setting (0.3 for realistic responses)
    """
    system_prompt = get_system_prompt_gallup()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_completion_tokens=50  # Single number, shorter response needed
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


def batch_survey_calls_gallup(country, personas, survey_text, batch_size=10):
    """
    Process single-question surveys for a country one persona at a time
    Returns list of responses with persona IDs
    
    Args:
        country: Country name
        personas: List of persona dictionaries
        survey_text: Survey question (single Cantril Ladder question)
        batch_size: Ignored (kept for compatibility, but processing is one-by-one)
    """
    all_responses = []
    
    # Process one persona at a time for better persona embodiment
    print(f"Processing {len(personas)} personas for {country} (single question, one at a time)...")
    
    for idx, persona in enumerate(personas, 1):
        print(f"  Processing persona {idx}/{len(personas)} (ID: {persona['id']})...")
        
        prompt = create_survey_prompt_gallup(
            persona["description"],
            country,
            survey_text
        )
        
        response = call_gpt_api_gallup(prompt, temperature=0.3)
        
        if response:
            # Validate that response has a single number
            list_match = re.search(r'\[([\d\.]+)\]', response)
            if list_match:
                try:
                    score = float(list_match.group(1))
                    if 0 <= score <= 10:
                        all_responses.append({
                            "persona_id": persona["id"],
                            "country": country,
                            "response_text": response
                        })
                        print(f"    ✓ Persona {persona['id']} completed successfully")
                    else:
                        print(f"    ✗ Warning: Persona {persona['id']} provided score {score} outside 0-10 range. Response: {response}")
                except:
                    print(f"    ✗ Warning: Could not parse response for persona {persona['id']}. Response: {response}")
            else:
                print(f"    ✗ Warning: No valid number format found for persona {persona['id']}. Response: {response}")
        else:
            print(f"    ✗ Failed to get response for persona {persona['id']}")
        
        # Small delay between personas to avoid rate limits
        if idx < len(personas):
            time.sleep(0.1)
    
    return all_responses
