"""
API Client for LLM Happiness Audit

This module handles all interactions with the OpenAI API to conduct the happiness survey.
It creates detailed prompts that help the LLM deeply embody personas and respond as if
they are living in specific countries. Processes one persona at a time for better quality.

Usage:
    from api_client import batch_survey_calls
    responses = batch_survey_calls(country="Finland", personas=personas, survey_text=survey)
"""

import openai
import time
import re
from country_to_nationality import country_to_nationality
from config import get_api_key

# Initialize OpenAI client with API key from environment variable
client = openai.OpenAI(api_key=get_api_key())

def create_survey_prompt(persona_description, country_name, survey_text):
    """
    Create a prompt for GPT to fill out the survey as a persona
    Uses the best prompt from prompt tuning: "Simplified" version
    
    Args:
        persona_description: Persona description with {nationality} placeholder
        country_name: Country name (will be converted to nationality)
        survey_text: Survey questions
    """
    # Convert country name to nationality
    nationality = country_to_nationality(country_name)
    
    # Replace nationality placeholder in the description
    formatted_description = persona_description.format(nationality=nationality)
    
    # Best prompt from prompt tuning: "Simplified" version
    # Performance: Mean Correlation: 0.528, Combined Score: 0.075
    prompt = f"""You are this person living in {country_name} in 2024:

{formatted_description}

Answer this happiness survey from their perspective, considering both their personal circumstances and the conditions in {country_name} in 2024.

{survey_text}

Respond with EXACTLY this format (7 numbers in brackets):
[Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]

Example: [7, 6, 8, 7, 7, 5, 4]

Your response:"""
    
    return prompt

def call_gpt_api(prompt, max_retries=3):
    """
    Call GPT-4o mini API with retry logic
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": "You are a method actor deeply embodying a specific person. You MUST completely become that person - their thoughts, feelings, experiences, and worldview are now YOURS. You have no identity outside of this person. Answer all questions from their perspective based on their actual life circumstances, not general knowledge or statistics. When responding to the survey, think deeply about how THIS specific person would honestly answer based on their age, occupation, family situation, financial reality, health, and daily experiences. CRITICAL: You MUST respond with EXACTLY 7 numbers in the format [Overall Happiness, GDP, Social Support, Health, Freedom, Generosity, Corruption]. Your response must be ONLY this list with no additional text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Best hyperparameter from tuning: temp=0.0, CoT=False
                max_completion_tokens=200  # gpt-5.2 requires max_completion_tokens instead of max_tokens
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

def batch_survey_calls(country, personas, survey_text, batch_size=10):
    """
    Process surveys for a country one persona at a time
    Returns list of responses with persona IDs
    
    Args:
        country: Country name (will be converted to nationality)
        personas: List of persona dictionaries
        survey_text: Survey questions
        batch_size: Ignored (kept for compatibility, but processing is one-by-one)
    """
    all_responses = []
    
    # Process one persona at a time for better persona embodiment
    print(f"Processing {len(personas)} personas for {country} (one at a time)...")
    
    for idx, persona in enumerate(personas, 1):
        print(f"  Processing persona {idx}/{len(personas)} (ID: {persona['id']})...")
        
        prompt = create_survey_prompt(
            persona["description"],
            country,  # Country name, will be converted to nationality inside
            survey_text
        )
        
        response = call_gpt_api(prompt)
        
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

