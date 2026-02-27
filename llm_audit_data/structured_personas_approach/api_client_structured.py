"""
API Client for Structured Personas Approach (Single Question Gallup)

Same single Cantril Ladder question as the single_question_gallup_approach,
but personas are structured fields (nationality, job, gender) instead of long text.
Nationality is the main focus in the prompt.

Usage:
    from api_client_structured import batch_survey_calls_structured
    responses = batch_survey_calls_structured(country="Finland", personas=personas, survey_text=survey)
"""

import openai
import time
import re
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from country_to_nationality import country_to_nationality
from config import get_api_key

client = openai.OpenAI(api_key=get_api_key())


def get_system_prompt_structured():
    """System prompt for single-question approach (same as Gallup)."""
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


def create_survey_prompt_structured(persona, country_name, survey_text):
    """
    Build the user prompt from all structured persona fields.
    Nationality is emphasized as the main identity focus.
    """
    nationality = country_to_nationality(country_name)
    nat_display = persona["nationality"].format(nationality=nationality)

    # Build full profile block; nationality first and highlighted
    lines = [
        f"- Nationality: {nat_display}  [main identity focus]",
        f"- Gender: {persona['gender']}",
        f"- Age: {persona['age']}",
        f"- Job: {persona['job']}",
        f"- Work / career: {persona['work_details']}",
        f"- Living situation: {persona['living_situation']}",
        f"- Family: {persona['family']}",
        f"- Hobbies / interests: {persona['hobbies']}",
        f"- Values / notes: {persona['values_or_notes']}",
    ]
    profile_block = "\n".join(lines)

    prompt = f"""You are this person living in {country_name} in 2024.

PERSON PROFILE (structured; nationality is the main focus):

{profile_block}

Answer this happiness survey from their perspective. Consider BOTH:
1. Their personal circumstances (especially their nationality and how it shapes their experience in {country_name}, plus their age, job, family, living situation, and everything above)
2. The actual conditions in {country_name} in 2024 - economic situation, infrastructure, social services, healthcare, education, safety, political stability, and quality of life

Be HONEST and REALISTIC. Base your score on the actual situation - both your personal life AND the real conditions in {country_name}. Many people worldwide score below 5. Low scores are valid and expected for people facing challenges.

{survey_text}

Respond with EXACTLY this format (single number in brackets):
[your score]

Example: [7]
Example for challenging situation: [3.5]
Example for good circumstances: [8.0]

Your response:"""

    return prompt


def call_gpt_api_structured(prompt, max_retries=3, temperature=0.3):
    """Call GPT for single-question response."""
    system_prompt = get_system_prompt_structured()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_completion_tokens=50
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            wait_time = (2 ** attempt) * 2
            print(f"Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2)
    return None


def batch_survey_calls_structured(country, personas, survey_text, batch_size=10):
    """
    Process single-question surveys for a country, one persona at a time.
    Personas must be structured dicts with id, nationality, gender, age, job, work_details, living_situation, family, hobbies, values_or_notes.
    """
    all_responses = []
    print(f"Processing {len(personas)} structured personas for {country} (single question, one at a time)...")

    for idx, persona in enumerate(personas, 1):
        print(f"  Processing persona {idx}/{len(personas)} (ID: {persona['id']})...")
        prompt = create_survey_prompt_structured(persona, country, survey_text)
        response = call_gpt_api_structured(prompt, temperature=0.3)

        if response:
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
                        print(f"    ✗ Warning: Persona {persona['id']} score {score} outside 0-10. Response: {response}")
                except Exception:
                    print(f"    ✗ Warning: Could not parse response for persona {persona['id']}. Response: {response}")
            else:
                print(f"    ✗ Warning: No valid number format for persona {persona['id']}. Response: {response}")
        else:
            print(f"    ✗ Failed to get response for persona {persona['id']}")

        if idx < len(personas):
            time.sleep(0.1)

    return all_responses
