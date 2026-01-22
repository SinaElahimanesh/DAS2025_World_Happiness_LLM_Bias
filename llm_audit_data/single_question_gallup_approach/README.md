# Single Question Gallup Approach

This approach uses ONLY the single Cantril Ladder question from the Gallup World Poll, with no sub-questions for factors.

## Overview

- **Survey**: Single question only
  - Cantril Ladder: "On which step do you feel you personally stand at this time?" (0-10)
  - No factor sub-questions
- **Methodology**: Pure Gallup World Poll approach
- **Temperature**: 0.3 (for realistic responses)
- **System Prompt**: Realistic, encourages honest scores including low scores

## Key Features

- **Single Question**: Only the life evaluation question, no factors
- **Country Context**: Considers actual conditions in country (unbiased)
- **Realistic Prompting**: Explicitly allows low scores (1-4)
- **Simplified Output**: Single number response [score]

## Files

- `run_audit_gallup.py`: Main pipeline script
- `survey_gallup.py`: Single question survey
- `api_client_gallup.py`: API client for single question

## Usage

```bash
cd single_question_gallup_approach
python run_audit_gallup.py
```

## Output

Results saved to `results/llm_audit_results.csv` with columns:
- persona_id, country, life_evaluation (the single score)
- gdp, social_support, health, freedom, generosity, corruption (all set to None)
- overall_happiness_explicit, overall_happiness_calculated, overall_happiness (all same value)
- response_text

**Note**: Factor columns (gdp, social_support, health, freedom, generosity, corruption) are set to None since they're not collected in this approach. Only `overall_happiness` is collected.
