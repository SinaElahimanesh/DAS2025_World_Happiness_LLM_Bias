# Initial Approach

This is the original LLM audit implementation with 7 questions (Overall Happiness + 6 factors).

## Overview

- **Survey**: 7 questions total
  - 1 Overall Happiness (Cantril Ladder)
  - 6 Factor questions (GDP, Social Support, Health, Freedom, Generosity, Corruption)
- **Prompt Style**: Simplified (best from prompt tuning)
- **Temperature**: 0.0 (best from hyperparameter tuning)
- **Chain of Thought**: False

## Files

- `run_audit.py`: Main pipeline script
- `survey.py`: Survey with 7 questions
- `api_client.py`: API client with simplified prompts

## Usage

```bash
cd initial_approach
python run_audit.py
```

## Output

Results saved to `results/llm_audit_results.csv` with columns:
- persona_id, country, life_evaluation, gdp, social_support, health, freedom, generosity, corruption
- overall_happiness_explicit, overall_happiness_calculated, overall_happiness
- response_text
