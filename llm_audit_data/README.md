# LLM Audit System

This directory contains three different approaches for conducting LLM happiness audits, each with different methodologies.

## Directory Structure

```
llm_audit_data/
├── initial_approach/              # Original 7-question approach
│   ├── run_audit.py
│   ├── survey.py
│   ├── api_client.py
│   └── README.md
│
├── few_shot_approach/              # Improved with few-shot examples
│   ├── run_audit_improved.py
│   ├── survey_improved.py
│   ├── api_client_improved.py
│   ├── compare_approaches.py
│   └── README.md
│
├── single_question_gallup_approach/ # Pure Gallup single question
│   ├── run_audit_gallup.py
│   ├── survey_gallup.py
│   ├── api_client_gallup.py
│   └── README.md
│
└── [Shared Files]                 # Used by all approaches
    ├── config.py                  # Configuration (API key management)
    ├── api_client.py              # API client (used by initial_approach & tuning scripts)
    ├── survey.py                  # Survey module (used by initial_approach & tuning scripts)
    ├── personas.py                # 20 diverse personas
    ├── country_to_nationality.py  # Country-nationality mapping
    ├── analyzer.py                # Statistical analysis utilities
    ├── analyze_llm_vs_real.py    # Compare LLM vs real data
    ├── analyze_bias.py            # Bias analysis by groups
    ├── hyperparameter_tuning.py  # Hyperparameter optimization
    ├── prompt_tuning.py           # Prompt optimization
    └── load_llm_results.py       # Load results for dashboard
```

## Three Approaches

### 1. Initial Approach (`initial_approach/`)

**Methodology**: 7 questions total
- 1 Overall Happiness (Cantril Ladder)
- 6 Factor questions (GDP, Social Support, Health, Freedom, Generosity, Corruption)

**Characteristics**:
- Original implementation
- Simplified prompt style (from prompt tuning)
- Temperature: 0.0
- No calibration or few-shot examples

### 2. Few-Shot Approach (`few_shot_approach/`)

**Methodology**: 7 questions + few-shot calibration
- Same 7 questions as initial
- Adds few-shot examples from real World Happiness Report data
- Percentile anchoring for score calibration
- Improved factor definitions

**Characteristics**:
- Few-shot examples from 5 diverse countries (selected by percentile)
- Percentile anchoring (global score distribution context)
- Explicit low score validation
- Realistic system prompts
- Temperature: 0.3
- Country context (unbiased - no income-level biasing)

### 3. Single Question Gallup Approach (`single_question_gallup_approach/`)

**Methodology**: Single question only
- Only the Cantril Ladder question: "On which step do you feel you personally stand?"
- No factor sub-questions
- Pure Gallup World Poll methodology

**Characteristics**:
- Simplest approach - just life evaluation
- No factor decomposition
- Country context consideration (unbiased)
- Realistic system prompts
- Temperature: 0.3

## Shared Components

All approaches use:
- **`personas.py`**: 20 diverse personas (ages 18-67, various occupations and life situations)
- **`country_to_nationality.py`**: Maps country names to nationalities for persona descriptions
- **`analyzer.py`**: Statistical analysis utilities (country/persona statistics)
- **`analyze_llm_vs_real.py`**: Compare LLM results with real World Happiness Report data
- **`analyze_bias.py`**: Systematic bias analysis across multiple groupings

## Setup

### Prerequisites

1. **Set OpenAI API Key**: All approaches require the `OPENAI_API_KEY` to be set.

   **Option 1: Use .env file (recommended)**
   ```bash
   # The .env file is already created in the project root
   # Just edit it and add your API key:
   OPENAI_API_KEY=your-api-key-here
   ```
   
   The config module will automatically load from `.env` if `python-dotenv` is installed.

   **Option 2: Export environment variable**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

2. **Install dependencies**:
   ```bash
   pip install -r ../requirements.txt
   # Or manually:
   pip install openai pandas numpy python-dotenv
   ```

3. **Data file**: Ensure `data.xlsx` exists in the project root directory

## Running Each Approach

### Initial Approach
```bash
cd initial_approach
python run_audit.py
```

### Few-Shot Approach
```bash
cd few_shot_approach
python run_audit_improved.py
```

### Single Question Gallup Approach
```bash
cd single_question_gallup_approach
python run_audit_gallup.py
```

## Output Format

All approaches produce results in the same CSV format:
- `results/llm_audit_results.csv` - Main results
- `results/statistics_*.csv` - Country statistics
- `results/persona_stats_*.csv` - Persona statistics

**Note**: Single question approach sets factor columns (gdp, social_support, etc.) to None since they're not collected.

## Comparison

| Aspect | Initial | Few-Shot | Single Question |
|--------|---------|----------|-----------------|
| **Questions** | 7 (1 overall + 6 factors) | 7 (1 overall + 6 factors) | 1 (overall only) |
| **Calibration** | None | Few-shot examples + percentile anchoring | None |
| **Temperature** | 0.0 | 0.3 | 0.3 |
| **Factor Data** | Collected | Collected | Not collected (None) |
| **Complexity** | Medium | High | Low |

## Analysis Tools

After running any approach, use:
- `analyze_llm_vs_real.py` - Compare with real data
- `analyze_bias.py` - Bias analysis by groups
- `few_shot_approach/compare_approaches.py` - Compare approaches (if you have results from multiple)
