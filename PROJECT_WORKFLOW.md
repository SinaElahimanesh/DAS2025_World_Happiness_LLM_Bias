# Project Workflow: World Happiness Analysis & LLM Audit

**Saarland University - Data and Society Seminar (Winter 2025)**

This document outlines the complete workflow and methodology of our project, from initial data analysis to the final improved LLM audit system.

---

## 1. Initial Data Analysis

### Data Source
- **Dataset**: World Happiness Report data (2013-2023) from `data.xlsx`
- **Coverage**: 147 countries across 11 years
- **Metrics**: Overall happiness score + 6 factors (GDP, Social Support, Health, Freedom, Generosity, Corruption)

### Analysis Methods Implemented

**Driver Analysis** (`driver_analysis.py`):
- **Method**: Weighted linear regression with standardized features
- **Technique**: StandardScaler normalization + LinearRegression from sklearn
- **Output**: Ranked factor importance by absolute coefficients
- **Regional Analysis**: Normalized coefficients by region for fair comparison

**Trend Analysis** (`trend_analysis.py`):
- **Global Trends**: Average happiness over time (2013-2023)
- **Regional Trends**: Happiness evolution by geographic region
- **Income-Level Trends**: Trends by economic classification
- **Volatility Metrics**: Stability analysis using standard deviation
- **Top Performers**: Biggest improvements and declines

**Group Comparisons** (`group_comparison.py`):
- **Statistical Testing**: T-tests and ANOVA for group differences
- **Gap Analysis**: Happiness gaps between regions and income levels
- **Factor Differences**: Comparative analysis of each factor by group
- **Significance Testing**: P-values for all comparisons

**Key Findings from Analysis**:
- GDP and Social Support are top drivers globally
- Regional differences in factor importance (e.g., Social Support more important in some regions)
- Global happiness relatively stable over decade (small changes)
- Significant gaps between high-income and low-income countries

---

## 2. Interactive Dashboard Creation

### Technology Stack
- **Framework**: Dash (Plotly)
- **Visualization**: Plotly Express and Graph Objects
- **Styling**: Custom CSS with modern gradient design

### Dashboard Components

**6 Main Tabs**:
1. **World Map**: Interactive choropleth map with year selector, country click details
2. **Driver Analysis**: Factor importance rankings, regional comparisons
3. **Trends**: Temporal visualizations, volatility analysis
4. **Group Comparisons**: Statistical comparisons with significance tests
5. **LLM Audit**: Integration with LLM audit results (real-time data loading)
6. **Overview**: Summary statistics and key metrics

**Methods**:
- Incremental data loading with caching
- Real-time LLM audit data integration
- Interactive callbacks for dynamic updates
- Responsive design with modern UI/UX

---

## 3. Initial LLM Audit Implementation

### First Approach: Initial 7-Question Method (`initial_approach/`)
- **Goal**: Compare LLM perceptions of happiness with real World Happiness Report data
- **Method**: Survey 20 diverse personas across all countries using GPT API

### Components Created

**Personas** (`personas.py`):
- 20 diverse personas: different ages (18-67), occupations, life situations
- Each persona has detailed description with {nationality} placeholder
- Covers: students, professionals, workers, retirees, unemployed, etc.
- **Shared across all three approaches**

**Survey** (`initial_approach/survey.py`):
- Based on World Happiness Report methodology (Cantril Ladder scale)
- 7 questions: Overall Happiness + 6 factors (GDP, Social Support, Health, Freedom, Generosity, Corruption)
- 0-10 scale for all metrics
- Strict output format: [Overall, GDP, Social, Health, Freedom, Generosity, Corruption]

**API Client** (`initial_approach/api_client.py`):
- OpenAI GPT API integration
- Batch processing (one persona at a time for quality)
- Retry logic with exponential backoff
- Response parsing and validation
- **Configuration**: Temperature: 0.0, Simplified prompt style (from prompt tuning)

**Pipeline** (`initial_approach/run_audit.py`):
- Processes all countries × all personas (147 countries × 20 personas = 2,940 combinations)
- Incremental saving after each country
- Resume capability (skips already processed items)
- Statistics generation

**Initial Results**:
- Generated LLM responses for all country-persona combinations
- Saved to `results/llm_audit_results.csv`
- Ready for comparison with real data

---

## 4. Hyperparameter Tuning

### Objective
Find optimal API hyperparameters to maximize correlation with real data.

### Method (`hyperparameter_tuning.py`)

**Hyperparameters Tested**:
- **Temperature**: [0.0, 0.5, 0.9] (determinism vs. creativity)
- **Chain of Thought**: [True, False] (explicit reasoning steps)

**Evaluation Method**:
1. Random sample of 10 countries
2. Run all personas for each country with each hyperparameter combination
3. Calculate country-level averages
4. Compare with real data using:
   - **Correlation** (Pearson correlation coefficient)
   - **Mean Absolute Error (MAE)**
   - **Combined Score**: Correlation - (MAE / 10) to balance both metrics

**Results**:
- **Best Temperature**: 0.0 (completely deterministic)
- **Best Chain of Thought**: False (direct responses better)
- **Best Performance**: Mean Correlation: 0.528, Combined Score: 0.075

**Conclusion**: Lower temperature (0.0) provides more consistent, less optimistic responses that better match real data.

---

## 5. Prompt Tuning

### Objective
Find optimal prompt structure using best hyperparameters (temp=0.0, CoT=False).

### Method (`prompt_tuning.py`)

**Prompt Variations Tested**:
1. **Current/Baseline**: Original detailed prompt with full persona embodiment
2. **Simplified**: Shorter, more concise version
3. **Concise Persona**: Reduced persona instructions
4. **JSON Format**: Output in JSON instead of brackets
5. **Few-shot**: Includes example responses
6. **Explicit Context**: More explicit country context
7. **Step-by-step**: Structured reasoning approach

**Evaluation Method**:
- Same as hyperparameter tuning: 10 random countries, compare with real data
- Metrics: Correlation, MAE, Combined Score

**Results**:
- **Best Prompt**: "Simplified" version
- **Performance**: Mean Correlation: 0.528, Combined Score: 0.075
- **Key Finding**: Shorter, more direct prompts work better than verbose instructions

**Final Configuration**:
- Temperature: 0.0
- Chain of Thought: False
- Prompt Style: Simplified

---

## 6. Overestimation Issues Identified

### Analysis Process

**Comparison Script** (`analyze_llm_vs_real.py`):
- Calculates country-level averages from LLM responses
- Compares with real World Happiness Report data (latest year)
- Computes differences, correlations, RMSE for all 7 metrics

**Bias Analysis** (`analyze_bias.py`):
- Systematic bias analysis across multiple groupings:
  - Global North vs Global South
  - World 1/2/3 classification (development levels)
  - Continents/Regions
  - Income levels
  - East vs West
- Statistical significance testing (t-tests, p-values)
- Effect sizes and bias direction

### Key Findings

**Systematic Overestimation**:
- LLM systematically overestimates all happiness scores
- Mean bias: ~6-10 points across all metrics
- Examples: Afghanistan (real=1.364, LLM=5.26), Finland (real=7.8, LLM=7.5)

**Root Causes Identified**:

1. **Scale Mismatch** (CRITICAL):
   - Real data uses regression contributions (0.0-1.8 range)
   - LLM provides 0-10 scale scores (3.0-8.5 range)
   - Different scales cause massive differences

2. **LLM Optimism Bias**:
   - LLMs default to "moderate to good" responses (5-7 range)
   - Very few responses below 4.0
   - Trained on human text, tends to be optimistic

3. **Lack of Calibration**:
   - No reference points (what does "average" mean?)
   - No examples of typical scores
   - LLM doesn't know score distributions

4. **Persona vs Country Context Confusion**:
   - Personas are generally positive/relatable
   - LLM may over-weight personal circumstances
   - Less consideration of country-level challenges

5. **Factor Interpretation Issues**:
   - Some factors ambiguous (Social Support: binary vs. continuous?)
   - Generosity: frequency vs. binary interpretation
   - LLM interprets differently than real data

6. **No Negative Examples**:
   - Survey doesn't emphasize low scores are valid
   - LLM assumes it should give "reasonable" (moderate) scores

**Quantified Impact**:
- Scale mismatch: ~4-6 points (PRIMARY ISSUE)
- Optimism bias: ~1-2 points
- Lack of calibration: ~0.5-1.5 points
- Other factors: ~0.5-1 point each
- **Total overestimation**: ~6-10 points (matches observed)

**Statistical Significance**:
- 345 significant findings (p < 0.05)
- Strongest biases in Global South, World 3 countries
- All metrics show systematic overestimation

---

## 7. Few-Shot Approach LLM Audit Version

### Objective
Create improved version addressing all overestimation issues while maintaining unbiased approach.

### Implementation (`few_shot_approach/`)

**Key Improvements**:

1. **Few-Shot Examples from Real Data** (`survey_improved.py`):
   - **Method**: Automatically loads real data from `data.xlsx`
   - **Selection**: 5 countries selected by percentile:
     - Highest happiness (rank #1)
     - Medium-high (top 25% percentile)
     - Medium (median, 50% percentile)
     - Medium-low (bottom 25% percentile)
     - Lowest happiness (rank #last)
   - **Purpose**: Calibration examples showing realistic score ranges
   - **Presentation**: Shown as examples, not targets (explicitly stated)

2. **Percentile Anchoring**:
   - Global score distribution context (9-10 = top 10%, 5-6 = middle 50%, etc.)
   - Helps LLM understand what scores mean globally
   - No country-specific biasing

3. **Explicit Low Score Validation**:
   - Clear statements that scores 1-4 are valid and expected
   - Examples of low scores (conflict zones, poverty)
   - Multiple reminders throughout survey

4. **Improved Factor Definitions**:
   - Detailed scale guides for each factor (0-2, 3-4, 5-6, 7-8, 9-10)
   - Social Support: Clarified as continuous scale, not binary
   - Generosity: Clarified as frequency AND amount, not binary

5. **Realistic System Prompt** (`api_client_improved.py`):
   - Emphasizes honesty and realism, not optimism
   - Validates low scores as normal
   - Reminds global average is ~5.5
   - Maintains method actor approach for persona embodiment

6. **Country Context (Unbiased)**:
   - Country name included in prompts for natural context consideration
   - LLM considers actual conditions (economic, infrastructure, services, safety)
   - **No income-level biasing**: Does NOT tell LLM what scores to expect
   - **No expected score ranges**: Avoids biasing based on country type

7. **Temperature Adjustment**:
   - Changed from 0.0 to 0.3
   - Slightly higher to reduce default optimism while maintaining consistency

### Technical Implementation

**Files Created**:
- `few_shot_approach/survey_improved.py`: Improved survey with few-shot examples and calibration
- `few_shot_approach/api_client_improved.py`: Realistic system prompts, unbiased country context
- `few_shot_approach/run_audit_improved.py`: Pipeline with identical output format to original
- `few_shot_approach/compare_approaches.py`: Comparison tool for original vs. improved

**Output Format**:
- **Identical to original**: Same CSV structure, same column names
- **Same file names**: `results/llm_audit_results.csv` (for compatibility)
- **Same statistics format**: Country and persona statistics match original

**Methods**:
- Few-shot examples loaded dynamically from real data
- Percentile-based country selection for diverse examples
- Unbiased prompt structure (country context without income-level labels)
- Realistic system prompt with explicit low score validation

### Expected Improvements

Based on analysis:
- **Mean Score Reduction**: From ~6-7 to ~5-6 (reduction of 1-2 points)
- **More Low Scores**: 20-30% of responses in 1-4 range (vs. <5% before)
- **Better Realism**: Scores reflect actual country conditions naturally
- **Reduced Overestimation**: Bias should decrease from ~6-10 points to ~2-4 points

---

## 8. Single Question Gallup Approach

### Objective
Implement the pure Gallup World Poll methodology using only the Cantril Ladder question, without factor sub-questions.

### Implementation (`single_question_gallup_approach/`)

**Key Characteristics**:
- **Single Question Only**: Just the Cantril Ladder life evaluation question
- **No Factor Decomposition**: No sub-questions for GDP, Social Support, Health, Freedom, Generosity, or Corruption
- **Pure Methodology**: Matches the original Gallup World Poll approach exactly

**Survey** (`single_question_gallup_approach/survey_gallup.py`):
- Only question: "Please imagine a ladder with steps numbered from 0 at the bottom to 10 at the top. The top represents the best possible life for you and the bottom the worst possible life. On which step do you feel you personally stand at this time?"
- Country context included naturally (unbiased)
- Realistic prompting with explicit low score validation
- Output format: Single number [score]

**API Client** (`single_question_gallup_approach/api_client_gallup.py`):
- Realistic system prompt (same as few-shot approach)
- Temperature: 0.3 (for realistic responses)
- Country context consideration without biasing
- Simplified output parsing (single number)

**Pipeline** (`single_question_gallup_approach/run_audit_gallup.py`):
- Same structure as other approaches
- Processes all countries × all personas
- Incremental saving and resume capability
- Output format: Factor columns set to None (not collected)

**Purpose**:
- Simplest approach - pure life evaluation
- Tests if factor decomposition affects LLM responses
- Comparison baseline for understanding factor influence
- Matches original Gallup methodology exactly

**Output Format**:
- Same CSV structure as other approaches for compatibility
- `overall_happiness` collected (single score)
- Factor columns (gdp, social_support, etc.) set to None
- All three happiness fields (explicit, calculated, overall) use same value

---

## 9. Analysis & Comparison Tools

### Comparison Scripts

**`analyze_llm_vs_real.py`**:
- Compares LLM country averages with real data
- Calculates correlations, differences, RMSE for all metrics
- Generates detailed comparison CSV and statistics report

**`analyze_bias.py`**:
- Systematic bias analysis across multiple groupings
- Statistical significance testing (t-tests, p-values)
- Identifies significant findings (p < 0.05)
- Generates bias summary and detailed reports

**`compare_approaches.py`** (few_shot_approach):
- Compares initial vs. few-shot approach results
- Shows mean differences, distributions, country-level comparisons
- Saves comparison to CSV
- Can be extended to compare all three approaches

### Statistical Methods Used

- **Correlation Analysis**: Pearson correlation coefficients
- **T-tests**: One-sample and independent samples t-tests
- **ANOVA**: Analysis of variance for group comparisons
- **Effect Sizes**: Mean differences and standardized effect sizes
- **Significance Testing**: P-values with multiple comparison corrections

---

## Summary

### Workflow Timeline

1. **Data Analysis** → Weighted regression, trend analysis, group comparisons
2. **Dashboard Creation** → Interactive web interface with 6 tabs
3. **Initial LLM Audit** → 20 personas × 147 countries = 2,940 responses (7 questions)
4. **Hyperparameter Tuning** → Found optimal: temp=0.0, CoT=False
5. **Prompt Tuning** → Found optimal: "Simplified" prompt style
6. **Overestimation Analysis** → Identified 6 root causes, 345 significant findings
7. **Few-Shot Approach** → Few-shot examples, calibration, realistic prompts, unbiased approach (7 questions)
8. **Single Question Approach** → Pure Gallup methodology, single question only (1 question)

### Key Methods

- **Statistical**: Weighted linear regression, t-tests, ANOVA, correlation analysis
- **LLM Optimization**: Hyperparameter grid search, prompt A/B testing
- **Bias Analysis**: Systematic grouping analysis with significance testing
- **Calibration**: Few-shot examples from real data, percentile anchoring
- **Unbiased Design**: Country context without income-level biasing

### Output

- **Dashboard**: Interactive web application with comprehensive visualizations
- **LLM Audit Results**: Three approaches (initial, few-shot, single question) with compatible formats
- **Analysis Reports**: Bias summaries, significant findings, comparison statistics
- **Documentation**: Complete workflow and methodology documentation

### Three LLM Audit Approaches Summary

| Approach | Questions | Calibration | Temperature | Purpose |
|----------|-----------|-------------|-------------|---------|
| **Initial** | 7 (1 overall + 6 factors) | None | 0.0 | Baseline implementation |
| **Few-Shot** | 7 (1 overall + 6 factors) | Few-shot examples + percentile anchoring | 0.3 | Reduced overestimation |
| **Single Question** | 1 (overall only) | None | 0.3 | Pure Gallup methodology |

---

