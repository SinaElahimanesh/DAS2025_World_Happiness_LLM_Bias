# Few-Shot Approach

This directory contains the few-shot approach for LLM happiness audit that addresses systematic overestimation issues where LLMs tend to give unrealistically high happiness scores (typically 6-7 range) compared to real-world data.

## Overview

The original LLM audit approach showed significant overestimation of happiness scores, with LLMs systematically giving scores 6-10 points higher than real data. This few-shot approach implements critical optimizations to reduce this bias **without introducing income-level or country-type biases**.

**Root Causes Addressed:**
1. **LLM Optimism Bias**: LLMs default to "moderate to good" responses (5-7 range), avoiding low scores
2. **Lack of Calibration**: No reference points provided for what scores mean globally
3. **Persona vs Country Context Confusion**: LLM may over-weight positive personal circumstances
4. **Factor Interpretation Issues**: Ambiguous scales for some factors (e.g., Social Support, Generosity)
5. **No Negative Examples**: Survey doesn't emphasize that low scores are valid and expected

**Important**: This approach considers country context (actual conditions in each country) but does NOT bias responses based on income level classifications or expected score ranges.

This is a parallel implementation that does NOT change the original flows - both approaches can run independently and produce identical output formats.

## Key Improvements

### 1. **Calibration and Anchoring** (HIGH PRIORITY)
- ✅ **Few-Shot Examples from Real Data**: Uses actual World Happiness Report data from 5 diverse countries selected by percentile:
  - Highest happiness country (rank #1)
  - Medium-high (top 25% percentile)
  - Medium (median, 50% percentile)
  - Medium-low (bottom 25% percentile)
  - Lowest happiness country (rank #last)
- ✅ **Percentile Anchoring**: Clear explanation of what scores mean globally (9-10 = top 10%, 5-6 = middle 50%, etc.)
- ✅ **Global Score Distribution Context**: Helps LLM understand the full range of possible scores

### 2. **Reduced Optimism Bias** (MEDIUM PRIORITY)
- ✅ **Explicitly Allow Low Scores**: Clear statement that scores 1-4 are valid and expected
- ✅ **Realistic System Prompt**: System prompt emphasizes honesty and realism, not optimism
- ✅ **Temperature Adjustment**: Uses 0.3 instead of 0.0 to reduce default optimism while maintaining consistency

### 3. **Improved Factor Definitions** (MEDIUM PRIORITY)
- ✅ **Clarified Scale Interpretations**: Each factor now has detailed scale guide (0-2, 3-4, 5-6, 7-8, 9-10)
- ✅ **Social Support**: Clarified as continuous scale, not binary
- ✅ **Generosity**: Clarified as frequency AND amount, not just binary

### 4. **Country Context Awareness (Unbiased)**
- ✅ **Country Name in Prompts**: LLM considers actual conditions in the country (economic, infrastructure, services, safety)
- ✅ **No Income-Level Biasing**: Does NOT tell LLM what scores to expect based on income level
- ✅ **Natural Context Consideration**: LLM naturally assesses country conditions without explicit biasing

## Files

### 1. `survey_improved.py`
**Purpose**: Improved survey with all optimizations

**Key Features**:
- ✅ Few-shot examples from real World Happiness Report data (5 countries selected by percentile: highest, top 25%, median, bottom 25%, lowest)
- ✅ Percentile anchoring (what scores mean globally)
- ✅ Explicit validation of low scores (1-4 are valid and expected)
- ✅ Improved factor definitions (detailed scale guides for each factor)
- ✅ Country context (country name included, but no income-level biasing)

**Functions**:
- `get_few_shot_examples_from_real_data()`: Loads real data and selects diverse country examples
- `get_survey_improved(country_name, income_level)`: Get optimized survey text (income_level not used for biasing)
- `parse_survey_response(response_text)`: Parse LLM responses (same as original)

### 2. `api_client_improved.py`
**Purpose**: Improved API client with realistic system prompts

**Key Features**:
- ✅ Realistic system prompt that encourages honest scores
- ✅ Country context awareness in prompts (actual conditions, not income labels)
- ✅ Optimized temperature (0.3 instead of 0.0) to reduce default optimism
- ✅ No income-level biasing in prompts

**Functions**:
- `get_realistic_system_prompt()`: Returns improved system prompt
- `create_survey_prompt_improved(...)`: Creates prompt with country context (no income-level biasing)
- `call_gpt_api_improved(...)`: Calls API with improved settings
- `batch_survey_calls_improved(...)`: Processes surveys for a country

### 3. `run_audit_improved.py`
**Purpose**: Main pipeline to run the improved audit

**Key Features**:
- ✅ Incremental saving (results saved after each country)
- ✅ Resume capability (skips already processed items)
- ✅ Progress tracking with detailed statistics
- ✅ Statistics generation after completion
- ✅ **Output format identical to original** (same CSV structure, same file names)

### 4. `compare_approaches.py`
**Purpose**: Compare original vs improved results

**Features**:
- Compares mean scores for all metrics
- Shows score distribution differences
- Country-level comparison
- Saves comparison to CSV

## Usage

### Prerequisites

1. **API Key**: Make sure the OpenAI API key is set in `api_client_improved.py` (line 19)
2. **Dependencies**: Ensure all required packages are installed:
   ```bash
   pip install openai pandas numpy
   ```
3. **Data File**: Ensure `data.xlsx` exists in the project root directory

### Run the Improved Audit

**Step 1: Navigate to the few_shot_approach directory**
```bash
cd llm_audit_data/few_shot_approach
```

**Step 2: Run the pipeline**
```bash
python run_audit_improved.py
```

**Note**: This approach uses few-shot examples from real data to calibrate the LLM's understanding of score ranges.

**What the script does:**
1. Loads all countries from the main dataset (`data.xlsx`)
2. Processes each country with all 20 personas (one at a time)
3. Saves results incrementally to `results/llm_audit_results.csv` after each country
4. Shows progress with country-level statistics
5. Generates final statistics after completion

**Expected Output:**
```
============================================================
LLM Happiness Audit - Starting
============================================================

Checking for existing results...
Found 0 already processed country-persona combinations

Loading countries...
Found 147 countries

Loading personas...
Found 20 personas

Total combinations: 2940
Already processed: 0
Pending: 2940

============================================================
Processing country 1/147: Afghanistan
  Personas for this country: 20/20
============================================================
Processing 20 personas for Afghanistan (one at a time)...
  Processing persona 1/20 (ID: 1)...
    ✓ Persona 1 completed successfully
  ...
  Saved 20 responses to CSV. Total processed: 20/2940
...
```

**Processing Time:**
- Each country takes approximately 1-2 minutes (20 personas × 3-6 seconds per persona)
- Full run for 147 countries: ~3-5 hours (depending on API rate limits)
- Results are saved incrementally, so you can stop and resume anytime

### Resume Interrupted Runs

The script automatically resumes from where it left off. If interrupted:
1. Simply run the script again: `python run_audit_improved.py`
2. It will detect already processed country-persona combinations
3. It will skip them and continue with pending items

**Example resume output:**
```
Checking for existing results...
Found 840 already processed country-persona combinations

Total combinations: 2940
Already processed: 840
Pending: 2100
```

### Check Which Countries Are Used in Few-Shot Examples

The few-shot examples are selected dynamically from your `data.xlsx`. The selected countries are **automatically printed** when the survey is generated. You'll see output like:

```
Few-shot examples selected from real data:
  Highest (rank #1): Finland (happiness: 7.80)
  Medium-high (top 25%): [Country] (happiness: X.XX)
  Medium (median): [Country] (happiness: X.XX)
  Medium-low (bottom 25%): [Country] (happiness: X.XX)
  Lowest (rank #last): Afghanistan (happiness: 2.40)
```

You can also check programmatically:

```python
from survey_improved import get_few_shot_examples_from_real_data

examples = get_few_shot_examples_from_real_data()  # verbose=True by default
```

This will print:
```
Few-shot examples selected from real data:
  Highest (rank #1): Finland (happiness: 7.80)
  Medium-high (top 25%): [Country] (happiness: X.XX)
  Medium (median): [Country] (happiness: X.XX)
  Medium-low (bottom 25%): [Country] (happiness: X.XX)
  Lowest (rank #last): Afghanistan (happiness: 2.40)
```

The exact countries depend on your dataset and will be:
- **Highest**: The country with the highest happiness_score in your data
- **Medium-high**: Country at the 25th percentile (approximately 1/4 down the sorted list)
- **Medium**: Country at the median (middle of the sorted list)
- **Medium-low**: Country at the 75th percentile (approximately 3/4 down the sorted list)
- **Lowest**: The country with the lowest happiness_score in your data

### Compare with Original Approach

```bash
python compare_approaches.py
```

This will compare:
- Mean scores between approaches
- Distribution of scores
- Overestimation reduction
- Correlation with real data

## Implementation Details

### Optimizations Implemented

#### Phase 1: Critical Fixes ✅
1. **Few-Shot Examples from Real Data**: Loads actual World Happiness Report data and shows diverse country examples
2. **Percentile Anchoring**: Global score distribution context (9-10 = top 10%, etc.)
3. **Global Calibration**: Helps LLM understand realistic score ranges without country-specific biasing

#### Phase 2: Bias Reduction ✅
4. **Explicitly Allow Low Scores**: Clear statements that 1-4 are valid
5. **Realistic System Prompt**: Emphasizes honesty over optimism
6. **Temperature Adjustment**: 0.3 instead of 0.0 to reduce default optimism

#### Phase 3: Refinement ✅
7. **Clarified Factor Definitions**: Detailed scale guides for each factor
8. **Country Context (Unbiased)**: Country name included for natural context consideration, but no income-level biasing
9. **Improved Examples**: Real data examples show full range of possible scores

### Technical Details

#### Temperature Setting
- Original: `temperature=0.0` (completely deterministic)
- Improved: `temperature=0.3` (slightly higher to reduce default optimism while maintaining consistency)

#### System Prompt
The improved system prompt explicitly:
- Encourages honest, realistic scores
- Validates low scores as normal
- Emphasizes reporting reality, not being "helpful"
- Reminds that global average is ~5.5
- Maintains method actor approach for persona embodiment

#### Survey Structure
The improved survey includes:
- Global score distribution context (percentile anchoring)
- Few-shot examples from real World Happiness Report data (diverse countries)
- Detailed scale guides for each factor
- Multiple reminders that low scores are valid
- Country name for context (but no income-level or expected score biasing)

#### Few-Shot Examples
- Automatically loads real data from `data.xlsx` (latest year available)
- Selects 5 diverse countries based on happiness score percentiles:
  1. **Highest happiness country** (rank #1, top of the list when sorted by happiness_score descending)
  2. **Medium-high happiness country** (top 25% percentile, at index = total_countries ÷ 4)
  3. **Medium happiness country** (median, at index = total_countries ÷ 2)
  4. **Medium-low happiness country** (bottom 25% percentile, at index = total_countries × 3 ÷ 4)
  5. **Lowest happiness country** (rank #last, bottom of the list when sorted by happiness_score descending)
- Shows actual scores from World Happiness Report for these countries
- **Countries are selected dynamically** based on the data in `data.xlsx`, so exact countries depend on your dataset
- To see which countries are used, call `get_few_shot_examples_from_real_data(verbose=True)` or check the survey prompt during execution
- Presented as calibration examples, not targets
- Helps LLM understand realistic score ranges across the full spectrum

## Expected Results

Based on analysis of the overestimation issues, we expect:

1. **Mean Score Reduction**: From ~6-7 to ~5-6 (reduction of 1-2 points)
2. **More Low Scores**: Should see 20-30% of responses in 1-4 range (vs <5% before)
3. **Better Realism**: Scores should reflect actual country conditions naturally
4. **Reduced Overestimation**: Bias should decrease from ~6-10 points to ~2-4 points

## Key Differences from Original

| Aspect | Original | Improved |
|--------|----------|----------|
| **Calibration** | None | Few-shot examples from real data + percentile anchoring |
| **Country Context** | Basic (country name only) | Country name + natural context consideration (no income-level biasing) |
| **Low Score Validation** | Implicit | Explicit (1-4 are valid) |
| **Examples** | None | Real data examples from diverse countries |
| **Factor Definitions** | Basic | Detailed scale guides |
| **System Prompt** | Method actor only | Realistic + method actor |
| **Temperature** | 0.0 | 0.3 |
| **Optimism Bias** | High | Reduced |
| **Output Format** | Standard | **Identical to original** |

## Testing Recommendations

After running the improved audit:

1. **Compare Means**: Check if average scores decreased by 1-2 points
2. **Check Distribution**: Verify more scores in 1-4 range
3. **Country Realism**: Ensure scores reflect actual country conditions naturally
4. **Correlation**: Compare correlation with real data (should improve)
5. **Bias Analysis**: Run bias analysis to measure overestimation reduction

## Results Location

Results are saved in:
- `results/llm_audit_results.csv` - Main results (same format as original)
- `results/statistics_*.csv` - Country statistics
- `results/persona_stats_*.csv` - Persona statistics

**Note**: Output format is identical to the original approach for compatibility.

## Files Structure

```
few_shot_approach/
├── survey_improved.py          # Improved survey with optimizations
├── api_client_improved.py      # Improved API client
├── run_audit_improved.py       # Main pipeline script
├── compare_approaches.py        # Comparison tool
├── README.md                    # This documentation
└── results/                    # Output directory (created when run)
    ├── llm_audit_results.csv   # Main results (same format as original)
    ├── statistics_*.csv         # Country statistics
    └── persona_stats_*.csv     # Persona statistics
```

## Dependencies

Same as original approach:
- `openai` (for API calls)
- `pandas` (for data processing)
- `numpy` (for calculations)
- Parent directory modules: `personas.py`, `country_to_nationality.py`, `analyzer.py`
- Main project modules: `data_loader.py`

All dependencies are already available in the project.

## Next Steps

1. Run `run_audit_improved.py` to generate new data
2. Run `compare_approaches.py` to see differences
3. Run bias analysis scripts to measure improvement
4. Compare with real data to verify reduced overestimation

## Notes

- ✅ Original flows are NOT changed - this is a parallel implementation
- ✅ Both approaches can run independently
- ✅ **Output format is identical to original** (same CSV structure, same file names)
- ✅ Results saved to `results/` directory (same as original, but can be in separate location if needed)
- ✅ All critical optimizations to address overestimation are implemented:
  - Few-shot examples from real data (diverse countries)
  - Percentile anchoring (global distribution context)
  - Reduced optimism bias (explicit low score validation, realistic system prompts)
  - Improved factor definitions (detailed scale guides for all factors)
  - Country context awareness (natural consideration, no income-level biasing)
