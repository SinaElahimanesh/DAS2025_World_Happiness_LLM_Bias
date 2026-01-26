# How Happy is the World? - World Happiness Analysis

**Saarland University - Data and Society Seminar (Winter 2025)**

**Team:** Sina Elahi Manesh, Yassal Arif, Akash Chavan

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [LLM Audit Pipelines](#llm-audit-pipelines)
- [Features](#features)
- [Data Files](#data-files)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This project provides a comprehensive analysis of the World Happiness Report data (2013-2023) through an interactive web dashboard and advanced LLM audit system. The project combines traditional statistical analysis with cutting-edge AI bias research.

### Key Components

1. **Interactive Dashboard** - Beautiful web interface for exploring happiness data
2. **Driver Analysis** - Weighted linear regression to identify key happiness factors
3. **Trend Analysis** - Temporal analysis of happiness changes over the decade
4. **Group Comparisons** - Regional and income-level comparisons with statistical testing
5. **LLM Audit System** - Comprehensive AI bias analysis comparing LLM perceptions with real data

---

## 🔑 API Key Setup

**Important**: This project requires an OpenAI API key to run LLM audit pipelines.

### Setting Up Your API Key

1. **Get your API key** from [OpenAI Platform](https://platform.openai.com/api-keys)

2. **Create .env file from .env.example**:
   ```bash
   # Copy the example file to create your .env file
   cp .env.example .env
   
   # Then edit .env and uncomment the line, then add your API key:
   # Change: # OPENAI_API_KEY=your-api-key-here
   # To:     OPENAI_API_KEY=sk-your-actual-key-here
   ```

3. **Alternative: Set environment variable**:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

4. **Install python-dotenv** (if not already installed):
   ```bash
   pip install python-dotenv
   ```

**Note**: 
- The `.env.example` file is provided as a template. Copy it to `.env` and add your actual API key.
- The API key is loaded from the `.env` file (if `python-dotenv` is installed) or from the `OPENAI_API_KEY` environment variable.
- The `.env` file is excluded from version control via `.gitignore` (never commit your actual API key!).

## 📁 Project Structure

```
Project/
├── app.py                      # Main Dash web application
├── data_loader.py              # Data loading and preprocessing
├── driver_analysis.py          # Factor importance analysis
├── trend_analysis.py           # Temporal trend analysis
├── group_comparison.py         # Regional/income comparisons
├── llm_audit.py                # LLM audit integration for dashboard
├── data.xlsx                   # World Happiness Report dataset
├── requirements.txt            # Python dependencies
├── run.sh                      # Quick start script
├── README.md                   # Project documentation
├── PROJECT_WORKFLOW.md         # Detailed workflow documentation
├── .env.example                # API key template (copy to .env)
├── .env                        # API key configuration (create from .env.example, not in git)
├── .gitignore                  # Git ignore rules
│
└── llm_audit_data/            # LLM Audit System
    ├── config.py              # Configuration (API key from .env)
    ├── personas.py            # 20 diverse persona definitions (shared)
    ├── country_to_nationality.py # Country-nationality mapping (shared)
    ├── analyzer.py            # Statistical analysis utilities (shared)
    ├── analyze_llm_vs_real.py # Compare LLM vs real data
    ├── analyze_bias.py        # Bias analysis by groups
    ├── hyperparameter_tuning.py # Find optimal API parameters
    ├── prompt_tuning.py       # Test different prompt strategies
    ├── load_llm_results.py   # Load results for dashboard
    ├── api_client.py          # Base API client (shared)
    ├── survey.py              # Base survey module (shared)
    ├── README.md              # LLM audit system documentation
    │
    ├── initial_approach/      # Original 7-question approach
    │   ├── run_audit.py       # Main execution script
    │   ├── survey.py          # Survey questions and parsing
    │   ├── api_client.py      # API client for this approach
    │   ├── README.md          # Approach documentation
    │   └── results/           # Generated results
    │       ├── llm_audit_results.csv
    │       ├── llm_vs_real_comparison.csv
    │       ├── bias_summary_*.csv
    │       ├── bias_analysis_data_*.csv
    │       ├── significant_findings_*.csv
    │       ├── statistics_*.csv
    │       ├── persona_stats_*.csv
    │       └── old_results/   # Archived older results
    │
    ├── few_shot_approach/     # Few-shot calibration approach
    │   ├── run_audit_improved.py # Main execution script
    │   ├── survey_improved.py    # Improved survey with few-shot examples
    │   ├── api_client_improved.py # Improved API client
    │   ├── compare_approaches.py # Compare different approaches
    │   ├── examples.txt          # Few-shot example data
    │   ├── README.md             # Approach documentation
    │   ├── results/              # Generated results
    │   │   ├── llm_audit_results.csv
    │   │   ├── statistics_*.csv
    │   │   ├── persona_stats_*.csv
    │   │   └── few_shots_countries.txt
    │   └── results_improved/     # Comparison results
    │       └── approach_comparison.csv
    │
    ├── single_question_gallup_approach/ # Pure Gallup single question
    │   ├── run_audit_gallup.py  # Main execution script
    │   ├── survey_gallup.py     # Single question survey
    │   ├── api_client_gallup.py # API client for Gallup approach
    │   ├── README.md            # Approach documentation
    │   └── results/             # Generated results
    │       ├── llm_audit_results.csv
    │       ├── statistics_*.csv
    │       └── persona_stats_*.csv
    │
    └── results/                # Shared results directory
        ├── llm_audit_results.csv
        ├── llm_vs_real_comparison.csv
        ├── bias_summary_*.csv
        ├── bias_analysis_data_*.csv
        ├── significant_findings_*.csv  # Auto-generated from bias_summary
        ├── llm_vs_real_statistics.txt
        └── bias_analysis_report_*.txt
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Navigate to project directory:**
   ```bash
   cd "/Users/sinaelahimanesh/Documents/Saarland/Semester 2/Data and Society/Project"
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   ```

3. **Activate virtual environment:**
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Running the Application

### Quick Start (Recommended)

```bash
./run.sh
```

### Manual Start

```bash
source venv/bin/activate
python app.py
```

Then open your browser to: **http://127.0.0.1:8050**

### Expected Output

```
Loading data...
✓ Real LLM audit data loader imported successfully
==================================================
Starting World Happiness Analysis Dashboard
==================================================
Data loaded: 1510 records
Years: 2013 - 2023
Countries: 166

Access the dashboard at: http://127.0.0.1:8050
==================================================
```

**To stop:** Press `Ctrl + C` in the terminal

---

## 🤖 LLM Audit Pipelines

The LLM audit system consists of **three different approaches**, each with its own methodology:

### Three Approaches Overview

1. **Initial Approach** (`initial_approach/`) - Original 7-question method
2. **Few-Shot Approach** (`few_shot_approach/`) - Improved with calibration examples
3. **Single Question Gallup Approach** (`single_question_gallup_approach/`) - Pure Gallup methodology

See `llm_audit_data/README.md` for detailed comparison and methodology.

### 1. Initial Approach (7 Questions)

**Purpose:** Baseline LLM audit with 7 questions (1 overall + 6 factors)

```bash
cd llm_audit_data/initial_approach
python run_audit.py
```

**What it does:**
- Processes all countries from the main dataset
- Uses 20 diverse personas per country
- 7 questions: Overall Happiness + 6 factors (GDP, Social Support, Health, Freedom, Generosity, Corruption)
- Temperature: 0.0, Simplified prompt style
- Saves progress incrementally after each country

**Output:**
- `results/llm_audit_results.csv` - All responses
- `results/statistics_*.csv` - Country-level statistics
- `results/persona_stats_*.csv` - Persona-level statistics

### 2. Few-Shot Approach (7 Questions + Calibration)

**Purpose:** Improved version with few-shot examples to reduce overestimation

```bash
cd llm_audit_data/few_shot_approach
python run_audit_improved.py
```

**What it does:**
- Same 7 questions as initial approach
- Adds few-shot examples from real World Happiness Report data (5 diverse countries)
- Percentile anchoring for score calibration
- Explicit low score validation
- Temperature: 0.3, Realistic system prompts
- Unbiased country context (no income-level biasing)

**Output:**
- Same format as initial approach (for compatibility)
- Expected: More realistic scores, reduced overestimation

### 3. Single Question Gallup Approach (1 Question Only)

**Purpose:** Pure Gallup World Poll methodology - only life evaluation question

```bash
cd llm_audit_data/single_question_gallup_approach
python run_audit_gallup.py
```

**What it does:**
- Single question: Cantril Ladder life evaluation only
- No factor sub-questions
- Pure Gallup methodology
- Temperature: 0.3, Realistic system prompts
- Country context consideration (unbiased)

**Output:**
- Same CSV structure (factor columns set to None)
- Only `overall_happiness` collected

**Note:** All approaches may take several hours due to API rate limits. Progress is saved incrementally, so you can resume if interrupted.

---

### 4. Compare LLM vs Real Data

**Purpose:** Compare LLM responses with actual World Happiness Report data

```bash
cd llm_audit_data
python analyze_llm_vs_real.py
```

**What it does:**
- Calculates country-level averages from LLM responses (by approach if multiple approaches exist)
- Compares with real 2024 (or latest) data
- Computes differences and correlations for all 7 metrics
- Identifies countries with largest estimation errors
- Generates comparison CSV and statistics report
- **Note:** If results from multiple approaches exist, the comparison file will include an 'approach' column for dashboard filtering

**Output:**
- `results/llm_vs_real_comparison.csv` - Detailed comparison (with 'approach' column if multiple approaches)
- `results/llm_vs_real_statistics.txt` - Summary statistics

---

### 5. Analyze Bias by Groups

**Purpose:** Systematic bias analysis across different country groupings

```bash
cd llm_audit_data
python analyze_bias.py
```

**What it does:**
- Analyzes bias across multiple groupings:
  - Global North vs Global South
  - Continents/Regions
  - Income levels (World 1/2/3)
  - East vs West
  - Developed/Undeveloped countries
- Calculates mean differences, statistical significance, effect sizes
- Generates detailed reports by group and metric
- **Note:** The dashboard's Statistical Significance tab shows a simplified analysis focusing on key groupings: Continent, World 1/2/3, Region, and Developed/Undeveloped, with separate results for each of the three approaches

**Output:**
- `results/bias_summary_*.csv` - Summary by groups
- `results/bias_analysis_data_*.csv` - Detailed bias data
- `results/bias_analysis_report_*.txt` - Comprehensive report
- `results/significant_findings_*.csv` - Statistically significant findings (auto-generated, p < 0.05)

---

### 6. Hyperparameter Tuning

**Purpose:** Find optimal API hyperparameters (temperature, chain-of-thought)

```bash
cd llm_audit_data
python hyperparameter_tuning.py
```

**What it does:**
- Tests different temperature values (0.3, 0.6, 0.9, 1.2)
- Tests with/without chain-of-thought prompting
- Uses random sample of 10 countries for efficiency
- Compares results with ground truth to find best parameters

**Output:**
- `results/hyperparameter_tuning_results_*.csv` - Results for each combination
- `results/hyperparameter_tuning_metrics_*.csv` - Performance metrics
- `results/hyperparameter_tuning_report_*.txt` - Analysis report
- `results/hyperparameter_tuning_logs.txt` - Execution logs

---

### 7. Prompt Tuning

**Purpose:** Test different prompt strategies to improve LLM accuracy

```bash
cd llm_audit_data
python prompt_tuning.py
```

**What it does:**
- Tests various prompt variations:
  - Baseline (original detailed prompt)
  - Simplified version
  - Concise persona instructions
  - JSON format output
  - Few-shot examples
  - Explicit context instructions
  - Step-by-step reasoning
- Uses optimal hyperparameters from step 4
- Evaluates which approach best matches real data

**Output:**
- `results/prompt_tuning_results_*.csv` - Results for each prompt
- `results/prompt_tuning_metrics_*.csv` - Performance metrics
- `results/prompt_tuning_report_*.txt` - Analysis report
- `results/prompt_tuning_logs_*.txt` - Execution logs

---

## ✨ Features

### Interactive Dashboard

The dashboard includes **6 main tabs** (LLM Audit has 4 sub-tabs):

1. **🌍 World Map** - Interactive choropleth map with country-level happiness scores
   - Year selector for temporal exploration
   - Click countries for detailed information
   - Hover for quick statistics

2. **📊 Driver Analysis** - Factor importance analysis
   - Weighted linear regression ranking
   - Regional differences in factor importance
   - R-squared values for model fit

3. **📈 Trends (2013-2023)** - Temporal analysis
   - Global average happiness over time
   - Regional and income-level trends
   - Top performers and biggest changers
   - Volatility analysis

4. **🔍 Group Comparisons** - Comparative analysis
   - Regional comparisons with statistical tests
   - Income level comparisons
   - Factor differences by group
   - Happiness gap analysis

5. **🤖 LLM Audit** - AI bias analysis with approach-specific filtering
   - **Overview:** Overall statistics and scatter plots (automatically filtered by selected approach)
   - **Key Findings:** Main insights from audit
   - **Bias by Groups:** Detailed analysis by groupings
   - **Statistical Significance:** Simplified significance tests showing:
     - **Key groupings only:** Continent, World 1/2/3, Region, and Developed/Undeveloped
     - **Separate results for each approach:** Initial, Few-Shot, and Single Question Gallup
     - **LLM vs Real comparison:** Shows LLM mean, Real mean, bias, p-values, and significance checkmarks
     - **Focus on Overall Happiness:** Tests whether LLM predictions differ significantly from real data

6. **📋 Overview** - Summary statistics and key metrics

### Analysis Methods

- **Weighted Linear Regression** - Factor importance analysis
- **Statistical Significance Testing** - T-tests, ANOVA
- **Correlation Analysis** - Pearson, Spearman correlations
- **Gap Analysis** - Between-group differences
- **Volatility Metrics** - Stability analysis over time

---

## 📊 Data Files

### Required Files

- ✅ `data.xlsx` - World Happiness Report dataset (required for dashboard)

### Optional LLM Audit Files

The dashboard automatically detects and uses real LLM audit data if available:

- ✅ `llm_audit_data/results/llm_vs_real_comparison.csv` - LLM vs real comparison (with 'approach' column for filtering)
- ✅ `llm_audit_data/results/bias_summary_*.csv` - Bias summary
- ✅ `llm_audit_data/results/significant_findings_*.csv` - Significant findings (auto-generated)
- ✅ `llm_audit_data/results/bias_analysis_data_*.csv` - Detailed bias data

**Note:** If LLM audit files are missing, the dashboard will use mock data for demonstration purposes.

### Approach-Specific Analysis

The LLM Audit tab includes an **Approach Filter** dropdown that allows you to:
- View **All Approaches** combined (default)
- View **Initial Approach** only
- View **Few-Shot Approach** only
- View **Single Question Gallup Approach** only

When a specific approach is selected:
- **Overview** tab shows statistics computed only from that approach's data
- **Key Findings** and **Bias by Groups** tabs show combined results (with a note)
- **Statistical Significance** tab shows separate results for each approach automatically, with simplified groupings:
  - **Continent** (Africa, Asia, Europe, etc.)
  - **World 1/2/3** (Development level classification)
  - **Region** (East Asia, Western Europe, Eastern Europe, etc.)
  - **Developed/Undeveloped** (Simplified binary classification)
- All visualizations and statistics update to reflect the selected approach

The dashboard automatically builds comparison data from raw LLM results (preferring multi-approach data with 'approach' column) to enable this filtering. The Statistical Significance tab displays results in clean tables showing LLM mean, Real mean, bias, p-values, and significance checkmarks (✓) for statistically significant findings.

---

## 🔧 Troubleshooting

### ModuleNotFoundError

**Solution:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### FileNotFoundError: data.xlsx

**Solution:** Ensure `data.xlsx` is in the project root directory.

### Port 8050 Already in Use

**Solution:** Modify `app.py` to use a different port:
```python
app.run(debug=True, port=8051)
```

### LLM Audit Data Not Loading

**Solution:** This is a warning, not an error. The dashboard will work with mock data. To use real data:
- Ensure `llm_audit_data/results/` directory exists
- Run the LLM audit pipelines to generate data files

### API Key Issues (LLM Audit)

**Solution:** 
1. Ensure `.env` file exists in project root with `OPENAI_API_KEY=your-key-here`
2. Install python-dotenv: `pip install python-dotenv`
3. Or export environment variable: `export OPENAI_API_KEY='your-key-here'`
4. See [API Key Setup](#-api-key-setup) section above

---

## 📝 Notes

- The LLM audit system uses OpenAI's GPT API (model: gpt-5.2)
- **Three approaches available**: Initial (7 questions), Few-Shot (7 questions + calibration), Single Question (1 question only)
- Results are automatically organized - newest files kept in `results/`, older versions moved to `results/old_results/`
- All approaches use the same 20 personas and produce compatible CSV output formats
- Regional and income classifications are simplified for analysis purposes
- For production use, consider using official World Bank or UN classifications
- API key is managed via `.env` file (see [API Key Setup](#-api-key-setup))

---

## 📄 License

This project is for academic purposes as part of the Data and Society Seminar at Saarland University.

---

## 👥 Team

- **Sina Elahi Manesh**
- **Yassal Arif**
- **Akash Chavanta**
