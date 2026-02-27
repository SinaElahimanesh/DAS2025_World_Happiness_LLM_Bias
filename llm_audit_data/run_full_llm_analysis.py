"""
Run full LLM analysis: build comparison from all approaches and run bias analysis.

This script:
1. Builds LLM vs Real comparison from all four approaches (initial, few_shot,
   single_question, structured_personas) and saves results/llm_vs_real_comparison.csv.
2. Runs analyze_bias.py to produce bias_summary_*.csv, bias_analysis_data_*.csv,
   significant_findings_*.csv, and bias_analysis_report_*.txt.

Run from project root or from llm_audit_data:
    python llm_audit_data/run_full_llm_analysis.py
    # or
    cd llm_audit_data && python run_full_llm_analysis.py
"""

import os
import sys
from pathlib import Path

# Ensure we run from llm_audit_data so analyze_bias finds results/
SCRIPT_DIR = Path(__file__).resolve().parent
if os.getcwd() != str(SCRIPT_DIR):
    os.chdir(SCRIPT_DIR)
    print(f"Working directory set to: {SCRIPT_DIR}")

# Parent for data_loader
sys.path.insert(0, str(SCRIPT_DIR.parent))

from load_llm_results import save_comparison_to_csv

def main():
    print("=" * 60)
    print("Full LLM Analysis (all approaches + bias)")
    print("=" * 60)

    print("\n1. Building comparison from all approaches...")
    path = save_comparison_to_csv()
    if path is None:
        print("   Failed to build comparison. Ensure LLM result CSVs exist for at least one approach.")
        sys.exit(1)
    print(f"   Saved: {path}")

    print("\n2. Running bias analysis...")
    import analyze_bias
    analyze_bias.main()

    print("\n" + "=" * 60)
    print("Done. Dashboard and reports now use all approaches.")
    print("=" * 60)

if __name__ == "__main__":
    main()
