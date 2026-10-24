"""
Master Pipeline Runner: Flight Delay Prediction Project (Phase 1 & 2)

Executes the entire end-to-end data preparation, splitting, feature engineering,
and validation pipeline:
  Step 1: src/cleaning.py   - Data cleaning, COVID-2020 exclusion, leakage assertion
  Step 2: src/split.py      - Chronological train/val/test splitting
  Step 3: src/features.py   - Feature engineering (tree & linear sets) and artifacts
  Step 4: src/validate.py   - Comprehensive validation, OOF checks, single-row checks

Optional flags:
  --eda      Also execute the initial EDA & visualization scripts (01-06)
  --help     Show help message
"""

import sys
import time
import argparse
import subprocess

def run_step(script_path: str, description: str):
    print("\n" + "=" * 76)
    print(f"  STEP: {description}")
    print(f"  Command: {sys.executable} {script_path}")
    print("=" * 76)
    start_time = time.time()
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    elapsed = time.time() - start_time
    if result.returncode != 0:
        print(f"\n[ERROR] Step {script_path} failed with exit code {result.returncode} after {elapsed:.2f}s")
        sys.exit(result.returncode)
    print(f"[OK] Completed {description} in {elapsed:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser(description="Flight Delay ML Pipeline - Master Runner (Phases 1-4)")
    parser.add_argument("--eda", action="store_true", help="Also run the initial exploratory data analysis scripts")
    parser.add_argument("--phase4", action="store_true", help="Also execute Phase 4 tuning and calibration pipeline")
    args = parser.parse_args()

    total_start = time.time()
    print("*" * 76)
    print("   US FLIGHT DELAY PREDICTION - END-TO-END DATA & FEATURE PIPELINE   ")
    print("*" * 76)

    # Optional EDA steps if requested
    if args.eda:
        print("\n--- Running EDA & Exploratory Scripts ---")
        run_step("scripts/01_convert_to_parquet.py", "Convert Raw CSV to Parquet")
        run_step("scripts/02_data_quality_audit.py", "Data Quality & Integrity Audit")
        run_step("scripts/03_target_analysis.py", "Target Distribution & Class Drift Analysis")
        run_step("scripts/04_delay_patterns_and_figures.py", "Generate High-Res Figures")
        run_step("scripts/05_delay_causes_and_correlations.py", "Delay Causes & Feature Associations")
        run_step("scripts/06_generate_report.py", "Generate EDA Markdown and HTML Reports")

    # Phase 1 & 2 Core Pipeline
    print("\n--- Running Phase 1: Cleaning & Splitting ---")
    run_step("src/cleaning.py", "Part A: Cleaning Pipeline & Leakage Assertion")
    run_step("src/split.py", "Part B: Chronological Train / Val / Test Split")

    print("\n--- Running Phase 2: Feature Engineering & Artifact Fitting ---")
    run_step("src/features.py", "Part C: Feature Engineering (Tree & Linear Sets)")

    print("\n--- Running Validation Suite ---")
    run_step("src/validate.py", "Sanity Checks, OOF Sanity, Leakage & PR-AUC Tests")

    # Phase 3: Baseline Models, Ablations, and Diagnostics
    print("\n--- Running Phase 3: Baseline Models, Feature Ablations & Diagnostics ---")
    run_step("src/research/train_baselines.py", "Phase 3 - Part 1: Train Baseline Models")
    run_step("src/research/ablation.py", "Phase 3 - Part 2: LightGBM Feature Ablation Study")
    run_step("src/research/diagnostics.py", "Phase 3 - Part 3: Sanity Diagnostics & Model Serialization")

    # Phase 4: Controlled Improvement, Tuning, and Calibration
    if args.phase4:
        print("\n--- Running Phase 4: Controlled Improvement & Calibration ---")
        run_step("src/research/feature_experiments.py", "Phase 4 - Part 1: Schedule-Only Feature Experiments")
        run_step("src/research/tune_lgbm.py", "Phase 4 - Part 2: Optuna LightGBM Tuning")
        run_step("src/research/train_alternatives.py", "Phase 4 - Part 3: Alternative Architectures & Blends")
        run_step("src/calibrate.py", "Phase 4 - Part 4: GroupKFold Monthly Calibration")
        run_step("src/research/champion_and_evaluation_v2.py", "Phase 4 - Part 5: Final Champion Packaging & Deciles")

    total_elapsed = time.time() - total_start
    print("\n" + "*" * 76)
    print("  ALL REQUESTED PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print(f"  Total Execution Time: {total_elapsed:.2f} seconds")
    print("*" * 76)
    print("Generated Datasets:")
    print("  - data/processed/cleaned_flights.parquet")
    print("  - data/processed/train.parquet, val.parquet, test.parquet")
    print("  - data/processed/features_tree_train.parquet, val.parquet, test.parquet")
    print("  - data/processed/features_linear_train.parquet, val.parquet, test.parquet")
    print("\nGenerated Artifacts (artifacts/):")
    print("  - feature_pipeline.joblib & feature_pipeline_v2.joblib")
    print("  - lookup_tables.joblib")
    print("  - scaler.joblib")
    print("  - category_lists.joblib")
    print("  - holiday_config.joblib")
    print("  - route_elapsed_medians.joblib")
    print("  - models/best_baseline_lgbm.joblib")
    print("  - models/champion_v2.joblib")
    print("  - models/calibrator_v2.joblib")
    print("  - models/model_metadata.json & model_metadata_v2.json")
    print("\nGenerated Reports & Diagnostic Figures:")
    print("  - reports/phase1_2_report.md")
    print("  - reports/phase3_report.md")
    print("  - reports/phase4_report.md")
    print("  - reports/figures/fig10_calibration_curve.png")
    print("  - reports/figures/fig11_feature_importance.png")
    print("  - reports/figures/fig12_shap_summary.png")
    print("  - reports/figures/fig13_learning_curve.png")
    print("  - reports/figures/fig14_calibration_comparison.png")
    print("  - reports/baseline_metrics.json")
    print("  - reports/ablation_metrics.json")
    print("  - reports/slice_metrics.json & slice_metrics_v2.json")
    print("  - reports/feature_experiments_metrics.json")
    print("  - reports/optuna_study.json & top5_lgbm_configs.json")
    print("  - reports/alternatives_metrics.json")
    print("  - reports/calibration_metrics.json")
    print("  - reports/decile_table_v2.json")
    print("  - reports/phase4_metrics.json")
    print("*" * 76)

if __name__ == "__main__":
    main()
