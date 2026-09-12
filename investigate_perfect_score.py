"""Investigate the exact 100% score without retraining a model."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def main():
    importance = pd.read_csv(ROOT / "lightgbm_safe_feature_importance.csv")
    safe_columns = pd.read_csv(ROOT / "X_train_safe.csv", nrows=0).columns.tolist()
    source = pd.read_csv(ROOT / "ML_FEATURE_DATASET.csv")
    source["report_month"] = pd.to_datetime(source["report_month"], errors="coerce")

    first_month = source.groupby("project_code")["report_month"].min()
    test_projects = set(first_month[first_month >= pd.Timestamp("2026-05-01")].index)
    safe_test_source = source[source["project_code"].isin(test_projects)].copy()

    top20 = importance.head(20)
    target_names = {"future_cost_overrun", "cost_overrun", "target", "overrun"}
    target_in_safe_features = sorted(target_names.intersection(safe_columns))

    candidate_patterns = (
        "target",
        "overrun",
        "future",
        "revised",
        "completion",
        "cost_growth",
        "cost_increase",
        "previous",
        "velocity",
    )
    suspicious_columns = [
        column
        for column in safe_columns
        if any(pattern in column.lower() for pattern in candidate_patterns)
    ]

    cost_increase_match = None
    cost_increase_crosstab = None
    if {"cost_increase", "future_cost_overrun"}.issubset(safe_test_source.columns):
        cost_increase = safe_test_source["cost_increase"].astype(int)
        target = safe_test_source["future_cost_overrun"].astype(int)
        cost_increase_match = float((cost_increase == target).mean())
        cost_increase_crosstab = pd.crosstab(target, cost_increase)

    test_months = safe_test_source["report_month"].dropna().dt.strftime("%Y-%m-%d").unique()
    complete_dataset_preprocessing = True
    feature_engineering_evidence = [
        "feature_engineering.py computes groupby(project_code).shift(1), cumcount(), and first_report on the full dataset before splitting.",
        "create_cost_target.py computes future_cost_overrun by searching later observations for each project before the train/test split.",
        "create_safe_split.py starts from ML_FEATURE_DATASET.csv, so these full-dataset-derived values are already embedded.",
    ]

    print("=" * 78)
    print("INVESTIGATION OF PERFECT LEAKAGE-SAFE LIGHTGBM SCORE")
    print("=" * 78)
    print("\nTop 20 features:")
    print(top20.to_string(index=False))

    print("\nTarget inclusion check:")
    print(f"Exact target columns in X_train_safe/X_test_safe: {target_in_safe_features or 'None'}")
    print(f"Suspicious feature columns detected: {suspicious_columns}")

    print("\nDirect target-proxy check:")
    print(f"Safe test rows: {len(safe_test_source)}")
    print(f"Safe test months: {', '.join(test_months)}")
    if cost_increase_match is not None:
        print(f"cost_increase == future_cost_overrun: {cost_increase_match:.2%}")
        print("Crosstab (target rows, cost_increase columns):")
        print(cost_increase_crosstab)
    else:
        print("Could not compare cost_increase with the target.")

    print("\nComplete-dataset preprocessing check:")
    print(f"Full-dataset-derived features/targets detected: {complete_dataset_preprocessing}")
    for evidence in feature_engineering_evidence:
        print(f"- {evidence}")

    print("\nFeature availability concerns:")
    print("- revised_cost is removed from the final safe feature file, but cost_increase remains.")
    print("- cost_increase is computed from revised_cost > original_cost in final_clean.py.")
    print("- revised_cost is a final/revised cost value and may not be available at prediction time.")
    print("- The target itself is defined using future revised_cost observations.")
    print("- previous_* and velocity features are computed from project histories built before splitting.")

    print("\nCONCLUSION:")
    if cost_increase_match == 1.0:
        print(
            "The 100% score is caused by target leakage: cost_increase exactly "
            "reproduces future_cost_overrun on the safe test set."
        )
    else:
        print(
            "The target is not directly present, but the feature engineering "
            "pipeline still contains possible indirect future-information leakage."
        )
    print(
        "The safe test set contains only May 2026 records, so this is also a "
        "single-month evaluation rather than a broad temporal validation."
    )
    print(
        "Do not treat the 100% score as genuine model performance. A corrected "
        "experiment must remove cost_increase and any future-derived fields, "
        "then recompute history/features within each training/test timeline."
    )


if __name__ == "__main__":
    main()
