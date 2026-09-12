
import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD FILES
# ============================================================

features = pd.read_csv(
    "JULY_2026_FEATURE_DATASET.csv"
)

predictions = pd.read_csv(
    "JULY_2026_RISK_PREDICTIONS.csv"
)

# ============================================================
# 2. MAKE PROJECT CODE CONSISTENT
# ============================================================

features["project_code"] = (
    features["project_code"]
    .astype(str)
)

predictions["project_code"] = (
    predictions["project_code"]
    .astype(str)
)

# ============================================================
# 3. SELECT DASHBOARD COLUMNS FROM FEATURES
# ============================================================

dashboard_columns = [
    "report_month",
    "project_code",
    "project_name",
    "state",
    "sector",
    "ministry_final",
    "agency_final",
    "original_cost",
    "expenditure",
    "physical_progress",
    "expenditure_ratio",
    "previous_progress",
    "progress_change",
    "previous_expenditure",
    "expenditure_change",
    "progress_velocity",
    "expenditure_velocity",
    "previous_observations",
    "months_observed",
    "days_since_first_report",
    "has_physical_progress",
    "has_expenditure",
    "has_original_cost"
]

# Keep only columns that exist
dashboard_columns = [
    col for col in dashboard_columns
    if col in features.columns
]

dashboard = features[dashboard_columns].copy()

# ============================================================
# 4. SELECT PREDICTION COLUMNS
# ============================================================

prediction_columns = [
    "project_code",
    "overrun_probability",
    "overrun_probability_percent",
    "risk_level"
]

prediction_columns = [
    col for col in prediction_columns
    if col in predictions.columns
]

predictions_selected = predictions[
    prediction_columns
].copy()

# ============================================================
# 5. MERGE PROJECT DATA + ML PREDICTIONS
# ============================================================

dashboard = dashboard.merge(
    predictions_selected,
    on="project_code",
    how="left"
)

# ============================================================
# 6. CLEAN RISK PROBABILITY
# ============================================================

if "overrun_probability" in dashboard.columns:

    dashboard["overrun_probability"] = pd.to_numeric(
        dashboard["overrun_probability"],
        errors="coerce"
    )

if "overrun_probability_percent" in dashboard.columns:

    dashboard["overrun_probability_percent"] = pd.to_numeric(
        dashboard["overrun_probability_percent"],
        errors="coerce"
    )

# ============================================================
# 7. CLEAN NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "original_cost",
    "expenditure",
    "physical_progress",
    "expenditure_ratio",
    "previous_progress",
    "progress_change",
    "previous_expenditure",
    "expenditure_change",
    "progress_velocity",
    "expenditure_velocity",
    "previous_observations",
    "months_observed",
    "days_since_first_report"
]

for col in numeric_columns:

    if col in dashboard.columns:

        dashboard[col] = pd.to_numeric(
            dashboard[col],
            errors="coerce"
        )

# ============================================================
# 8. CLEAN TEXT COLUMNS
# ============================================================

text_columns = [
    "state",
    "sector",
    "ministry_final",
    "agency_final"
]

for col in text_columns:

    if col in dashboard.columns:

        dashboard[col] = (
            dashboard[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

# ============================================================
# 9. REMOVE DUPLICATE PROJECTS
# ============================================================

dashboard = dashboard.drop_duplicates(
    subset=["project_code"]
)

# ============================================================
# 10. SORT BY RISK
# ============================================================

if "overrun_probability" in dashboard.columns:

    dashboard = dashboard.sort_values(
        "overrun_probability",
        ascending=False
    )

# ============================================================
# 11. SAVE MASTER DASHBOARD DATASET
# ============================================================

output_file = "PAIMANA_DASHBOARD_DATA.csv"

dashboard.to_csv(
    output_file,
    index=False
)

# ============================================================
# 12. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PAIMANA DASHBOARD DATA CREATED")
print("=" * 70)

print(
    "Total projects:",
    len(dashboard)
)

print(
    "Total columns:",
    len(dashboard.columns)
)

print("\nRisk distribution:")

if "risk_level" in dashboard.columns:

    print(
        dashboard["risk_level"]
        .value_counts()
    )

print("\nTop 10 highest-risk projects:")

display_columns = [
    "project_code",
    "project_name",
    "state",
    "sector",
    "overrun_probability_percent",
    "risk_level"
]

display_columns = [
    col for col in display_columns
    if col in dashboard.columns
]

print(
    dashboard[display_columns]
    .head(10)
    .to_string(index=False)
)

print("\nSaved file:")
print(output_file)

print("=" * 70)

