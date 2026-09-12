
import pandas as pd
import numpy as np

print("=" * 70)
print("JULY 2026 RISK PREDICTION VALIDATION")
print("=" * 70)

# ============================================================
# 1. LOAD FILES
# ============================================================

pred_file = "JULY_2026_RISK_PREDICTIONS.csv"
feature_file = "JULY_2026_FEATURE_DATASET.csv"

pred = pd.read_csv(pred_file)
features = pd.read_csv(feature_file)

print("\nPrediction dataset shape:", pred.shape)
print("Feature dataset shape:", features.shape)


# ============================================================
# 2. CHECK COLUMNS
# ============================================================

print("\nPrediction columns:")
print(pred.columns.tolist())

print("\nFeature columns:")
print(features.columns.tolist())


# ============================================================
# 3. MERGE PREDICTIONS WITH FEATURES
# ============================================================

# Keep one row per project
pred = pred.drop_duplicates(subset=["project_code"])

features = features.drop_duplicates(
    subset=["project_code"]
)

df = pred.merge(
    features,
    on="project_code",
    how="left",
    suffixes=("", "_feature")
)

print("\nMerged dataset shape:", df.shape)


# ============================================================
# 4. BASIC RISK DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("RISK DISTRIBUTION")
print("=" * 70)

risk_counts = df["risk_level"].value_counts()

print(risk_counts)

risk_percent = (
    df["risk_level"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nRisk percentages:")
print(risk_percent)


# ============================================================
# 5. NUMERIC FEATURE COMPARISON
# ============================================================

numeric_features = [
    "physical_progress",
    "expenditure",
    "original_cost",
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

numeric_features = [
    col for col in numeric_features
    if col in df.columns
]

print("\n" + "=" * 70)
print("HIGH vs MEDIUM vs LOW FEATURE COMPARISON")
print("=" * 70)

summary = (
    df.groupby("risk_level")[numeric_features]
    .mean()
    .round(2)
)

print(summary)


# ============================================================
# 6. MEDIAN COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MEDIAN VALUES")
print("=" * 70)

median_summary = (
    df.groupby("risk_level")[numeric_features]
    .median()
    .round(2)
)

print(median_summary)


# ============================================================
# 7. MIN / MAX FOR IMPORTANT FEATURES
# ============================================================

important_features = [
    "physical_progress",
    "expenditure_ratio",
    "expenditure",
    "original_cost",
    "progress_velocity",
    "expenditure_change"
]

important_features = [
    col for col in important_features
    if col in df.columns
]

print("\n" + "=" * 70)
print("IMPORTANT FEATURE RANGES")
print("=" * 70)

for col in important_features:

    print("\n" + "-" * 60)
    print("FEATURE:", col)

    for risk in ["HIGH", "MEDIUM", "LOW"]:

        subset = df[
            df["risk_level"] == risk
        ][col].dropna()

        if len(subset) > 0:

            print(
                f"{risk:8s} | "
                f"min={subset.min():.2f} | "
                f"median={subset.median():.2f} | "
                f"max={subset.max():.2f}"
            )


# ============================================================
# 8. TOP 20 HIGH-RISK PROJECTS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 HIGH-RISK PROJECTS")
print("=" * 70)

top_high = (
    df[df["risk_level"] == "HIGH"]
    .sort_values(
        "overrun_probability",
        ascending=False
    )
    .head(20)
)

display_columns = [
    "project_code",
    "project_name",
    "state",
    "sector",
    "original_cost",
    "expenditure",
    "physical_progress",
    "expenditure_ratio",
    "progress_velocity",
    "expenditure_change",
    "overrun_probability",
    "risk_level"
]

display_columns = [
    col for col in display_columns
    if col in top_high.columns
]

print(
    top_high[display_columns].to_string(
        index=False
    )
)


# ============================================================
# 9. TOP 20 MEDIUM-RISK PROJECTS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 MEDIUM-RISK PROJECTS")
print("=" * 70)

top_medium = (
    df[df["risk_level"] == "MEDIUM"]
    .sort_values(
        "overrun_probability",
        ascending=False
    )
    .head(20)
)

print(
    top_medium[display_columns].to_string(
        index=False
    )
)


# ============================================================
# 10. STATE-WISE RISK
# ============================================================

if "state" in df.columns:

    print("\n" + "=" * 70)
    print("STATE-WISE RISK DISTRIBUTION")
    print("=" * 70)

    state_risk = pd.crosstab(
        df["state"],
        df["risk_level"]
    )

    # Make sure all columns exist
    for col in ["LOW", "MEDIUM", "HIGH"]:

        if col not in state_risk.columns:
            state_risk[col] = 0

    state_risk["TOTAL"] = state_risk[
        ["LOW", "MEDIUM", "HIGH"]
    ].sum(axis=1)

    state_risk["HIGH_%"] = (
        state_risk["HIGH"]
        / state_risk["TOTAL"]
        * 100
    ).round(2)

    state_risk = state_risk.sort_values(
        "HIGH_%",
        ascending=False
    )

    print(state_risk)


# ============================================================
# 11. SECTOR-WISE RISK
# ============================================================

if "sector" in df.columns:

    print("\n" + "=" * 70)
    print("SECTOR-WISE RISK DISTRIBUTION")
    print("=" * 70)

    sector_risk = pd.crosstab(
        df["sector"],
        df["risk_level"]
    )

    for col in ["LOW", "MEDIUM", "HIGH"]:

        if col not in sector_risk.columns:
            sector_risk[col] = 0

    sector_risk["TOTAL"] = sector_risk[
        ["LOW", "MEDIUM", "HIGH"]
    ].sum(axis=1)

    sector_risk["HIGH_%"] = (
        sector_risk["HIGH"]
        / sector_risk["TOTAL"]
        * 100
    ).round(2)

    sector_risk = sector_risk.sort_values(
        "HIGH_%",
        ascending=False
    )

    print(sector_risk)


# ============================================================
# 12. HIGH-RISK PROJECT STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("HIGH-RISK PROJECT ANALYSIS")
print("=" * 70)

high = df[
    df["risk_level"] == "HIGH"
]

print("\nNumber of HIGH-risk projects:", len(high))

if len(high) > 0:

    print(
        "\nAverage HIGH-risk probability:",
        round(
            high["overrun_probability"].mean() * 100,
            2
        ),
        "%"
    )

    if "expenditure_ratio" in high.columns:
        print(
            "Average expenditure ratio:",
            round(
                high["expenditure_ratio"].mean(),
                2
            )
        )

    if "physical_progress" in high.columns:
        print(
            "Average physical progress:",
            round(
                high["physical_progress"].mean(),
                2
            ),
            "%"
        )

    if "original_cost" in high.columns:
        print(
            "Average original cost:",
            round(
                high["original_cost"].mean(),
                2
            ),
            "crore"
        )

    if "expenditure" in high.columns:
        print(
            "Average expenditure:",
            round(
                high["expenditure"].mean(),
                2
            ),
            "crore"
        )


# ============================================================
# 13. SAVE VALIDATION FILES
# ============================================================

df.to_csv(
    "JULY_2026_VALIDATED_PREDICTIONS.csv",
    index=False
)

summary.to_csv(
    "JULY_2026_RISK_FEATURE_SUMMARY.csv"
)

state_risk.to_csv(
    "JULY_2026_STATE_RISK.csv"
)

sector_risk.to_csv(
    "JULY_2026_SECTOR_RISK.csv"
)

print("\n" + "=" * 70)
print("VALIDATION COMPLETED")
print("=" * 70)

print("\nFiles created:")

print("1. JULY_2026_VALIDATED_PREDICTIONS.csv")
print("2. JULY_2026_RISK_FEATURE_SUMMARY.csv")
print("3. JULY_2026_STATE_RISK.csv")
print("4. JULY_2026_SECTOR_RISK.csv")

print("\nDONE")
print("=" * 70)

