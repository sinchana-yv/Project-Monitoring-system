import pandas as pd
import numpy as np

print("=" * 60)
print("CREATING LEAKAGE-SAFE FEATURE DATASET")
print("=" * 60)


# ============================================================
# 1. LOAD FULL CLEAN DATA
# ============================================================

df = pd.read_csv("FINAL_CLEAN_ONGOING_PROJECTS.csv")

print("\nOriginal records:", len(df))

# Remove records without project code
df = df.dropna(subset=["project_code"]).copy()

# Convert report month
df["report_month"] = pd.to_datetime(
    df["report_month"],
    errors="coerce"
)

# Convert numeric columns
numeric_columns = [
    "original_cost",
    "revised_cost",
    "expenditure",
    "physical_progress"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Sort chronologically
df = df.sort_values(
    ["project_code", "report_month"]
).reset_index(drop=True)


# ============================================================
# 2. BASIC CURRENT FEATURES
# ============================================================

print("\nCreating current-project features...")


# ------------------------------------------------------------
# Expenditure ratio
# ------------------------------------------------------------
# How much of the original project cost has already been spent

df["expenditure_ratio"] = np.where(
    df["original_cost"] > 0,
    (df["expenditure"] / df["original_cost"]) * 100,
    np.nan
)


# ============================================================
# 3. HISTORICAL FEATURES
# ============================================================

print("Creating historical project features...")


# Previous physical progress
df["previous_progress"] = (
    df.groupby("project_code")["physical_progress"]
      .shift(1)
)


# Change in physical progress
df["progress_change"] = (
    df["physical_progress"]
    - df["previous_progress"]
)


# Previous expenditure
df["previous_expenditure"] = (
    df.groupby("project_code")["expenditure"]
      .shift(1)
)


# Change in expenditure
df["expenditure_change"] = (
    df["expenditure"]
    - df["previous_expenditure"]
)


# Previous report month
df["previous_report_month"] = (
    df.groupby("project_code")["report_month"]
      .shift(1)
)


# ============================================================
# 4. TIME GAP
# ============================================================

df["days_since_previous"] = (
    df["report_month"]
    - df["previous_report_month"]
).dt.days


# ============================================================
# 5. PROGRESS VELOCITY
# ============================================================
# Progress percentage gained per day

df["progress_velocity"] = np.where(
    df["days_since_previous"] > 0,
    df["progress_change"] / df["days_since_previous"],
    np.nan
)


# ============================================================
# 6. EXPENDITURE VELOCITY
# ============================================================
# Expenditure increase per day

df["expenditure_velocity"] = np.where(
    df["days_since_previous"] > 0,
    df["expenditure_change"] / df["days_since_previous"],
    np.nan
)


# ============================================================
# 7. PROJECT HISTORY
# ============================================================

# Number of observations BEFORE current observation
df["previous_observations"] = (
    df.groupby("project_code")
      .cumcount()
)


# Number of observations INCLUDING current observation
df["months_observed"] = (
    df["previous_observations"] + 1
)


# ============================================================
# 8. PROJECT AGE IN DATASET
# ============================================================

first_report = (
    df.groupby("project_code")["report_month"]
      .transform("min")
)

df["days_since_first_report"] = (
    df["report_month"] - first_report
).dt.days


# ============================================================
# 9. PROGRESS / EXPENDITURE QUALITY FLAGS
# ============================================================

df["has_physical_progress"] = (
    df["physical_progress"].notna().astype(int)
)

df["has_expenditure"] = (
    df["expenditure"].notna().astype(int)
)

df["has_original_cost"] = (
    df["original_cost"].notna().astype(int)
)


# ============================================================
# 10. CURRENT TARGET
# ============================================================

# Load the correctly generated target file

target_df = pd.read_csv(
    "cost_overrun_training_data.csv"
)

target_df["report_month"] = pd.to_datetime(
    target_df["report_month"],
    errors="coerce"
)

target_df["future_cost_overrun"] = pd.to_numeric(
    target_df["future_cost_overrun"],
    errors="coerce"
)


# Only keep the columns needed for target matching

target_df = target_df[
    [
        "project_code",
        "report_month",
        "future_cost_overrun"
    ]
].copy()


# ============================================================
# 11. MERGE TARGET
# ============================================================

df = df.merge(
    target_df,
    on=["project_code", "report_month"],
    how="left"
)


# ============================================================
# 12. SEPARATE TRAINING DATA
# ============================================================

training_df = df[
    df["future_cost_overrun"].notna()
].copy()

training_df["future_cost_overrun"] = (
    training_df["future_cost_overrun"]
    .astype(int)
)


# ============================================================
# 13. SEPARATE JULY 2026 LIVE DATA
# ============================================================

july_2026 = df[
    df["report_month"] == pd.Timestamp("2026-07-01")
].copy()

# July 2026 must remain unlabelled

july_2026["future_cost_overrun"] = np.nan


# ============================================================
# 14. REMOVE FUTURE / LEAKAGE FEATURES
# ============================================================

# These columns must NOT be used by the ML model.

leakage_columns = [
    "revised_cost",
    "future_cost_overrun"
]

# We do not delete them yet.
# We keep them in the feature dataset for inspection.
#
# The ML preparation step will remove them before training.


# ============================================================
# 15. SAVE DATASETS
# ============================================================

training_df.to_csv(
    "ML_FEATURE_DATASET.csv",
    index=False
)

july_2026.to_csv(
    "JULY_2026_FEATURE_DATASET.csv",
    index=False
)


# ============================================================
# 16. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 60)

print(
    "\nFull records:",
    len(df)
)

print(
    "Training records:",
    len(training_df)
)

print(
    "Training projects:",
    training_df["project_code"].nunique()
)

print(
    "July 2026 live records:",
    len(july_2026)
)

print(
    "July 2026 projects:",
    july_2026["project_code"].nunique()
)


print("\nTarget distribution:")

print(
    training_df[
        "future_cost_overrun"
    ].value_counts()
)


print("\nTarget percentage:")

print(
    training_df[
        "future_cost_overrun"
    ]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 17. FEATURE CHECK
# ============================================================

feature_columns = [
    "physical_progress",
    "expenditure",
    "original_cost",
    "expenditure_ratio",
    "previous_progress",
    "progress_change",
    "previous_expenditure",
    "expenditure_change",
    "days_since_previous",
    "progress_velocity",
    "expenditure_velocity",
    "previous_observations",
    "months_observed",
    "days_since_first_report",
    "has_physical_progress",
    "has_expenditure",
    "has_original_cost",
    "state",
    "sector",
    "ministry_final",
    "agency_final"
]

print("\n" + "=" * 60)
print("FEATURE AVAILABILITY")
print("=" * 60)

for col in feature_columns:

    if col in training_df.columns:

        available = (
            training_df[col].notna().sum()
        )

        missing = (
            training_df[col].isna().sum()
        )

        print(
            f"{col:30s} "
            f"Available: {available:5d} "
            f"Missing: {missing:5d}"
        )


# ============================================================
# 18. HISTORY CHECK
# ============================================================

print("\n" + "=" * 60)
print("PROJECT HISTORY CHECK")
print("=" * 60)

print(
    training_df[
        "months_observed"
    ].describe()
)


print("\nHistory distribution:")

print(
    training_df[
        "months_observed"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 19. JULY 2026 CHECK
# ============================================================

print("\n" + "=" * 60)
print("JULY 2026 LIVE DATA CHECK")
print("=" * 60)

print(
    "July 2026 records:",
    len(july_2026)
)

print(
    "July 2026 labelled:",
    july_2026[
        "future_cost_overrun"
    ].notna().sum()
)

print(
    "\nJuly 2026 remains UNLABELLED."
)


# ============================================================
# 20. FILES CREATED
# ============================================================

print("\n" + "=" * 60)
print("FILES CREATED")
print("=" * 60)

print("1. ML_FEATURE_DATASET.csv")
print("2. JULY_2026_FEATURE_DATASET.csv")

print("\nDONE")
print("=" * 60)