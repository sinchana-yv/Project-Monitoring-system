import pandas as pd
import numpy as np

print("=" * 60)
print("CREATING FINAL COST OVERRUN TARGET")
print("=" * 60)

# ---------------------------------------------------------
# 1. LOAD CLEAN DATA
# ---------------------------------------------------------

df = pd.read_csv("FINAL_CLEAN_ONGOING_PROJECTS.csv")

print("\nOriginal records:", len(df))

# Remove records without project code
df = df.dropna(subset=["project_code"]).copy()

# Convert dates
df["report_month"] = pd.to_datetime(
    df["report_month"],
    errors="coerce"
)

# Convert cost columns
df["original_cost"] = pd.to_numeric(
    df["original_cost"],
    errors="coerce"
)

df["revised_cost"] = pd.to_numeric(
    df["revised_cost"],
    errors="coerce"
)

# Sort chronologically
df = df.sort_values(
    ["project_code", "report_month"]
).reset_index(drop=True)


# ---------------------------------------------------------
# 2. CREATE FUTURE COST OVERRUN TARGET
# ---------------------------------------------------------

targets = []

for project_code, group in df.groupby("project_code"):

    group = group.sort_values("report_month")

    for index, row in group.iterrows():

        current_date = row["report_month"]
        current_original = row["original_cost"]

        # -------------------------------------------------
        # No original cost → cannot create target
        # -------------------------------------------------

        if pd.isna(current_original):

            targets.append(np.nan)
            continue

        # -------------------------------------------------
        # ONLY FUTURE OBSERVATIONS
        # -------------------------------------------------

        future = group[
            group["report_month"] > current_date
        ]

        # Keep future observations having revised cost
        future_revised = future[
            pd.notna(future["revised_cost"])
        ]

        # -------------------------------------------------
        # No future evidence
        # -------------------------------------------------

        if future_revised.empty:

            targets.append(np.nan)
            continue

        # -------------------------------------------------
        # Check future cost overrun
        #
        # Future revised cost > CURRENT original cost
        # -------------------------------------------------

        future_overrun = (
            future_revised["revised_cost"]
            > current_original
        )

        if future_overrun.any():

            targets.append(1)

        else:

            targets.append(0)


# ---------------------------------------------------------
# 3. ADD TARGET
# ---------------------------------------------------------

df["future_cost_overrun"] = targets


# ---------------------------------------------------------
# 4. SEPARATE LIVE JULY 2026 DATA
# ---------------------------------------------------------

july_2026 = df[
    df["report_month"] == pd.Timestamp("2026-07-01")
].copy()

print("\nJuly 2026 records:", len(july_2026))

# July 2026 has no future observation in our dataset,
# therefore it should be treated as live/unlabelled data.

july_2026["future_cost_overrun"] = np.nan

july_2026.to_csv(
    "JULY_2026_LIVE_DATA.csv",
    index=False
)

print("Saved: JULY_2026_LIVE_DATA.csv")


# ---------------------------------------------------------
# 5. REMOVE UNKNOWN TARGETS FROM TRAINING DATA
# ---------------------------------------------------------

training_df = df[
    df["future_cost_overrun"].notna()
].copy()

training_df["future_cost_overrun"] = (
    training_df["future_cost_overrun"]
    .astype(int)
)


# ---------------------------------------------------------
# 6. SAVE TRAINING DATA
# ---------------------------------------------------------

training_df.to_csv(
    "cost_overrun_training_data.csv",
    index=False
)


# ---------------------------------------------------------
# 7. SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TARGET SUMMARY")
print("=" * 60)

print(
    "\nTotal records after project-code cleaning:",
    len(df)
)

print(
    "Training records:",
    len(training_df)
)

print(
    "Unknown/unlabelled records:",
    df["future_cost_overrun"].isna().sum()
)

print(
    "Unique training projects:",
    training_df["project_code"].nunique()
)

print("\nTarget distribution:")

print(
    training_df["future_cost_overrun"]
    .value_counts()
)

print("\nTarget percentage:")

print(
    training_df["future_cost_overrun"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nTraining records by report month:")

print(
    training_df
    .groupby("report_month")
    .size()
    .sort_index()
)

print("\nTarget by report month:")

print(
    pd.crosstab(
        training_df["report_month"],
        training_df["future_cost_overrun"]
    )
)


# ---------------------------------------------------------
# 8. VERIFY JULY 2026
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("JULY 2026 LIVE DATA CHECK")
print("=" * 60)

print(
    "July 2026 projects:",
    len(july_2026)
)

print(
    "July 2026 labels:",
    july_2026["future_cost_overrun"]
    .notna()
    .sum()
)

print(
    "\nJuly 2026 is kept UNLABELLED for live prediction."
)


print("\n" + "=" * 60)
print("FILES CREATED")
print("=" * 60)

print("1. cost_overrun_training_data.csv")
print("2. JULY_2026_LIVE_DATA.csv")

print("\nDONE")
print("=" * 60)