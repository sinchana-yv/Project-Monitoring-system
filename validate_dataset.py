import pandas as pd
import os

folder = r"C:\Users\Sinchana\OneDrive\Desktop\Data Set"

file = os.path.join(
    folder,
    "clean_ongoing_projects_v2.csv"
)

df = pd.read_csv(file)

print("\n" + "=" * 60)
print("DATASET VALIDATION")
print("=" * 60)

# --------------------------------------------------
# BASIC INFORMATION
# --------------------------------------------------

print("\n1. BASIC INFORMATION")
print("-" * 40)

print("Total records:", len(df))
print("Unique projects:", df["project_code"].nunique())
print("Number of months:", df["report_month"].nunique())

print("\nMonths:")
print(sorted(df["report_month"].unique()))


# --------------------------------------------------
# PROJECT FREQUENCY
# --------------------------------------------------

print("\n2. PROJECT HISTORY")
print("-" * 40)

project_frequency = (
    df.groupby("project_code")
    .size()
    .value_counts()
    .sort_index()
)

print("Number of months each project appears:")
print(project_frequency)

freq = df.groupby("project_code").size()

for n in [2, 3, 4, 5, 6, 8, 10, 11]:
    print(
        f"Projects appearing {n}+ months:",
        (freq >= n).sum()
    )


# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------

print("\n3. MISSING VALUES")
print("-" * 40)

important_columns = [
    "project_code",
    "project_name",
    "state",
    "sector",
    "ministry",
    "agency",
    "approval_date",
    "start_date",
    "original_cost",
    "revised_cost",
    "expenditure",
    "physical_progress"
]

for col in important_columns:

    if col in df.columns:

        missing = df[col].isna().sum()

        percentage = (
            missing / len(df) * 100
        )

        print(
            f"{col:25} {missing:6} missing "
            f"({percentage:.2f}%)"
        )


# --------------------------------------------------
# COST ANALYSIS
# --------------------------------------------------

print("\n4. COST DATA")
print("-" * 40)

if "original_cost" in df.columns:

    print(
        "Original cost available:",
        df["original_cost"].notna().sum()
    )

if "revised_cost" in df.columns:

    print(
        "Revised cost available:",
        df["revised_cost"].notna().sum()
    )

if "original_cost" in df.columns and "revised_cost" in df.columns:

    valid_cost = df[
        df["original_cost"].notna() &
        df["revised_cost"].notna() &
        (df["original_cost"] > 0)
    ]

    cost_increased = (
        valid_cost["revised_cost"] >
        valid_cost["original_cost"]
    ).sum()

    print(
        "Records where revised cost > original:",
        cost_increased
    )


# --------------------------------------------------
# PHYSICAL PROGRESS
# --------------------------------------------------

print("\n5. PHYSICAL PROGRESS")
print("-" * 40)

if "physical_progress" in df.columns:

    progress = df["physical_progress"].dropna()

    print("Available:", len(progress))
    print("Minimum:", progress.min())
    print("Maximum:", progress.max())
    print("Mean:", progress.mean())
    print("Median:", progress.median())

    print(
        "Progress > 100:",
        (progress > 100).sum()
    )

    print(
        "Progress < 0:",
        (progress < 0).sum()
    )


# --------------------------------------------------
# EXPENDITURE
# --------------------------------------------------

print("\n6. EXPENDITURE")
print("-" * 40)

if "expenditure" in df.columns:

    expenditure = df["expenditure"].dropna()

    print("Available:", len(expenditure))
    print("Minimum:", expenditure.min())
    print("Maximum:", expenditure.max())


# --------------------------------------------------
# DUPLICATES
# --------------------------------------------------

print("\n7. DUPLICATE CHECK")
print("-" * 40)

duplicates = df.duplicated(
    subset=["project_code", "report_month"]
).sum()

print(
    "Project Code + Month duplicates:",
    duplicates
)


# --------------------------------------------------
# SAVE PROJECT FREQUENCY
# --------------------------------------------------

frequency_df = (
    freq
    .reset_index()
    .rename(columns={0: "number_of_months"})
)

frequency_file = os.path.join(
    folder,
    "project_history_frequency.csv"
)

frequency_df.to_csv(
    frequency_file,
    index=False
)

print("\nFrequency file saved:")
print(frequency_file)

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)