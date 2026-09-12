import pandas as pd
import os

folder = r"C:\Users\Sinchana\OneDrive\Desktop\Data Set"

input_file = os.path.join(
    folder,
    "clean_ongoing_projects_v2.csv"
)

df = pd.read_csv(input_file)

print("Original records:", len(df))


# ============================================================
# HELPER FUNCTION
# ============================================================

def combine_columns(df, columns):

    existing = [
        col for col in columns
        if col in df.columns
    ]

    if not existing:
        return pd.Series(index=df.index, dtype="float64")

    result = df[existing[0]].copy()

    for col in existing[1:]:
        result = result.fillna(df[col])

    return result


# ============================================================
# STANDARDIZE IMPORTANT NUMERIC COLUMNS
# ============================================================

# Original Cost
df["original_cost"] = combine_columns(
    df,
    [
        "original_cost_rs._crore",
        "cost_original_rs_cr",
        "original_cost_rs_crore"
    ]
)

# Revised Cost
df["revised_cost"] = combine_columns(
    df,
    [
        "revised_cost_rs._crore",
        "cost_revised_rs_cr",
        "revised_cost_rs_crore"
    ]
)

# Expenditure
df["expenditure"] = combine_columns(
    df,
    [
        "cumulative_expenditure_rs._crore",
        "cumulative_expenditure_rs_cr",
        "cumulative_expenditure_rs_crore"
    ]
)

# Physical Progress
df["physical_progress"] = pd.to_numeric(
    df["physical_progress_%"],
    errors="coerce"
)


# ============================================================
# STANDARDIZE TEXT COLUMNS
# ============================================================

# Agency
if "agency" in df.columns:
    df["agency_final"] = df["agency"]

if "implementing_agency" in df.columns:
    df["agency_final"] = df["agency_final"].fillna(
        df["implementing_agency"]
    )

if "agency_name" in df.columns:
    df["agency_final"] = df["agency_final"].fillna(
        df["agency_name"]
    )


# Ministry
if "ministry" in df.columns:
    df["ministry_final"] = df["ministry"]

if "ministry_department" in df.columns:
    df["ministry_final"] = df["ministry_final"].fillna(
        df["ministry_department"]
    )


# ============================================================
# STANDARDIZE DATE / PROJECT COLUMNS
# ============================================================

# Approval date
if "approval_date" not in df.columns:
    df["approval_date"] = None

if "approval_date_start_date_mm_yyyy" in df.columns:
    df["approval_date"] = df["approval_date"].fillna(
        df["approval_date_start_date_mm_yyyy"]
    )


# Start date
if "start_date" not in df.columns:
    df["start_date"] = None

if "approval_date_start_date_mm_yyyy" in df.columns:
    df["start_date"] = df["start_date"].fillna(
        df["approval_date_start_date_mm_yyyy"]
    )


# ============================================================
# KEEP ONLY IMPORTANT COLUMNS
# ============================================================

columns_to_keep = [
    "report_month",
    "project_code",
    "project_name",
    "state",
    "sector",
    "ministry_final",
    "agency_final",
    "approval_date",
    "start_date",
    "original_doc",
    "revised_doc",
    "original_cost",
    "revised_cost",
    "expenditure",
    "physical_progress",
    "legacy_ocms_code",
    "pmgid",
    "source_file",
    "source_sheet"
]

# Keep only columns that actually exist
columns_to_keep = [
    col for col in columns_to_keep
    if col in df.columns
]

final_df = df[columns_to_keep].copy()


# ============================================================
# REMOVE DUPLICATES AGAIN
# ============================================================

before = len(final_df)

final_df = final_df.drop_duplicates(
    subset=["project_code", "report_month"],
    keep="first"
)

after = len(final_df)

print("Duplicates removed:", before - after)


# ============================================================
# NUMERIC CLEANING
# ============================================================

numeric_columns = [
    "original_cost",
    "revised_cost",
    "expenditure",
    "physical_progress"
]

for col in numeric_columns:

    if col in final_df.columns:

        final_df[col] = pd.to_numeric(
            final_df[col],
            errors="coerce"
        )


# ============================================================
# PHYSICAL PROGRESS VALIDATION
# ============================================================

if "physical_progress" in final_df.columns:

    # Invalid values become missing
    invalid_progress = (
        (final_df["physical_progress"] < 0) |
        (final_df["physical_progress"] > 100)
    )

    print(
        "Invalid physical progress values:",
        invalid_progress.sum()
    )

    final_df.loc[
        invalid_progress,
        "physical_progress"
    ] = None


# ============================================================
# COST OVERRUN INDICATOR
# ============================================================

final_df["cost_increase"] = (
    final_df["revised_cost"] >
    final_df["original_cost"]
)


# ============================================================
# SORT
# ============================================================

final_df = final_df.sort_values(
    ["project_code", "report_month"]
)


# ============================================================
# SAVE FINAL DATASET
# ============================================================

output_file = os.path.join(
    folder,
    "FINAL_CLEAN_ONGOING_PROJECTS.csv"
)

final_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("FINAL CLEAN DATASET")
print("=" * 60)

print("Records:", len(final_df))

print(
    "Unique projects:",
    final_df["project_code"].nunique()
)

print(
    "Physical progress available:",
    final_df["physical_progress"].notna().sum()
)

print(
    "Original cost available:",
    final_df["original_cost"].notna().sum()
)

print(
    "Revised cost available:",
    final_df["revised_cost"].notna().sum()
)

print(
    "Expenditure available:",
    final_df["expenditure"].notna().sum()
)

print("\nRecords by month:")
print(
    final_df["report_month"]
    .value_counts()
    .sort_index()
)

print("\nMissing values:")
print(
    final_df.isna().sum()
)

print("\nSaved to:")
print(output_file)

print("=" * 60)