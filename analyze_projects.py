import pandas as pd
import glob
import os
import re

# ============================================================
# SETTINGS
# ============================================================

FOLDER = r"C:\Users\Sinchana\OneDrive\Desktop\Data Set"

# ============================================================
# COLUMN NORMALIZATION
# ============================================================

COLUMN_MAP = {
    "Project Code": "project_code",
    "Project Name": "project_name",
    "State": "state",
    "Sector": "sector",
    "Ministry": "ministry",
    "Ministry/Department": "ministry",
    "Agency": "agency",
    "Implementing Agency": "agency",
    "Date of Approval": "approval_date",
    "Date of Approval (MM/YYYY)": "approval_date",
    "Date of Approval (Start Date)": "approval_date",
    "Approval Date (Start Date) MM/YYYY": "approval_date",

    "Start Date": "start_date",
    "Start Date (MM/YYYY)": "start_date",
    "Revised Start Date": "revised_start_date",

    "Commissioning Original": "original_doc",
    "Original/Target DoC": "original_doc",
    "Original Target DoC MM/YYYY": "original_doc",
    "Target DoC (Original)": "original_doc",
    "DoC - Original (MM/YYYY)": "original_doc",

    "Commissioning Revised": "revised_doc",
    "Revised DoC": "revised_doc",
    "Revised DoC (MM/YYYY)": "revised_doc",
    "Target DoC (Revised)": "revised_doc",
    "DoC - Revised (MM/YYYY)": "revised_doc",

    "Original Cost (Rs Cr)": "original_cost",
    "Original Cost (Rs. Crore)": "original_cost",
    "Original Cost (Rs Crore)": "original_cost",

    "Cost Original (Rs Cr)": "original_cost",

    "Revised Cost (Rs Cr)": "revised_cost",
    "Revised Cost (Rs. Crore)": "revised_cost",
    "Revised Cost (Rs Crore)": "revised_cost",

    "Cost Revised (Rs Cr)": "revised_cost",

    "Cumulative Expenditure (Rs Cr)": "expenditure",
    "Cumulative Expenditure (Rs. Crore)": "expenditure",

    "Physical Progress (%)": "physical_progress",
    "Progress (%)": "physical_progress",

    "Actual Date of Completion": "actual_completion",
    "Actual Date of Completion MM/YYYY": "actual_completion",
    "Actual/Revised Date of Completion (MM/YYYY)": "actual_completion",
}


# ============================================================
# FIND EXCEL FILES
# ============================================================

files = glob.glob(os.path.join(FOLDER, "*.xlsx"))

print("\n" + "=" * 70)
print("SIH26103 PROJECT DATA AUDIT")
print("=" * 70)

print(f"\nExcel files found: {len(files)}")

for f in files:
    print(" -", os.path.basename(f))


# ============================================================
# READ PROJECT SHEETS
# ============================================================

all_rows = []
completed_rows = []

for file in files:

    filename = os.path.basename(file)

    try:
        excel = pd.ExcelFile(file)

        for sheet in excel.sheet_names:

            # Read sheet
            df = pd.read_excel(file, sheet_name=sheet)

            # Normalize column names
            renamed = {}

            for col in df.columns:

                col_clean = str(col).strip()

                if col_clean in COLUMN_MAP:
                    renamed[col_clean] = COLUMN_MAP[col_clean]

            df = df.rename(columns=renamed)

            # Must contain Project Code
            if "project_code" not in df.columns:
                continue

            # Ignore empty rows
            df = df.dropna(subset=["project_code"])

            if len(df) == 0:
                continue

            # Identify completed sheets
            sheet_lower = sheet.lower()

            is_completed = (
                "completed" in sheet_lower
                or "completion" in sheet_lower
            )

            # Add source information
            df["source_file"] = filename
            df["source_sheet"] = sheet

            # Keep project data
            all_rows.append(df)

            if is_completed:
                completed_rows.append(df)

            print(
                f"\nFOUND: {filename} | {sheet} | "
                f"{len(df)} project rows"
            )

    except Exception as e:

        print(
            f"\nERROR reading {filename}: {e}"
        )


# ============================================================
# COMBINE DATA
# ============================================================

if not all_rows:

    print("\nNO PROJECT DATA FOUND.")
    exit()

projects = pd.concat(
    all_rows,
    ignore_index=True,
    sort=False
)

print("\n" + "=" * 70)
print("BASIC DATASET SUMMARY")
print("=" * 70)

print("\nTotal project records:", len(projects))

print(
    "Unique Project Codes:",
    projects["project_code"].nunique()
)


# ============================================================
# PROJECT FREQUENCY
# ============================================================

frequency = (
    projects
    .groupby("project_code")
    .size()
    .sort_values(ascending=False)
)

print("\n" + "=" * 70)
print("PROJECT OBSERVATION FREQUENCY")
print("=" * 70)

print(
    "\nProjects appearing 2+ times:",
    (frequency >= 2).sum()
)

print(
    "Projects appearing 5+ times:",
    (frequency >= 5).sum()
)

print(
    "Projects appearing 10+ times:",
    (frequency >= 10).sum()
)

print(
    "Projects appearing 12+ times:",
    (frequency >= 12).sum()
)


# ============================================================
# COST OVERRUN
# ============================================================

if "original_cost" in projects.columns and \
   "revised_cost" in projects.columns:

    projects["original_cost"] = pd.to_numeric(
        projects["original_cost"],
        errors="coerce"
    )

    projects["revised_cost"] = pd.to_numeric(
        projects["revised_cost"],
        errors="coerce"
    )

    valid_cost = projects.dropna(
        subset=["original_cost", "revised_cost"]
    )

    cost_overrun = (
        valid_cost["revised_cost"]
        > valid_cost["original_cost"]
    )

    print("\n" + "=" * 70)
    print("COST OVERRUN")
    print("=" * 70)

    print(
        "\nRecords with revised cost > original cost:",
        cost_overrun.sum()
    )

    print(
        "Percentage:",
        round(cost_overrun.mean() * 100, 2),
        "%"
    )


# ============================================================
# PHYSICAL PROGRESS
# ============================================================

if "physical_progress" in projects.columns:

    projects["physical_progress"] = pd.to_numeric(
        projects["physical_progress"],
        errors="coerce"
    )

    print("\n" + "=" * 70)
    print("PHYSICAL PROGRESS")
    print("=" * 70)

    print(
        "\nRecords with physical progress available:",
        projects["physical_progress"].notna().sum()
    )


# ============================================================
# EXPENDITURE
# ============================================================

if "expenditure" in projects.columns:

    projects["expenditure"] = pd.to_numeric(
        projects["expenditure"],
        errors="coerce"
    )

    print("\n" + "=" * 70)
    print("EXPENDITURE")
    print("=" * 70)

    print(
        "\nRecords with expenditure available:",
        projects["expenditure"].notna().sum()
    )


# ============================================================
# COMPLETED PROJECTS
# ============================================================

print("\n" + "=" * 70)
print("COMPLETED PROJECTS")
print("=" * 70)

if completed_rows:

    completed = pd.concat(
        completed_rows,
        ignore_index=True,
        sort=False
    )

    completed = completed.dropna(
        subset=["project_code"]
    )

    print(
        "\nCompleted project records:",
        len(completed)
    )

    print(
        "Unique completed projects:",
        completed["project_code"].nunique()
    )

else:

    print("\nNo completed project sheets detected.")


# ============================================================
# TOP PROJECTS WITH MOST MONTHLY RECORDS
# ============================================================

print("\n" + "=" * 70)
print("TOP PROJECTS BY NUMBER OF OBSERVATIONS")
print("=" * 70)

top_projects = frequency.head(20)

print("\n")

for code, count in top_projects.items():

    print(
        f"{str(code):30} → {count} records"
    )


# ============================================================
# SAVE FREQUENCY REPORT
# ============================================================

frequency_df = frequency.reset_index()

frequency_df.columns = [
    "project_code",
    "number_of_records"
]

output_file = os.path.join(
    FOLDER,
    "project_frequency_report.xlsx"
)

frequency_df.to_excel(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(
    "\nFrequency report saved to:",
    output_file
)