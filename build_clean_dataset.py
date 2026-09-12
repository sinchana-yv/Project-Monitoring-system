import pandas as pd
import glob
import os
import re

# ============================================================
# SETTINGS
# ============================================================

FOLDER = r"C:\Users\Sinchana\OneDrive\Desktop\Data Set"

OUTPUT_ONGOING = os.path.join(
    FOLDER,
    "clean_ongoing_projects.csv"
)

OUTPUT_COMPLETED = os.path.join(
    FOLDER,
    "clean_completed_projects.csv"
)

# ============================================================
# COLUMN STANDARDIZATION
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

    "Revised Start Date (MM/YYYY)": "revised_start_date",

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
# HELPER FUNCTION
# ============================================================

def standardize_columns(df):

    renamed = {}

    for col in df.columns:

        col = str(col).strip()

        if col in COLUMN_MAP:
            renamed[col] = COLUMN_MAP[col]

    return df.rename(columns=renamed)


# ============================================================
# IDENTIFY REPORT MONTH
# ============================================================

def get_report_month(filename, sheet):

    text = (filename + " " + sheet).lower()

    # 2026
    if "april-may 2026" in text:
        return "2026-05"

    if "june-july 2026" in text:
        return "2026-07"

    # 2025
    if "jan-feb 2025" in text:
        return "2025-02"

    if "march" in text and "q1" in text:
        return "2025-03"

    if "apr-may 2025" in text:
        return "2025-05"

    if "june-july 2025" in text:
        return "2025-07"

    if "aug-oct 2025" in text:

        if "oct" in sheet.lower():
            return "2025-10"

        return "2025-08"

    if "nov-dec 2025" in text:

        if "december" in sheet.lower():
            return "2025-12"

        return "2025-11"

    return None


# ============================================================
# FIND EXCEL FILES
# ============================================================

files = glob.glob(
    os.path.join(FOLDER, "*.xlsx")
)

# Don't accidentally read generated report files
files = [
    f for f in files
    if "frequency" not in os.path.basename(f).lower()
]

print("\n" + "=" * 70)
print("SIH26103 CLEAN DATASET BUILDER")
print("=" * 70)

print("\nExcel files:", len(files))


ongoing_data = []
completed_data = []


# ============================================================
# READ FILES
# ============================================================

for file in files:

    filename = os.path.basename(file)

    print("\n" + "-" * 70)
    print("FILE:", filename)
    print("-" * 70)

    try:

        excel = pd.ExcelFile(file)

        for sheet in excel.sheet_names:

            sheet_lower = sheet.lower()

            # ------------------------------------------------
            # ONLY SELECT MAIN ONGOING SHEETS
            # ------------------------------------------------

            is_ongoing = (
                "all ongoing" in sheet_lower
                or "all_projects" in sheet_lower
                or "all_projects" in sheet_lower
                or "allongoing" in sheet_lower
            )

            # ------------------------------------------------
            # ONLY SELECT COMPLETED SHEETS
            # ------------------------------------------------

            is_completed = (
                "completed" in sheet_lower
            )

            # Skip everything else
            if not is_ongoing and not is_completed:
                continue

            # Skip duplicate combined sheets
            if "combined" in sheet_lower:
                print("SKIP:", sheet, "(combined duplicate)")
                continue

            # Read
            df = pd.read_excel(
                file,
                sheet_name=sheet
            )

            # Standardize
            df = standardize_columns(df)

            # Project code is mandatory
            if "project_code" not in df.columns:

                print(
                    "SKIP:",
                    sheet,
                    "(no Project Code)"
                )

                continue

            # Remove empty project codes
            df = df.dropna(
                subset=["project_code"]
            )

            # Add report month
            report_month = get_report_month(
                filename,
                sheet
            )

            df["report_month"] = report_month

            # Source information
            df["source_file"] = filename
            df["source_sheet"] = sheet

            # -----------------------------------------------
            # PROJECT CODE CLEANING
            # -----------------------------------------------

            df["project_code"] = (
                df["project_code"]
                .astype(str)
                .str.strip()
            )

            # Remove Excel-style .0
            df["project_code"] = (
                df["project_code"]
                .str.replace(
                    r"\.0$",
                    "",
                    regex=True
                )
            )

            # -----------------------------------------------
            # NUMERIC CLEANING
            # -----------------------------------------------

            for col in [
                "original_cost",
                "revised_cost",
                "expenditure",
                "physical_progress"
            ]:

                if col in df.columns:

                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(
                            ",",
                            "",
                            regex=False
                        )
                        .str.replace(
                            "%",
                            "",
                            regex=False
                        )
                        .str.strip()
                    )

                    df[col] = pd.to_numeric(
                        df[col],
                        errors="coerce"
                    )

            # -----------------------------------------------
            # ADD TO CORRECT DATASET
            # -----------------------------------------------

            if is_completed:

                completed_data.append(df)

                print(
                    "KEEP COMPLETED:",
                    sheet,
                    "→",
                    len(df),
                    "rows"
                )

            else:

                ongoing_data.append(df)

                print(
                    "KEEP ONGOING:",
                    sheet,
                    "→",
                    len(df),
                    "rows"
                )

    except Exception as e:

        print(
            "ERROR:",
            filename,
            e
        )


# ============================================================
# COMBINE ONGOING
# ============================================================

if ongoing_data:

    ongoing = pd.concat(
        ongoing_data,
        ignore_index=True,
        sort=False
    )

else:

    ongoing = pd.DataFrame()


# ============================================================
# COMBINE COMPLETED
# ============================================================

if completed_data:

    completed = pd.concat(
        completed_data,
        ignore_index=True,
        sort=False
    )

else:

    completed = pd.DataFrame()


# ============================================================
# REMOVE DUPLICATE ONGOING OBSERVATIONS
# ============================================================

if not ongoing.empty:

    before = len(ongoing)

    ongoing = ongoing.drop_duplicates(
        subset=[
            "project_code",
            "report_month"
        ],
        keep="last"
    )

    after = len(ongoing)

    print("\n" + "=" * 70)
    print("ONGOING DUPLICATE REMOVAL")
    print("=" * 70)

    print(
        "\nBefore:",
        before
    )

    print(
        "After:",
        after
    )

    print(
        "Duplicates removed:",
        before - after
    )


# ============================================================
# REMOVE DUPLICATE COMPLETED OBSERVATIONS
# ============================================================

if not completed.empty:

    before = len(completed)

    completed = completed.drop_duplicates(
        subset=[
            "project_code",
            "report_month"
        ],
        keep="last"
    )

    after = len(completed)

    print("\n" + "=" * 70)
    print("COMPLETED DUPLICATE REMOVAL")
    print("=" * 70)

    print(
        "\nBefore:",
        before
    )

    print(
        "After:",
        after
    )

    print(
        "Duplicates removed:",
        before - after
    )


# ============================================================
# SORT
# ============================================================

if not ongoing.empty:

    ongoing = ongoing.sort_values(
        by=[
            "project_code",
            "report_month"
        ]
    )

if not completed.empty:

    completed = completed.sort_values(
        by=[
            "project_code",
            "report_month"
        ]
    )


# ============================================================
# SAVE
# ============================================================

if not ongoing.empty:

    ongoing.to_csv(
        OUTPUT_ONGOING,
        index=False
    )

    print(
        "\nClean ongoing dataset saved:"
    )

    print(
        OUTPUT_ONGOING
    )


if not completed.empty:

    completed.to_csv(
        OUTPUT_COMPLETED,
        index=False
    )

    print(
        "\nClean completed dataset saved:"
    )

    print(
        OUTPUT_COMPLETED
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEAN DATASET SUMMARY")
print("=" * 70)

if not ongoing.empty:

    print(
        "\nONGOING"
    )

    print(
        "Records:",
        len(ongoing)
    )

    print(
        "Unique projects:",
        ongoing["project_code"].nunique()
    )

    print(
        "Months:",
        ongoing["report_month"].dropna().unique()
    )

if not completed.empty:

    print(
        "\nCOMPLETED"
    )

    print(
        "Records:",
        len(completed)
    )

    print(
        "Unique projects:",
        completed["project_code"].nunique()
    )

print(
    "\n" + "=" * 70
)

print(
    "CLEANING COMPLETE"
)

print(
    "=" * 70
)