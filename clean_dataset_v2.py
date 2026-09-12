import pandas as pd
import os

folder = r"C:\Users\Sinchana\OneDrive\Desktop\Data Set"

# Exact sheets we want
sheets_to_use = {
    "JAN-FEB 2025.xlsx": [
        ("Table6_All_Ongoing_Projects", "2025-02")
    ],

    "March n Q1 2025.xlsx": [
        ("QPISR_T7_AllOngoing", "2025-03")
    ],

    "Apr-May 2025.xlsx": [
        ("All Ongoing Projects", "2025-05")
    ],

    "June-July 2025.xlsx": [
        ("June_Table7_All_Projects", "2025-06"),
        ("July_Table4_All_Projects", "2025-07")
    ],

    "Aug-Oct 2025.xlsx": [
        ("Aug - All Ongoing", "2025-08"),
        ("Oct - All Ongoing", "2025-10")
    ],

    "Nov-Dec 2025.xlsx": [
        ("November_T6_All_Projects", "2025-11"),
        ("December_T6_All_Projects", "2025-12")
    ],

    "April-May 2026.xlsx": [
        ("All Ongoing Projects", "2026-05")
    ],

    "June-July 2026.xlsx": [
        ("All Ongoing Projects", "2026-07")
    ]
}


def clean_number(x):
    if pd.isna(x):
        return None

    x = str(x).replace(",", "").replace("%", "").strip()

    try:
        return float(x)
    except:
        return None


all_data = []

for filename, sheets in sheets_to_use.items():

    filepath = os.path.join(folder, filename)

    print("\nProcessing:", filename)

    for sheet_name, report_month in sheets:

        print("  Sheet:", sheet_name, "->", report_month)

        df = pd.read_excel(filepath, sheet_name=sheet_name)

        # Clean column names
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("/", "_")
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )

        # Rename columns where possible
        rename_map = {
            "sl.no": "sl_no",
            "project_code": "project_code",
            "project_name": "project_name",
            "state": "state",
            "sector": "sector",
            "ministry": "ministry",
            "agency": "agency",
            "date_of_approval": "approval_date",
            "date_of_approval_mm_yyyy": "approval_date",
            "start_date_mm_yyyy": "start_date",
            "original_doc": "original_doc",
            "revised_doc": "revised_doc",
            "original_target_doc": "original_doc",
            "revised_doc": "revised_doc",
            "original_cost": "original_cost",
            "revised_cost": "revised_cost",
            "cumulative_expenditure": "expenditure",
            "physical_progress": "physical_progress",
            "legacy_ocms_code": "legacy_ocms_code",
            "pmgid": "pmgid"
        }

        df.rename(columns=rename_map, inplace=True)

        # Project code is essential
        if "project_code" not in df.columns:
            print("    SKIPPED - no project_code")
            continue

        # Convert project code to clean string
        df["project_code"] = (
            df["project_code"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

        # Remove empty/invalid project codes
        df = df[
            (df["project_code"] != "") &
            (df["project_code"].str.lower() != "nan")
        ]

        # Add report month
        df["report_month"] = report_month

        # Add source information
        df["source_file"] = filename
        df["source_sheet"] = sheet_name

        # Numeric columns
        numeric_columns = [
            "original_cost",
            "revised_cost",
            "expenditure",
            "physical_progress"
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_number)

        all_data.append(df)


# Combine everything
final_df = pd.concat(all_data, ignore_index=True)

print("\n======================================")
print("BEFORE DUPLICATE REMOVAL")
print("======================================")

print("Total records:", len(final_df))
print("Unique projects:", final_df["project_code"].nunique())


# Remove exact duplicate project/month observations
before = len(final_df)

final_df = final_df.drop_duplicates(
    subset=["project_code", "report_month"],
    keep="first"
)

after = len(final_df)

print("Duplicates removed:", before - after)


# Sort
final_df = final_df.sort_values(
    ["project_code", "report_month"]
)


# Save
output_file = os.path.join(
    folder,
    "clean_ongoing_projects_v2.csv"
)

final_df.to_csv(
    output_file,
    index=False
)


print("\n======================================")
print("FINAL DATASET")
print("======================================")

print("Records:", len(final_df))
print("Unique projects:", final_df["project_code"].nunique())

print("\nRecords by month:")
print(final_df["report_month"].value_counts().sort_index())

print("\nSaved to:")
print(output_file)