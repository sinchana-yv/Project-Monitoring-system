import pandas as pd
import os

print("\n========== PROJECT SHEET SEARCH ==========\n")

files = [f for f in os.listdir(".") if f.lower().endswith(".xlsx")]

for file in files:

    print("\n" + "=" * 70)
    print("FILE:", file)
    print("=" * 70)

    try:
        excel = pd.ExcelFile(file)

        for sheet in excel.sheet_names:

            # Read only the first few rows
            df = pd.read_excel(file, sheet_name=sheet, nrows=3)

            columns = " ".join(str(col).lower() for col in df.columns)

            # Detect project-level sheets based on important columns
            if (
                "project code" in columns
                and "project name" in columns
                and (
                    "physical progress" in columns
                    or "revised cost" in columns
                    or "original cost" in columns
                )
            ):

                print("\n⭐ PROJECT DATA SHEET FOUND")
                print("Sheet:", sheet)
                print("Columns:", len(df.columns))

                print("\nColumn names:")
                for col in df.columns:
                    print("  -", col)

                print()

    except Exception as e:
        print("ERROR:", e)

print("\n========== SEARCH COMPLETE ==========\n")