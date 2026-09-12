import pandas as pd
import os

folder = "."

files = [f for f in os.listdir(folder) if f.endswith(".xlsx")]

print("\n========== PAIMANA DATASET INSPECTION ==========\n")

for file in files:
    path = os.path.join(folder, file)

    print("\n" + "=" * 70)
    print(f"FILE: {file}")
    print("=" * 70)

    try:
        excel_file = pd.ExcelFile(path)

        print("Sheets:")
        for sheet in excel_file.sheet_names:
            print(f"  - {sheet}")

            df = pd.read_excel(path, sheet_name=sheet)

            print(f"    Rows: {len(df)}")
            print(f"    Columns: {len(df.columns)}")
            print("    Column names:")

            for column in df.columns:
                print(f"      - {column}")

    except Exception as e:
        print(f"ERROR: {e}")

print("\n========== INSPECTION COMPLETE ==========\n")