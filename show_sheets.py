import pandas as pd
import glob
import os

folder = r"C:\Users\Sinchana\OneDrive\Desktop\Data Set"

for file in glob.glob(os.path.join(folder, "*.xlsx")):
    print("\n" + "=" * 60)
    print(os.path.basename(file))
    print("=" * 60)

    sheets = pd.ExcelFile(file).sheet_names

    for sheet in sheets:
        print("-", sheet)