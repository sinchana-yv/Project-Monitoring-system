import pandas as pd

print("=" * 60)
print("TARGET VALIDATION")
print("=" * 60)

df = pd.read_csv("ML_COST_OVERRUN_DATASET.csv")

df["report_month"] = pd.to_datetime(df["report_month"])

print("\nTotal records:", len(df))
print("Unique projects:", df["project_code"].nunique())

print("\nTarget distribution:")
print(df["future_cost_overrun"].value_counts())

print("\nTarget percentage:")
print(
    df["future_cost_overrun"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nRecords by report month:")
print(
    df.groupby("report_month")
      .size()
      .sort_index()
)

print("\nTarget by report month:")
print(
    pd.crosstab(
        df["report_month"],
        df["future_cost_overrun"]
    )
)

print("\nProjects with multiple labelled records:")

history = (
    df.groupby("project_code")
      .size()
      .value_counts()
      .sort_index()
)

print(history)

print("\n" + "=" * 60)
print("CHECKING SAME PROJECT ACROSS MONTHS")
print("=" * 60)

# Show a few projects with multiple observations
counts = df["project_code"].value_counts()

multi_projects = counts[counts >= 3].head(5).index

for project in multi_projects:

    temp = df[df["project_code"] == project].sort_values(
        "report_month"
    )

    print("\nProject:", project)

    cols = [
        "report_month",
        "original_cost",
        "revised_cost",
        "future_cost_overrun"
    ]

    print(temp[cols].to_string(index=False))

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)