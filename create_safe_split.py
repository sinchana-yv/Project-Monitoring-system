"""Create a project-disjoint, time-aware PAIMANA train/test dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
CUTOFF = pd.Timestamp("2026-05-01")


def main():
    source = pd.read_csv(ROOT / "ML_FEATURE_DATASET.csv")
    source["report_month"] = pd.to_datetime(source["report_month"], errors="coerce")
    source["project_code"] = source["project_code"].astype(str).str.strip()
    source = source.dropna(subset=["project_code", "report_month", "future_cost_overrun"])
    source["future_cost_overrun"] = source["future_cost_overrun"].astype(int)

    # Assign each project using its first observed month. This keeps every
    # monthly record for a project in exactly one partition.
    first_month = source.groupby("project_code")["report_month"].min()
    train_projects = set(first_month[first_month < CUTOFF].index)
    test_projects = set(first_month[first_month >= CUTOFF].index)

    train_df = source[source["project_code"].isin(train_projects)].copy()
    test_df = source[source["project_code"].isin(test_projects)].copy()

    overlap = train_projects.intersection(test_projects)
    if overlap:
        raise RuntimeError("Project overlap detected in the safe split.")

    target = "future_cost_overrun"
    columns_to_drop = [
        target,
        "revised_cost",
        "cost_growth",
        "has_current_revision",
        "project_code",
        "project_name",
        "report_month",
        "previous_report_month",
        "source_file",
        "source_sheet",
    ]
    columns_to_drop = [column for column in columns_to_drop if column in train_df.columns]

    y_train = train_df[target].reset_index(drop=True)
    y_test = test_df[target].reset_index(drop=True)
    x_train = train_df.drop(columns=columns_to_drop).reset_index(drop=True)
    x_test = test_df.drop(columns=columns_to_drop).reset_index(drop=True)

    date_columns = x_train.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    x_train = x_train.drop(columns=date_columns)
    x_test = x_test.drop(columns=date_columns)

    categorical_columns = x_train.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_columns = [column for column in x_train.columns if column not in categorical_columns]
    empty_numeric_columns = [
        column for column in numeric_columns if x_train[column].notna().sum() == 0
    ]
    if empty_numeric_columns:
        x_train = x_train.drop(columns=empty_numeric_columns)
        x_test = x_test.drop(columns=empty_numeric_columns)
        numeric_columns = [
            column for column in numeric_columns if column not in empty_numeric_columns
        ]

    for column in numeric_columns:
        x_train[column] = pd.to_numeric(x_train[column], errors="coerce")
        x_test[column] = pd.to_numeric(x_test[column], errors="coerce")
    x_train[numeric_columns] = x_train[numeric_columns].replace([np.inf, -np.inf], np.nan)
    x_test[numeric_columns] = x_test[numeric_columns].replace([np.inf, -np.inf], np.nan)

    if numeric_columns:
        numeric_imputer = SimpleImputer(strategy="median")
        x_train[numeric_columns] = numeric_imputer.fit_transform(x_train[numeric_columns])
        x_test[numeric_columns] = numeric_imputer.transform(x_test[numeric_columns])

    if categorical_columns:
        categorical_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
        train_categories = categorical_imputer.fit_transform(x_train[categorical_columns])
        test_categories = categorical_imputer.transform(x_test[categorical_columns])
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        train_encoded = encoder.fit_transform(train_categories)
        test_encoded = encoder.transform(test_categories)
        encoded_names = encoder.get_feature_names_out(categorical_columns)
        x_train = pd.concat(
            [x_train.drop(columns=categorical_columns),
             pd.DataFrame(train_encoded, columns=encoded_names)],
            axis=1,
        )
        x_test = pd.concat(
            [x_test.drop(columns=categorical_columns),
             pd.DataFrame(test_encoded, columns=encoded_names)],
            axis=1,
        )

    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
    x_train.to_csv(ROOT / "X_train_safe.csv", index=False)
    x_test.to_csv(ROOT / "X_test_safe.csv", index=False)
    y_train.to_csv(ROOT / "y_train_safe.csv", index=False)
    y_test.to_csv(ROOT / "y_test_safe.csv", index=False)

    train_dates = train_df["report_month"]
    test_dates = test_df["report_month"]
    print("=" * 70)
    print("LEAKAGE-SAFE PROJECT-DISJOINT SPLIT")
    print("=" * 70)
    print(f"Cutoff date: {CUTOFF.date()}")
    print(f"Training rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print(f"Unique projects in training: {len(train_projects)}")
    print(f"Unique projects in testing: {len(test_projects)}")
    print(f"Overlapping project IDs: {len(overlap)}")
    print(f"Training date range: {train_dates.min().date()} to {train_dates.max().date()}")
    print(f"Test date range: {test_dates.min().date()} to {test_dates.max().date()}")
    print(f"Training features: {x_train.shape[1]}")
    print("\nSaved:")
    print("  X_train_safe.csv")
    print("  X_test_safe.csv")
    print("  y_train_safe.csv")
    print("  y_test_safe.csv")


if __name__ == "__main__":
    main()
