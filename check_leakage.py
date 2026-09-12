"""Check project-level overlap in the current PAIMANA train/test split."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "leakage_check_report.csv"


def find_source_files():
    """Find CSVs that retain both a project identifier and a month column."""
    ignored_prefixes = (
        "X_",
        "y_",
        "catboost_",
        "lightgbm_",
        "xgboost_",
        "ensemble_",
        "random_forest_",
        "logistic_regression_",
        "feature_importance",
        "shap_",
    )
    source_files = []
    for path in ROOT.glob("*.csv"):
        if path.name.startswith(ignored_prefixes):
            continue
        try:
            columns = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        lower_columns = {str(column).lower() for column in columns}
        has_project_id = "project_code" in lower_columns or "project_id" in lower_columns
        has_month = any(
            "month" in column or "date" in column for column in lower_columns
        )
        if has_project_id and has_month:
            source_files.append(path.name)
    return sorted(source_files)


def main():
    source_files = find_source_files()
    feature_path = ROOT / "ML_FEATURE_DATASET.csv"
    if not feature_path.exists():
        raise FileNotFoundError("ML_FEATURE_DATASET.csv is required for this check.")

    source = pd.read_csv(feature_path)
    project_column = next(
        column
        for column in ("project_code", "project_id")
        if column in source.columns
    )
    month_column = next(
        column
        for column in ("report_month", "month", "Month")
        if column in source.columns
    )

    source[project_column] = source[project_column].astype(str).str.strip()
    source[month_column] = pd.to_datetime(source[month_column], errors="coerce")

    project_month_counts = source.groupby(project_column)[month_column].nunique()
    total_projects = source[project_column].nunique()
    multi_month_projects = int((project_month_counts > 1).sum())

    # This reproduces the split used to produce the current 4,751/1,833 files.
    train_rows = source[source[month_column] < pd.Timestamp("2026-05-01")]
    test_rows = source[source[month_column] == pd.Timestamp("2026-05-01")]
    train_projects = set(train_rows[project_column])
    test_projects = set(test_rows[project_column])
    overlap = train_projects.intersection(test_projects)

    x_train_rows = len(pd.read_csv(ROOT / "X_train.csv"))
    x_test_rows = len(pd.read_csv(ROOT / "X_test.csv"))
    split_matches_current_files = (
        len(train_rows) == x_train_rows and len(test_rows) == x_test_rows
    )

    report = pd.DataFrame(
        [
            ["project_id_column", project_column, "Identifier retained in ML_FEATURE_DATASET.csv"],
            ["month_column", month_column, "Monthly observation column"],
            ["source_files_with_project_and_month", len(source_files), "; ".join(source_files)],
            ["total_unique_projects", total_projects, "Across ML_FEATURE_DATASET.csv"],
            ["projects_with_multiple_months", multi_month_projects, "Projects with more than one distinct report month"],
            ["train_rows", len(train_rows), "Rows before 2026-05-01"],
            ["test_rows", len(test_rows), "Rows from 2026-05-01"],
            ["current_X_train_rows", x_train_rows, "Rows in X_train.csv"],
            ["current_X_test_rows", x_test_rows, "Rows in X_test.csv"],
            ["split_matches_current_files", split_matches_current_files, "Confirms source split corresponds to current files"],
            ["train_unique_projects", len(train_projects), "Unique IDs in inferred training partition"],
            ["test_unique_projects", len(test_projects), "Unique IDs in inferred test partition"],
            ["projects_overlapping_train_test", len(overlap), "Same project ID appears in both partitions"],
            ["project_level_leakage", bool(overlap), "True when any project crosses train/test"],
        ],
        columns=["Metric", "Value", "Details"],
    )
    report.to_csv(REPORT_PATH, index=False)

    print("=" * 70)
    print("PAIMANA PROJECT-LEVEL LEAKAGE CHECK")
    print("=" * 70)
    print(f"Project ID column: {project_column}")
    print(f"Monthly column   : {month_column}")
    print(f"Source CSVs found: {', '.join(source_files)}")
    print(f"Total unique projects: {total_projects}")
    print(f"Projects in multiple months: {multi_month_projects}")
    print(f"Projects overlapping train/test: {len(overlap)}")
    print(f"Current split matches X_train/X_test: {split_matches_current_files}")
    print(
        "\nCONCLUSION: PROJECT-LEVEL DATA LEAKAGE IS PRESENT."
        if overlap
        else "\nCONCLUSION: NO PROJECT-LEVEL OVERLAP WAS FOUND."
    )
    print(
        "The same project IDs occur in both partitions, so the current metrics "
        "may benefit from repeated monthly observations of known projects."
    )
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
