"""Validate the saved schedule-slippage model on July 2026 data."""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "JULY_2026_FEATURE_DATASET.csv"
MODEL_FILE = ROOT / "time_delay_model.txt"
NUMERIC_IMPUTER_FILE = ROOT / "time_delay_numeric_imputer.pkl"
CATEGORICAL_IMPUTER_FILE = ROOT / "time_delay_categorical_imputer.pkl"
ENCODER_FILE = ROOT / "time_delay_onehot_encoder.pkl"
FEATURE_NAMES_FILE = ROOT / "time_delay_feature_names.pkl"
OUTPUT = ROOT / "JULY_2026_TIME_DELAY_PREDICTIONS.csv"
AUDIT = ROOT / "time_delay_july_validation.txt"

NUMERIC_FEATURES = [
    "physical_progress",
    "previous_progress",
    "progress_change",
    "progress_velocity",
    "expenditure",
    "expenditure_ratio",
    "previous_expenditure",
    "expenditure_change",
    "expenditure_velocity",
    "original_cost",
    "months_observed",
    "days_since_first_report",
    "has_physical_progress",
    "has_expenditure",
]
CATEGORICAL_FEATURES = [
    "state",
    "sector",
    "ministry_final",
    "agency_final",
]
FORBIDDEN_FIELDS = [
    "progress_gap",
    "expected_progress",
    "planned_duration_days",
    "planned_duration_days_raw",
    "elapsed_duration_days",
    "actual_completion",
    "actual_completion_date",
    "revised_completion",
    "revised_completion_date",
    "future_completion",
    "future_progress",
    "future_expenditure",
    "future_cost",
    "final_project_status",
]


def risk_level(probability):
    if probability >= 0.70:
        return "HIGH"
    if probability >= 0.40:
        return "MEDIUM"
    return "LOW"


def main():
    data = pd.read_csv(INPUT, low_memory=False)
    missing = [
        field
        for field in NUMERIC_FEATURES + CATEGORICAL_FEATURES
        if field not in data.columns
    ]
    if missing:
        raise RuntimeError(
            "July dataset is missing model features: " + ", ".join(missing)
        )

    forbidden_present = sorted(
        set(FORBIDDEN_FIELDS).intersection(data.columns)
    )
    x = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for field in NUMERIC_FEATURES:
        x[field] = pd.to_numeric(x[field], errors="coerce")

    numeric_imputer = joblib.load(NUMERIC_IMPUTER_FILE)
    categorical_imputer = joblib.load(CATEGORICAL_IMPUTER_FILE)
    encoder = joblib.load(ENCODER_FILE)
    feature_names = joblib.load(FEATURE_NAMES_FILE)
    model = lgb.Booster(model_file=str(MODEL_FILE))

    numeric_values = numeric_imputer.transform(x[NUMERIC_FEATURES])
    categorical_values = categorical_imputer.transform(
        x[CATEGORICAL_FEATURES]
    )
    encoded_values = encoder.transform(categorical_values)
    encoded_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    transformed_names = NUMERIC_FEATURES + encoded_names.tolist()
    transformed = np.column_stack([numeric_values, encoded_values])
    transformed = pd.DataFrame(
        transformed,
        columns=transformed_names,
    ).reindex(columns=feature_names, fill_value=0)

    if transformed.shape[1] != len(model.feature_name()):
        raise RuntimeError(
            "Feature compatibility failed: transformed columns do not match "
            "the saved LightGBM model."
        )

    probabilities = np.asarray(model.predict(transformed), dtype=float)
    predictions = (probabilities >= 0.50).astype(int)
    output = data[
        [
            column
            for column in [
                "project_code",
                "project_name",
                "report_month",
                "state",
                "sector",
            ]
            if column in data.columns
        ]
    ].copy()
    output["time_delay_probability"] = probabilities
    output["time_delay_probability_percent"] = probabilities * 100
    output["time_delay_prediction"] = np.where(
        predictions == 1,
        "Current Schedule Slippage Risk",
        "No Current Schedule Slippage Risk",
    )
    output["time_delay_risk_level"] = [
        risk_level(probability)
        for probability in probabilities
    ]
    output.to_csv(OUTPUT, index=False)

    counts = output["time_delay_risk_level"].value_counts()
    audit_lines = [
        "JULY 2026 TIME DELAY MODEL VALIDATION",
        "=" * 78,
        "Model interpretation: current schedule slippage risk, not confirmed",
        "eventual completion-delay prediction.",
        "",
        "ARTIFACTS USED",
        "---------------",
        MODEL_FILE.name,
        NUMERIC_IMPUTER_FILE.name,
        CATEGORICAL_IMPUTER_FILE.name,
        ENCODER_FILE.name,
        FEATURE_NAMES_FILE.name,
        "Preprocessing was loaded from saved artifacts and was not refit.",
        "",
        "DATA",
        "----",
        f"Total July rows: {len(data)}",
        f"Valid projects predicted: {len(output)}",
        f"Report months: {sorted(data['report_month'].dropna().unique().tolist()) if 'report_month' in data else []}",
        "",
        "RISK LEVEL COUNTS",
        "-----------------",
        f"HIGH (>= 0.70): {int(counts.get('HIGH', 0))}",
        f"MEDIUM (0.40-0.69): {int(counts.get('MEDIUM', 0))}",
        f"LOW (< 0.40): {int(counts.get('LOW', 0))}",
        "",
        "PROBABILITY SUMMARY",
        "-------------------",
        f"Minimum: {probabilities.min():.6f}",
        f"Maximum: {probabilities.max():.6f}",
        f"Average: {probabilities.mean():.6f}",
        f"Threshold used: 0.50",
        f"Predicted current slippage risk: {int(predictions.sum())}",
        f"Predicted no current slippage risk: {int((predictions == 0).sum())}",
        "",
        "FEATURE COMPATIBILITY",
        "---------------------",
        f"Required numeric features present: {len(NUMERIC_FEATURES)}",
        f"Required categorical features present: {len(CATEGORICAL_FEATURES)}",
        f"Transformed feature count: {transformed.shape[1]}",
        f"Saved model feature count: {len(model.feature_name())}",
        "Feature compatibility: PASSED",
        "Raw date strings were not used as model inputs.",
        "",
        "LEAKAGE CHECK",
        "-------------",
        "Forbidden target-defining fields present in source: "
        + (", ".join(forbidden_present) if forbidden_present else "none"),
        "Forbidden fields were not selected or transformed.",
        "Actual/revised completion and future information were not used.",
        "Target leakage check: PASSED",
        "",
        "OUTPUT",
        "------",
        OUTPUT.name,
    ]
    AUDIT.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
