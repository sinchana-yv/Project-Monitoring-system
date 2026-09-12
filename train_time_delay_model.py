"""Train and audit a baseline LightGBM current schedule-slippage model."""

from pathlib import Path
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "DELAY_FEATURE_DATASET.csv"
MODEL_OUTPUT = ROOT / "time_delay_model.txt"
AUDIT_OUTPUT = ROOT / "time_delay_model_audit.txt"
NUMERIC_IMPUTER_OUTPUT = ROOT / "time_delay_numeric_imputer.pkl"
CATEGORICAL_IMPUTER_OUTPUT = ROOT / "time_delay_categorical_imputer.pkl"
ENCODER_OUTPUT = ROOT / "time_delay_onehot_encoder.pkl"
FEATURE_NAMES_OUTPUT = ROOT / "time_delay_feature_names.pkl"
METADATA_OUTPUT = ROOT / "time_delay_preprocessing_metadata.pkl"

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
FORBIDDEN_COLUMNS = {
    "progress_gap",
    "expected_progress",
    "planned_duration_days",
    "planned_duration_days_raw",
    "elapsed_duration_days",
    "time_delay_risk",
    "actual_completion",
    "actual_completion_date",
    "revised_completion",
    "revised_completion_date",
    "future_completion",
    "future_progress",
    "future_expenditure",
    "future_cost",
    "final_project_status",
}


def metric_row(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }


def main():
    data = pd.read_csv(INPUT, low_memory=False)
    valid = data[
        data["schedule_valid"].astype(str).str.lower() == "true"
    ].copy()
    valid["time_delay_risk"] = (
        pd.to_numeric(valid["progress_gap"], errors="coerce") < -20
    ).astype(int)

    missing_features = [
        column
        for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES
        if column not in valid.columns
    ]
    if missing_features:
        raise RuntimeError(
            "Required baseline features are missing: "
            + ", ".join(missing_features)
        )

    leakage_used = sorted(
        FORBIDDEN_COLUMNS.intersection(
            NUMERIC_FEATURES + CATEGORICAL_FEATURES
        )
    )
    if leakage_used:
        raise RuntimeError(
            "Forbidden target-defining columns selected: "
            + ", ".join(leakage_used)
        )

    x = valid[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = valid["time_delay_risk"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(
        strategy="most_frequent",
        missing_values=np.nan,
    )
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )
    numeric_train = numeric_imputer.fit_transform(x_train[NUMERIC_FEATURES])
    numeric_test = numeric_imputer.transform(x_test[NUMERIC_FEATURES])
    categorical_train = categorical_imputer.fit_transform(
        x_train[CATEGORICAL_FEATURES]
    )
    categorical_test = categorical_imputer.transform(
        x_test[CATEGORICAL_FEATURES]
    )
    encoded_train = encoder.fit_transform(categorical_train)
    encoded_test = encoder.transform(categorical_test)
    encoded_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names = NUMERIC_FEATURES + encoded_names.tolist()
    x_train_final = np.column_stack([numeric_train, encoded_train])
    x_test_final = np.column_stack([numeric_test, encoded_test])

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train_final, y_train)
    probabilities = model.predict_proba(x_test_final)[:, 1]
    roc_auc = roc_auc_score(y_test, probabilities)
    threshold_results = [
        metric_row(y_test, probabilities, threshold)
        for threshold in [
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
        ]
    ]
    recommended = max(
        threshold_results,
        key=lambda result: (
            result["f1"],
            result["recall"],
            -abs(result["threshold"] - 0.50),
        ),
    )
    recommended_predictions = (
        probabilities >= recommended["threshold"]
    ).astype(int)
    cm = confusion_matrix(y_test, recommended_predictions)
    report = classification_report(
        y_test,
        recommended_predictions,
        target_names=["not_delayed", "delayed"],
        zero_division=0,
    )
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)
    performance_warning = (
        recommended["accuracy"] >= 0.995
        or recommended["f1"] >= 0.995
        or roc_auc >= 0.995
    )

    model.booster_.save_model(str(MODEL_OUTPUT))
    joblib.dump(numeric_imputer, NUMERIC_IMPUTER_OUTPUT)
    joblib.dump(categorical_imputer, CATEGORICAL_IMPUTER_OUTPUT)
    joblib.dump(encoder, ENCODER_OUTPUT)
    joblib.dump(feature_names, FEATURE_NAMES_OUTPUT)
    joblib.dump(
        {
            "target": "time_delay_risk",
            "target_definition": "progress_gap < -20",
            "numeric_columns": NUMERIC_FEATURES,
            "categorical_columns": CATEGORICAL_FEATURES,
            "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
            "threshold": recommended["threshold"],
        },
        METADATA_OUTPUT,
    )

    lines = [
        "TIME DELAY / SCHEDULE SLIPPAGE LIGHTGBM MODEL AUDIT",
        "=" * 78,
        "Interpretation: current schedule slippage risk, not confirmed eventual",
        "completion-delay prediction.",
        "",
        "TARGET",
        "------",
        "time_delay_risk = 1 if progress_gap < -20 else 0",
        f"Valid rows used: {len(valid)}",
        f"Delayed rows: {int(y.sum())}",
        f"Not-delayed rows: {int((y == 0).sum())}",
        "",
        "FEATURES USED",
        "-------------",
        "Numeric: " + ", ".join(NUMERIC_FEATURES),
        "Categorical: " + ", ".join(CATEGORICAL_FEATURES),
        "Raw date strings were not passed to LightGBM.",
        "",
        "LEAKAGE CHECK",
        "-------------",
        "Target-defining columns selected: "
        + (", ".join(leakage_used) if leakage_used else "none"),
        "Target leakage check: PASSED",
        "The target formula columns were used only to construct y and were not",
        "included in X. Actual/revised/future completion and future measurements",
        "were not used.",
        f"Suspicious-performance check: {'WARNING' if performance_warning else 'PASSED'}",
        "",
        "DATA SPLIT",
        "----------",
        f"Training rows: {len(x_train)}",
        f"Testing rows: {len(x_test)}",
        f"Training delayed/not-delayed: {int(y_train.sum())}/{int((y_train == 0).sum())}",
        f"Testing delayed/not-delayed: {int(y_test.sum())}/{int((y_test == 0).sum())}",
        "Split: stratified 80/20, random_state=42",
        "",
        "MODEL",
        "-----",
        "LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,",
        "max_depth=-1, subsample=0.8, colsample_bytree=0.8,",
        "class_weight='balanced', random_state=42, n_jobs=-1)",
        "",
        "RECOMMENDED THRESHOLD METRICS",
        "-----------------------------",
        f"Threshold: {recommended['threshold']:.2f}",
        f"Accuracy: {recommended['accuracy']:.4f}",
        f"Precision: {recommended['precision']:.4f}",
        f"Recall: {recommended['recall']:.4f}",
        f"F1: {recommended['f1']:.4f}",
        f"ROC-AUC: {roc_auc:.4f}",
        f"Confusion matrix [[TN, FP], [FN, TP]]: {cm.tolist()}",
        "",
        "CLASSIFICATION REPORT",
        "---------------------",
        report,
        "THRESHOLD COMPARISON",
        "--------------------",
    ]
    for result in threshold_results:
        lines.append(
            f"{result['threshold']:.2f}: accuracy={result['accuracy']:.4f}, "
            f"precision={result['precision']:.4f}, "
            f"recall={result['recall']:.4f}, f1={result['f1']:.4f}"
        )
    lines.extend(
        [
            "",
            "TOP FEATURE IMPORTANCES",
            "-----------------------",
        ]
    )
    lines.extend(
        f"{feature}: {value:.4f}"
        for feature, value in importance.head(20).items()
    )
    lines.extend(
        [
            "",
            "ARTIFACTS",
            "---------",
            MODEL_OUTPUT.name,
            NUMERIC_IMPUTER_OUTPUT.name,
            CATEGORICAL_IMPUTER_OUTPUT.name,
            ENCODER_OUTPUT.name,
            FEATURE_NAMES_OUTPUT.name,
            METADATA_OUTPUT.name,
        ]
    )
    AUDIT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
