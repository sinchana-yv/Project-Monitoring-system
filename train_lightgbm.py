"""Train LightGBM on the same fixed split used by the XGBoost model."""

import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent


def clean_feature_name(name):
    """Apply the same feature-name cleaning used by train_xgboost.py."""
    name = str(name).replace("[", "(").replace("]", ")").replace("<", "_")
    return re.sub(r"[^A-Za-z0-9_().\-]", "_", name)


def make_unique(names):
    counts = {}
    unique_names = []
    for name in names:
        count = counts.get(name, 0)
        unique_names.append(name if count == 0 else f"{name}_duplicate_{count}")
        counts[name] = count + 1
    return unique_names


def load_data():
    """Load the existing train/test files without changing their split."""
    x_train = pd.read_csv(ROOT / "X_train.csv")
    x_test = pd.read_csv(ROOT / "X_test.csv")
    y_train = pd.read_csv(ROOT / "y_train.csv").squeeze()
    y_test = pd.read_csv(ROOT / "y_test.csv").squeeze()

    names = make_unique([clean_feature_name(column) for column in x_train.columns])
    x_train.columns = names
    x_test.columns = names

    x_train = x_train.replace([np.inf, -np.inf], np.nan)
    x_test = x_test.replace([np.inf, -np.inf], np.nan)

    categorical_columns = x_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_columns = [
        column for column in x_train.columns if column not in categorical_columns
    ]

    for column in numerical_columns:
        median = x_train[column].median()
        median = 0 if pd.isna(median) else median
        x_train[column] = x_train[column].fillna(median)
        x_test[column] = x_test[column].fillna(median)

    for column in categorical_columns:
        train_values = x_train[column].fillna("Unknown").astype(str)
        test_values = x_test[column].fillna("Unknown").astype(str)
        categories = pd.Index(train_values.unique())
        x_train[column] = pd.Categorical(train_values, categories=categories)
        x_test[column] = pd.Categorical(
            test_values.where(test_values.isin(categories), "Unknown"),
            categories=categories,
        )

    return x_train, x_test, y_train, y_test, categorical_columns


def calculate_metrics(y_true, y_pred, probabilities):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
    }


def main():
    print("=" * 60)
    print("LIGHTGBM COST OVERRUN MODEL")
    print("=" * 60)

    x_train, x_test, y_train, y_test, categorical_columns = load_data()
    print(f"\nTraining records: {len(x_train)}")
    print(f"Testing records : {len(x_test)}")
    print(f"Features        : {x_train.shape[1]}")
    print(f"Categorical features: {len(categorical_columns)}")

    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()
    scale_pos_weight = negative / positive
    print(f"No Overrun     : {negative}")
    print(f"Cost Overrun   : {positive}")
    print(f"Scale pos weight: {scale_pos_weight:.4f}")

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    print("\nTraining LightGBM model...")
    model.fit(x_train, y_train, categorical_feature=categorical_columns)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)
    metrics = calculate_metrics(y_test, predictions, probabilities)

    print("\n" + "=" * 60)
    print("LIGHTGBM RESULTS")
    print("=" * 60)
    for name, value in metrics.items():
        print(f"{name:10}: {value:.4f} ({value * 100:.2f}%)")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Overrun", "Cost Overrun"],
            zero_division=0,
        )
    )

    importance = pd.DataFrame(
        {"feature": x_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\nTop 20 important features:")
    print(importance.head(20).to_string(index=False))
    importance.to_csv(ROOT / "lightgbm_feature_importance.csv", index=False)

    pd.DataFrame(
        {
            "actual": y_test,
            "predicted": predictions,
            "risk_probability": probabilities,
        }
    ).to_csv(ROOT / "lightgbm_predictions.csv", index=False)
    model.booster_.save_model(str(ROOT / "lightgbm_cost_overrun_model.txt"))

    comparison = pd.DataFrame(
        [
            ["XGBoost", 0.8385, 0.6505, 0.8172, 0.7244, 0.9075],
            ["CatBoost", 0.8232, 0.6254, 0.7962, 0.7006, 0.8989],
            ["LightGBM", metrics["Accuracy"], metrics["Precision"], metrics["Recall"], metrics["F1"], metrics["ROC-AUC"]],
        ],
        columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
    )
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nFiles created:")
    print("  lightgbm_feature_importance.csv")
    print("  lightgbm_predictions.csv")
    print("  lightgbm_cost_overrun_model.txt")


if __name__ == "__main__":
    main()
