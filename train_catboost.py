"""Train and evaluate CatBoost using the same split as train_xgboost.py."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
    """Use the same feature-name cleaning as the existing XGBoost script."""
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
    """Load the exact train/test files used by the existing XGBoost workflow."""
    x_train = pd.read_csv(ROOT / "X_train.csv")
    x_test = pd.read_csv(ROOT / "X_test.csv")
    y_train = pd.read_csv(ROOT / "y_train.csv").squeeze()
    y_test = pd.read_csv(ROOT / "y_test.csv").squeeze()

    feature_names = make_unique(
        [clean_feature_name(column) for column in x_train.columns]
    )
    x_train.columns = feature_names
    x_test.columns = feature_names

    x_train = x_train.replace([np.inf, -np.inf], np.nan)
    x_test = x_test.replace([np.inf, -np.inf], np.nan)

    categorical_columns = x_train.select_dtypes(include=["object", "category"]).columns.tolist()
    for column in categorical_columns:
        x_train[column] = x_train[column].fillna("Unknown").astype(str)
        x_test[column] = x_test[column].fillna("Unknown").astype(str)

    numerical_columns = [column for column in x_train.columns if column not in categorical_columns]
    for column in numerical_columns:
        median = x_train[column].median()
        median = 0 if pd.isna(median) else median
        x_train[column] = x_train[column].fillna(median)
        x_test[column] = x_test[column].fillna(median)

    return x_train, x_test, y_train, y_test, categorical_columns


def main():
    print("=" * 60)
    print("CATBOOST COST OVERRUN MODEL")
    print("=" * 60)

    x_train, x_test, y_train, y_test, categorical_columns = load_data()
    print(f"\nTraining records: {len(x_train)}")
    print(f"Testing records : {len(x_test)}")
    print(f"Features        : {x_train.shape[1]}")
    print(f"Categorical features: {len(categorical_columns)}")

    model = CatBoostClassifier(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        verbose=100,
        thread_count=-1,
        auto_class_weights="Balanced",
    )

    print("\nTraining CatBoost model...")
    model.fit(x_train, y_train, cat_features=categorical_columns)

    y_probability = model.predict_proba(x_test)[:, 1]
    y_pred = (y_probability >= 0.50).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_probability),
    }

    print("\n" + "=" * 60)
    print("CATBOOST RESULTS")
    print("=" * 60)
    for name, value in metrics.items():
        print(f"{name:10}: {value:.4f} ({value * 100:.2f}%)")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Overrun", "Cost Overrun"],
            zero_division=0,
        )
    )

    importance = pd.DataFrame(
        {"feature": x_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(ROOT / "catboost_feature_importance.csv", index=False)

    predictions = pd.DataFrame(
        {
            "actual": y_test,
            "predicted": y_pred,
            "risk_probability": y_probability,
        }
    )
    predictions.to_csv(ROOT / "catboost_predictions.csv", index=False)
    model.save_model(ROOT / "catboost_cost_overrun_model.cbm")

    print("\nFiles created:")
    print("  catboost_feature_importance.csv")
    print("  catboost_predictions.csv")
    print("  catboost_cost_overrun_model.cbm")


if __name__ == "__main__":
    main()
