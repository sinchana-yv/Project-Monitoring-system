"""Train LightGBM on the leakage-safe project-disjoint datasets."""

import re
from pathlib import Path

import lightgbm as lgb
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


def main():
    print("=" * 70)
    print("LEAKAGE-SAFE LIGHTGBM COST OVERRUN MODEL")
    print("=" * 70)

    x_train = pd.read_csv(ROOT / "X_train_safe.csv")
    x_test = pd.read_csv(ROOT / "X_test_safe.csv")
    y_train = pd.read_csv(ROOT / "y_train_safe.csv").squeeze()
    y_test = pd.read_csv(ROOT / "y_test_safe.csv").squeeze()
    feature_names = make_unique([clean_feature_name(column) for column in x_train.columns])
    x_train.columns = feature_names
    x_test.columns = feature_names

    print(f"\nTraining records: {len(x_train)}")
    print(f"Testing records : {len(x_test)}")
    print(f"Features        : {x_train.shape[1]}")

    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()
    print(f"No Overrun     : {negative}")
    print(f"Cost Overrun   : {positive}")
    print(f"Class ratio    : {negative / positive:.4f}")

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

    print("\nTraining leakage-safe LightGBM model...")
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    print("\n" + "=" * 70)
    print("SAFE LIGHTGBM RESULTS")
    print("=" * 70)
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
    importance.to_csv(ROOT / "lightgbm_safe_feature_importance.csv", index=False)

    pd.DataFrame(
        {
            "actual": y_test,
            "predicted": predictions,
            "risk_probability": probabilities,
        }
    ).to_csv(ROOT / "lightgbm_safe_predictions.csv", index=False)
    model.booster_.save_model(str(ROOT / "lightgbm_safe_model.txt"))

    previous = {
        "Accuracy": 0.8811,
        "Precision": 0.7452,
        "Recall": 0.8235,
        "F1": 0.7824,
        "ROC-AUC": 0.9416,
    }
    comparison = pd.DataFrame(
        [previous, metrics],
        index=["Previous LightGBM", "Safe LightGBM"],
    )
    print("\n" + "=" * 70)
    print("PREVIOUS VS LEAKAGE-SAFE LIGHTGBM")
    print("=" * 70)
    print(comparison.to_string(float_format=lambda value: f"{value:.4f}"))

    print("\nConclusion:")
    if metrics["ROC-AUC"] > previous["ROC-AUC"]:
        print("The leakage-safe model performs better on ROC-AUC.")
    elif metrics["ROC-AUC"] < previous["ROC-AUC"]:
        print("The leakage-safe model performs worse on ROC-AUC.")
    else:
        print("The leakage-safe model has the same ROC-AUC.")
    print("The safe result is the more realistic estimate because projects do not overlap.")

    print("\nFiles created:")
    print("  lightgbm_safe_model.txt")
    print("  lightgbm_safe_predictions.csv")
    print("  lightgbm_safe_feature_importance.csv")


if __name__ == "__main__":
    main()
