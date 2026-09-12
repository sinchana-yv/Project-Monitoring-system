"""Evaluate probability-averaging ensembles without retraining base models."""

from pathlib import Path

import matplotlib.pyplot as plt
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


def load_prediction_file(filename, actual_column, probability_column):
    """Load and standardize one saved model prediction file."""
    data = pd.read_csv(ROOT / filename)
    required = {actual_column, probability_column}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
    return data[[actual_column, probability_column]].rename(
        columns={actual_column: "file_actual", probability_column: "probability"}
    )


def calculate_metrics(actual, prediction, probability):
    return {
        "Accuracy": accuracy_score(actual, prediction),
        "Precision": precision_score(actual, prediction, zero_division=0),
        "Recall": recall_score(actual, prediction, zero_division=0),
        "F1": f1_score(actual, prediction, zero_division=0),
        "ROC_AUC": roc_auc_score(actual, probability),
    }


def evaluate(name, actual, probability):
    prediction = (probability >= 0.50).astype(int)
    metrics = calculate_metrics(actual, prediction, probability)
    print(f"\n{name}")
    print("-" * len(name))
    for metric, value in metrics.items():
        print(f"{metric:10}: {value:.4f} ({value * 100:.2f}%)")
    print("\nConfusion matrix:")
    print(confusion_matrix(actual, prediction))
    print("\nClassification report:")
    print(
        classification_report(
            actual,
            prediction,
            target_names=["No Overrun", "Cost Overrun"],
            zero_division=0,
        )
    )
    return metrics, prediction


def main():
    print("=" * 70)
    print("MODEL ENSEMBLE EVALUATION")
    print("=" * 70)

    y_test = pd.read_csv(ROOT / "y_test.csv").squeeze().reset_index(drop=True)
    xgboost = load_prediction_file(
        "xgboost_predictions.csv", "Actual", "Overrun_Probability"
    )
    catboost = load_prediction_file(
        "catboost_predictions.csv", "actual", "risk_probability"
    )
    lightgbm = load_prediction_file(
        "lightgbm_predictions.csv", "actual", "risk_probability"
    )

    predictions = pd.DataFrame(
        {
            "actual_target": y_test,
            "xgboost_probability": xgboost["probability"],
            "catboost_probability": catboost["probability"],
            "lightgbm_probability": lightgbm["probability"],
        }
    )
    if len(predictions) != 1833 or not (
        len(xgboost) == len(catboost) == len(lightgbm) == len(y_test)
    ):
        raise ValueError("Prediction files and y_test.csv are not row-aligned.")
    for filename, actual_values in (
        ("xgboost_predictions.csv", xgboost["file_actual"]),
        ("catboost_predictions.csv", catboost["file_actual"]),
        ("lightgbm_predictions.csv", lightgbm["file_actual"]),
    ):
        if not actual_values.astype(int).equals(y_test.astype(int)):
            raise ValueError(f"{filename} actual targets do not match y_test.csv.")
    print(f"\nVerified {len(predictions)} aligned test records.")

    predictions["soft_voting_probability"] = predictions[
        ["xgboost_probability", "catboost_probability", "lightgbm_probability"]
    ].mean(axis=1)
    predictions["weighted_ensemble_probability"] = (
        0.30 * predictions["xgboost_probability"]
        + 0.20 * predictions["catboost_probability"]
        + 0.50 * predictions["lightgbm_probability"]
    )

    results = {}
    results["XGBoost"], _ = evaluate(
        "XGBoost", y_test, predictions["xgboost_probability"]
    )
    results["CatBoost"], _ = evaluate(
        "CatBoost", y_test, predictions["catboost_probability"]
    )
    results["LightGBM"], _ = evaluate(
        "LightGBM", y_test, predictions["lightgbm_probability"]
    )
    results["Simple Soft Voting"], soft_prediction = evaluate(
        "Simple Soft Voting", y_test, predictions["soft_voting_probability"]
    )
    results["Weighted Ensemble"], weighted_prediction = evaluate(
        "Weighted Ensemble", y_test, predictions["weighted_ensemble_probability"]
    )

    predictions["soft_voting_prediction"] = soft_prediction
    predictions["weighted_ensemble_prediction"] = weighted_prediction
    predictions.to_csv(ROOT / "ensemble_predictions.csv", index=False)

    comparison = pd.DataFrame(results).T.reset_index(names="Model")
    comparison.to_csv(ROOT / "ensemble_model_comparison.csv", index=False)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    plot_data = comparison.set_index("Model")[["ROC_AUC", "F1"]]
    ax = plot_data.plot(kind="bar", figsize=(10, 6), rot=20)
    ax.set_title("Model ROC-AUC and F1-score Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    figure = ax.get_figure()
    figure.tight_layout()
    figure.savefig(ROOT / "ensemble_model_comparison.png", dpi=150)
    plt.close(figure)

    best = comparison.sort_values(["ROC_AUC", "F1"], ascending=False).iloc[0]
    print(
        "\nFinal conclusion: Based on the same test dataset, the best-performing "
        f"model is {best['Model']} because it achieved "
        f"{best['ROC_AUC']:.4f} ROC-AUC and {best['F1']:.4f} F1-score."
    )
    print(
        "\nLeakage warning: PAIMANA data commonly contains repeated monthly "
        "observations for the same project. The current transformed X_test has "
        "no project-code or report-month identifier, so this experiment cannot "
        "measure project overlap directly. Features such as months_observed, "
        "days_since_first_report, previous_progress, and previous_expenditure "
        "show that temporal history is represented. A later leakage-safe "
        "validation should split by project before feature engineering, and "
        "should use a time-ordered holdout so no project's future observations "
        "appear in training."
    )


if __name__ == "__main__":
    main()
