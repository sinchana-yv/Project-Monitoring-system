"""Tune the classification threshold using saved clean LightGBM probabilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parent


def main():
    predictions = pd.read_csv(ROOT / "lightgbm_clean_predictions.csv")
    actual = predictions["actual"].astype(int)
    probability = predictions["risk_probability"]
    thresholds = np.round(np.arange(0.20, 0.701, 0.05), 2)
    rows = []

    for threshold in thresholds:
        predicted = (probability >= threshold).astype(int)
        matrix = confusion_matrix(actual, predicted, labels=[0, 1])
        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(actual, predicted),
                "precision": precision_score(actual, predicted, zero_division=0),
                "recall": recall_score(actual, predicted, zero_division=0),
                "f1": f1_score(actual, predicted, zero_division=0),
                "tn": matrix[0, 0],
                "fp": matrix[0, 1],
                "fn": matrix[1, 0],
                "tp": matrix[1, 1],
                "confusion_matrix": str(matrix.tolist()),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "threshold_results.csv", index=False)

    # A balanced recommendation: maximize the harmonic mean of precision and recall.
    results["precision_recall_balance"] = (
        2 * results["precision"] * results["recall"]
        / (results["precision"] + results["recall"]).replace(0, np.nan)
    )
    recommended = results.loc[
        results["precision_recall_balance"].idxmax()
    ]

    print("=" * 70)
    print("CLEAN LIGHTGBM THRESHOLD TUNING")
    print("=" * 70)
    print(results.drop(columns="precision_recall_balance").to_string(index=False))
    print("\nRecommended threshold:")
    print(
        f"{recommended['threshold']:.2f} "
        f"(precision {recommended['precision']:.4f}, "
        f"recall {recommended['recall']:.4f}, "
        f"F1 {recommended['f1']:.4f})"
    )
    print(
        "This recommendation balances precision and recall using their harmonic "
        "mean, while retaining recall as the key safety metric."
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(results["threshold"], results["f1"], marker="o")
    axes[0].axvline(recommended["threshold"], color="red", linestyle="--")
    axes[0].set_title("Threshold vs F1-score")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("F1-score")
    axes[0].grid(alpha=0.3)

    axes[1].plot(results["threshold"], results["recall"], marker="o", color="darkorange")
    axes[1].axvline(recommended["threshold"], color="red", linestyle="--")
    axes[1].set_title("Threshold vs Recall")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Recall")
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(ROOT / "threshold_vs_f1_recall.png", dpi=150)
    plt.close(figure)
    print("\nSaved:")
    print("  threshold_results.csv")
    print("  threshold_vs_f1_recall.png")


if __name__ == "__main__":
    main()
