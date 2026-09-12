"""Explain the clean LightGBM model with SHAP on the safe test data."""

import re
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parent


def clean_feature_name(name):
    name = str(name).replace("[", "(").replace("]", ")").replace("<", "_")
    return re.sub(r"[^A-Za-z0-9_().\-]", "_", name)


def make_unique(names):
    counts = {}
    result = []
    for name in names:
        count = counts.get(name, 0)
        result.append(name if count == 0 else f"{name}_duplicate_{count}")
        counts[name] = count + 1
    return result


def align_test_features(test_data, model):
    names = make_unique([clean_feature_name(column) for column in test_data.columns])
    test_data.columns = names
    model_features = model.feature_name()
    missing = [column for column in model_features if column not in test_data.columns]
    if missing:
        raise ValueError(f"Test data is missing model features: {missing[:10]}")
    return test_data[model_features]


def main():
    print("=" * 70)
    print("SHAP ANALYSIS FOR CLEAN LIGHTGBM")
    print("=" * 70)

    model = lgb.Booster(model_file=str(ROOT / "lightgbm_clean_model.txt"))
    test_data = pd.read_csv(ROOT / "X_test_safe.csv")
    y_test = pd.read_csv(ROOT / "y_test_safe.csv").squeeze()
    test_data = align_test_features(test_data, model)
    print(f"Test rows: {len(test_data)}")
    print(f"Model features used: {test_data.shape[1]}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_data)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    values = pd.DataFrame(shap_values, columns=test_data.columns)
    importance = pd.DataFrame(
        {
            "feature": values.columns,
            "mean_abs_shap": values.abs().mean().values,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(ROOT / "shap_feature_importance.csv", index=False)

    plt.figure()
    shap.summary_plot(shap_values, test_data, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(ROOT / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values,
        test_data,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(ROOT / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    probabilities = model.predict(test_data)
    sample_index = int(pd.Series(probabilities).idxmax())
    sample_values = values.iloc[sample_index]
    sample_features = test_data.iloc[sample_index]
    increasing = sample_values[sample_values > 0].sort_values(ascending=False).head(15)
    decreasing = sample_values[sample_values < 0].sort_values().head(15)

    explanation_rows = []
    for feature, shap_value in pd.concat([increasing, decreasing]).items():
        explanation_rows.append(
            {
                "sample_index": sample_index,
                "actual_target": int(y_test.iloc[sample_index]),
                "predicted_probability": probabilities[sample_index],
                "feature": feature,
                "feature_value": sample_features[feature],
                "shap_value": shap_value,
                "effect": "increases risk" if shap_value > 0 else "decreases risk",
                "base_value": expected_value,
            }
        )
    pd.DataFrame(explanation_rows).to_csv(
        ROOT / "shap_sample_explanation.csv", index=False
    )

    print("\nTop 15 global features:")
    print(importance.head(15).to_string(index=False))
    print("\nSample high-risk project:")
    print(f"Test row: {sample_index}")
    print(f"Actual target: {int(y_test.iloc[sample_index])}")
    print(f"Predicted probability: {probabilities[sample_index]:.4f}")
    print("\nFeatures increasing risk:")
    print(increasing.to_string())
    print("\nFeatures decreasing risk:")
    print(decreasing.to_string())
    print("\nSaved:")
    print("  shap_feature_importance.csv")
    print("  shap_summary.png")
    print("  shap_bar.png")
    print("  shap_sample_explanation.csv")


if __name__ == "__main__":
    main()
