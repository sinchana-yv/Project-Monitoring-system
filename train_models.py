import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# XGBoost
from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING ML DATASET")
print("=" * 60)

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").iloc[:, 0]
y_test = pd.read_csv("y_test.csv").iloc[:, 0]

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

print("y_train:", len(y_train))
print("y_test :", len(y_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# 2. HANDLE INF / NAN
# ============================================================

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)


# ============================================================
# 3. FUNCTION TO EVALUATE MODEL
# ============================================================

results = []


def evaluate_model(name, model):

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    # Train
    print("Training model...")
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Probability
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = 0

    cm = confusion_matrix(y_test, y_pred)

    print("\nAccuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")
    print("ROC-AUC  :", round(roc_auc * 100, 2), "%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["No Overrun", "Overrun"],
        zero_division=0
    ))

    # Save results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc
    })

    # Save predictions
    prediction_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })

    if y_prob is not None:
        prediction_df["Overrun_Probability"] = y_prob

    filename = name.lower().replace(" ", "_") + "_predictions.csv"

    prediction_df.to_csv(filename, index=False)

    print("\nPredictions saved:", filename)

    return model


# ============================================================
# 4. LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="liblinear",
    random_state=42
)

logistic_model = evaluate_model(
    "Logistic Regression",
    logistic_model
)


# ============================================================
# 5. RANDOM FOREST
# ============================================================

random_forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

random_forest_model = evaluate_model(
    "Random Forest",
    random_forest_model
)

# ============================================================
# CLEAN FEATURE NAMES FOR XGBOOST
# ============================================================

import re

def clean_feature_names(columns):
    cleaned = []

    for col in columns:
        col = str(col)

        # Replace characters that XGBoost does not allow
        col = re.sub(r"[\[\]<>]", "_", col)

        # Replace other problematic characters
        col = re.sub(r"[^a-zA-Z0-9_]", "_", col)

        cleaned.append(col)

    # Make duplicate names unique
    seen = {}
    unique_names = []

    for name in cleaned:
        if name not in seen:
            seen[name] = 0
            unique_names.append(name)
        else:
            seen[name] += 1
            unique_names.append(
                f"{name}_{seen[name]}"
            )

    return unique_names


# Save original names
original_feature_names = X_train.columns.tolist()

# Clean names
cleaned_feature_names = clean_feature_names(
    X_train.columns
)

X_train.columns = cleaned_feature_names
X_test.columns = cleaned_feature_names

print("\nFeature names cleaned for XGBoost.")
print("Total features:", len(cleaned_feature_names))

# ============================================================
# 6. XGBOOST
# ============================================================

# Calculate class imbalance
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nXGBoost scale_pos_weight:", round(scale_pos_weight, 2))


xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model = evaluate_model(
    "XGBoost",
    xgb_model
)


# ============================================================
# 7. COMPARE MODELS
# ============================================================

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results)

results_df["Accuracy"] *= 100
results_df["Precision"] *= 100
results_df["Recall"] *= 100
results_df["F1"] *= 100
results_df["ROC_AUC"] *= 100

results_df = results_df.round(2)

print("\n")
print(results_df.to_string(index=False))

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nModel comparison saved:")
print("model_comparison.csv")


# ============================================================
# 8. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n")
print("=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

rf_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": random_forest_model.feature_importances_
})

rf_importance = rf_importance.sort_values(
    by="Importance",
    ascending=False
)

rf_importance.to_csv(
    "random_forest_feature_importance.csv",
    index=False
)

print("\nTop 20 Random Forest Features:")
print(rf_importance.head(20).to_string(index=False))

print("\nSaved:")
print("random_forest_feature_importance.csv")


# ============================================================
# 9. XGBOOST FEATURE IMPORTANCE
# ============================================================

print("\n")
print("=" * 60)
print("XGBOOST FEATURE IMPORTANCE")
print("=" * 60)

xgb_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": xgb_model.feature_importances_
})

xgb_importance = xgb_importance.sort_values(
    by="Importance",
    ascending=False
)

xgb_importance.to_csv(
    "xgboost_feature_importance.csv",
    index=False
)

print("\nTop 20 XGBoost Features:")
print(xgb_importance.head(20).to_string(index=False))

print("\nSaved:")
print("xgboost_feature_importance.csv")


# ============================================================
# 10. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print("""
Files created:

1. logistic_regression_predictions.csv
2. random_forest_predictions.csv
3. xgboost_predictions.csv
4. model_comparison.csv
5. random_forest_feature_importance.csv
6. xgboost_feature_importance.csv

Next step:
→ Select the best model
→ Apply SHAP explainability
→ Predict risk for July 2026 live projects
→ Build the SIH dashboard
""")