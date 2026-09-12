import pandas as pd
import numpy as np
import re

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("=" * 60)
print("XGBOOST COST OVERRUN MODEL")
print("=" * 60)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")
print(f"Features        : {X_train.shape[1]}")


# ---------------------------------------------------------
# 2. CLEAN FEATURE NAMES
# ---------------------------------------------------------

def clean_feature_name(name):
    name = str(name)

    # Replace characters XGBoost does not like
    name = name.replace("[", "(")
    name = name.replace("]", ")")
    name = name.replace("<", "_")

    # Replace other special characters
    name = re.sub(r"[^A-Za-z0-9_().\-]", "_", name)

    return name


# Clean names
cleaned_names = [clean_feature_name(col) for col in X_train.columns]


# ---------------------------------------------------------
# 3. MAKE DUPLICATE FEATURE NAMES UNIQUE
# ---------------------------------------------------------

def make_unique(names):

    counts = {}
    unique_names = []

    for name in names:

        if name not in counts:
            counts[name] = 0
            unique_names.append(name)

        else:
            counts[name] += 1
            unique_names.append(
                f"{name}_duplicate_{counts[name]}"
            )

    return unique_names


unique_names = make_unique(cleaned_names)

X_train.columns = unique_names
X_test.columns = unique_names

print("\nFeature names cleaned and made unique.")


# ---------------------------------------------------------
# 4. CHECK FOR MISSING / INFINITE VALUES
# ---------------------------------------------------------

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# XGBoost cannot handle unexpected NaN patterns reliably here,
# so fill remaining missing values using training medians.

for column in X_train.columns:

    if X_train[column].isna().any():

        median_value = X_train[column].median()

        if pd.isna(median_value):
            median_value = 0

        X_train[column] = X_train[column].fillna(median_value)
        X_test[column] = X_test[column].fillna(median_value)


# ---------------------------------------------------------
# 5. CLASS DISTRIBUTION
# ---------------------------------------------------------

no_overrun = (y_train == 0).sum()
cost_overrun = (y_train == 1).sum()

scale_pos_weight = no_overrun / cost_overrun

print(f"\nNo Overrun   : {no_overrun}")
print(f"Cost Overrun : {cost_overrun}")
print(f"Scale Pos Weight: {scale_pos_weight:.4f}")


# ---------------------------------------------------------
# 6. CREATE XGBOOST MODEL
# ---------------------------------------------------------

model = XGBClassifier(

    n_estimators=300,

    max_depth=5,

    learning_rate=0.05,

    min_child_weight=5,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=42,

    n_jobs=-1
)


# ---------------------------------------------------------
# 7. TRAIN
# ---------------------------------------------------------

print("\nTraining XGBoost model...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ---------------------------------------------------------
# 8. PREDICTION
# ---------------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------
# 9. EVALUATION
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n" + "=" * 60)
print("XGBOOST RESULTS")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score  : {f1:.4f} ({f1*100:.2f}%)")
print(f"ROC-AUC   : {roc_auc:.4f} ({roc_auc*100:.2f}%)")


# ---------------------------------------------------------
# 10. CONFUSION MATRIX
# ---------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ---------------------------------------------------------
# 11. CLASSIFICATION REPORT
# ---------------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Overrun",
            "Cost Overrun"
        ],
        zero_division=0
    )
)


# ---------------------------------------------------------
# 12. FEATURE IMPORTANCE
# ---------------------------------------------------------

importance = pd.DataFrame({

    "feature": X_train.columns,

    "importance": model.feature_importances_

})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

importance.to_csv(
    "xgboost_feature_importance.csv",
    index=False
)


# ---------------------------------------------------------
# 13. SAVE PREDICTIONS
# ---------------------------------------------------------

predictions = pd.DataFrame({

    "actual": y_test,

    "predicted": y_pred,

    "overrun_probability": y_probability

})

predictions.to_csv(
    "xgboost_predictions.csv",
    index=False
)


# ---------------------------------------------------------
# 14. TOP FEATURES
# ---------------------------------------------------------

print("\nTop 20 Important Features:")

print(
    importance.head(20).to_string(index=False)
)

print("\nFiles created:")
print("- xgboost_feature_importance.csv")
print("- xgboost_predictions.csv")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)