
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import re

# ============================================================
# 1. LOAD DATA
# ============================================================

train_df = pd.read_csv("ML_FEATURE_DATASET.csv")
july_df = pd.read_csv("JULY_2026_FEATURE_DATASET.csv")

train_df["report_month"] = pd.to_datetime(
    train_df["report_month"], errors="coerce"
)

july_df["report_month"] = pd.to_datetime(
    july_df["report_month"], errors="coerce"
)

# ============================================================
# 2. SELECT THE SAME HISTORICAL TRAINING DATA
# ============================================================

train_df = train_df[
    train_df["future_cost_overrun"].notna()
    & (train_df["report_month"] < "2026-05-01")
].copy()

y_train = train_df["future_cost_overrun"].astype(int)

print("Training rows:", len(train_df))

# ============================================================
# 3. DROP TARGET / LEAKAGE / ID COLUMNS
# ============================================================

drop_cols = [
    "future_cost_overrun",
    "revised_cost",
    "cost_growth",
    "cost_increase",
    "has_current_revision",
    "project_code",
    "project_name",
    "report_month",
    "previous_report_month",
    "source_file",
    "source_sheet"
]


X_train = train_df.drop(
    columns=[c for c in drop_cols if c in train_df.columns],
    errors="ignore"
)

X_july = july_df.drop(
    columns=[c for c in drop_cols if c in july_df.columns],
    errors="ignore"
)

# ============================================================
# 4. REMOVE DATETIME COLUMNS
# ============================================================

datetime_cols = X_train.select_dtypes(
    include=["datetime64[ns]", "datetime64[ns, UTC]"]
).columns.tolist()

X_train = X_train.drop(
    columns=datetime_cols,
    errors="ignore"
)

X_july = X_july.drop(
    columns=datetime_cols,
    errors="ignore"
)

# ============================================================
# 5. IDENTIFY NUMERIC + CATEGORICAL COLUMNS
# ============================================================

categorical_cols = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

numeric_cols = X_train.select_dtypes(
    include=[np.number]
).columns.tolist()

print("Initial numeric columns:", len(numeric_cols))
print("Initial categorical columns:", len(categorical_cols))

# ============================================================
# 6. REMOVE COMPLETELY EMPTY COLUMNS
# ============================================================

empty_numeric = [
    c for c in numeric_cols
    if X_train[c].notna().sum() == 0
]

empty_categorical = [
    c for c in categorical_cols
    if X_train[c].notna().sum() == 0
]

empty_columns = empty_numeric + empty_categorical

print("Completely empty columns removed:", empty_columns)

X_train = X_train.drop(
    columns=empty_columns,
    errors="ignore"
)

X_july = X_july.drop(
    columns=empty_columns,
    errors="ignore"
)

# Recalculate column types after removing empty columns
categorical_cols = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

numeric_cols = X_train.select_dtypes(
    include=[np.number]
).columns.tolist()

print("Final numeric columns:", len(numeric_cols))
print("Final categorical columns:", len(categorical_cols))

# ============================================================
# 7. NUMERIC IMPUTATION
# ============================================================

for col in numeric_cols:

    median_value = X_train[col].median()

    X_train[col] = X_train[col].fillna(median_value)

    X_july[col] = X_july[col].fillna(median_value)

# ============================================================
# 8. CATEGORICAL IMPUTATION
# ============================================================

for col in categorical_cols:

    X_train[col] = (
        X_train[col]
        .fillna("Unknown")
        .astype(str)
    )

    X_july[col] = (
        X_july[col]
        .fillna("Unknown")
        .astype(str)
    )

# ============================================================
# 9. ONE-HOT ENCODING
# ============================================================

X_train_encoded = pd.get_dummies(
    X_train,
    columns=categorical_cols,
    dtype=float
)

X_july_encoded = pd.get_dummies(
    X_july,
    columns=categorical_cols,
    dtype=float
)

# ============================================================
# 10. ALIGN JULY FEATURES WITH TRAINING FEATURES
# ============================================================

X_july_encoded = X_july_encoded.reindex(
    columns=X_train_encoded.columns,
    fill_value=0
)

X_july_encoded = X_july_encoded[
    X_train_encoded.columns
]

# ============================================================
# 11. FINAL NaN CLEANING
# ============================================================

X_train_encoded = X_train_encoded.fillna(0)
X_july_encoded = X_july_encoded.fillna(0)

print(
    "Training encoded shape:",
    X_train_encoded.shape
)

print(
    "July encoded shape:",
    X_july_encoded.shape
)

# ============================================================
# 12. CLEAN FEATURE NAMES FOR XGBOOST
# ============================================================

def clean_feature_names(columns):

    cleaned = []

    for col in columns:

        name = re.sub(
            r"[\[\]<>]",
            "_",
            str(col)
        )

        cleaned.append(name)

    # Make names unique
    seen = {}
    final_names = []

    for name in cleaned:

        if name not in seen:

            seen[name] = 0
            final_names.append(name)

        else:

            seen[name] += 1

            final_names.append(
                f"{name}_{seen[name]}"
            )

    return final_names


feature_names = clean_feature_names(
    X_train_encoded.columns
)

X_train_encoded.columns = feature_names
X_july_encoded.columns = feature_names

# ============================================================
# 13. CHECK FOR OBJECT COLUMNS
# ============================================================

object_columns = X_train_encoded.select_dtypes(
    include=["object"]
).columns.tolist()

if len(object_columns) > 0:

    print(
        "ERROR: Object columns still exist:"
    )

    print(object_columns)

    raise ValueError(
        "Object columns remain after preprocessing."
    )

print("All features are numeric.")

# ============================================================
# 14. CLASS WEIGHT
# ============================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(
    "Scale positive weight:",
    scale_pos_weight
)

# ============================================================
# 15. TRAIN SAME XGBOOST MODEL
# ============================================================

model = xgb.XGBClassifier(
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

model.fit(
    X_train_encoded,
    y_train
)

print("XGBoost model trained successfully.")

# ============================================================
# 16. CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_july_encoded
)

print(
    "SHAP values calculated."
)

print(
    "SHAP shape:",
    shap_values.shape
)

# ============================================================
# 17. LOAD JULY PREDICTIONS
# ============================================================

predictions = pd.read_csv(
    "JULY_2026_RISK_PREDICTIONS.csv"
)

predictions["project_code"] = (
    predictions["project_code"]
    .astype(str)
)

july_df["project_code"] = (
    july_df["project_code"]
    .astype(str)
)

# ============================================================
# 18. EXPLAIN A SINGLE PROJECT
# ============================================================

def explain_project(index):

    row = X_july_encoded.iloc[index]

    shap_row = shap_values[index]

    probability = model.predict_proba(
        X_july_encoded.iloc[[index]]
    )[0][1]

    contribution = pd.DataFrame({

        "feature": X_july_encoded.columns,

        "shap_value": shap_row,

        "absolute_shap": np.abs(shap_row),

        "feature_value": row.values
    })

    contribution = contribution.sort_values(
        "absolute_shap",
        ascending=False
    )

    top = contribution.head(10).copy()

    top["direction"] = np.where(
        top["shap_value"] > 0,
        "INCREASES RISK",
        "DECREASES RISK"
    )

    project_code = july_df.iloc[index][
        "project_code"
    ]

    project_name = july_df.iloc[index].get(
        "project_name",
        "Unknown"
    )

    print("\n")
    print("=" * 75)
    print("INDIVIDUAL SHAP EXPLANATION")
    print("=" * 75)

    print(
        "Project:",
        project_name
    )

    print(
        "Project Code:",
        project_code
    )

    print(
        "Risk Probability:",
        round(probability * 100, 2),
        "%"
    )

    print("\nTop contributing factors:\n")

    print(
        top[
            [
                "feature",
                "feature_value",
                "shap_value",
                "direction"
            ]
        ].to_string(index=False)
    )

    return top


# ============================================================
# 19. FIND HIGHEST-RISK JULY PROJECT
# ============================================================

july_probabilities = model.predict_proba(
    X_july_encoded
)[:, 1]

top_index = np.argmax(
    july_probabilities
)

# ============================================================
# 20. EXPLAIN HIGHEST-RISK PROJECT
# ============================================================

top_explanation = explain_project(
    top_index
)

# ============================================================
# 21. SAVE EXPLANATION
# ============================================================

top_explanation.to_csv(
    "TOP_PROJECT_SHAP_EXPLANATION.csv",
    index=False
)

print("\n")
print("=" * 75)
print("Saved:")
print("TOP_PROJECT_SHAP_EXPLANATION.csv")
print("=" * 75)

