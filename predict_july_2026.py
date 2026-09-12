import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier


print("=" * 60)
print("JULY 2026 LIVE RISK PREDICTION")
print("=" * 60)


# ============================================================
# 1. LOAD DATA
# ============================================================

train_df = pd.read_csv("ML_FEATURE_DATASET.csv")
july_df = pd.read_csv("JULY_2026_FEATURE_DATASET.csv")

print("\nTraining dataset:", train_df.shape)
print("July 2026 dataset:", july_df.shape)


# ============================================================
# 2. CONVERT REPORT MONTH
# ============================================================

train_df["report_month"] = pd.to_datetime(
    train_df["report_month"],
    errors="coerce"
)

july_df["report_month"] = pd.to_datetime(
    july_df["report_month"],
    errors="coerce"
)


# ============================================================
# 3. KEEP ONLY LABELLED TRAINING DATA
# ============================================================

train_df = train_df[
    train_df["future_cost_overrun"].notna()
].copy()

train_df["future_cost_overrun"] = (
    train_df["future_cost_overrun"].astype(int)
)

print("\nLabelled training records:", len(train_df))


# ============================================================
# 4. TIME-BASED TRAINING DATA
# ============================================================

train_df = train_df[
    train_df["report_month"] <
    pd.Timestamp("2026-05-01")
].copy()

print(
    "Historical training records:",
    len(train_df)
)

print("\nTraining months:")
print(
    train_df["report_month"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 5. TARGET
# ============================================================

y_train = train_df[
    "future_cost_overrun"
].reset_index(drop=True)


# ============================================================
# 6. EXACT SAME COLUMN DROPS AS ORIGINAL PIPELINE
# ============================================================

columns_to_drop = [
    "future_cost_overrun",
    "revised_cost",
    "cost_growth",
    "has_current_revision",

    "project_code",
    "project_name",

    "report_month",
    "previous_report_month",

    "source_file",
    "source_sheet"
]

# Drop only columns that exist
columns_to_drop = [
    col for col in columns_to_drop
    if col in train_df.columns
]


X_train = train_df.drop(
    columns=columns_to_drop
).copy()


X_july = july_df.drop(
    columns=[
        col
        for col in columns_to_drop
        if col in july_df.columns
    ]
).copy()


# ============================================================
# 7. REMOVE DATETIME COLUMNS
# ============================================================

date_columns = X_train.select_dtypes(
    include=["datetime64[ns]"]
).columns.tolist()

if date_columns:

    print(
        "\nRemoving datetime columns:",
        date_columns
    )

    X_train = X_train.drop(
        columns=date_columns
    )

    X_july = X_july.drop(
        columns=[
            col for col in date_columns
            if col in X_july.columns
        ]
    )


# ============================================================
# 8. IDENTIFY NUMERIC / CATEGORICAL
# ============================================================

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_columns = X_train.select_dtypes(
    include=[np.number]
).columns.tolist()


print("\nInitial categorical features:")
print(categorical_columns)

print(
    "\nInitial numeric features:",
    len(numeric_columns)
)


# ============================================================
# 9. REMOVE COMPLETELY EMPTY FEATURES
# ============================================================

empty_numeric_columns = [
    col
    for col in numeric_columns
    if X_train[col].notna().sum() == 0
]

empty_categorical_columns = [
    col
    for col in categorical_columns
    if X_train[col].notna().sum() == 0
]

empty_columns = (
    empty_numeric_columns
    + empty_categorical_columns
)

if empty_columns:

    print("\nRemoving completely empty features:")

    for col in empty_columns:
        print(" -", col)

    X_train = X_train.drop(
        columns=empty_columns
    )

    X_july = X_july.drop(
        columns=[
            col for col in empty_columns
            if col in X_july.columns
        ]
    )

    numeric_columns = [
        col
        for col in numeric_columns
        if col not in empty_columns
    ]

    categorical_columns = [
        col
        for col in categorical_columns
        if col not in empty_columns
    ]


print(
    "\nFinal numeric features:",
    len(numeric_columns)
)

print(
    "Final categorical features:",
    len(categorical_columns)
)


# ============================================================
# 10. MAKE JULY COLUMNS MATCH TRAINING
# ============================================================

# Add missing training columns to July
for col in X_train.columns:

    if col not in X_july.columns:
        X_july[col] = np.nan


# Remove any extra July columns
extra_columns = [
    col
    for col in X_july.columns
    if col not in X_train.columns
]

if extra_columns:

    X_july = X_july.drop(
        columns=extra_columns
    )


# Exact same order
X_july = X_july[
    X_train.columns
]


# ============================================================
# 11. CLEAN NUMERIC DATA
# ============================================================

for col in numeric_columns:

    X_train[col] = pd.to_numeric(
        X_train[col],
        errors="coerce"
    )

    X_july[col] = pd.to_numeric(
        X_july[col],
        errors="coerce"
    )


X_train[numeric_columns] = (
    X_train[numeric_columns]
    .replace([np.inf, -np.inf], np.nan)
)

X_july[numeric_columns] = (
    X_july[numeric_columns]
    .replace([np.inf, -np.inf], np.nan)
)


# ============================================================
# 12. NUMERIC IMPUTATION
# ============================================================

if numeric_columns:

    numeric_imputer = SimpleImputer(
        strategy="median"
    )

    train_numeric_array = (
        numeric_imputer.fit_transform(
            X_train[numeric_columns]
        )
    )

    july_numeric_array = (
        numeric_imputer.transform(
            X_july[numeric_columns]
        )
    )

    numeric_feature_names = (
        numeric_imputer.get_feature_names_out(
            numeric_columns
        )
    )

    X_train_numeric = pd.DataFrame(
        train_numeric_array,
        columns=numeric_feature_names
    )

    X_july_numeric = pd.DataFrame(
        july_numeric_array,
        columns=numeric_feature_names
    )

else:

    X_train_numeric = pd.DataFrame()

    X_july_numeric = pd.DataFrame()


# ============================================================
# 13. CATEGORICAL IMPUTATION
# ============================================================

if categorical_columns:

    categorical_imputer = SimpleImputer(
        strategy="constant",
        fill_value="Unknown"
    )

    train_cat_array = (
        categorical_imputer.fit_transform(
            X_train[categorical_columns]
        )
    )

    july_cat_array = (
        categorical_imputer.transform(
            X_july[categorical_columns]
        )
    )

    X_train_cat = pd.DataFrame(
        train_cat_array,
        columns=categorical_columns
    )

    X_july_cat = pd.DataFrame(
        july_cat_array,
        columns=categorical_columns
    )

else:

    X_train_cat = pd.DataFrame()

    X_july_cat = pd.DataFrame()


# ============================================================
# 14. ONE-HOT ENCODING
# ============================================================

if categorical_columns:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    train_encoded_array = (
        encoder.fit_transform(
            X_train_cat
        )
    )

    july_encoded_array = (
        encoder.transform(
            X_july_cat
        )
    )

    encoded_feature_names = (
        encoder.get_feature_names_out(
            categorical_columns
        )
    )

    X_train_encoded = pd.DataFrame(
        train_encoded_array,
        columns=encoded_feature_names
    )

    X_july_encoded = pd.DataFrame(
        july_encoded_array,
        columns=encoded_feature_names
    )

else:

    X_train_encoded = pd.DataFrame()

    X_july_encoded = pd.DataFrame()


# ============================================================
# 15. COMBINE FEATURES
# ============================================================

X_train_final = pd.concat(
    [
        X_train_numeric.reset_index(drop=True),
        X_train_encoded.reset_index(drop=True)
    ],
    axis=1
)

X_july_final = pd.concat(
    [
        X_july_numeric.reset_index(drop=True),
        X_july_encoded.reset_index(drop=True)
    ],
    axis=1
)


# ============================================================
# 16. ALIGN JULY WITH TRAINING
# ============================================================

X_july_final = X_july_final.reindex(
    columns=X_train_final.columns,
    fill_value=0
)


# ============================================================
# 17. FINAL SAFETY CHECK
# ============================================================

X_train_final = (
    X_train_final
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

X_july_final = (
    X_july_final
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(
    "Training shape:",
    X_train_final.shape
)

print(
    "July 2026 shape:",
    X_july_final.shape
)


# ============================================================
# 18. VERIFY FEATURE COUNT
# ============================================================

if X_train_final.shape[1] != X_july_final.shape[1]:

    raise ValueError(
        "Training and July feature counts do not match!"
    )

print(
    "\nFeature count matches:",
    X_train_final.shape[1]
)


# ============================================================
# 19. CLASS WEIGHT
# ============================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(
    "\nScale positive weight:",
    round(scale_pos_weight, 2)
)


# ============================================================
# 20. TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

model = XGBClassifier(
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

import re

# Clean feature names for XGBoost
clean_feature_names = []

for col in X_train_final.columns:
    clean_col = re.sub(r'[\[\]<>]', '_', str(col))
    clean_feature_names.append(clean_col)

# Make names unique
seen = {}
unique_feature_names = []

for name in clean_feature_names:
    if name not in seen:
        seen[name] = 0
        unique_feature_names.append(name)
    else:
        seen[name] += 1
        unique_feature_names.append(
            f"{name}_{seen[name]}"
        )

X_train_final.columns = unique_feature_names
X_july_final.columns = unique_feature_names

print("\nFeature names cleaned for XGBoost.")

model.fit(
    X_train_final,
    y_train
)

print("XGBoost training completed.")


# ============================================================
# 21. PREDICT JULY
# ============================================================

print("\n" + "=" * 60)
print("GENERATING JULY 2026 PREDICTIONS")
print("=" * 60)

probabilities = model.predict_proba(
    X_july_final
)[:, 1]


# ============================================================
# 22. RISK LEVEL
# ============================================================

def get_risk(probability):

    if probability >= 0.66:
        return "HIGH"

    elif probability >= 0.33:
        return "MEDIUM"

    else:
        return "LOW"


risk_levels = [
    get_risk(p)
    for p in probabilities
]


# ============================================================
# 23. CREATE RESULT
# ============================================================

result = july_df.copy()

result["overrun_probability"] = probabilities

result["overrun_probability_percent"] = (
    probabilities * 100
).round(2)

result["risk_level"] = risk_levels


# ============================================================
# 24. SORT
# ============================================================

result = result.sort_values(
    "overrun_probability",
    ascending=False
)


# ============================================================
# 25. SAVE
# ============================================================

OUTPUT_FILE = (
    "JULY_2026_RISK_PREDICTIONS.csv"
)

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 26. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("JULY 2026 RISK SUMMARY")
print("=" * 60)

print(
    "\nTotal projects:",
    len(result)
)

print("\nRisk distribution:")

print(
    result["risk_level"]
    .value_counts()
)


print("\nProbability statistics:")

print(
    result["overrun_probability"]
    .describe()
)


# ============================================================
# 27. TOP 20
# ============================================================

print(
    "\nTop 20 highest-risk projects:"
)

display_columns = [
    col
    for col in [
        "project_code",
        "project_name",
        "overrun_probability_percent",
        "risk_level"
    ]
    if col in result.columns
]

print(
    result[
        display_columns
    ].head(20).to_string(index=False)
)


print("\n" + "=" * 60)
print("PREDICTION COMPLETED")
print("=" * 60)

print("\nSaved:")
print(OUTPUT_FILE)