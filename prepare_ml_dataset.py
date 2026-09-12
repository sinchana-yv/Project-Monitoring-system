import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

print("=" * 60)
print("PREPARING LEAKAGE-SAFE ML DATASET")
print("=" * 60)


# ============================================================
# 1. LOAD FEATURE DATASET
# ============================================================

df = pd.read_csv("ML_FEATURE_DATASET.csv")

print("\nTotal records:", len(df))


# ============================================================
# 2. CONVERT REPORT MONTH
# ============================================================

df["report_month"] = pd.to_datetime(
    df["report_month"],
    errors="coerce"
)


# ============================================================
# 3. REMOVE UNKNOWN TARGETS
# ============================================================

df = df[
    df["future_cost_overrun"].notna()
].copy()

df["future_cost_overrun"] = (
    df["future_cost_overrun"]
    .astype(int)
)

print(
    "Records with valid target:",
    len(df)
)


# ============================================================
# 4. SORT BY TIME
# ============================================================

df = df.sort_values(
    "report_month"
).reset_index(drop=True)


# ============================================================
# 5. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

train_df = df[
    df["report_month"] < pd.Timestamp("2026-05-01")
].copy()

test_df = df[
    df["report_month"] == pd.Timestamp("2026-05-01")
].copy()


print("\nTRAINING DATA")
print("Records:", len(train_df))

print("\nTraining months:")

print(
    train_df[
        "report_month"
    ]
    .value_counts()
    .sort_index()
)


print("\nTEST DATA")
print("Records:", len(test_df))

print("\nTesting months:")

print(
    test_df[
        "report_month"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 6. TARGET
# ============================================================

target_column = "future_cost_overrun"

y_train = train_df[
    target_column
].copy()

y_test = test_df[
    target_column
].copy()


# ============================================================
# 7. REMOVE TARGET / LEAKAGE / ID COLUMNS
# ============================================================

columns_to_drop = [
    # Target
    "future_cost_overrun",

    # Future information
    "revised_cost",

    # Leakage
    "cost_growth",
    "has_current_revision",

    # Identifiers
    "project_code",
    "project_name",

    # Used only for splitting
    "report_month",

    # Raw previous date
    "previous_report_month",

    # Source information
    "source_file",
    "source_sheet"
]


columns_to_drop = [
    col
    for col in columns_to_drop
    if col in train_df.columns
]


X_train = train_df.drop(
    columns=columns_to_drop
)

X_test = test_df.drop(
    columns=columns_to_drop
)


# ============================================================
# 8. REMOVE DATETIME COLUMNS
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

    X_test = X_test.drop(
        columns=date_columns
    )


# ============================================================
# 9. IDENTIFY CATEGORICAL AND NUMERIC FEATURES
# ============================================================

# Use object + string explicitly to avoid pandas warning

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_columns = X_train.select_dtypes(
    include=[np.number]
).columns.tolist()


print("\nCategorical features:")

print(categorical_columns)

print(
    "\nNumber of categorical features:",
    len(categorical_columns)
)

print(
    "Number of numeric features:",
    len(numeric_columns)
)


# ============================================================
# 10. REMOVE COMPLETELY EMPTY FEATURES
# ============================================================

# A feature with 100% missing values in training data
# cannot provide useful information.

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

    print(
        "\nRemoving completely empty features:"
    )

    for col in empty_columns:
        print(" -", col)

    X_train = X_train.drop(
        columns=empty_columns
    )

    X_test = X_test.drop(
        columns=empty_columns
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
# 11. CLEAN NUMERIC DATA
# ============================================================

for col in numeric_columns:

    X_train[col] = pd.to_numeric(
        X_train[col],
        errors="coerce"
    )

    X_test[col] = pd.to_numeric(
        X_test[col],
        errors="coerce"
    )


# Replace infinity

X_train[numeric_columns] = (
    X_train[numeric_columns]
    .replace([np.inf, -np.inf], np.nan)
)

X_test[numeric_columns] = (
    X_test[numeric_columns]
    .replace([np.inf, -np.inf], np.nan)
)


# ============================================================
# 12. NUMERIC IMPUTATION
# ============================================================

if numeric_columns:

    numeric_imputer = SimpleImputer(
        strategy="median"
    )

    X_train_numeric_array = (
        numeric_imputer.fit_transform(
            X_train[numeric_columns]
        )
    )

    X_test_numeric_array = (
        numeric_imputer.transform(
            X_test[numeric_columns]
        )
    )

    # IMPORTANT:
    # Get the actual columns returned by the imputer

    numeric_feature_names = (
        numeric_imputer.get_feature_names_out(
            numeric_columns
        )
    )

    X_train_numeric = pd.DataFrame(
        X_train_numeric_array,
        columns=numeric_feature_names,
        index=X_train.index
    )

    X_test_numeric = pd.DataFrame(
        X_test_numeric_array,
        columns=numeric_feature_names,
        index=X_test.index
    )

else:

    X_train_numeric = pd.DataFrame(
        index=X_train.index
    )

    X_test_numeric = pd.DataFrame(
        index=X_test.index
    )


# ============================================================
# 13. CATEGORICAL IMPUTATION
# ============================================================

if categorical_columns:

    categorical_imputer = SimpleImputer(
        strategy="constant",
        fill_value="Unknown"
    )

    X_train_cat_array = (
        categorical_imputer.fit_transform(
            X_train[categorical_columns]
        )
    )

    X_test_cat_array = (
        categorical_imputer.transform(
            X_test[categorical_columns]
        )
    )

    categorical_feature_names = (
        categorical_imputer.get_feature_names_out(
            categorical_columns
        )
    )

    X_train_cat = pd.DataFrame(
        X_train_cat_array,
        columns=categorical_feature_names,
        index=X_train.index
    )

    X_test_cat = pd.DataFrame(
        X_test_cat_array,
        columns=categorical_feature_names,
        index=X_test.index
    )

else:

    X_train_cat = pd.DataFrame(
        index=X_train.index
    )

    X_test_cat = pd.DataFrame(
        index=X_test.index
    )


# ============================================================
# 14. ONE-HOT ENCODING
# ============================================================

if categorical_columns:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train_encoded_array = (
        encoder.fit_transform(
            X_train_cat
        )
    )

    X_test_encoded_array = (
        encoder.transform(
            X_test_cat
        )
    )

    encoded_feature_names = (
        encoder.get_feature_names_out(
            categorical_columns
        )
    )

    X_train_encoded = pd.DataFrame(
        X_train_encoded_array,
        columns=encoded_feature_names,
        index=X_train.index
    )

    X_test_encoded = pd.DataFrame(
        X_test_encoded_array,
        columns=encoded_feature_names,
        index=X_test.index
    )

else:

    X_train_encoded = pd.DataFrame(
        index=X_train.index
    )

    X_test_encoded = pd.DataFrame(
        index=X_test.index
    )


# ============================================================
# 15. COMBINE FEATURES
# ============================================================

X_train_final = pd.concat(
    [
        X_train_numeric,
        X_train_encoded
    ],
    axis=1
)

X_test_final = pd.concat(
    [
        X_test_numeric,
        X_test_encoded
    ],
    axis=1
)


# ============================================================
# 16. ALIGN TEST WITH TRAINING FEATURES
# ============================================================

X_test_final = X_test_final.reindex(
    columns=X_train_final.columns,
    fill_value=0
)


# ============================================================
# 17. FINAL NaN / INF SAFETY
# ============================================================

X_train_final = X_train_final.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test_final = X_test_final.replace(
    [np.inf, -np.inf],
    np.nan
)


# Any unexpected remaining NaN → 0
#
# Normally there should be none because all usable
# numeric/categorical columns were already imputed.

X_train_final = X_train_final.fillna(0)

X_test_final = X_test_final.fillna(0)


# ============================================================
# 18. RESET INDEX
# ============================================================

X_train_final = X_train_final.reset_index(
    drop=True
)

X_test_final = X_test_final.reset_index(
    drop=True
)

y_train = y_train.reset_index(
    drop=True
)

y_test = y_test.reset_index(
    drop=True
)


# ============================================================
# 19. SAVE ML DATASETS
# ============================================================

X_train_final.to_csv(
    "X_train.csv",
    index=False
)

X_test_final.to_csv(
    "X_test.csv",
    index=False
)

y_train.to_csv(
    "y_train.csv",
    index=False
)

y_test.to_csv(
    "y_test.csv",
    index=False
)


# ============================================================
# 20. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL ML DATASET SUMMARY")
print("=" * 60)

print(
    "\nTraining records:",
    len(X_train_final)
)

print(
    "Testing records:",
    len(X_test_final)
)

print(
    "Final number of features:",
    X_train_final.shape[1]
)


print("\nTraining target distribution:")

print(
    y_train.value_counts()
)


print("\nTraining target percentage:")

print(
    y_train
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


print("\nTesting target distribution:")

print(
    y_test.value_counts()
)


print("\nTesting target percentage:")

print(
    y_test
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 21. LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 60)
print("LEAKAGE CHECK")
print("=" * 60)

dangerous_columns = [
    "revised_cost",
    "cost_growth",
    "has_current_revision",
    "future_cost_overrun",
    "project_code",
    "project_name"
]

remaining_dangerous = [
    col
    for col in dangerous_columns
    if col in X_train_final.columns
]

if remaining_dangerous:

    print(
        "WARNING! Potential leakage columns:",
        remaining_dangerous
    )

else:

    print(
        "No target / future-cost leakage columns found."
    )


# ============================================================
# 22. MISSING VALUE CHECK
# ============================================================

print("\nMissing values in X_train:")

print(
    X_train_final.isna().sum().sum()
)

print("\nMissing values in X_test:")

print(
    X_test_final.isna().sum().sum()
)


# ============================================================
# 23. FINAL FEATURE LIST
# ============================================================

print("\n" + "=" * 60)
print("FINAL FEATURE COUNT")
print("=" * 60)

print(
    "Numeric features:",
    len(numeric_feature_names)
    if numeric_columns
    else 0
)

print(
    "Encoded categorical features:",
    len(encoded_feature_names)
    if categorical_columns
    else 0
)

print(
    "Total features:",
    X_train_final.shape[1]
)


# ============================================================
# 24. FILES CREATED
# ============================================================

print("\n" + "=" * 60)
print("FILES CREATED")
print("=" * 60)

print("1. X_train.csv")
print("2. X_test.csv")
print("3. y_train.csv")
print("4. y_test.csv")

print("\nDONE")
print("=" * 60)