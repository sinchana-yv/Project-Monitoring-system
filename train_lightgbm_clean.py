"""Train LightGBM after removing target and future-information features."""

import re
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
CUTOFF = pd.Timestamp("2026-05-01")


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


def prepare_data():
    source = pd.read_csv(ROOT / "ML_FEATURE_DATASET.csv")
    source["report_month"] = pd.to_datetime(source["report_month"], errors="coerce")
    source["project_code"] = source["project_code"].astype(str).str.strip()
    source = source.dropna(subset=["project_code", "report_month", "future_cost_overrun"])
    source["future_cost_overrun"] = source["future_cost_overrun"].astype(int)

    first_month = source.groupby("project_code")["report_month"].min()
    train_projects = set(first_month[first_month < CUTOFF].index)
    test_projects = set(first_month[first_month >= CUTOFF].index)
    train_df = source[source["project_code"].isin(train_projects)].copy()
    test_df = source[source["project_code"].isin(test_projects)].copy()
    overlap = train_projects.intersection(test_projects)
    if overlap:
        raise RuntimeError("Project overlap detected.")

    target = "future_cost_overrun"
    future_or_target_columns = [
        target,
        "cost_increase",
        "revised_cost",
        "revised_doc",
        "cost_growth",
        "has_current_revision",
        "actual_completion",
        "actual_completion_date",
        "project_code",
        "project_name",
        "report_month",
        "previous_report_month",
        "source_file",
        "source_sheet",
    ]
    dropped = [column for column in future_or_target_columns if column in train_df.columns]

    y_train = train_df[target].reset_index(drop=True)
    y_test = test_df[target].reset_index(drop=True)
    x_train = train_df.drop(columns=dropped).reset_index(drop=True)
    x_test = test_df.drop(columns=dropped).reset_index(drop=True)

    date_columns = x_train.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    x_train = x_train.drop(columns=date_columns)
    x_test = x_test.drop(columns=date_columns)

    categorical = x_train.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric = [column for column in x_train.columns if column not in categorical]
    empty_numeric = [column for column in numeric if x_train[column].notna().sum() == 0]
    if empty_numeric:
        x_train = x_train.drop(columns=empty_numeric)
        x_test = x_test.drop(columns=empty_numeric)
        numeric = [column for column in numeric if column not in empty_numeric]

    for column in numeric:
        x_train[column] = pd.to_numeric(x_train[column], errors="coerce")
        x_test[column] = pd.to_numeric(x_test[column], errors="coerce")
    x_train[numeric] = x_train[numeric].replace([np.inf, -np.inf], np.nan)
    x_test[numeric] = x_test[numeric].replace([np.inf, -np.inf], np.nan)

    numeric_imputer = None
    if numeric:
        numeric_imputer = SimpleImputer(strategy="median")
        x_train[numeric] = numeric_imputer.fit_transform(x_train[numeric])
        x_test[numeric] = numeric_imputer.transform(x_test[numeric])

    categorical_imputer = None
    encoder = None
    if categorical:
        categorical_imputer = SimpleImputer(
            strategy="constant",
            fill_value="Unknown",
        )
        train_cat = categorical_imputer.fit_transform(x_train[categorical])
        test_cat = categorical_imputer.transform(x_test[categorical])
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        train_encoded = encoder.fit_transform(train_cat)
        test_encoded = encoder.transform(test_cat)
        names = encoder.get_feature_names_out(categorical)
        x_train = pd.concat(
            [x_train.drop(columns=categorical), pd.DataFrame(train_encoded, columns=names)],
            axis=1,
        )
        x_test = pd.concat(
            [x_test.drop(columns=categorical), pd.DataFrame(test_encoded, columns=names)],
            axis=1,
        )

    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
    names = make_unique([clean_feature_name(column) for column in x_train.columns])
    x_train.columns = names
    x_test.columns = names
    preprocessing = {
        "numeric_imputer": numeric_imputer,
        "categorical_imputer": categorical_imputer,
        "onehot_encoder": encoder,
        "categorical_columns": categorical,
        "numeric_columns": numeric,
        "dropped_columns": dropped,
        "feature_names": names,
        "threshold": 0.50,
    }
    return (
        x_train,
        x_test,
        y_train,
        y_test,
        dropped,
        train_projects,
        test_projects,
        preprocessing,
    )


def main():
    print("=" * 70)
    print("CLEAN LEAKAGE-SAFE LIGHTGBM")
    print("=" * 70)
    (
        x_train,
        x_test,
        y_train,
        y_test,
        dropped,
        train_projects,
        test_projects,
        preprocessing,
    ) = prepare_data()
    print(f"\nDropped target/future-information columns: {dropped}")
    print(f"Training rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print(f"Training projects: {len(train_projects)}")
    print(f"Testing projects: {len(test_projects)}")
    print(f"Project overlap: {len(train_projects.intersection(test_projects))}")
    print(f"Features after cleaning: {x_train.shape[1]}")
    print("\nAudit: cost_increase is excluded completely.")
    print("Audit: target future_cost_overrun is excluded completely.")
    print("Audit: revised cost/document and completion-derived columns are excluded.")
    print("Audit: imputation and one-hot encoding are fitted on training data only.")

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
    print("CLEAN LIGHTGBM RESULTS")
    print("=" * 70)
    for name, value in metrics.items():
        print(f"{name:10}: {value:.4f} ({value * 100:.2f}%)")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["No Overrun", "Cost Overrun"], zero_division=0))

    importance = pd.DataFrame(
        {"feature": x_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\nTop 20 important features:")
    print(importance.head(20).to_string(index=False))
    importance.to_csv(ROOT / "lightgbm_clean_feature_importance.csv", index=False)
    pd.DataFrame(
        {"actual": y_test, "predicted": predictions, "risk_probability": probabilities}
    ).to_csv(ROOT / "lightgbm_clean_predictions.csv", index=False)
    model.booster_.save_model(str(ROOT / "lightgbm_clean_model.txt"))
    joblib.dump(
        preprocessing["numeric_imputer"],
        ROOT / "lightgbm_numeric_imputer.pkl",
    )
    joblib.dump(
        preprocessing["categorical_imputer"],
        ROOT / "lightgbm_categorical_imputer.pkl",
    )
    joblib.dump(
        preprocessing["onehot_encoder"],
        ROOT / "lightgbm_onehot_encoder.pkl",
    )
    joblib.dump(
        preprocessing["feature_names"],
        ROOT / "lightgbm_feature_names.pkl",
    )
    joblib.dump(
        preprocessing,
        ROOT / "lightgbm_preprocessing_metadata.pkl",
    )

    print("\nComparison:")
    print("Previous LightGBM ROC-AUC: 94.16%")
    print("Invalid leaked-model ROC-AUC: 100.00%")
    print(f"Clean LightGBM ROC-AUC: {metrics['ROC-AUC'] * 100:.2f}%")
    print(
        "The clean result is the valid comparison because cost_increase and "
        "known future-derived fields were removed."
    )


if __name__ == "__main__":
    main()
