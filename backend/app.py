
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import os
import re
import shap

# ============================================================
# CREATE FLASK APP
# ============================================================

app = Flask(__name__)

# Allow frontend to communicate with backend
CORS(app)

# ============================================================
# LOAD DASHBOARD DATA
# ============================================================

DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "PAIMANA_DASHBOARD_DATA.csv"
)

DATA_FILE = os.path.abspath(DATA_FILE)

df = pd.read_csv(DATA_FILE)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_FILE = os.path.join(PROJECT_ROOT, "lightgbm_clean_model.txt")
JULY_FEATURE_FILE = os.path.join(PROJECT_ROOT, "JULY_2026_FEATURE_DATASET.csv")
NUMERIC_IMPUTER_FILE = os.path.join(PROJECT_ROOT, "lightgbm_numeric_imputer.pkl")
CATEGORICAL_IMPUTER_FILE = os.path.join(
    PROJECT_ROOT,
    "lightgbm_categorical_imputer.pkl",
)
ONEHOT_ENCODER_FILE = os.path.join(
    PROJECT_ROOT,
    "lightgbm_onehot_encoder.pkl",
)
FEATURE_NAMES_FILE = os.path.join(PROJECT_ROOT, "lightgbm_feature_names.pkl")
PREPROCESSING_METADATA_FILE = os.path.join(
    PROJECT_ROOT,
    "lightgbm_preprocessing_metadata.pkl",
)
TIME_DELAY_MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "time_delay_model.txt",
)
TIME_DELAY_NUMERIC_IMPUTER_FILE = os.path.join(
    PROJECT_ROOT,
    "time_delay_numeric_imputer.pkl",
)
TIME_DELAY_CATEGORICAL_IMPUTER_FILE = os.path.join(
    PROJECT_ROOT,
    "time_delay_categorical_imputer.pkl",
)
TIME_DELAY_ENCODER_FILE = os.path.join(
    PROJECT_ROOT,
    "time_delay_onehot_encoder.pkl",
)
TIME_DELAY_FEATURE_NAMES_FILE = os.path.join(
    PROJECT_ROOT,
    "time_delay_feature_names.pkl",
)
TIME_DELAY_METADATA_FILE = os.path.join(
    PROJECT_ROOT,
    "time_delay_preprocessing_metadata.pkl",
)


class LightGBMProbabilityModel:
    """Expose predict_proba for the native LightGBM model format."""

    def __init__(self, model_file):
        self.booster = lgb.Booster(model_file=model_file)

    def predict_proba(self, features):
        probability = np.asarray(self.booster.predict(features), dtype=float)
        return np.column_stack((1.0 - probability, probability))

    def feature_name(self):
        return self.booster.feature_name()


try:
    prediction_model = LightGBMProbabilityModel(MODEL_FILE)
    shap_explainer = shap.TreeExplainer(
        prediction_model.booster,
        model_output="raw",
    )
    numeric_imputer = joblib.load(NUMERIC_IMPUTER_FILE)
    categorical_imputer = joblib.load(CATEGORICAL_IMPUTER_FILE)
    onehot_encoder = joblib.load(ONEHOT_ENCODER_FILE)
    feature_names = joblib.load(FEATURE_NAMES_FILE)
    preprocessing_metadata = joblib.load(PREPROCESSING_METADATA_FILE)
    july_features = pd.read_csv(JULY_FEATURE_FILE)
    july_features["project_code"] = (
        july_features["project_code"].astype(str).str.strip()
    )
    july_features["report_month"] = pd.to_datetime(
        july_features["report_month"],
        errors="coerce",
    )
    print("LightGBM model loaded:", MODEL_FILE)
    print("LightGBM preprocessing artifacts loaded")
    print("Model features:", len(prediction_model.feature_name()))
    print("July dataset rows:", len(july_features))
except FileNotFoundError as exc:
    prediction_model = None
    shap_explainer = None
    numeric_imputer = None
    categorical_imputer = None
    onehot_encoder = None
    feature_names = None
    preprocessing_metadata = None
    july_features = None
    print("LightGBM prediction assets missing:", exc)
except Exception as exc:
    prediction_model = None
    shap_explainer = None
    numeric_imputer = None
    categorical_imputer = None
    onehot_encoder = None
    feature_names = None
    preprocessing_metadata = None
    july_features = None
    print("LightGBM prediction assets could not be loaded:", exc)

try:
    time_delay_model = lgb.Booster(model_file=TIME_DELAY_MODEL_FILE)
    time_delay_numeric_imputer = joblib.load(
        TIME_DELAY_NUMERIC_IMPUTER_FILE
    )
    time_delay_categorical_imputer = joblib.load(
        TIME_DELAY_CATEGORICAL_IMPUTER_FILE
    )
    time_delay_onehot_encoder = joblib.load(TIME_DELAY_ENCODER_FILE)
    time_delay_feature_names = joblib.load(TIME_DELAY_FEATURE_NAMES_FILE)
    time_delay_metadata = joblib.load(TIME_DELAY_METADATA_FILE)
    print("Time Delay model loaded:", TIME_DELAY_MODEL_FILE)
    print("Time Delay preprocessing artifacts loaded")
except FileNotFoundError as exc:
    time_delay_model = None
    time_delay_numeric_imputer = None
    time_delay_categorical_imputer = None
    time_delay_onehot_encoder = None
    time_delay_feature_names = None
    time_delay_metadata = None
    print("Time Delay prediction assets missing:", exc)
except Exception as exc:
    time_delay_model = None
    time_delay_numeric_imputer = None
    time_delay_categorical_imputer = None
    time_delay_onehot_encoder = None
    time_delay_feature_names = None
    time_delay_metadata = None
    print("Time Delay prediction assets could not be loaded:", exc)

try:
    time_delay_shap_explainer = shap.TreeExplainer(
        time_delay_model,
        model_output="raw",
    ) if time_delay_model is not None else None
except Exception as exc:
    time_delay_shap_explainer = None
    print("Time Delay SHAP explainer could not be loaded:", exc)

print("=" * 70)
print("PAIMANA BACKEND")
print("=" * 70)

print("Data file:", DATA_FILE)
print("Projects loaded:", len(df))
print("Columns:", len(df.columns))

# ============================================================
# BASIC CLEANING
# ============================================================

df["project_code"] = df["project_code"].astype(str)

df["risk_level"] = (
    df["risk_level"]
    .fillna("UNKNOWN")
    .astype(str)
    .str.upper()
)


def records_without_nan(frame):
    """Convert dataframe records to JSON-safe values."""
    return frame.astype(object).where(pd.notna(frame), None).to_dict(
        orient="records"
    )


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


def prepare_prediction_features(project_row):
    if preprocessing_metadata is None:
        raise RuntimeError("LightGBM preprocessing artifacts are unavailable.")

    dropped = preprocessing_metadata["dropped_columns"]
    numeric = preprocessing_metadata["numeric_columns"]
    categorical = preprocessing_metadata["categorical_columns"]
    required = [column for column in numeric + categorical if column not in project_row]
    if required:
        raise RuntimeError(
            "Required model feature columns are missing: "
            + ", ".join(required)
        )

    features = project_row.drop(
        columns=[column for column in dropped if column in project_row],
        errors="ignore",
    ).to_frame().T.reset_index(drop=True)
    date_columns = features.select_dtypes(
        include=["datetime64[ns]"]
    ).columns.tolist()
    features = features.drop(columns=date_columns)

    for column in numeric:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan)
    if numeric:
        features[numeric] = numeric_imputer.transform(features[numeric])

    if categorical:
        categorical_values = categorical_imputer.transform(features[categorical])
        encoded = onehot_encoder.transform(categorical_values)
        encoded_names = onehot_encoder.get_feature_names_out(categorical)
        features = pd.concat(
            [
                features.drop(columns=categorical),
                pd.DataFrame(encoded, columns=encoded_names),
            ],
            axis=1,
        )

    features = features.reindex(
        columns=[column for column in features.columns if column in numeric
                 or column in onehot_encoder.get_feature_names_out(categorical)],
        fill_value=0,
    )
    features.columns = make_unique(
        [clean_feature_name(column) for column in features.columns]
    )
    return features.reindex(columns=feature_names, fill_value=0)


def prepare_time_delay_features(project_row):
    if (
        time_delay_metadata is None
        or time_delay_numeric_imputer is None
        or time_delay_categorical_imputer is None
        or time_delay_onehot_encoder is None
        or time_delay_feature_names is None
    ):
        raise RuntimeError(
            "Time Delay preprocessing artifacts are unavailable."
        )

    numeric = time_delay_metadata["numeric_columns"]
    categorical = time_delay_metadata["categorical_columns"]
    required = [
        column
        for column in numeric + categorical
        if column not in project_row
    ]
    if required:
        raise RuntimeError(
            "Required Time Delay feature columns are missing: "
            + ", ".join(required)
        )

    numeric_values = project_row[numeric].to_frame().T.copy()
    for column in numeric:
        numeric_values[column] = pd.to_numeric(
            numeric_values[column],
            errors="coerce",
        )
    numeric_values = numeric_values.replace([np.inf, -np.inf], np.nan)
    numeric_values = time_delay_numeric_imputer.transform(numeric_values)

    categorical_values = time_delay_categorical_imputer.transform(
        project_row[categorical].to_frame().T
    )
    encoded_values = time_delay_onehot_encoder.transform(categorical_values)
    encoded_names = (
        time_delay_onehot_encoder
        .get_feature_names_out(categorical)
        .tolist()
    )
    transformed = pd.DataFrame(
        np.column_stack([numeric_values, encoded_values]),
        columns=numeric + encoded_names,
    )
    return transformed.reindex(
        columns=time_delay_feature_names,
        fill_value=0,
    )


def risk_level(probability):
    if probability >= 0.70:
        return "HIGH"
    if probability >= 0.40:
        return "MEDIUM"
    return "LOW"


def readable_feature_name(feature):
    prefixes = {
        "state_": "State: ",
        "sector_": "Sector: ",
        "ministry_final_": "Ministry: ",
        "agency_final_": "Agency: ",
    }
    for prefix, label in prefixes.items():
        if feature.startswith(prefix):
            value = feature[len(prefix):]
            value = value.replace("___", " & ").replace("__", " / ")
            value = value.replace("_", " ")
            return label + value

    return feature.replace("_", " ").title()


def positive_class_shap_values(shap_values):
    if isinstance(shap_values, list):
        if len(shap_values) == 1:
            shap_values = shap_values[0]
        elif len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            raise RuntimeError("Unsupported SHAP list output format.")

    values = np.asarray(shap_values)
    if values.ndim == 3:
        if values.shape[0] != 1 or values.shape[2] < 2:
            raise RuntimeError("Unsupported SHAP array output format.")
        values = values[0, :, 1]
    elif values.ndim == 2:
        if values.shape[0] != 1:
            raise RuntimeError("Expected one-row SHAP output.")
        values = values[0]
    elif values.ndim != 1:
        raise RuntimeError("Unsupported SHAP output format.")
    return values.astype(float)


def shap_base_value(explainer):
    expected = np.asarray(explainer.expected_value)
    if expected.ndim == 0:
        return float(expected)
    if expected.size == 1:
        return float(expected.reshape(-1)[0])
    return float(expected.reshape(-1)[-1])


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "message": "PAIMANA AI Project Monitoring API",
        "status": "running",
        "projects": len(df)
    })


# ============================================================
# SUMMARY API
# ============================================================

@app.route("/api/summary")
def summary():

    total_projects = len(df)

    high = int(
        (df["risk_level"] == "HIGH").sum()
    )

    medium = int(
        (df["risk_level"] == "MEDIUM").sum()
    )

    low = int(
        (df["risk_level"] == "LOW").sum()
    )

    average_probability = round(
        df["overrun_probability_percent"]
        .mean(),
        2
    )

    return jsonify({

        "total_projects": total_projects,

        "high_risk": high,

        "medium_risk": medium,

        "low_risk": low,

        "average_risk_probability":
            average_probability
    })


# ============================================================
# ALL PROJECTS API
# ============================================================

@app.route("/api/projects")
def projects():

    result = df.copy()

    # Optional risk filter
    risk = request.args.get("risk")

    if risk:

        result = result[
            result["risk_level"]
            == risk.upper()
        ]

    # Optional search
    search = request.args.get("search")

    if search:

        search = search.lower()

        mask = (
            result["project_name"]
            .fillna("")
            .str.lower()
            .str.contains(search, na=False)
        )

        result = result[mask]

    # Sort by risk probability
    result = result.sort_values(
        "overrun_probability_percent",
        ascending=False
    )

    return jsonify(records_without_nan(result))


# ============================================================
# SINGLE PROJECT API
# ============================================================

@app.route("/api/projects/<project_code>")
def project_details(project_code):

    result = df[
        df["project_code"]
        == str(project_code)
    ]

    if result.empty:

        return jsonify({
            "error": "Project not found"
        }), 404

    project = records_without_nan(result.iloc[[0]])[0]

    return jsonify(project)


@app.route("/api/projects/<project_code>/prediction")
def project_prediction(project_code):
    if prediction_model is None or july_features is None:
        return jsonify({
            "error": "LightGBM model or preprocessing artifacts are unavailable."
        }), 500

    result = july_features[
        (july_features["project_code"] == str(project_code))
        & (july_features["report_month"] == pd.Timestamp("2026-07-01"))
    ]
    if result.empty:
        return jsonify({"error": "July 2026 project not found"}), 404

    try:
        features = prepare_prediction_features(result.iloc[0])
        probability = float(prediction_model.predict_proba(features)[0, 1])
        threshold = float(preprocessing_metadata["threshold"])
        prediction = (
            "Cost Overrun" if probability >= threshold else "No Overrun"
        )
        project = result.iloc[0]
        return jsonify({
            "project_code": str(project["project_code"]),
            "project_name": project.get("project_name"),
            "report_month": "2026-07-01",
            "overrun_probability": probability,
            "overrun_probability_percent": round(probability * 100, 2),
            "prediction": prediction,
            "risk_level": risk_level(probability),
            "threshold": threshold,
        })
    except Exception as exc:
        print("LightGBM prediction failed:", exc)
        return jsonify({
            "error": "LightGBM prediction failed",
            "details": str(exc),
        }), 500


@app.route("/api/projects/<project_code>/time-delay")
def project_time_delay(project_code):
    if time_delay_model is None or july_features is None:
        return jsonify({
            "error": "Time Delay model or preprocessing artifacts are unavailable."
        }), 500

    result = july_features[
        (july_features["project_code"] == str(project_code))
        & (july_features["report_month"] == pd.Timestamp("2026-07-01"))
    ]
    if result.empty:
        return jsonify({"error": "July 2026 project not found"}), 404

    try:
        project = result.iloc[0]
        features = prepare_time_delay_features(project)
        probability = float(time_delay_model.predict(features)[0])
        threshold = float(time_delay_metadata["threshold"])
        prediction = (
            "Current Schedule Slippage Risk"
            if probability >= threshold
            else "No Current Schedule Slippage Risk"
        )
        return jsonify({
            "project_code": str(project["project_code"]),
            "project_name": project.get("project_name"),
            "report_month": "2026-07-01",
            "time_delay_probability": probability,
            "time_delay_probability_percent": round(probability * 100, 2),
            "prediction": prediction,
            "risk_level": risk_level(probability),
            "threshold": threshold,
        })
    except Exception as exc:
        print("Time Delay prediction failed:", exc)
        return jsonify({
            "error": "Time Delay prediction failed",
            "details": str(exc),
        }), 500


@app.route("/api/projects/<project_code>/time-delay-explanation")
def project_time_delay_explanation(project_code):
    if (
        time_delay_model is None
        or time_delay_shap_explainer is None
        or july_features is None
    ):
        return jsonify({
            "error": (
                "Time Delay model, preprocessing artifacts, or SHAP "
                "is unavailable."
            )
        }), 500

    result = july_features[
        (july_features["project_code"] == str(project_code))
        & (july_features["report_month"] == pd.Timestamp("2026-07-01"))
    ]
    if result.empty:
        return jsonify({"error": "July 2026 project not found"}), 404

    try:
        project = result.iloc[0]
        features = prepare_time_delay_features(project)
        probability = float(time_delay_model.predict(features)[0])

        shap_output = time_delay_shap_explainer.shap_values(
            features,
            check_additivity=True,
        )
        contributions = positive_class_shap_values(shap_output)
        if len(contributions) != len(time_delay_feature_names):
            raise RuntimeError(
                "Time Delay SHAP output does not match the model feature count."
            )

        base_value = shap_base_value(time_delay_shap_explainer)
        raw_prediction = float(
            time_delay_model.predict(
                features,
                raw_score=True,
            )[0]
        )
        additivity_error = float(
            base_value + float(contributions.sum()) - raw_prediction
        )

        contribution_frame = pd.DataFrame({
            "feature": time_delay_feature_names,
            "shap_value": contributions,
        })
        increasing = contribution_frame[
            contribution_frame["shap_value"] > 0
        ].sort_values("shap_value", ascending=False).head(10)
        decreasing = contribution_frame[
            contribution_frame["shap_value"] < 0
        ].sort_values("shap_value", ascending=True).head(10)

        def format_contributions(frame, direction):
            return [
                {
                    "feature": readable_feature_name(row["feature"]),
                    "shap_value": float(row["shap_value"]),
                    "direction": direction,
                }
                for _, row in frame.iterrows()
            ]

        return jsonify({
            "project_code": str(project["project_code"]),
            "project_name": project.get("project_name"),
            "time_delay_probability": probability,
            "risk_level": risk_level(probability),
            "risk_factors": format_contributions(
                increasing,
                "increases_risk",
            ),
            "protective_factors": format_contributions(
                decreasing,
                "decreases_risk",
            ),
            "shap_additivity_error": additivity_error,
        })
    except Exception as exc:
        print("Time Delay SHAP explanation failed:", exc)
        return jsonify({
            "error": "Time Delay SHAP explanation failed",
            "details": str(exc),
        }), 500


@app.route("/api/projects/<project_code>/explanation")
def project_explanation(project_code):
    if (
        prediction_model is None
        or shap_explainer is None
        or july_features is None
    ):
        return jsonify({
            "error": "LightGBM model, preprocessing artifacts, or SHAP is unavailable."
        }), 500

    result = july_features[
        (july_features["project_code"] == str(project_code))
        & (july_features["report_month"] == pd.Timestamp("2026-07-01"))
    ]
    if result.empty:
        return jsonify({"error": "July 2026 project not found"}), 404

    try:
        features = prepare_prediction_features(result.iloc[0])
        probability = float(prediction_model.predict_proba(features)[0, 1])
        threshold = float(preprocessing_metadata["threshold"])
        prediction = (
            "Cost Overrun" if probability >= threshold else "No Overrun"
        )

        shap_output = shap_explainer.shap_values(
            features,
            check_additivity=True,
        )
        contributions = positive_class_shap_values(shap_output)
        if len(contributions) != len(feature_names):
            raise RuntimeError(
                "SHAP output does not match the model feature count."
            )

        base_value = shap_base_value(shap_explainer)
        raw_prediction = float(
            prediction_model.booster.predict(
                features,
                raw_score=True,
            )[0]
        )
        additivity_error = float(
            base_value + float(contributions.sum()) - raw_prediction
        )

        contribution_frame = pd.DataFrame({
            "feature": feature_names,
            "shap_value": contributions,
        })
        increasing = contribution_frame[
            contribution_frame["shap_value"] > 0
        ].sort_values("shap_value", ascending=False).head(10)
        decreasing = contribution_frame[
            contribution_frame["shap_value"] < 0
        ].sort_values("shap_value", ascending=True).head(10)

        def format_contributions(frame, direction):
            return [
                {
                    "feature": readable_feature_name(row["feature"]),
                    "shap_value": float(row["shap_value"]),
                    "direction": direction,
                }
                for _, row in frame.iterrows()
            ]

        project = result.iloc[0]
        return jsonify({
            "project_code": str(project["project_code"]),
            "project_name": project.get("project_name"),
            "report_month": "2026-07-01",
            "overrun_probability": probability,
            "overrun_probability_percent": round(probability * 100, 2),
            "prediction": prediction,
            "risk_level": risk_level(probability),
            "threshold": threshold,
            "top_risk_factors": format_contributions(
                increasing,
                "increases_risk",
            ),
            "top_protective_factors": format_contributions(
                decreasing,
                "decreases_risk",
            ),
            "explanation_note": (
                "These features contributed most to the model's risk prediction; "
                "they should not be interpreted as causal effects."
            ),
            "shap_output_space": "raw LightGBM margin",
            "shap_base_value": base_value,
            "shap_additivity_error": additivity_error,
        })
    except Exception as exc:
        print("SHAP explanation failed:", exc)
        return jsonify({
            "error": "SHAP explanation failed",
            "details": str(exc),
        }), 500


# ============================================================
# TOP HIGH-RISK PROJECTS
# ============================================================

@app.route("/api/top-risk")
def top_risk():

    limit = request.args.get(
        "limit",
        default=10,
        type=int
    )

    result = df.sort_values(
        "overrun_probability_percent",
        ascending=False
    ).head(limit)

    return jsonify(records_without_nan(result))


# ============================================================
# STATE-WISE SUMMARY
# ============================================================

@app.route("/api/states")
def states():

    result = (
        df.groupby("state")
        .agg(
            total_projects=(
                "project_code",
                "count"
            ),
            high_risk=(
                "risk_level",
                lambda x:
                (x == "HIGH").sum()
            ),
            medium_risk=(
                "risk_level",
                lambda x:
                (x == "MEDIUM").sum()
            ),
            low_risk=(
                "risk_level",
                lambda x:
                (x == "LOW").sum()
            )
        )
        .reset_index()
    )

    result["high_risk_percent"] = (
        result["high_risk"]
        / result["total_projects"]
        * 100
    ).round(2)

    result = result.sort_values(
        "high_risk_percent",
        ascending=False
    )

    return jsonify(records_without_nan(result))


# ============================================================
# SECTOR-WISE SUMMARY
# ============================================================

@app.route("/api/sectors")
def sectors():

    result = (
        df.groupby("sector")
        .agg(
            total_projects=(
                "project_code",
                "count"
            ),
            high_risk=(
                "risk_level",
                lambda x:
                (x == "HIGH").sum()
            ),
            medium_risk=(
                "risk_level",
                lambda x:
                (x == "MEDIUM").sum()
            ),
            low_risk=(
                "risk_level",
                lambda x:
                (x == "LOW").sum()
            )
        )
        .reset_index()
    )

    result["high_risk_percent"] = (
        result["high_risk"]
        / result["total_projects"]
        * 100
    ).round(2)

    result = result.sort_values(
        "high_risk_percent",
        ascending=False
    )

    return jsonify(records_without_nan(result))


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("\nStarting PAIMANA backend...")
    print("Open: http://127.0.0.1:5001")

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )
