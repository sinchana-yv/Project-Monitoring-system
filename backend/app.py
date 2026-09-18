from datetime import date
from io import BytesIO
from datetime import datetime, timedelta, timezone
import hashlib
import secrets
import smtplib
import sqlite3
from email.message import EmailMessage

from flask import Flask, jsonify, request, send_file, session
from flask_cors import CORS
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import os
import re
from werkzeug.security import check_password_hash, generate_password_hash
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from notification_service import (
    ensure_notification_table,
    generate_notifications_for_user,
    notification_record,
)

# ============================================================
# CREATE FLASK APP
# ============================================================

app = Flask(__name__)

app.config.update(
    SECRET_KEY=os.environ.get("AVLOKAN_SECRET_KEY") or os.environ.get("NIRIKSHAN_SECRET_KEY") or secrets.token_hex(32),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("AVLOKAN_COOKIE_SECURE", os.environ.get("NIRIKSHAN_COOKIE_SECURE", "0")) == "1",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)

# Allow frontend to communicate with backend
CORS(app, supports_credentials=True)

AUTH_DB_FILE = os.path.join(os.path.dirname(__file__), "nirikshan_auth.db")
RESET_TOKEN_LIFETIME = timedelta(minutes=30)


def auth_db_connection():
    connection = sqlite3.connect(AUTH_DB_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_auth_database():
    with auth_db_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                contact_number TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                ministry TEXT NOT NULL,
                organization TEXT NOT NULL,
                state TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                expires_at TEXT NOT NULL,
                used_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )


initialize_auth_database()
with auth_db_connection() as connection:
    ensure_notification_table(connection)

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
HISTORY_FEATURE_FILE = os.path.join(PROJECT_ROOT, "DELAY_FEATURE_DATASET.csv")
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
    history_features = pd.read_csv(HISTORY_FEATURE_FILE)
    history_features["project_code"] = (
        history_features["project_code"].astype(str).str.strip()
    )
    history_features["report_month"] = pd.to_datetime(
        history_features["report_month"],
        errors="coerce",
    )
    print("LightGBM model loaded:", MODEL_FILE)
    print("LightGBM preprocessing artifacts loaded")
    print("Model features:", len(prediction_model.feature_name()))
    print("July dataset rows:", len(july_features))
except FileNotFoundError as exc:
    prediction_model = None
    numeric_imputer = None
    categorical_imputer = None
    onehot_encoder = None
    feature_names = None
    preprocessing_metadata = None
    july_features = None
    history_features = None
    print("LightGBM prediction assets missing:", exc)
except Exception as exc:
    prediction_model = None
    numeric_imputer = None
    categorical_imputer = None
    onehot_encoder = None
    feature_names = None
    preprocessing_metadata = None
    july_features = None
    history_features = None
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

print("=" * 70)
print("AVLOKAN BACKEND")
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


def auth_error(message, status=400):
    return jsonify({"error": message}), status


def request_json():
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def normalized_email(value):
    return str(value or "").strip().lower()


def valid_password(password):
    password = str(password or "")
    return (
        len(password) >= 8
        and re.search(r"[A-Z]", password)
        and re.search(r"[a-z]", password)
        and re.search(r"\d", password)
        and re.search(r"[^A-Za-z0-9]", password)
    )


def user_response(user):
    return {
        "id": user["id"],
        "first_name": user["first_name"],
        "last_name": user["last_name"],
        "email": user["email"],
        "contact_number": user["contact_number"],
        "ministry": user["ministry"],
        "organization": user["organization"],
        "state": user["state"],
    }


def current_auth_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    with auth_db_connection() as connection:
        return connection.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()


def send_password_reset_email(user, token):
    smtp_host = os.environ.get("AVLOKAN_SMTP_HOST") or os.environ.get("NIRIKSHAN_SMTP_HOST")
    smtp_port = int(os.environ.get("AVLOKAN_SMTP_PORT") or os.environ.get("NIRIKSHAN_SMTP_PORT", "587"))
    smtp_username = os.environ.get("AVLOKAN_SMTP_USERNAME") or os.environ.get("NIRIKSHAN_SMTP_USERNAME")
    smtp_password = os.environ.get("AVLOKAN_SMTP_PASSWORD") or os.environ.get("NIRIKSHAN_SMTP_PASSWORD")
    sender = os.environ.get("AVLOKAN_SMTP_FROM") or os.environ.get("NIRIKSHAN_SMTP_FROM", smtp_username or "")
    frontend_url = (
        os.environ.get("AVLOKAN_FRONTEND_URL")
        or os.environ.get("NIRIKSHAN_FRONTEND_URL", "http://127.0.0.1:5173")
    ).rstrip("/")

    if not all([smtp_host, smtp_username, smtp_password, sender]):
        raise RuntimeError(
            "Password reset email is not configured. Set "
            "AVLOKAN_SMTP_HOST, AVLOKAN_SMTP_PORT, "
            "AVLOKAN_SMTP_USERNAME, AVLOKAN_SMTP_PASSWORD, "
            "AVLOKAN_SMTP_FROM, and AVLOKAN_FRONTEND_URL."
        )

    reset_url = f"{frontend_url}/reset-password?token={token}"
    message = EmailMessage()
    message["Subject"] = "AVLOKAN password reset"
    message["From"] = sender
    message["To"] = user["email"]
    message.set_content(
        "A password reset was requested for your AVLOKAN account.\n\n"
        f"Open this secure link to create a new password:\n{reset_url}\n\n"
        "This link expires in 30 minutes. If you did not request this, "
        "you can ignore this email."
    )

    with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as smtp:
        smtp.starttls()
        smtp.login(smtp_username, smtp_password)
        smtp.send_message(message)


@app.post("/api/auth/register")
def register_user():
    payload = request_json()
    fields = {
        "first_name": str(payload.get("first_name", "")).strip(),
        "last_name": str(payload.get("last_name", "")).strip(),
        "contact_number": str(payload.get("contact_number", "")).strip(),
        "email": normalized_email(payload.get("email")),
        "ministry": str(payload.get("ministry", "")).strip(),
        "organization": str(payload.get("organization", "")).strip(),
        "state": str(payload.get("state", "")).strip(),
        "password": str(payload.get("password", "")),
    }
    errors = {}
    name_pattern = re.compile(r"^[A-Za-z][A-Za-z .'-]{0,79}$")
    if not name_pattern.fullmatch(fields["first_name"]):
        errors["first_name"] = "First Name is required and must contain valid characters."
    if not name_pattern.fullmatch(fields["last_name"]):
        errors["last_name"] = "Last Name is required and must contain valid characters."
    if not re.fullmatch(r"(?:\+91[- ]?)?[6-9]\d{9}", fields["contact_number"]):
        errors["contact_number"] = "Enter a valid Indian mobile number."
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", fields["email"]):
        errors["email"] = "Enter a valid email address."
    for field, label in [
        ("ministry", "Ministry/Department"),
        ("organization", "Organization"),
        ("state", "State"),
    ]:
        if not fields[field]:
            errors[field] = f"{label} is required."
    if not valid_password(fields["password"]):
        errors["password"] = (
            "Password must be at least 8 characters and include uppercase, "
            "lowercase, number, and special character."
        )
    if payload.get("password") != payload.get("confirm_password"):
        errors["confirm_password"] = "Passwords do not match."
    if errors:
        return jsonify({"errors": errors}), 400

    try:
        with auth_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    first_name, last_name, contact_number, email, ministry,
                    organization, state, password_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fields["first_name"],
                    fields["last_name"],
                    fields["contact_number"],
                    fields["email"],
                    fields["ministry"],
                    fields["organization"],
                    fields["state"],
                    generate_password_hash(fields["password"]),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    except sqlite3.IntegrityError:
        return auth_error("An account with this email already exists.", 409)

    return jsonify({
        "message": "Registration submitted successfully. Your account is ready for login."
    }), 201


@app.post("/api/auth/login")
def login_user():
    payload = request_json()
    email = normalized_email(payload.get("email"))
    password = str(payload.get("password", ""))
    with auth_db_connection() as connection:
        user = connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    if user is None or not check_password_hash(user["password_hash"], password):
        return auth_error(
            "Invalid email or password. Please check your credentials and try again.",
            401,
        )
    session.clear()
    session.permanent = True
    session["user_id"] = user["id"]
    return jsonify({"user": user_response(user)})


@app.get("/api/auth/me")
def auth_me():
    user = current_auth_user()
    if user is None:
        return auth_error("Authentication required.", 401)
    return jsonify({"user": user_response(user)})


@app.post("/api/auth/logout")
def logout_user():
    session.clear()
    return jsonify({"message": "Logged out successfully."})


@app.post("/api/auth/forgot-password")
def forgot_password():
    payload = request_json()
    email = normalized_email(payload.get("email"))
    generic_message = (
        "If an account exists for this email, a password reset link has been sent. "
        "Please check your inbox and spam folder."
    )
    with auth_db_connection() as connection:
        user = connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,),
        ).fetchone()
        if user is None:
            return jsonify({"message": generic_message})
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        expires_at = (datetime.now(timezone.utc) + RESET_TOKEN_LIFETIME).isoformat()
        connection.execute(
            "DELETE FROM password_reset_tokens WHERE user_id = ? OR expires_at < ?",
            (user["id"], datetime.now(timezone.utc).isoformat()),
        )
        connection.execute(
            "INSERT INTO password_reset_tokens (user_id, token_hash, expires_at) VALUES (?, ?, ?)",
            (user["id"], token_hash, expires_at),
        )
    try:
        send_password_reset_email(user, raw_token)
    except Exception as exc:
        with auth_db_connection() as connection:
            connection.execute(
                "DELETE FROM password_reset_tokens WHERE token_hash = ?",
                (token_hash,),
            )
        print("Password reset email delivery failed:", exc)
        return auth_error(
            "Unable to send the password reset email. Configure the SMTP environment variables and try again.",
            503,
        )
    return jsonify({"message": generic_message})


@app.post("/api/auth/reset-password")
def reset_password():
    payload = request_json()
    raw_token = str(payload.get("token", ""))
    password = str(payload.get("password", ""))
    if not raw_token or not valid_password(password):
        return auth_error(
            "Password must be at least 8 characters and include uppercase, lowercase, number, and special character."
        )
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc)
    with auth_db_connection() as connection:
        token = connection.execute(
            "SELECT * FROM password_reset_tokens WHERE token_hash = ? AND used_at IS NULL",
            (token_hash,),
        ).fetchone()
        if token is None:
            return auth_error(
                "This password reset link is invalid or has expired. Please request a new reset link.",
                400,
            )
        expires_at = datetime.fromisoformat(token["expires_at"])
        if expires_at <= now:
            return auth_error(
                "This password reset link is invalid or has expired. Please request a new reset link.",
                400,
            )
        connection.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (generate_password_hash(password), token["user_id"]),
        )
        connection.execute(
            "UPDATE password_reset_tokens SET used_at = ? WHERE id = ?",
            (now.isoformat(), token["id"]),
        )
    return jsonify({
        "message": "Your password has been reset successfully. You can now sign in with your new password."
    })


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

    project_row = project_row.copy()
    numeric = time_delay_metadata["numeric_columns"]
    categorical = time_delay_metadata["categorical_columns"]
    for column in numeric + categorical:
        if column not in project_row:
            project_row[column] = np.nan

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


def predict_historical_risk(project_row):
    if prediction_model is None or preprocessing_metadata is None:
        return None
    try:
        project_row = project_row.copy()
        required_columns = (
            preprocessing_metadata["numeric_columns"]
            + preprocessing_metadata["categorical_columns"]
        )
        for column in required_columns:
            if column not in project_row:
                project_row[column] = np.nan
        features = prepare_prediction_features(project_row)
        probability = float(prediction_model.predict_proba(features)[0, 1])
        return {
            "probability": probability,
            "risk_level": risk_level(probability),
        }
    except Exception as exc:
        print("Historical risk calculation failed:", exc)
        return None


def prepare_historical_prediction_features(frame):
    if prediction_model is None or preprocessing_metadata is None:
        return pd.DataFrame()
    dropped = preprocessing_metadata["dropped_columns"]
    numeric = preprocessing_metadata["numeric_columns"]
    categorical = preprocessing_metadata["categorical_columns"]
    features = frame.copy()
    for column in numeric + categorical:
        if column not in features:
            features[column] = np.nan
    features = features.drop(columns=dropped, errors="ignore")
    date_columns = features.select_dtypes(
        include=["datetime64[ns]"]
    ).columns.tolist()
    features = features.drop(columns=date_columns)
    for column in numeric:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan)
    numeric_values = numeric_imputer.transform(features[numeric])
    categorical_values = categorical_imputer.transform(features[categorical])
    encoded = onehot_encoder.transform(categorical_values)
    encoded_names = onehot_encoder.get_feature_names_out(categorical)
    transformed = pd.concat(
        [
            pd.DataFrame(numeric_values, columns=numeric, index=features.index),
            pd.DataFrame(
                encoded,
                columns=encoded_names,
                index=features.index,
            ),
        ],
        axis=1,
    )
    transformed.columns = make_unique(
        [clean_feature_name(column) for column in transformed.columns]
    )
    return transformed.reindex(columns=feature_names, fill_value=0)


def generate_current_user_notifications(user_id):
    if history_features is None:
        return 0
    predicted_history = history_features.copy()
    try:
        batch_features = prepare_historical_prediction_features(predicted_history)
    except Exception as exc:
        print("Historical risk feature preparation failed:", exc)
        batch_features = pd.DataFrame()
    if not batch_features.empty:
        probabilities = prediction_model.predict_proba(batch_features)[:, 1]
        predicted_history["_risk_probability"] = np.nan
        predicted_history["_risk_level"] = None
        for index, probability in zip(predicted_history.index, probabilities):
            predicted_history.at[index, "_risk_probability"] = float(probability)
            predicted_history.at[index, "_risk_level"] = risk_level(probability)
    else:
        predicted_history = pd.DataFrame()
    with auth_db_connection() as connection:
        return generate_notifications_for_user(
            connection,
            user_id,
            predicted_history,
            predict_historical_risk,
        )


def notification_user():
    user = current_auth_user()
    if user is None:
        return None, auth_error("Authentication required.", 401)
    return user, None


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


def native_shap_explanation(model, features, names, positive_label):
    """Return native LightGBM SHAP contributions without the SHAP Python package."""
    contributions = np.asarray(
        model.predict(features, pred_contrib=True),
        dtype=float,
    )
    if contributions.ndim != 2 or contributions.shape[0] != 1:
        raise RuntimeError("Unexpected native LightGBM contribution shape.")
    if contributions.shape[1] != len(names) + 1:
        raise RuntimeError("Native contributions do not match model features.")

    values = contributions[0, :-1]
    frame = pd.DataFrame({"feature": names, "shap_value": values})
    increasing = frame[frame["shap_value"] > 0].sort_values(
        "shap_value", ascending=False
    ).head(10)
    decreasing = frame[frame["shap_value"] < 0].sort_values(
        "shap_value", ascending=True
    ).head(10)

    def format_factors(source, direction):
        return [
            {
                "feature": readable_feature_name(row["feature"]),
                "feature_name": str(row["feature"]),
                "shap_value": float(row["shap_value"]),
                "direction": direction,
                "explanation": (
                    f"{readable_feature_name(row['feature'])} "
                    f"{direction} the model's raw {positive_label} score by "
                    f"{abs(float(row['shap_value'])):.4f}."
                ),
            }
            for _, row in source.iterrows()
        ]

    return {
        "risk_factors": format_factors(increasing, "increases"),
        "protective_factors": format_factors(decreasing, "decreases"),
        "shap_base_value": float(contributions[0, -1]),
        "shap_value_sum": float(values.sum()),
        "shap_method": "Native LightGBM pred_contrib SHAP values",
    }


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/")
def home():
    return jsonify({
        "message": "AVLOKAN AI Project Monitoring API",
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


def report_value(value, suffix="", decimals=2):
    if value is None or pd.isna(value):
        return "Unavailable"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{decimals}f}{suffix}"
    return f"{value}{suffix}"


def report_text(value):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return "Unavailable"
    return str(value)


def report_table(rows, widths):
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C4CF")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def build_project_report(project):
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=19, leading=23, textColor=colors.HexColor("#17324D"),
        alignment=TA_LEFT, spaceAfter=4,
    )
    subtitle = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], fontName="Helvetica",
        fontSize=9.5, leading=13, textColor=colors.HexColor("#52606D"),
        spaceAfter=12,
    )
    section = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=12, leading=15, textColor=colors.HexColor("#17324D"),
        spaceBefore=12, spaceAfter=7,
    )
    body = ParagraphStyle(
        "ReportBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9, leading=12, textColor=colors.HexColor("#263238"),
        spaceAfter=5,
    )
    small = ParagraphStyle(
        "ReportSmall", parent=body, fontSize=8, leading=10,
        textColor=colors.HexColor("#52606D"),
    )
    bullet = ParagraphStyle(
        "ReportBullet", parent=body, leftIndent=10, firstLineIndent=-7,
        spaceAfter=4,
    )
    label = ParagraphStyle(
        "ReportLabel", parent=body, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#17324D"),
    )

    def p(value, style=body):
        from xml.sax.saxutils import escape
        return Paragraph(escape(str(value)), style)

    def visual_indicator(label_text, value, color):
        if value is None or pd.isna(value):
            return Table([[p(label_text, label), p("Unavailable", body)]], colWidths=[62 * mm, 114 * mm])
        bounded = max(0.0, min(float(value), 100.0))
        filled = max(1.0, 94.0 * bounded / 100.0)
        empty = max(1.0, 94.0 - filled)
        bar = Table([["", ""]], colWidths=[filled * mm, empty * mm], rowHeights=[5 * mm])
        bar.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(color)),
            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E3E9EE")),
            ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB7C4")),
        ]))
        value_cell = Table([[bar], [p(f"{bounded:.2f}%", small)]], colWidths=[98 * mm])
        value_cell.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        return Table([[p(label_text, label), value_cell]], colWidths=[62 * mm, 114 * mm])

    def trend_chart(history):
        chart = Drawing(176 * mm, 48 * mm)
        chart.add(Rect(14 * mm, 8 * mm, 150 * mm, 32 * mm, fillColor=colors.white, strokeColor=colors.HexColor("#B8C4CF")))
        chart.add(String(15 * mm, 42 * mm, "Physical progress (%) and expenditure (crore)", fontName="Helvetica-Bold", fontSize=8, fillColor=colors.HexColor("#17324D")))
        if len(history) < 2:
            chart.add(String(55 * mm, 22 * mm, "Insufficient historical observations for a trend chart", fontSize=8, fillColor=colors.HexColor("#52606D")))
            return chart
        progress = pd.to_numeric(history["physical_progress"], errors="coerce")
        spend = pd.to_numeric(history["expenditure"], errors="coerce")
        if progress.notna().sum() < 2 or spend.notna().sum() < 2:
            chart.add(String(55 * mm, 22 * mm, "Insufficient historical values for a trend chart", fontSize=8, fillColor=colors.HexColor("#52606D")))
            return chart
        max_progress = max(float(progress.max()), 1.0)
        max_spend = max(float(spend.max()), 1.0)
        x_points = np.linspace(18 * mm, 158 * mm, len(history))
        previous = None
        for x, value in zip(x_points, progress):
            if pd.isna(value):
                previous = None
                continue
            y = 10 * mm + 25 * mm * float(value) / max_progress
            if previous is not None:
                chart.add(Line(previous[0], previous[1], x, y, strokeColor=colors.HexColor("#1F6F8B"), strokeWidth=1.4))
            previous = (x, y)
        previous = None
        for x, value in zip(x_points, spend):
            if pd.isna(value):
                previous = None
                continue
            y = 10 * mm + 25 * mm * float(value) / max_spend
            if previous is not None:
                chart.add(Line(previous[0], previous[1], x, y, strokeColor=colors.HexColor("#C4513A"), strokeWidth=1.4))
            previous = (x, y)
        chart.add(String(116 * mm, 42 * mm, "Progress", fontSize=7, fillColor=colors.HexColor("#1F6F8B")))
        chart.add(String(140 * mm, 42 * mm, "Expenditure", fontSize=7, fillColor=colors.HexColor("#C4513A")))
        return chart

    def field_rows(fields):
        rows = []
        for index in range(0, len(fields), 2):
            row = []
            for field in fields[index:index + 2]:
                row.extend([p(field[0], label), p(field[1])])
            if len(row) == 2:
                row.extend([p(""), p("")])
            rows.append(row)
        table = Table(rows, colWidths=[33 * mm, 55 * mm, 33 * mm, 55 * mm])
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#D5DDE3")),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return table

    original_cost = project.get("original_cost")
    expenditure = project.get("expenditure")
    ratio = project.get("expenditure_ratio")
    physical = project.get("physical_progress")
    previous_physical = project.get("previous_progress")
    progress_change = project.get("progress_change")
    progress_velocity = project.get("progress_velocity")
    observations = project.get("months_observed")

    history = pd.DataFrame()
    if history_features is not None:
        history = history_features[
            history_features["project_code"] == str(project["project_code"])
        ].sort_values("report_month")

    prediction = None
    july_project = None
    if prediction_model is not None and july_features is not None:
        july_result = july_features[
            (july_features["project_code"] == str(project["project_code"]))
            & (july_features["report_month"] == pd.Timestamp("2026-07-01"))
        ]
        if not july_result.empty:
            july_project = july_result.iloc[0]
            try:
                features = prepare_prediction_features(july_project)
                probability = float(prediction_model.predict_proba(features)[0, 1])
                prediction = {
                    "probability": probability,
                    "risk_level": risk_level(probability),
                    "prediction": "Cost Overrun" if probability >= float(preprocessing_metadata["threshold"]) else "No Overrun",
                }
            except Exception as exc:
                print("Report cost prediction failed:", exc)

    time_delay = None
    if time_delay_model is not None and july_project is not None:
        try:
            features = prepare_time_delay_features(july_project)
            probability = float(time_delay_model.predict(features)[0])
            threshold = float(time_delay_metadata["threshold"])
            time_delay = {
                "probability": probability,
                "risk_level": risk_level(probability),
                "prediction": (
                    "Current Schedule Slippage Risk"
                    if probability >= threshold
                    else "No Current Schedule Slippage Risk"
                ),
            }
        except Exception as exc:
            print("Report time-delay prediction failed:", exc)

    cost_explanation = None
    if prediction is not None and july_project is not None:
        try:
            cost_features = prepare_prediction_features(july_project)
            cost_explanation = native_shap_explanation(
                prediction_model.booster,
                cost_features,
                feature_names,
                "cost-overrun risk",
            )
        except Exception as exc:
            print("Report cost explanation failed:", exc)

    delay_explanation = None
    if time_delay is not None and july_project is not None:
        try:
            delay_features = prepare_time_delay_features(july_project)
            delay_explanation = native_shap_explanation(
                time_delay_model,
                delay_features,
                time_delay_feature_names,
                "schedule-slippage risk",
            )
        except Exception as exc:
            print("Report time-delay explanation failed:", exc)

    warnings = []
    if pd.notna(original_cost) and pd.notna(expenditure) and expenditure > original_cost:
        warnings.append(
            f"Expenditure is {report_value(expenditure, ' crore')} against an original cost of {report_value(original_cost, ' crore')}; cumulative expenditure is above the recorded baseline."
        )
    if pd.notna(physical) and physical < 100:
        warnings.append(
            f"Physical progress is {report_value(physical, '%')}; the project is below 100% recorded physical progress."
        )
    if pd.notna(ratio) and ratio >= 80:
        warnings.append(
            f"Expenditure ratio is {report_value(ratio, '%')}; this is a high monitoring signal in the current record."
        )
    if pd.notna(progress_velocity) and abs(float(progress_velocity)) < 0.1:
        warnings.append(
            f"Progress velocity is {report_value(progress_velocity, ' percentage points/day', 4)}; the recorded progress change is relatively slow."
        )
    if pd.notna(observations) and observations < 6:
        warnings.append(
            f"The record contains {report_value(observations)} observations; limited history can reduce trend confidence."
        )
    if not warnings:
        warnings.append("No rule-based early warning was triggered from the available project fields.")

    actions = []
    if pd.notna(expenditure) and pd.notna(original_cost) and expenditure > original_cost:
        actions.append("Officials may review the expenditure record against the approved baseline and supporting financial documentation.")
    if pd.notna(physical) and physical < 100:
        actions.append("Further verification is recommended for current physical progress and remaining execution activities.")
    if pd.notna(progress_velocity) and abs(float(progress_velocity)) < 0.1:
        actions.append("The project may require closer monitoring of near-term progress updates and the causes of slow movement.")
    if not actions:
        actions.append("Officials may continue routine monitoring and verify future progress and expenditure updates as they become available.")

    trend_rows = [[
        p("Report month", label),
        p("Physical progress", label),
        p("Expenditure", label),
        p("Change from previous record", label),
    ]]
    if history.empty:
        trend_rows.append([p(report_text(project.get("report_month"))), p(report_value(physical, "%")), p(report_value(expenditure, " crore")), p(report_value(progress_change, " pp"))])
    else:
        for _, record in history.tail(12).iterrows():
            trend_rows.append([
                p(report_text(record.get("report_month"))),
                p(report_value(record.get("physical_progress"), "%")),
                p(report_value(record.get("expenditure"), " crore")),
                p(report_value(record.get("progress_change"), " pp")),
            ])

    executive_summary = (
        f"The latest available record shows physical progress of {report_value(physical, '%')} "
        f"and cumulative expenditure of {report_value(expenditure, ' crore')}, corresponding to an "
        f"expenditure ratio of {report_value(ratio, '%')}. "
    )
    if prediction:
        executive_summary += f"The cost-overrun model estimates {report_value(prediction['probability'] * 100, '%')} risk and classifies the project as {prediction['risk_level']}. "
    else:
        executive_summary += "A cost-overrun model result is unavailable for this project. "
    if time_delay:
        executive_summary += f"The time-delay model classifies current schedule-slippage risk as {time_delay['risk_level']}. "
    else:
        executive_summary += "A time-delay model result is unavailable for this project. "
    executive_summary += f"Interpretation should account for {report_value(observations)} recorded observations and the availability of only the supplied historical records."

    story = [
        p("AVLOKAN", title),
        p("Intelligent Infrastructure Project Monitoring and Risk Assessment", subtitle),
        report_table([
            [p("Programme", label), p("SIH26103"), p("Generated", label), p(date.today().isoformat())],
            [p("Project", label), p(report_text(project.get("project_name"))), p("Project code", label), p(report_text(project.get("project_code")))],
        ], [31 * mm, 57 * mm, 31 * mm, 57 * mm]),
        Spacer(1, 5),
        p("This report is an AI-assisted monitoring aid generated by the AVLOKAN prototype. It does not replace official project verification.", small),
        p("1. Executive Project Status", section),
        field_rows([
            ("Project name", report_text(project.get("project_name"))),
            ("Project code", report_text(project.get("project_code"))),
            ("Ministry / department", report_text(project.get("ministry_final"))),
            ("Agency", report_text(project.get("agency_final"))),
            ("State", report_text(project.get("state"))),
            ("Sector", report_text(project.get("sector"))),
            ("Report month", report_text(project.get("report_month"))),
            ("Project status", "Unavailable in the supplied project dataset"),
        ]),
        p(executive_summary, body),
        p("2. Financial Baseline and Progress", section),
        field_rows([
            ("Original project cost", report_value(original_cost, " crore")),
            ("Revised cost", "Unavailable in the supplied project dataset"),
            ("Cumulative expenditure", report_value(expenditure, " crore")),
            ("Expenditure ratio", report_value(ratio, "%")),
            ("Previous expenditure", report_value(project.get("previous_expenditure"), " crore")),
            ("Expenditure change", report_value(project.get("expenditure_change"), " crore")),
            ("Cost variance", "Unavailable: revised cost is not available"),
        ]),
        p("The expenditure ratio is a relationship between recorded cumulative expenditure and the recorded original cost. It is not automatically the same as a confirmed cost overrun.", small),
        p("3. Physical Progress and Execution Trend", section),
        field_rows([
            ("Current physical progress", report_value(physical, "%")),
            ("Previous physical progress", report_value(previous_physical, "%")),
            ("Progress change", report_value(progress_change, " percentage points")),
            ("Progress velocity", report_value(progress_velocity, " percentage points/day", 4)),
            ("Number of observations", report_value(observations)),
            ("Days since first report", report_value(project.get("days_since_first_report"))),
        ]),
        report_table(trend_rows, [42 * mm, 39 * mm, 39 * mm, 56 * mm]),
        trend_chart(history),
        p("4. AI Risk Assessment", section),
    ]

    risk_rows = [[p("Measure", label), p("Result", label)]]
    if prediction:
        risk_rows.extend([
            [p("Cost-overrun probability"), p(report_value(prediction["probability"] * 100, "%"))],
            [p("Risk level"), p(prediction["risk_level"])],
            [p("Model prediction"), p(prediction["prediction"])],
        ])
    else:
        risk_rows.append([p("Cost-overrun model"), p("Unavailable for this project")])
    if time_delay:
        risk_rows.extend([
            [p("Time-delay probability"), p(report_value(time_delay["probability"] * 100, "%"))],
            [p("Time-delay prediction"), p(time_delay["prediction"])],
            [p("Time-delay risk level"), p(time_delay["risk_level"])],
        ])
    else:
        risk_rows.append([p("Time-delay prediction"), p("Unavailable: existing time-delay model did not return a result")])
    story.append(report_table(risk_rows, [67 * mm, 109 * mm]))
    story.extend([
        p("5. Visual Risk Summary", section),
        visual_indicator("AI Cost-Overrun Probability", prediction["probability"] * 100 if prediction else None, "#C4513A"),
        visual_indicator("AI Time-Delay Probability", time_delay["probability"] * 100 if time_delay else None, "#D18B2C"),
        visual_indicator("Physical Progress", physical, "#1F6F8B"),
        visual_indicator("Expenditure Ratio", ratio, "#7A5C9E"),
    ])
    story.extend([
        p("6. Early Warning Indicators", section),
        *[p("• " + warning, bullet) for warning in warnings],
        p("7. AI Observations and Recommended Monitoring Actions", section),
        *[p("• " + action, bullet) for action in actions],
        p("These are monitoring suggestions based only on available data and are not official government orders.", small),
        p("8. Model Explainability — SHAP Analysis", section),
    ])
    if cost_explanation:
        story.append(p(f"Method: {cost_explanation['shap_method']}. Values are contributions to the model's raw cost-overrun margin; they are not causal effects.", small))
        story.append(report_table(
            [[p("Top risk factor", label), p("SHAP value", label), p("Explanation", label)]]
            + [[p(item["feature"]), p(report_value(item["shap_value"], "", 4)), p(item["explanation"])] for item in cost_explanation["risk_factors"]],
            [47 * mm, 27 * mm, 102 * mm],
        ))
        story.append(report_table(
            [[p("Protective factor", label), p("SHAP value", label), p("Explanation", label)]]
            + [[p(item["feature"]), p(report_value(item["shap_value"], "", 4)), p(item["explanation"])] for item in cost_explanation["protective_factors"]],
            [47 * mm, 27 * mm, 102 * mm],
        ))
    else:
        story.append(p("Native LightGBM SHAP contributions were not available for this project.", body))
    if delay_explanation:
        story.append(p("Time-delay model SHAP factors", body))
        story.append(report_table(
            [[p("Factor", label), p("SHAP value", label), p("Explanation", label)]]
            + [[p(item["feature"]), p(report_value(item["shap_value"], "", 4)), p(item["explanation"])] for item in delay_explanation["risk_factors"]],
            [47 * mm, 27 * mm, 102 * mm],
        ))
        story.append(report_table(
            [[p("Protective factor", label), p("SHAP value", label), p("Explanation", label)]]
            + [[p(item["feature"]), p(report_value(item["shap_value"], "", 4)), p(item["explanation"])] for item in delay_explanation["protective_factors"]],
            [47 * mm, 27 * mm, 102 * mm],
        ))
    story.extend([
        p("9. Milestone and Schedule Information", section),
        p("Milestone and schedule details are not available in the current dataset.", body),
        p("10. Sector Benchmarking", section),
        p("Sector benchmarking is not available in the current dataset.", body),
        p("11. Data Availability and Limitations", section),
        *[p("• " + item, bullet) for item in [
            "Revised cost is unavailable.", "Completion dates are unavailable.",
            "Milestone dates are unavailable.", "Contractor details are unavailable.",
            "Sector benchmarks are unavailable.", "Exact delay days are unavailable.",
            "Project status is unavailable in the supplied dataset.",
            "Full monthly history is limited to records available in the historical feature dataset.",
        ]],
        p("These are model-based estimates and are not confirmed financial losses or confirmed completion delays. No predicted final cost, completion date, exact delay days, future expenditure, or unsupported milestone values are included.", small),
    ])

    buffer = BytesIO()
    _report_project_code = project.get("project_code", "")
    document = BaseDocTemplate(
        buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=22 * mm, bottomMargin=18 * mm,
        title=f"AVLOKAN {_report_project_code} Project Report",
        author="AVLOKAN prototype",
    )
    frame = Frame(document.leftMargin, document.bottomMargin, document.width, document.height, id="normal")

    def draw_page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#B8C4CF"))
        canvas.line(doc.leftMargin, A4[1] - 16 * mm, A4[0] - doc.rightMargin, A4[1] - 16 * mm)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(colors.HexColor("#17324D"))
        canvas.drawString(doc.leftMargin, A4[1] - 12 * mm, "AVLOKAN | SIH26103")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#52606D"))
        canvas.drawString(doc.leftMargin, 10 * mm, "Generated by AVLOKAN prototype | SIH26103")
        canvas.drawRightString(A4[0] - doc.rightMargin, 10 * mm, f"Page {doc.page}")
        canvas.drawCentredString(A4[0] / 2, 6 * mm, "This report is an AI-assisted monitoring aid and does not replace official project verification.")
        canvas.restoreState()

    document.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=draw_page)])
    document.build(story)
    buffer.seek(0)
    return buffer


@app.route("/api/project/<project_code>/report")
def project_report(project_code):
    result = df[
        df["project_code"]
        == str(project_code)
    ]

    if result.empty:
        return jsonify({
            "error": "Project not found"
        }), 404

    project = records_without_nan(result.iloc[[0]])[0]
    pdf = build_project_report(project)
    return send_file(
        pdf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"AVLOKAN_{project_code}_Project_Report.pdf",
    )


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
        or time_delay_feature_names is None
        or july_features is None
    ):
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
        explanation = native_shap_explanation(
            time_delay_model,
            features,
            time_delay_feature_names,
            "schedule-slippage risk",
        )
        raw_prediction = float(time_delay_model.predict(features, raw_score=True)[0])
        additivity_error = float(
            explanation["shap_base_value"]
            + explanation["shap_value_sum"]
            - raw_prediction
        )

        return jsonify({
            "project_code": str(project["project_code"]),
            "project_name": project.get("project_name"),
            "time_delay_probability": probability,
            "risk_level": risk_level(probability),
            "risk_factors": explanation["risk_factors"],
            "protective_factors": explanation["protective_factors"],
            "shap_method": explanation["shap_method"],
            "shap_base_value": explanation["shap_base_value"],
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
        or feature_names is None
        or july_features is None
    ):
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
        explanation = native_shap_explanation(
            prediction_model.booster,
            features,
            feature_names,
            "cost-overrun risk",
        )
        raw_prediction = float(prediction_model.booster.predict(features, raw_score=True)[0])
        additivity_error = float(
            explanation["shap_base_value"]
            + explanation["shap_value_sum"]
            - raw_prediction
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
            "top_risk_factors": explanation["risk_factors"],
            "top_protective_factors": explanation["protective_factors"],
            "shap_method": explanation["shap_method"],
            "shap_base_value": explanation["shap_base_value"],
            "explanation_note": (
                "These features contributed most to the model's risk prediction; "
                "they should not be interpreted as causal effects."
            ),
            "shap_output_space": "raw LightGBM margin",
            "shap_additivity_error": additivity_error,
        })
    except Exception as exc:
        print("SHAP explanation failed:", exc)
        return jsonify({
            "error": "SHAP explanation failed",
            "details": str(exc),
        }), 500


def historical_risk_history(project_code):
    if history_features is None:
        return []
    target_code = str(project_code).strip()
    rows = history_features[
        history_features["project_code"].astype(str).str.strip() == target_code
    ].copy()
    if rows.empty:
        return []
    rows["report_month"] = pd.to_datetime(rows["report_month"], errors="coerce")
    rows = (
        rows.dropna(subset=["report_month"])
        .sort_values("report_month")
        .drop_duplicates(subset=["project_code", "report_month"], keep="last")
    )
    records = []
    for _, row in rows.iterrows():
        cost_probability = None
        cost_level = None
        if prediction_model is not None:
            cost_prediction = predict_historical_risk(row)
            if cost_prediction is not None:
                cost_probability = cost_prediction["probability"]
                cost_level = cost_prediction["risk_level"]

        time_probability = None
        time_level = None
        if time_delay_model is not None:
            try:
                delay_features = prepare_time_delay_features(row)
                time_probability = float(time_delay_model.predict(delay_features)[0])
                time_level = risk_level(time_probability)
            except Exception as exc:
                print("Historical time-delay calculation failed:", exc)

        def value_or_none(column):
            if column not in row or pd.isna(row[column]):
                return None
            val = row[column]
            if isinstance(val, (np.floating, float)) and (np.isnan(val) or np.isinf(val)):
                return None
            return val

        records.append({
            "project_code": target_code,
            "project_name": value_or_none("project_name"),
            "report_month": row["report_month"].strftime("%Y-%m-%d"),
            "cost_overrun_probability": cost_probability,
            "time_delay_probability": time_probability,
            "cost_risk_level": cost_level,
            "time_delay_risk_level": time_level,
            "physical_progress": value_or_none("physical_progress"),
            "expenditure_ratio": value_or_none("expenditure_ratio"),
        })
    previous_record = None
    escalation_targets = {"LOW -> MEDIUM", "LOW -> HIGH", "MEDIUM -> HIGH"}
    for record in records:
        record["cost_risk_change"] = None
        record["time_delay_risk_change"] = None
        record["escalation"] = "No escalation"
        if previous_record is not None:
            if (
                previous_record["cost_risk_level"] is not None
                and record["cost_risk_level"] is not None
            ):
                cost_change = f"{previous_record['cost_risk_level']} -> {record['cost_risk_level']}"
                record["cost_risk_change"] = cost_change
                if cost_change in escalation_targets:
                    record["escalation"] = cost_change
            if (
                previous_record["time_delay_risk_level"] is not None
                and record["time_delay_risk_level"] is not None
            ):
                time_change = f"{previous_record['time_delay_risk_level']} -> {record['time_delay_risk_level']}"
                record["time_delay_risk_change"] = time_change
                if time_change in escalation_targets and record["escalation"] == "No escalation":
                    record["escalation"] = time_change
        previous_record = record
    return records


@app.get("/api/notifications")
def notifications():
    user, error = notification_user()
    if error:
        return error
    try:
        generate_current_user_notifications(user["id"])
        with auth_db_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM notifications
                WHERE user_id = ?
                ORDER BY date(current_report_month) DESC, datetime(created_at) DESC, id DESC
                """,
                (user["id"],),
            ).fetchall()
        records = [notification_record(row) for row in rows]
        return jsonify({
            "success": True,
            "notifications": records,
            "unread_count": sum(not record["is_read"] for record in records),
        })
    except Exception as exc:
        print("Notification retrieval failed:", exc)
        return jsonify({
            "error": "Unable to load risk escalation notifications.",
            "details": str(exc),
        }), 500


@app.get("/api/notifications/unread-count")
def notification_unread_count():
    user, error = notification_user()
    if error:
        return error
    try:
        generate_current_user_notifications(user["id"])
        with auth_db_connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS unread_count
                FROM notifications
                WHERE user_id = ? AND is_read = 0
                """,
                (user["id"],),
            ).fetchone()
        return jsonify({
            "success": True,
            "unread_count": int(row["unread_count"]),
        })
    except Exception as exc:
        print("Unread notification count failed:", exc)
        return jsonify({
            "error": "Unable to load unread notification count.",
            "details": str(exc),
        }), 500


@app.patch("/api/notifications/<int:notification_id>/read")
def mark_notification_read(notification_id):
    user, error = notification_user()
    if error:
        return error
    with auth_db_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE notifications
            SET is_read = 1
            WHERE id = ? AND user_id = ?
            """,
            (notification_id, user["id"]),
        )
        if cursor.rowcount == 0:
            return jsonify({"error": "Notification not found."}), 404
    return jsonify({"success": True, "notification_id": notification_id})


@app.patch("/api/notifications/mark-all-read")
def mark_all_notifications_read():
    user, error = notification_user()
    if error:
        return error
    with auth_db_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE notifications
            SET is_read = 1
            WHERE user_id = ? AND is_read = 0
            """,
            (user["id"],),
        )
    return jsonify({
        "success": True,
        "marked_read": cursor.rowcount,
    })


@app.get("/api/notifications/project/<project_code>")
def project_notifications(project_code):
    user, error = notification_user()
    if error:
        return error
    try:
        generate_current_user_notifications(user["id"])
        with auth_db_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM notifications
                WHERE user_id = ? AND project_code = ?
                ORDER BY datetime(created_at) DESC, id DESC
                """,
                (user["id"], str(project_code)),
            ).fetchall()
        return jsonify({
            "success": True,
            "project_code": str(project_code),
            "notifications": [notification_record(row) for row in rows],
        })
    except Exception as exc:
        print("Project notifications retrieval failed:", exc)
        return jsonify({
            "error": "Unable to load project notifications.",
            "details": str(exc),
        }), 500


@app.get("/api/projects/<project_code>/risk-history")
def project_risk_history(project_code):
    try:
        records = historical_risk_history(project_code)
        if not records:
            return jsonify({
                "success": True,
                "project_code": str(project_code),
                "records": [],
                "message": "No historical records are available for this project.",
            })

        message = None
        previous = None
        current = None
        if len(records) == 1:
            message = "Only one historical observation is available for this project."
        else:
            valid_records = [
                record for record in records
                if record["cost_risk_level"] is not None
                or record["time_delay_risk_level"] is not None
            ]
            if len(valid_records) >= 2:
                previous = valid_records[-2]
                current = valid_records[-1]

        return jsonify({
            "success": True,
            "project_code": str(project_code),
            "project_name": records[-1]["project_name"],
            "records": records,
            "message": message,
            "previous": previous,
            "current": current,
        })
    except Exception as exc:
        print("Project risk history failed:", exc)
        return jsonify({
            "error": "Unable to load project risk history.",
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

    print("\nStarting AVLOKAN backend...")
    print("Open: http://127.0.0.1:5001")

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False
    )
