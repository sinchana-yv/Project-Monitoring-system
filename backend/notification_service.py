"""Idempotent in-app risk escalation notifications for AVLOKAN."""

from datetime import datetime, timezone
import sqlite3

import pandas as pd


def ensure_notification_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            project_code TEXT NOT NULL,
            project_name TEXT,
            previous_risk TEXT NOT NULL,
            current_risk TEXT NOT NULL,
            previous_probability REAL,
            current_probability REAL,
            previous_report_month TEXT,
            current_report_month TEXT,
            message TEXT NOT NULL,
            notification_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            is_read INTEGER NOT NULL DEFAULT 0,
            severity TEXT NOT NULL,
            unique_key TEXT NOT NULL,
            UNIQUE(user_id, unique_key)
        )
        """
    )
    connection.commit()


def risk_rank(level):
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(str(level).upper(), -1)


def _notification_key(project_code, previous_row, current_row, previous_risk, current_risk):
    return "|".join(
        [
            str(project_code),
            str(previous_row["report_month"]),
            str(current_row["report_month"]),
            previous_risk,
            current_risk,
        ]
    )


def generate_notifications_for_user(
    connection,
    user_id,
    history_frame,
    predict_observation,
):
    """Persist every observed upward risk transition once for one user."""
    ensure_notification_table(connection)
    if history_frame is None or history_frame.empty:
        return 0

    inserted = 0
    ordered = history_frame.copy()
    ordered["project_code"] = ordered["project_code"].astype(str).str.strip()
    ordered["report_month"] = pd.to_datetime(
        ordered["report_month"],
        errors="coerce",
    )
    ordered = ordered.dropna(subset=["project_code", "report_month"])

    for project_code, project_rows in ordered.groupby("project_code"):
        project_rows = project_rows.sort_values("report_month")
        previous = None
        for _, current in project_rows.iterrows():
            if "_risk_level" in current and "_risk_probability" in current:
                current_prediction = {
                    "risk_level": current["_risk_level"],
                    "probability": current["_risk_probability"],
                }
            else:
                current_prediction = predict_observation(current)
            if current_prediction is None:
                previous = None
                continue
            if previous is not None:
                previous_risk = previous["risk_level"]
                current_risk = current_prediction["risk_level"]
                if risk_rank(current_risk) > risk_rank(previous_risk):
                    unique_key = _notification_key(
                        project_code,
                        previous["row"],
                        current,
                        previous_risk,
                        current_risk,
                    )
                    current_month = current["report_month"].strftime("%Y-%m-%d")
                    previous_month = previous["row"]["report_month"].strftime(
                        "%Y-%m-%d"
                    )
                    project_name = current.get("project_name")
                    message = (
                        f"Project {project_code} has escalated from "
                        f"{previous_risk} risk to {current_risk} risk based "
                        "on the latest available project observation. "
                        "Immediate attention is recommended."
                    )
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO notifications (
                            user_id, project_code, project_name,
                            previous_risk, current_risk,
                            previous_probability, current_probability,
                            previous_report_month, current_report_month,
                            message, notification_type, created_at,
                            is_read, severity, unique_key
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                        """,
                        (
                            user_id,
                            project_code,
                            None if pd.isna(project_name) else str(project_name),
                            previous_risk,
                            current_risk,
                            previous["probability"],
                            current_prediction["probability"],
                            previous_month,
                            current_month,
                            message,
                            "risk_escalation",
                            datetime.now(timezone.utc).isoformat(),
                            current_risk,
                            unique_key,
                        ),
                    )
                    inserted += int(cursor.rowcount > 0)
            previous = {
                "row": current,
                "risk_level": current_prediction["risk_level"],
                "probability": current_prediction["probability"],
            }
    connection.commit()
    return inserted


def notification_record(row):
    return {
        "id": row["id"],
        "project_code": row["project_code"],
        "project_name": row["project_name"],
        "previous_risk": row["previous_risk"],
        "current_risk": row["current_risk"],
        "previous_probability": row["previous_probability"],
        "current_probability": row["current_probability"],
        "previous_report_month": row["previous_report_month"],
        "current_report_month": row["current_report_month"],
        "report_month": row["current_report_month"],
        "message": row["message"],
        "notification_type": row["notification_type"],
        "created_at": row["created_at"],
        "is_read": bool(row["is_read"]),
        "severity": row["severity"],
        "action": {
            "type": "view_project",
            "project_code": row["project_code"],
        },
    }
