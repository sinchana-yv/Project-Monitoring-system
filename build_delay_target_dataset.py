"""Build a leakage-safe binary time-delay target dataset."""

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
COMPLETED_FILE = ROOT / "clean_completed_projects.csv"
FEATURE_FILE = ROOT / "ML_FEATURE_DATASET.csv"
OUTPUT_FILE = ROOT / "delay_target_dataset.csv"
AUDIT_FILE = ROOT / "delay_target_audit.txt"

SAFE_FEATURES = [
    "report_month",
    "project_code",
    "original_cost",
    "expenditure",
    "expenditure_ratio",
    "physical_progress",
    "previous_progress",
    "progress_change",
    "progress_velocity",
    "previous_expenditure",
    "expenditure_change",
    "expenditure_velocity",
    "previous_observations",
    "months_observed",
    "days_since_first_report",
    "has_physical_progress",
    "has_expenditure",
    "has_original_cost",
    "state",
    "sector",
    "ministry_final",
    "agency_final",
    "start_date",
    "original_doc",
]

ORIGINAL_FIELDS = [
    "original_doc",
    "Original/Target DoC (MM/YYYY)",
    "Original Target DoC (MM/YYYY)",
    "Target Completion Date (MM/YYYY)",
    "Original Target DoC (MM/YYYY)",
]

ACTUAL_FIELDS = [
    "Actual Date of Completion",
    "Actual Completion Date",
    "Actual Date of Completion (MM/YYYY)",
]

AMBIGUOUS_COMPLETION_FIELDS = [
    "Actual/Revised Date of Completion (MM/YYYY)",
    "actual_completion",
]

REVISED_FIELDS = [
    "Revised Completion Date (MM/YYYY)",
    "Revised DoC MM/YYYY",
    "revised_doc",
]


def parse_month(value):
    """Parse the source's month/year values without inventing day precision."""
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if not text or text.lower() in {"nan", "n.a.", "na", "-", "--"}:
        return pd.NaT
    text = re.sub(r"[()]", "", text).strip()
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=False)
    if pd.isna(parsed):
        parsed = pd.to_datetime(
            text.replace("-", "/"),
            errors="coerce",
            dayfirst=False,
        )
    return parsed.to_period("M").to_timestamp() if pd.notna(parsed) else pd.NaT


def first_date(row, fields):
    for field in fields:
        if field in row.index:
            parsed = parse_month(row[field])
            if pd.notna(parsed):
                return parsed, field
    return pd.NaT, ""


def build_outcomes(completed):
    records = []
    for row_number, row in completed.iterrows():
        original, original_source = first_date(row, ORIGINAL_FIELDS)
        actual, actual_source = first_date(row, ACTUAL_FIELDS)
        if pd.notna(actual):
            outcome = actual
            outcome_source = actual_source
            outcome_kind = "actual"
        else:
            ambiguous, ambiguous_source = first_date(
                row,
                AMBIGUOUS_COMPLETION_FIELDS,
            )
            revised, revised_source = first_date(row, REVISED_FIELDS)
            if pd.notna(ambiguous):
                outcome = ambiguous
                outcome_source = ambiguous_source
            else:
                outcome = revised
                outcome_source = revised_source
            outcome_kind = "revised"

        project_code = str(row.get("project_code", "")).strip()
        if not project_code or project_code.lower() == "nan":
            continue

        report_month = parse_month(row.get("report_month"))
        valid_dates = pd.notna(original) and pd.notna(outcome)
        negative_delay = valid_dates and outcome < original
        records.append(
            {
                "project_code": project_code,
                "original_completion_date": original,
                "eventual_completion_date": outcome,
                "outcome_kind": outcome_kind,
                "original_date_source": original_source,
                "outcome_date_source": outcome_source,
                "source_report_month": report_month,
                "source_row": row_number,
                "negative_delay": bool(negative_delay),
            }
        )

    outcomes = pd.DataFrame(records)
    if outcomes.empty:
        return outcomes

    outcomes["valid_outcome"] = (
        outcomes["original_completion_date"].notna()
        & outcomes["eventual_completion_date"].notna()
        & ~outcomes["negative_delay"]
    )
    outcomes["actual_rank"] = (outcomes["outcome_kind"] == "actual").astype(int)
    outcomes = outcomes.sort_values(
        [
            "project_code",
            "valid_outcome",
            "actual_rank",
            "eventual_completion_date",
            "source_report_month",
            "source_row",
        ],
        ascending=[True, False, False, False, False, False],
    )
    return outcomes.drop_duplicates("project_code", keep="first")


def main():
    completed = pd.read_csv(COMPLETED_FILE, low_memory=False)
    features = pd.read_csv(FEATURE_FILE, low_memory=False)
    features["project_code"] = features["project_code"].astype(str).str.strip()
    features["report_month"] = features["report_month"].map(parse_month)

    outcomes_all = build_outcomes(completed)
    valid_outcomes = outcomes_all[outcomes_all["valid_outcome"]].copy()
    valid_outcomes["delay_months"] = (
        valid_outcomes["eventual_completion_date"]
        - valid_outcomes["original_completion_date"]
    ).dt.days / 30.4375
    valid_outcomes["future_time_delay"] = (
        valid_outcomes["delay_months"] > 0
    ).astype(int)

    outcome_lookup = valid_outcomes.set_index("project_code")
    matched = features[features["project_code"].isin(outcome_lookup.index)].copy()
    matched["original_completion_date"] = matched["project_code"].map(
        outcome_lookup["original_completion_date"]
    )
    matched["eventual_completion_date"] = matched["project_code"].map(
        outcome_lookup["eventual_completion_date"]
    )
    pre_outcome = matched[
        matched["report_month"] < matched["eventual_completion_date"]
    ].copy()
    pre_outcome["future_time_delay"] = pre_outcome["project_code"].map(
        outcome_lookup["future_time_delay"]
    ).astype(int)

    output_columns = [
        column
        for column in SAFE_FEATURES
        if column in pre_outcome.columns
    ] + ["future_time_delay"]
    pre_outcome[output_columns].to_csv(OUTPUT_FILE, index=False)

    examples_valid = valid_outcomes.head(5)
    examples_rejected = outcomes_all[
        ~outcomes_all["valid_outcome"]
    ].head(5)
    audit_lines = [
        "TIME DELAY TARGET AUDIT",
        "=" * 70,
        f"Completed project records: {len(completed)}",
        f"Completed projects (unique): {completed['project_code'].nunique()}",
        "Valid original completion date records: "
        f"{outcomes_all['original_completion_date'].notna().sum()}",
        "Valid actual/revised completion date records: "
        f"{outcomes_all['eventual_completion_date'].notna().sum()}",
        f"Valid project-level delay labels: {len(valid_outcomes)}",
        f"Delayed count: {(valid_outcomes['future_time_delay'] == 1).sum()}",
        f"Not-delayed count: {(valid_outcomes['future_time_delay'] == 0).sum()}",
        "Invalid/negative-delay records rejected: "
        f"{(~outcomes_all['valid_outcome']).sum()}",
        f"Matched ML projects: {matched['project_code'].nunique()}",
        f"Pre-outcome snapshot rows: {len(pre_outcome)}",
        f"Pre-outcome unique projects: {pre_outcome['project_code'].nunique()}",
        "Outcome date range: "
        f"{valid_outcomes['eventual_completion_date'].min()} to "
        f"{valid_outcomes['eventual_completion_date'].max()}",
        f"Output rows: {len(pre_outcome)}",
        f"Output unique projects: {pre_outcome['project_code'].nunique()}",
        "Class distribution: "
        f"{pre_outcome['future_time_delay'].value_counts().to_dict()}",
        "",
        "Deduplication rule:",
        "One record per project; valid outcomes are preferred over invalid, "
        "explicit actual completion over revised completion, then the latest "
        "eventual completion date, source report month, and source row.",
        "",
        "Rejected records (first 5):",
    ]
    for _, row in examples_rejected.iterrows():
        audit_lines.append(
            f"{row['project_code']} | original={row['original_completion_date']} | "
            f"eventual={row['eventual_completion_date']} | "
            f"source={row['outcome_date_source']} | "
            f"negative={row['negative_delay']}"
        )
    audit_lines.append("")
    audit_lines.append("Valid records (first 5):")
    for _, row in examples_valid.iterrows():
        audit_lines.append(
            f"{row['project_code']} | original={row['original_completion_date']} | "
            f"eventual={row['eventual_completion_date']} | "
            f"delay_months={row['delay_months']:.2f} | "
            f"future_time_delay={row['future_time_delay']} | "
            f"kind={row['outcome_kind']}"
        )
    audit_lines.extend(
        [
            "",
            "Leakage controls:",
            "Only ML rows with report_month strictly before eventual completion "
            "date were retained.",
            "Actual/revised completion dates, revised_cost, cost_increase, "
            "future progress/expenditure, and final status are not output features.",
            "project_code is preserved for grouping/audit but is not a model feature.",
        ]
    )
    AUDIT_FILE.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
