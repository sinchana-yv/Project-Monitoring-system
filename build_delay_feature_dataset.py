"""Build a separate point-in-time feature dataset for schedule-delay analysis."""

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "FINAL_CLEAN_ONGOING_PROJECTS.csv"
OUTPUT = ROOT / "DELAY_FEATURE_DATASET.csv"
AUDIT = ROOT / "delay_feature_audit.txt"

OUTPUT_COLUMNS = [
    "project_code",
    "project_name",
    "report_month",
    "start_date",
    "original_doc",
    "physical_progress",
    "previous_progress",
    "progress_change",
    "progress_velocity",
    "expenditure",
    "expenditure_ratio",
    "previous_expenditure",
    "expenditure_change",
    "expenditure_velocity",
    "original_cost",
    "months_observed",
    "days_since_first_report",
    "has_physical_progress",
    "has_expenditure",
    "state",
    "sector",
    "ministry_final",
    "agency_final",
    "planned_duration_days_raw",
    "elapsed_duration_days",
    "expected_progress_raw",
    "expected_progress",
    "progress_gap",
    "schedule_valid",
    "schedule_invalid_reason",
    "source_file",
    "source_sheet",
]


def parse_month(value):
    if pd.isna(value):
        return pd.NaT
    text = re.sub(r"[()]", "", str(value)).strip()
    if not text or text.lower() in {"nan", "n.a.", "na", "-", "--"}:
        return pd.NaT
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=False)
    if pd.isna(parsed):
        parsed = pd.to_datetime(
            text.replace("-", "/"),
            errors="coerce",
            dayfirst=False,
        )
    return parsed.to_period("M").to_timestamp() if pd.notna(parsed) else pd.NaT


def main():
    source = pd.read_csv(SOURCE, low_memory=False)
    source["project_code"] = source["project_code"].astype(str).str.strip()
    for field in ["start_date", "original_doc", "report_month"]:
        source[field] = source[field].map(parse_month)
    for field in [
        "physical_progress",
        "previous_progress",
        "progress_change",
        "progress_velocity",
        "expenditure",
        "expenditure_ratio",
        "previous_expenditure",
        "expenditure_change",
        "expenditure_velocity",
        "original_cost",
        "months_observed",
        "days_since_first_report",
        "has_physical_progress",
        "has_expenditure",
    ]:
        if field not in source:
            source[field] = np.nan
        source[field] = pd.to_numeric(source[field], errors="coerce")

    source = source.sort_values(["project_code", "report_month"]).copy()
    grouped = source.groupby("project_code", sort=False)
    source["start_date"] = grouped["start_date"].ffill()
    source["original_doc"] = grouped["original_doc"].ffill()
    if source["previous_progress"].isna().all():
        source["previous_progress"] = grouped["physical_progress"].shift(1)
    if source["progress_change"].isna().all():
        source["progress_change"] = (
            source["physical_progress"] - source["previous_progress"]
        )
    if source["previous_expenditure"].isna().all():
        source["previous_expenditure"] = grouped["expenditure"].shift(1)
    if source["expenditure_change"].isna().all():
        source["expenditure_change"] = (
            source["expenditure"] - source["previous_expenditure"]
        )
    days_since_previous = (
        source["report_month"] - grouped["report_month"].shift(1)
    ).dt.days
    if source["progress_velocity"].isna().all():
        source["progress_velocity"] = (
            source["progress_change"] / days_since_previous
        )
    if source["expenditure_velocity"].isna().all():
        source["expenditure_velocity"] = (
            source["expenditure_change"] / days_since_previous
        )
    if source["expenditure_ratio"].isna().all():
        source["expenditure_ratio"] = np.where(
            source["original_cost"] > 0,
            source["expenditure"] / source["original_cost"] * 100,
            np.nan,
        )
    if source["months_observed"].isna().all():
        source["months_observed"] = grouped.cumcount() + 1
    if source["days_since_first_report"].isna().all():
        first_report = grouped["report_month"].transform("min")
        source["days_since_first_report"] = (
            source["report_month"] - first_report
        ).dt.days
    if source["has_physical_progress"].isna().all():
        source["has_physical_progress"] = (
            source["physical_progress"].notna().astype(int)
        )
    if source["has_expenditure"].isna().all():
        source["has_expenditure"] = source["expenditure"].notna().astype(int)

    source["duplicate_project_month"] = source.duplicated(
        ["project_code", "report_month"],
        keep=False,
    )
    duplicate_rows = int(source["duplicate_project_month"].sum())
    duplicate_groups = int(
        source.loc[source["duplicate_project_month"]]
        .groupby(["project_code", "report_month"])
        .ngroups
    )

    # The normalized source has one canonical ongoing-project row per
    # project-month. If duplicates ever appear, retain the most complete row
    # deterministically and preserve the duplicate count in the audit.
    source["_completeness"] = source.notna().sum(axis=1)
    source = (
        source.sort_values(
            ["project_code", "report_month", "_completeness", "source_file", "source_sheet"],
            ascending=[True, True, False, True, True],
        )
        .drop_duplicates(["project_code", "report_month"], keep="first")
        .copy()
    )

    source["planned_duration_days_raw"] = (
        source["original_doc"] - source["start_date"]
    ).dt.days
    source["elapsed_duration_days"] = (
        source["report_month"] - source["start_date"]
    ).dt.days
    source["expected_progress_raw"] = np.where(
        source["planned_duration_days_raw"] > 0,
        source["elapsed_duration_days"]
        / source["planned_duration_days_raw"]
        * 100,
        np.nan,
    )
    source["expected_progress"] = source["expected_progress_raw"].clip(
        upper=100
    )
    source["progress_gap"] = (
        source["physical_progress"] - source["expected_progress"]
    )

    reasons = pd.Series("", index=source.index, dtype="object")

    def add_reason(mask, text):
        nonlocal reasons
        empty = mask & reasons.eq("")
        nonempty = mask & reasons.ne("")
        reasons = reasons.mask(empty, text)
        reasons = reasons.mask(nonempty, reasons + "; " + text)

    add_reason(source["start_date"].isna(), "missing start_date")
    add_reason(source["original_doc"].isna(), "missing original_doc")
    add_reason(
        source["report_month"].isna()
        | (
            source["start_date"].notna()
            & source["original_doc"].notna()
            & source["report_month"].isna()
        ),
        "invalid dates",
    )
    add_reason(
        source["planned_duration_days_raw"].le(0),
        "planned duration <= 0",
    )
    add_reason(
        source["report_month"].notna()
        & source["start_date"].notna()
        & source["report_month"].lt(source["start_date"]),
        "report month before start date",
    )
    add_reason(
        source["physical_progress"].notna()
        & ~source["physical_progress"].between(0, 100),
        "invalid physical progress",
    )
    add_reason(
        source["original_doc"].notna()
        & source["start_date"].notna()
        & source["original_doc"].lt(source["start_date"]),
        "invalid dates",
    )

    source["schedule_invalid_reason"] = reasons
    source["schedule_valid"] = reasons.eq("")

    source[OUTPUT_COLUMNS].to_csv(OUTPUT, index=False)

    source_files = sorted(source["source_file"].dropna().unique())
    both_dates = source["start_date"].notna() & source["original_doc"].notna()
    valid = source["schedule_valid"]
    july = source["report_month"].eq(pd.Timestamp("2026-07-01"))
    july_both = july & both_dates

    audit_lines = [
        "DELAY FEATURE DATASET AUDIT",
        "=" * 78,
        "Source used:",
        f"  {SOURCE.name}",
        "",
        "The source is the normalized PAIMANA ongoing-project dataset. Each",
        "row retains source_file and source_sheet provenance from the original",
        "PAIMANA Excel workbooks. No actual/revised completion, future progress,",
        "future expenditure, future cost, or final status fields were used.",
        "",
        "Original source files represented in the source:",
    ]
    audit_lines.extend(f"  {name}" for name in source_files)
    audit_lines.extend(
        [
            "",
            "Source field provenance:",
            "  start_date: normalized source column start_date, populated from",
            "    PAIMANA Start Date / Start Date (MM/YYYY) variants.",
            "  original_doc: normalized source column original_doc, populated from",
            "    PAIMANA Original/Target DoC and Original Target DoC variants.",
            "  report_month: normalized source report_month.",
            "  source_file/source_sheet: preserved from the PAIMANA ingestion step.",
            "",
            "DATA QUALITY",
            "------------",
            f"Total rows: {len(source)}",
            f"Unique projects: {source['project_code'].nunique()}",
            f"Date range: {source['report_month'].min()} to {source['report_month'].max()}",
            f"Start-date coverage: {source['start_date'].notna().sum()} "
            f"({source['start_date'].notna().mean() * 100:.2f}%)",
            f"Original-completion-date coverage: {source['original_doc'].notna().sum()} "
            f"({source['original_doc'].notna().mean() * 100:.2f}%)",
            f"Both dates available: {both_dates.sum()} "
            f"({both_dates.mean() * 100:.2f}%)",
            f"Physical-progress coverage: {source['physical_progress'].notna().sum()} "
            f"({source['physical_progress'].notna().mean() * 100:.2f}%)",
            f"Valid schedule rows: {valid.sum()}",
            f"Invalid rows: {(~valid).sum()}",
            f"Duplicate project-month records found before resolution: {duplicate_rows}",
            f"Duplicate project-month groups found: {duplicate_groups}",
            "Duplicate resolution: retain the most complete row, then sort by",
            "source_file and source_sheet for a deterministic tie-break.",
            "",
            "INVALID-ROW REASONS",
            "-------------------",
        ]
    )
    for reason in [
        "missing start_date",
        "missing original_doc",
        "invalid dates",
        "planned duration <= 0",
        "report month before start date",
        "invalid physical progress",
    ]:
        audit_lines.append(
            f"  {reason}: {source['schedule_invalid_reason'].str.contains(reason, regex=False).sum()}"
        )
    audit_lines.extend(
        [
            "",
            "DERIVED-FEATURE DEFINITIONS",
            "---------------------------",
            "planned_duration_days_raw = original_doc - start_date",
            "elapsed_duration_days = report_month - start_date",
            "expected_progress_raw = elapsed_duration_days / planned_duration_days_raw * 100",
            "expected_progress = expected_progress_raw capped at 100",
            "progress_gap = physical_progress - expected_progress",
            "",
            "JULY 2026 CHECK",
            "---------------",
            f"July 2026 rows: {july.sum()}",
            f"July 2026 rows with both dates: {july_both.sum()}",
            f"July 2026 valid schedule rows: {(july & valid).sum()}",
            "",
            "OUTPUT COLUMNS",
            "--------------",
        ]
    )
    audit_lines.extend(f"  {column}" for column in OUTPUT_COLUMNS)
    audit_lines.extend(
        [
            "",
            "POINT-IN-TIME AND LEAKAGE CHECK",
            "-------------------------------",
            "Only current row fields and historical/lag fields already present in",
            "the normalized source were retained. Revised/actual completion dates,",
            "future values, and final project status were excluded. project_code is",
            "preserved for identity and grouping, not as a model feature.",
            "Invalid schedule rows remain in the output with schedule_valid=0 and",
            "schedule_invalid_reason; they were not silently deleted.",
        ]
    )
    AUDIT.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
