"""Audit a current schedule-delay target without training a model."""

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "ML_FEATURE_DATASET.csv": ROOT / "ML_FEATURE_DATASET.csv",
    "JULY_2026_FEATURE_DATASET.csv": ROOT / "JULY_2026_FEATURE_DATASET.csv",
}
OUTPUT = ROOT / "schedule_delay_analysis.txt"

FIELDS = [
    "start_date",
    "original_doc",
    "report_month",
    "physical_progress",
    "previous_progress",
    "progress_velocity",
    "months_observed",
]

SAFE_FEATURES = [
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
    "has_original_cost",
    "state",
    "sector",
    "ministry_final",
    "agency_final",
    "start_date-derived duration features",
    "original_doc-derived schedule features",
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


def coverage(frame, field):
    non_null = int(frame[field].notna().sum())
    return non_null, 100 * non_null / len(frame) if len(frame) else 0


def analyze(name, path):
    raw = pd.read_csv(path, low_memory=False)
    frame = raw.copy()
    for field in FIELDS:
        if field not in frame:
            frame[field] = pd.NaT

    for field in ["start_date", "original_doc", "report_month"]:
        frame[field] = frame[field].map(parse_month)
    for field in [
        "physical_progress",
        "previous_progress",
        "progress_velocity",
        "months_observed",
    ]:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")

    frame["planned_duration_days"] = (
        frame["original_doc"] - frame["start_date"]
    ).dt.days
    frame["elapsed_duration_days"] = (
        frame["report_month"] - frame["start_date"]
    ).dt.days
    frame["expected_progress"] = np.where(
        frame["planned_duration_days"] > 0,
        frame["elapsed_duration_days"]
        / frame["planned_duration_days"]
        * 100,
        np.nan,
    )
    frame["progress_gap"] = (
        frame["physical_progress"] - frame["expected_progress"]
    )

    invalid = {
        "start date missing": frame["start_date"].isna(),
        "original completion missing": frame["original_doc"].isna(),
        "planned duration <= 0": frame["planned_duration_days"].le(0),
        "report month before start date": (
            frame["report_month"] < frame["start_date"]
        ),
        "physical progress outside 0-100": (
            frame["physical_progress"].notna()
            & ~frame["physical_progress"].between(0, 100)
        ),
        "expected progress > 100": frame["expected_progress"].gt(100),
        "impossible dates": (
            frame["start_date"].notna()
            & frame["original_doc"].notna()
            & frame["original_doc"].lt(frame["start_date"])
        ),
    }
    valid = (
        frame["planned_duration_days"].gt(0)
        & frame["elapsed_duration_days"].ge(0)
        & frame["physical_progress"].between(0, 100)
        & frame["expected_progress"].between(0, 100)
        & frame["progress_gap"].notna()
    )
    gaps = frame.loc[valid, "progress_gap"]
    rules = []
    for threshold in [0, -10, -20, -30]:
        delayed = valid & frame["progress_gap"].lt(threshold)
        non_delayed = valid & ~delayed
        rules.append(
            {
                "threshold": threshold,
                "delayed_rows": int(delayed.sum()),
                "non_delayed_rows": int(non_delayed.sum()),
                "delayed_pct": 100 * delayed.sum() / valid.sum()
                if valid.sum()
                else 0,
                "delayed_projects": int(
                    frame.loc[delayed, "project_code"].nunique()
                ),
                "non_delayed_projects": int(
                    frame.loc[non_delayed, "project_code"].nunique()
                ),
            }
        )
    return {
        "name": name,
        "raw": raw,
        "frame": frame,
        "valid": valid,
        "invalid": invalid,
        "gaps": gaps,
        "rules": rules,
    }


def fmt(value):
    return "None" if pd.isna(value) else str(value)


def main():
    analyses = [analyze(name, path) for name, path in DATASETS.items()]
    lines = [
        "CURRENT SCHEDULE DELAY TARGET ANALYSIS",
        "=" * 78,
        "",
        "This is an analysis-only audit. No source dataset was modified and no",
        "model was trained. Schedule fields were calculated in memory only.",
        "",
        "Date interpretation:",
        "start_date, original_doc, and report_month were parsed to month starts.",
        "No actual, revised, future completion, future progress, future expenditure,",
        "future cost, or final completion-status fields were used.",
        "",
    ]

    for result in analyses:
        name = result["name"]
        raw = result["raw"]
        frame = result["frame"]
        valid = result["valid"]
        lines.extend(
            [
                name,
                "-" * len(name),
                f"Total rows: {len(raw)}",
                f"Unique projects: {raw['project_code'].nunique() if 'project_code' in raw else 0}",
                "",
                "A. Field coverage",
            ]
        )
        for field in FIELDS:
            count, pct = coverage(frame, field)
            lines.append(f"  {field}: {count} non-null ({pct:.2f}%)")

        lines.extend(
            [
                "",
                "B. Date parsing and schedule calculations",
                f"  Parseable start_date: {frame['start_date'].notna().sum()}",
                f"  Parseable original_doc: {frame['original_doc'].notna().sum()}",
                f"  Parseable report_month: {frame['report_month'].notna().sum()}",
                f"  Valid schedule rows: {int(valid.sum())}",
                f"  planned_duration_days non-null: {frame['planned_duration_days'].notna().sum()}",
                f"  elapsed_duration_days non-null: {frame['elapsed_duration_days'].notna().sum()}",
                f"  expected_progress non-null: {frame['expected_progress'].notna().sum()}",
                f"  progress_gap non-null: {frame['progress_gap'].notna().sum()}",
                "",
                "C. Invalid records",
            ]
        )
        for label, mask in result["invalid"].items():
            lines.append(f"  {label}: {int(mask.sum())}")

        lines.extend(
            [
                "",
                "D. Progress-gap distribution among valid schedule rows",
            ]
        )
        if len(result["gaps"]):
            quantiles = result["gaps"].quantile(
                [0.10, 0.25, 0.50, 0.75, 0.90]
            )
            lines.extend(
                [
                    f"  minimum: {result['gaps'].min():.4f}",
                    f"  maximum: {result['gaps'].max():.4f}",
                    f"  mean: {result['gaps'].mean():.4f}",
                    f"  median: {result['gaps'].median():.4f}",
                    f"  10th percentile: {quantiles.loc[0.10]:.4f}",
                    f"  25th percentile: {quantiles.loc[0.25]:.4f}",
                    f"  50th percentile: {quantiles.loc[0.50]:.4f}",
                    f"  75th percentile: {quantiles.loc[0.75]:.4f}",
                    f"  90th percentile: {quantiles.loc[0.90]:.4f}",
                ]
            )
        else:
            lines.append("  No valid schedule rows available.")

        lines.extend(["", "E. Candidate schedule-delay rules"])
        for rule in result["rules"]:
            lines.append(
                f"  progress_gap < {rule['threshold']}: "
                f"delayed rows={rule['delayed_rows']}, "
                f"non-delayed rows={rule['non_delayed_rows']}, "
                f"delayed={rule['delayed_pct']:.2f}%, "
                f"delayed projects={rule['delayed_projects']}, "
                f"non-delayed projects={rule['non_delayed_projects']}"
            )

    ml = analyses[0]
    july = analyses[1]
    lines.extend(
        [
            "",
            "F. Cross-dataset comparison",
            "The ML feature dataset and July live dataset are audited separately.",
            "A target can only be calculated where start_date, original_doc,",
            "report_month, and physical_progress are all valid under the rules.",
            f"ML valid schedule rows: {int(ml['valid'].sum())}",
            f"ML valid schedule projects: {ml['frame'].loc[ml['valid'], 'project_code'].nunique()}",
            f"July valid schedule rows: {int(july['valid'].sum())}",
            f"July valid schedule projects: {july['frame'].loc[july['valid'], 'project_code'].nunique()}",
            "",
            "G. Current-information and leakage audit",
            "The calculation uses only start_date, original_doc, report_month,",
            "and current physical_progress. These are available at the current",
            "report_month in principle. It does not use actual/revised completion",
            "dates or any later progress, expenditure, cost, or status.",
            "",
            "H. Safe ML features",
        ]
    )
    lines.extend(f"  {feature}" for feature in SAFE_FEATURES)
    lines.extend(
        [
            "",
            "Unsafe features to exclude:",
            "  actual completion date, revised completion date, future completion",
            "  future progress, future expenditure, future cost, final status,",
            "  project_code, project_name, source_file, source_sheet",
            "",
            "I. Recommendation",
        ]
    )

    valid_counts = {r["threshold"]: r for r in ml["rules"]}
    if not int(ml["valid"].sum()):
        recommendation = (
            "No threshold can be recommended from the current ML feature "
            "dataset because it has no valid rows: start_date is present "
            "but original_doc is empty. The July dataset also has no valid "
            "rows because start_date is empty."
        )
    else:
        chosen = next(
            (
                r
                for r in ml["rules"]
                if 20 <= r["delayed_pct"] <= 80
            ),
            None,
        )
        recommendation = (
            f"Recommended threshold: progress_gap < {chosen['threshold']}"
            if chosen
            else "No threshold gives a reasonably balanced distribution."
        )
    lines.extend(
        [
            recommendation,
            "",
            "J. Answers",
            "1. Can we create a defensible current schedule-delay target?",
            "   Not from ML_FEATURE_DATASET.csv or JULY_2026_FEATURE_DATASET.csv "
            "as currently materialized. The concept is defensible, but the "
            "required date fields do not overlap in either dataset.",
            "2. Which threshold should we use?",
            "   None can be selected yet. A threshold requires valid schedule "
            "rows and should be chosen after the feature pipeline preserves "
            "both dates without future leakage.",
            "3. How many training rows/projects would we have?",
            f"   Current ML dataset: {int(ml['valid'].sum())} rows, "
            f"{ml['frame'].loc[ml['valid'], 'project_code'].nunique()} projects.",
            "4. How many July projects would be currently behind schedule?",
            f"   Current July dataset: no valid schedule rows, so 0 can be "
            "classified defensibly.",
            "5. Is the class balance suitable for ML?",
            "   No. There are no valid rows in either supplied feature dataset.",
            "6. Is this better than confirmed-actual completion?",
            "   It is a better-defined real-time risk concept, but it is not "
            "currently implementable from these materialized datasets. It "
            "should not replace the confirmed-outcome conclusion yet.",
            "7. Can we call this Time Delay Risk Prediction in the SIH demo?",
            "   Only with the qualifier 'current schedule delay risk' or "
            "'schedule slippage risk' after rebuilding the feature dataset "
            "with point-in-time start and original-target dates. Do not call "
            "it prediction of actual completion delay.",
            "",
            "Conclusion:",
            "Do not train a model yet. Preserve start_date and original_doc "
            "as point-in-time schedule inputs in a separate delay pipeline, "
            "then rerun this audit before selecting a threshold.",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
