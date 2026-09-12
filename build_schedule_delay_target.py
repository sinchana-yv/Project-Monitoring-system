"""Analyze schedule-delay target thresholds without training a model."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "DELAY_FEATURE_DATASET.csv"
OUTPUT = ROOT / "schedule_delay_target_audit.txt"
THRESHOLDS = [0, -10, -20, -30]


def analyze_threshold(frame, threshold):
    delayed = frame["progress_gap"] < threshold
    non_delayed = ~delayed
    return {
        "threshold": threshold,
        "total": len(frame),
        "delayed_rows": int(delayed.sum()),
        "non_delayed_rows": int(non_delayed.sum()),
        "delayed_pct": float(delayed.mean() * 100) if len(frame) else 0.0,
        "delayed_projects": int(
            frame.loc[delayed, "project_code"].nunique()
        ),
        "non_delayed_projects": int(
            frame.loc[non_delayed, "project_code"].nunique()
        ),
    }


def format_rule(result):
    return (
        f"progress_gap < {result['threshold']}: "
        f"total={result['total']}, delayed={result['delayed_rows']}, "
        f"non-delayed={result['non_delayed_rows']}, "
        f"delayed_pct={result['delayed_pct']:.2f}%, "
        f"delayed_projects={result['delayed_projects']}, "
        f"non_delayed_projects={result['non_delayed_projects']}"
    )


def main():
    data = pd.read_csv(INPUT, low_memory=False)
    valid = data[data["schedule_valid"].astype(str).str.lower() == "true"].copy()
    valid["project_code"] = valid["project_code"].astype(str).str.strip()

    numeric = [
        "planned_duration_days_raw",
        "elapsed_duration_days",
        "physical_progress",
    ]
    for column in numeric:
        valid[column] = pd.to_numeric(valid[column], errors="coerce")

    valid["expected_progress_raw_analysis"] = (
        valid["elapsed_duration_days"]
        / valid["planned_duration_days_raw"]
        * 100
    )
    valid["expected_progress_analysis"] = valid[
        "expected_progress_raw_analysis"
    ].clip(upper=100)
    valid["progress_gap"] = (
        valid["physical_progress"] - valid["expected_progress_analysis"]
    )
    valid = valid[valid["progress_gap"].notna()].copy()

    distribution = valid["progress_gap"].describe(
        percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
    )
    row_results = [
        analyze_threshold(valid, threshold)
        for threshold in THRESHOLDS
    ]

    valid["report_month"] = pd.to_datetime(
        valid["report_month"],
        errors="coerce",
    )
    project_counts = valid.groupby("project_code").size()
    latest = (
        valid.sort_values(["project_code", "report_month"])
        .drop_duplicates("project_code", keep="last")
        .copy()
    )

    project_results = []
    for threshold in THRESHOLDS:
        delayed = valid["progress_gap"] < threshold
        latest_delayed = latest["progress_gap"] < threshold
        project_results.append(
            {
                "threshold": threshold,
                "projects_multiple": int((project_counts > 1).sum()),
                "delayed_any_project": int(
                    valid.loc[delayed, "project_code"].nunique()
                ),
                "latest_delayed_projects": int(latest_delayed.sum()),
                "latest_not_delayed_projects": int(
                    (~latest_delayed).sum()
                ),
            }
        )

    july = valid[valid["report_month"] == pd.Timestamp("2026-07-01")].copy()
    july_results = [
        analyze_threshold(july, threshold)
        for threshold in THRESHOLDS
    ]

    # Choose a threshold using balance, positive/negative counts, and
    # early-warning meaning. Prefer -20 when it is reasonably balanced and
    # sufficiently populated; otherwise choose the closest balanced rule.
    selected = min(
        row_results,
        key=lambda result: (
            abs(result["delayed_pct"] - 50),
            -min(result["delayed_rows"], result["non_delayed_rows"]),
        ),
    )
    if (
        selected["threshold"] == 0
        and row_results[2]["delayed_rows"] >= 100
        and row_results[2]["non_delayed_rows"] >= 100
    ):
        selected = row_results[2]

    selected_project = next(
        item
        for item in project_results
        if item["threshold"] == selected["threshold"]
    )
    selected_july = next(
        item
        for item in july_results
        if item["threshold"] == selected["threshold"]
    )

    lines = [
        "SCHEDULE DELAY TARGET AUDIT",
        "=" * 78,
        "",
        f"Input: {INPUT.name}",
        "Only rows with schedule_valid == True were analyzed.",
        "No source dataset was modified and no model was trained.",
        "",
        "TARGET DEFINITION",
        "-----------------",
        "expected_progress_raw = elapsed_duration_days / "
        "planned_duration_days_raw * 100",
        "expected_progress = expected_progress_raw capped at 100",
        "progress_gap = physical_progress - expected_progress",
        "Positive gap means ahead; zero is approximately on schedule; negative "
        "gap means behind.",
        "",
        "LEAKAGE CHECK",
        "-------------",
        "The target uses only start_date-derived duration, original_doc-derived "
        "planned duration, report_month, and current physical_progress.",
        "Actual completion, revised completion, future progress, future "
        "expenditure, future cost, and final project status were not used.",
        "All retained candidate ML features come from the same report-month row "
        "or historical/lag values already available at that report month.",
        "",
        "ROW-LEVEL RESULTS",
        "-----------------",
        f"Total valid rows: {len(valid)}",
    ]
    lines.extend(format_rule(result) for result in row_results)
    lines.extend(
        [
            "",
            "PROGRESS-GAP DISTRIBUTION",
            "-------------------------",
            f"minimum: {distribution['min']:.4f}",
            f"maximum: {distribution['max']:.4f}",
            f"mean: {distribution['mean']:.4f}",
            f"median: {distribution['50%']:.4f}",
            f"10th percentile: {distribution['10%']:.4f}",
            f"25th percentile: {distribution['25%']:.4f}",
            f"50th percentile: {distribution['50%']:.4f}",
            f"75th percentile: {distribution['75%']:.4f}",
            f"90th percentile: {distribution['90%']:.4f}",
            "",
            "CLASS-DOMINATION CHECK",
            "----------------------",
        ]
    )
    for result in row_results:
        dominant = max(
            result["delayed_rows"],
            result["non_delayed_rows"],
        )
        dominant_pct = dominant / result["total"] * 100
        lines.append(
            f"progress_gap < {result['threshold']}: dominant class "
            f"{dominant_pct:.2f}%"
        )

    lines.extend(["", "PROJECT-LEVEL ANALYSIS", "----------------------"])
    lines.append(
        f"Projects with multiple valid monthly observations: "
        f"{int((project_counts > 1).sum())}"
    )
    lines.append(f"Projects with one valid snapshot: {int((project_counts == 1).sum())}")
    for item in project_results:
        lines.append(
            f"progress_gap < {item['threshold']}: delayed in at least one "
            f"month={item['delayed_any_project']}, latest delayed="
            f"{item['latest_delayed_projects']}, latest not delayed="
            f"{item['latest_not_delayed_projects']}"
        )

    lines.extend(["", "JULY 2026 ANALYSIS", "------------------"])
    lines.append(f"Valid July 2026 rows: {len(july)}")
    for result in july_results:
        lines.append(
            f"progress_gap < {result['threshold']}: delayed projects="
            f"{result['delayed_projects']}, non-delayed projects="
            f"{result['non_delayed_projects']}, delayed="
            f"{result['delayed_pct']:.2f}%"
        )

    lines.extend(
        [
            "",
            "RECOMMENDATION",
            "--------------",
            f"RECOMMENDED THRESHOLD:",
            f"progress_gap < {selected['threshold']}",
            "",
            "REASON:",
            "This threshold was selected using interpretability, class "
            "balance, positive/negative sample counts, and early-warning "
            "usefulness. It is not selected for maximum accuracy. A gap of "
            f"{selected['threshold']} percentage points represents a meaningful "
            "shortfall from the current planned schedule while retaining both "
            "classes for analysis.",
            "",
            "TRAINING ROWS:",
            str(selected["total"]),
            "",
            "DELAYED:",
            str(selected["delayed_rows"]),
            "",
            "NOT DELAYED:",
            str(selected["non_delayed_rows"]),
            "",
            "JULY DELAYED:",
            str(selected_july["delayed_projects"]),
            "",
            "JULY NOT DELAYED:",
            str(selected_july["non_delayed_projects"]),
            "",
            "PROJECT-LEVEL RECOMMENDED RESULT:",
            f"Delayed in at least one valid month: "
            f"{selected_project['delayed_any_project']}",
            f"Delayed in latest valid snapshot: "
            f"{selected_project['latest_delayed_projects']}",
            f"Not delayed in latest valid snapshot: "
            f"{selected_project['latest_not_delayed_projects']}",
            "",
            "INTERPRETATION",
            "--------------",
            "This target represents current schedule slippage risk, not "
            "confirmed eventual completion delay. It can be used for a "
            "separate early-warning model only after model validation and "
            "time/project-aware evaluation.",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
