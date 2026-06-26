import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_paths import (
    FIGURES_DIR,
    PREDICTIONS_DIR,
    SUMMARIES_DIR,
    ensure_output_dirs,
    first_existing,
    legacy_output_path,
)


def build_suffix(log_target):
    return "_log" if log_target else ""


def build_paths(log_target):
    suffix = build_suffix(log_target)
    prediction_filename = f"test_set_predictions_cost{suffix}.csv"
    return {
        "input_file": first_existing(
            PREDICTIONS_DIR / prediction_filename,
            legacy_output_path(prediction_filename),
        )
        or str(PREDICTIONS_DIR / prediction_filename),
        "output_file": str(SUMMARIES_DIR / f"processed_op_performance{suffix}.csv"),
        "figure_file": str(FIGURES_DIR / f"operator_accuracy_distribution{suffix}.png"),
    }


def extract_operator_performance(input_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"input file not found: {input_file}")

    df = pd.read_csv(input_file, low_memory=False)
    op_type_cols = [col for col in df.columns if col.startswith("op_type_")]

    records = []
    for _, row in df.iterrows():
        op_name = None
        for col in op_type_cols:
            val = row[col]
            if pd.notna(val) and str(val).strip() != "":
                op_name = col.replace("op_type_", "")
                break

        if op_name is None:
            continue

        true_time = row.get("op_process_time")
        pred_time = row.get("op_process_time_pred")
        if pd.isna(true_time) or pd.isna(pred_time):
            continue

        try:
            true_time = float(true_time)
            pred_time = float(pred_time)
        except (TypeError, ValueError):
            continue

        if not (np.isfinite(true_time) and np.isfinite(pred_time)):
            continue

        absolute_deviation = abs(true_time - pred_time)
        prediction_accuracy_score = 1.0 - (absolute_deviation / (true_time + 1e-9))

        records.append(
            {
                "op_name": op_name,
                "op_process_time": true_time,
                "op_process_time_pred": pred_time,
                "absolute_deviation": absolute_deviation,
                "prediction_accuracy_score": prediction_accuracy_score,
            }
        )

    return pd.DataFrame(records)


def draw(processed_file, figure_file):
    df = pd.read_csv(processed_file, low_memory=False)
    df["prediction_accuracy_score"] = pd.to_numeric(
        df["prediction_accuracy_score"],
        errors="coerce",
    )
    df = df.dropna(subset=["prediction_accuracy_score"]).copy()

    summary = (
        df.groupby("op_name", dropna=False)
        .agg(
            low_acc=("prediction_accuracy_score", lambda s: int((s < 0.9).sum())),
            high_acc=("prediction_accuracy_score", lambda s: int((s >= 0.9).sum())),
        )
        .reset_index()
    )

    summary["total"] = summary["low_acc"] + summary["high_acc"]
    summary = summary.sort_values("total", ascending=True)

    y_labels = summary["op_name"]
    x_low = summary["low_acc"]
    x_high = summary["high_acc"]

    fig, ax = plt.subplots(figsize=(10, max(6, len(y_labels) * 0.4)))
    ax.barh(y_labels, x_low, color="#F54E57", label="Accuracy < 90%")
    ax.barh(y_labels, x_high, left=x_low, color="#4A90E2", label="Accuracy >= 90%")

    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Operator (op_name)")
    ax.set_title("Operator Prediction Accuracy Distribution")
    ax.legend(loc="lower right")

    for i, (low, high) in enumerate(zip(x_low, x_high)):
        if low > 0:
            ax.text(low / 2, i, str(low), va="center", ha="center", color="white", fontsize=9)
        if high > 0:
            ax.text(low + high / 2, i, str(high), va="center", ha="center", color="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(figure_file, dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Summarize operator prediction accuracy.")
    parser.add_argument(
        "--log_target",
        action="store_true",
        help="Read/write *_log files generated from log-target cost prediction runs.",
    )
    args = parser.parse_args()
    ensure_output_dirs()

    paths = build_paths(args.log_target)
    result_df = extract_operator_performance(paths["input_file"])
    if result_df.empty:
        print("no valid operator prediction rows found.")
        return

    result_df.to_csv(paths["output_file"], index=False)
    print(f"saved {len(result_df)} rows to: {paths['output_file']}")
    print(result_df.head(3))

    draw(paths["output_file"], paths["figure_file"])
    print(f"saved figure: {paths['figure_file']}")


if __name__ == "__main__":
    main()
