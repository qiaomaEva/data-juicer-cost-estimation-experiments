import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import REPORTS_DIR, SUMMARIES_DIR, ensure_output_dirs


LABEL = "op_process_time"
PRED_LABEL = "op_process_time_pred"

DISPLAY_NAME = {
    "overall": "总和采集数据",
    "audio": "音频采集数据",
    "image": "图像采集数据",
    "text": "文本采集数据",
}


def normalize_suffix(suffix):
    if not suffix:
        return ""
    return suffix if suffix.startswith("_") else f"_{suffix}"


def infer_suffix(pred_path):
    stem = Path(pred_path).stem
    prefix = "test_set_predictions_cost"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return f"_{stem}"


def format_number(value, digits=2):
    if pd.isna(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def calculate_metrics(df, datatype, label):
    subset = df.copy()
    if datatype in {"audio", "image", "text"}:
        subset = subset[subset["ds_type"] == datatype].copy()

    subset = subset.dropna(subset=[LABEL, PRED_LABEL]).copy()
    y_true = pd.to_numeric(subset[LABEL], errors="coerce")
    y_pred = pd.to_numeric(subset[PRED_LABEL], errors="coerce")
    valid_mask = (
        y_true.notna()
        & y_pred.notna()
        & np.isfinite(y_true)
        & np.isfinite(y_pred)
        & (y_true > 0)
    )
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return {
            "datatype": datatype,
            "display_name": label,
            "sample_count": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "accuracy": np.nan,
        }

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {
        "datatype": datatype,
        "display_name": label,
        "sample_count": int(len(y_true)),
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "accuracy": float(100 - mape),
    }


def write_markdown_table(file_obj, df):
    headers = ["数据范围", "算子样本数", "RMSE (秒)", "MAE (秒)", "MAPE", "准确率"]
    file_obj.write("| " + " | ".join(headers) + " |\n")
    file_obj.write("|---|---:|---:|---:|---:|---:|\n")
    for _, row in df.iterrows():
        file_obj.write(
            "| {name} | {samples} | {rmse} | {mae} | {mape}% | {acc}% |\n".format(
                name=row["display_name"],
                samples=int(row["sample_count"]),
                rmse=format_number(row["rmse"]),
                mae=format_number(row["mae"]),
                mape=format_number(row["mape"]),
                acc=format_number(row["accuracy"]),
            )
        )
    file_obj.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Compute cost operator-only accuracy from one prediction CSV.")
    parser.add_argument(
        "--pred_path",
        required=True,
        help="Prediction CSV containing op_process_time and op_process_time_pred.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Output suffix. Defaults to suffix inferred from --pred_path.",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    pred_path = Path(args.pred_path)
    if not pred_path.exists():
        raise FileNotFoundError(f"prediction csv not found: {pred_path}")

    df = pd.read_csv(pred_path, low_memory=False)
    required_cols = {LABEL, PRED_LABEL, "ds_type"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"prediction csv missing required columns: {missing_cols}")

    rows = [
        calculate_metrics(df, "overall", DISPLAY_NAME["overall"]),
        calculate_metrics(df, "audio", DISPLAY_NAME["audio"]),
        calculate_metrics(df, "image", DISPLAY_NAME["image"]),
        calculate_metrics(df, "text", DISPLAY_NAME["text"]),
    ]
    summary_df = pd.DataFrame(rows)

    suffix = normalize_suffix(args.suffix) if args.suffix else infer_suffix(pred_path)
    summary_path = SUMMARIES_DIR / f"cost_operator_accuracy{suffix}.csv"
    report_path = REPORTS_DIR / f"代价估计算子级准确率报告{suffix}.md"
    summary_df.to_csv(summary_path, index=False, float_format="%.4f")

    with open(report_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("# 代价估计算子级准确率报告\n\n")
        file_obj.write(f"- 预测文件：`{pred_path}`\n")
        file_obj.write("- 指标口径：只统计算子级预测，不计算 pipeline 级汇总。\n")
        file_obj.write("- `准确率 = 100 - MAPE`，其中 MAPE 基于 `op_process_time` 与 `op_process_time_pred` 计算。\n\n")
        write_markdown_table(file_obj, summary_df)

    print(f"saved summary csv: {summary_path}")
    print(f"saved report: {report_path}")
    print("\n算子级结果")
    print("=" * 78)
    print(f"{'数据范围':<16}{'算子样本':>10}{'RMSE':>12}{'MAE':>12}{'MAPE':>12}{'准确率':>12}")
    print("-" * 78)
    for _, row in summary_df.iterrows():
        print(
            f"{row['display_name']:<16}"
            f"{int(row['sample_count']):>10}"
            f"{format_number(row['rmse']):>12}"
            f"{format_number(row['mae']):>12}"
            f"{format_number(row['mape']) + '%':>12}"
            f"{format_number(row['accuracy']) + '%':>12}"
        )


if __name__ == "__main__":
    main()
