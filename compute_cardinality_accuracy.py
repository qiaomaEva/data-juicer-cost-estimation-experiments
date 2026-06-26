import argparse
import os

import numpy as np
import pandas as pd

from project_paths import (
    PREDICTIONS_DIR,
    REPORTS_DIR,
    SUMMARIES_DIR,
    ensure_output_dirs,
    legacy_output_path,
)

LABEL = "ds_output_count"
PRED_LABEL = "ds_output_count_pred"
INPUT_COUNT_COL = "ds_input_count"
RATIO_TRUE_COL = "cardinality_ratio_true"
RATIO_PRED_COL = "cardinality_ratio_pred"
TARGET_OUTPUT = "output"
TARGET_RATIO = "ratio"
TARGET_SMOOTH_LOG_RATIO = "smooth_log_ratio"

DISPLAY_NAME = {
    "overall": "总和采集数据",
    "audio": "音频采集数据",
    "image": "图像采集数据",
    "text": "文本采集数据",
}


def normalize_extra_suffix(extra_suffix):
    if not extra_suffix:
        return ""
    return extra_suffix if extra_suffix.startswith("_") else f"_{extra_suffix}"


def build_suffix(log_target, extra_suffix="", target_mode=TARGET_OUTPUT):
    if target_mode == TARGET_RATIO:
        target_suffix = "_ratio"
    elif target_mode == TARGET_SMOOTH_LOG_RATIO:
        target_suffix = "_smooth_log_ratio"
    else:
        target_suffix = ""
    log_suffix = "_log" if log_target else ""
    return f"{target_suffix}{log_suffix}{normalize_extra_suffix(extra_suffix)}"


def is_ratio_like_mode(target_mode):
    return target_mode in {TARGET_RATIO, TARGET_SMOOTH_LOG_RATIO}


def build_input_candidates(log_target, target_mode, single_model=False, input_suffix=""):
    suffix = build_suffix(log_target, input_suffix, target_mode=target_mode)
    overall_file = str(PREDICTIONS_DIR / f"test_set_predictions_cardinality{suffix}.csv")
    legacy_overall_file = legacy_output_path(f"test_set_predictions_cardinality{suffix}.csv")
    if single_model:
        return {
            "overall": [overall_file, legacy_overall_file],
            "audio": [overall_file, legacy_overall_file],
            "image": [overall_file, legacy_overall_file],
            "text": [overall_file, legacy_overall_file],
        }

    return {
        "overall": [overall_file, legacy_overall_file],
        "audio": [
            str(PREDICTIONS_DIR / f"test_set_predictions_cardinality_audio{suffix}.csv"),
            legacy_output_path(f"test_set_predictions_cardinality_audio{suffix}.csv"),
        ],
        "image": [
            str(PREDICTIONS_DIR / f"test_set_predictions_cardinality_image{suffix}.csv"),
            legacy_output_path(f"test_set_predictions_cardinality_image{suffix}.csv"),
        ],
        "text": [
            str(PREDICTIONS_DIR / f"test_set_predictions_cardinality_text{suffix}.csv"),
            legacy_output_path(f"test_set_predictions_cardinality_text{suffix}.csv"),
        ],
    }


def build_summary_csv(log_target, extra_suffix, target_mode):
    suffix = build_suffix(log_target, extra_suffix, target_mode)
    return str(SUMMARIES_DIR / f"cardinality_performance_summary{suffix}.csv")


def build_report_md(log_target, extra_suffix, target_mode):
    suffix = build_suffix(log_target, extra_suffix, target_mode)
    return str(REPORTS_DIR / f"基数估计准确率汇总报告{suffix}.md")


def first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def build_ratio_metrics(df):
    if RATIO_TRUE_COL not in df.columns or RATIO_PRED_COL not in df.columns:
        return np.nan, np.nan
    if INPUT_COUNT_COL not in df.columns:
        return np.nan, np.nan

    ratio_df = df.dropna(subset=[RATIO_TRUE_COL, RATIO_PRED_COL, INPUT_COUNT_COL]).copy()
    ratio_df = ratio_df[ratio_df[INPUT_COUNT_COL] > 0].copy()
    if len(ratio_df) == 0:
        return np.nan, np.nan

    y_true = pd.to_numeric(ratio_df[RATIO_TRUE_COL], errors="coerce")
    y_pred = pd.to_numeric(ratio_df[RATIO_PRED_COL], errors="coerce")
    valid_mask = y_true.notna() & y_pred.notna() & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    if len(y_true) == 0:
        return np.nan, np.nan

    ratio_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ratio_mae = float(np.mean(np.abs(y_true - y_pred)))
    return ratio_rmse, ratio_mae


def build_metric_dict(datatype, input_file, y_true, y_pred, ratio_rmse=np.nan, ratio_mae=np.nan):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    accuracy = float(100 - mape)

    return {
        "datatype": datatype,
        "data_range": DISPLAY_NAME[datatype],
        "sample_count": int(len(y_true)),
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "accuracy": accuracy,
        "ratio_rmse": ratio_rmse,
        "ratio_mae": ratio_mae,
        "input_file": input_file,
    }


def load_metrics(datatype, log_target, target_mode, single_model=False, input_suffix=""):
    input_file = first_existing(
        build_input_candidates(log_target, target_mode, single_model, input_suffix)[datatype]
    )
    if input_file is None:
        raise FileNotFoundError(
            "no cardinality prediction file found for "
            f"{datatype}, log_target={log_target}, target_mode={target_mode}"
        )

    full_df = pd.read_csv(input_file, low_memory=False)
    if LABEL not in full_df.columns or PRED_LABEL not in full_df.columns:
        raise ValueError(f"missing required columns in {input_file}")

    if datatype in {"audio", "image", "text"} and "ds_type" in full_df.columns:
        full_df = full_df[full_df["ds_type"] == datatype].copy()

    ratio_rmse, ratio_mae = (
        build_ratio_metrics(full_df) if is_ratio_like_mode(target_mode) else (np.nan, np.nan)
    )

    df = full_df.dropna(subset=[LABEL, PRED_LABEL]).copy()
    df = df[df[LABEL] > 0].copy()

    y_true = pd.to_numeric(df[LABEL], errors="coerce")
    y_pred = pd.to_numeric(df[PRED_LABEL], errors="coerce")
    valid_mask = y_true.notna() & y_pred.notna() & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        raise ValueError(f"no valid rows after filtering: {input_file}")

    return build_metric_dict(datatype, input_file, y_true, y_pred, ratio_rmse, ratio_mae)


def build_summary(log_target, target_mode, single_model=False, input_suffix=""):
    rows = []
    for datatype in ["overall", "audio", "image", "text"]:
        rows.append(load_metrics(datatype, log_target, target_mode, single_model, input_suffix))
    return pd.DataFrame(rows)


def print_table(summary_df):
    print("\n" + "=" * 96)
    print("基数估计结果表")
    print("=" * 96)
    print(
        "{:<20} {:>10} {:>12} {:>12} {:>10} {:>12} {:>12} {:>12}".format(
            "数据范围",
            "样本数",
            "RMSE",
            "MAE",
            "MAPE",
            "准确率",
            "ratio_RMSE",
            "ratio_MAE",
        )
    )
    print("-" * 96)

    for _, row in summary_df.iterrows():
        print(
            "{:<20} {:>10} {:>12.2f} {:>12.2f} {:>9.2f}% {:>11.2f}% {:>12} {:>12}".format(
                row["data_range"],
                int(row["sample_count"]),
                row["rmse"],
                row["mae"],
                row["mape"],
                row["accuracy"],
                format_optional_metric(row.get("ratio_rmse")),
                format_optional_metric(row.get("ratio_mae")),
            )
        )


def format_optional_metric(value):
    if pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def write_report(summary_df, report_md, extra_suffix, target_mode):
    mode_text = extra_suffix if extra_suffix else "(default)"

    with open(report_md, "w", encoding="utf-8") as file_obj:
        file_obj.write("# 基数估计准确率汇总报告\n\n")
        file_obj.write("## 结果标识\n\n")
        file_obj.write(f"- 当前结果后缀：`{mode_text}`\n\n")
        file_obj.write(f"- 目标模式：`{target_mode}`\n\n")
        file_obj.write("## 指标口径\n\n")
        file_obj.write("- 统计对象：`ds_output_count` 的算子级测试集预测结果。\n")
        file_obj.write("- `准确率` 定义为 `100 - MAPE`。\n\n")
        if is_ratio_like_mode(target_mode):
            file_obj.write(
                "- `ratio_rmse` / `ratio_mae` 仅在 `ds_input_count > 0` 且 ratio 有效的样本上计算。\n\n"
            )
        file_obj.write("## 最终结果表\n\n")
        if is_ratio_like_mode(target_mode):
            file_obj.write("| 数据范围 | 样本数 | RMSE | MAE | MAPE | 准确率 | ratio_RMSE | ratio_MAE |\n")
            file_obj.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        else:
            file_obj.write("| 数据范围 | 样本数 | RMSE | MAE | MAPE | 准确率 |\n")
            file_obj.write("|---|---:|---:|---:|---:|---:|\n")

        for _, row in summary_df.iterrows():
            if is_ratio_like_mode(target_mode):
                file_obj.write(
                    "| {0} | {1} | {2:.2f} | {3:.2f} | {4:.2f}% | {5:.2f}% | {6} | {7} |\n".format(
                        row["data_range"],
                        int(row["sample_count"]),
                        row["rmse"],
                        row["mae"],
                        row["mape"],
                        row["accuracy"],
                        format_optional_metric(row.get("ratio_rmse")),
                        format_optional_metric(row.get("ratio_mae")),
                    )
                )
            else:
                file_obj.write(
                    "| {0} | {1} | {2:.2f} | {3:.2f} | {4:.2f}% | {5:.2f}% |\n".format(
                        row["data_range"],
                        int(row["sample_count"]),
                        row["rmse"],
                        row["mae"],
                        row["mape"],
                        row["accuracy"],
                    )
                )

        file_obj.write("\n## 数据文件来源\n\n")
        for _, row in summary_df.iterrows():
            file_obj.write(f"- `{row['data_range']}`：`{row['input_file']}`\n")


def main():
    parser = argparse.ArgumentParser(description="Compute cardinality prediction summary.")
    parser.add_argument(
        "--log_target",
        action="store_true",
        help="Read *_log files generated from log-target cardinality prediction runs.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Extra suffix appended to output summary filenames.",
    )
    parser.add_argument(
        "--input_suffix",
        default="",
        help=(
            "Extra suffix appended to input prediction filenames, for example alpha0p1 "
            "when prediction_cardinality.py was run with --model_suffix alpha0p1."
        ),
    )
    parser.add_argument(
        "--target_mode",
        choices=[TARGET_OUTPUT, TARGET_RATIO, TARGET_SMOOTH_LOG_RATIO],
        default=TARGET_OUTPUT,
        help="Read cardinality predictions generated by this target mode.",
    )
    parser.add_argument(
        "--single_model",
        action="store_true",
        help=(
            "Use the overall prediction file for overall/audio/image/text metrics, "
            "then filter by ds_type instead of requiring per-modality prediction files."
        ),
    )
    args = parser.parse_args()
    ensure_output_dirs()

    if args.target_mode == TARGET_SMOOTH_LOG_RATIO and args.log_target:
        raise ValueError("smooth_log_ratio 模式本身已经是 log 目标，汇总时不要再叠加 --log_target。")

    report_suffix = args.suffix
    if not report_suffix:
        report_suffix = args.input_suffix
    if args.single_model and not report_suffix:
        report_suffix = "single_model"
    elif args.single_model and "single_model" not in report_suffix:
        report_suffix = f"{report_suffix}_single_model"

    summary_df = build_summary(
        args.log_target,
        args.target_mode,
        args.single_model,
        args.input_suffix,
    )
    summary_csv = build_summary_csv(args.log_target, report_suffix, args.target_mode)
    report_md = build_report_md(args.log_target, report_suffix, args.target_mode)

    summary_df.to_csv(summary_csv, index=False, float_format="%.4f")
    print(f"saved summary csv: {summary_csv}")
    print_table(summary_df)
    write_report(summary_df, report_md, report_suffix, args.target_mode)
    print(f"saved report: {report_md}")


if __name__ == "__main__":
    main()
