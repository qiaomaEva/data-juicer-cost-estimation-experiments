import argparse
import os

import numpy as np
import pandas as pd

from project_paths import (
    EXISTING_MODEL_COMPARISON_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    SUMMARIES_DIR,
    ensure_output_dirs,
    legacy_output_path,
)

LABEL = "op_process_time"
PRED_LABEL = "op_process_time_pred"

DISPLAY_NAME = {
    "overall": "总和采集数据",
    "audio": "音频采集数据",
    "image": "图像采集数据",
    "text": "文本采集数据",
}

LEGACY_EXISTING_MODEL_COMPARISON_DIR = legacy_output_path(
    "existing_model_fixed_testset_comparison"
)

MODE_OVERALL_ONLY = "overall_only"
MODE_AUDIO_IMAGE_OVERALL_TEXT_ONLY = "audio_image_overall_text_only"


def normalize_extra_suffix(extra_suffix):
    if not extra_suffix:
        return ""
    return extra_suffix if extra_suffix.startswith("_") else f"_{extra_suffix}"


def build_suffix(log_target, extra_suffix=""):
    return ("_log" if log_target else "") + normalize_extra_suffix(extra_suffix)


def build_operator_file_candidates(log_target, single_model=False):
    log_suffix = build_suffix(log_target)
    overall_file = str(PREDICTIONS_DIR / f"test_set_predictions_cost{log_suffix}.csv")
    legacy_overall_file = legacy_output_path(f"test_set_predictions_cost{log_suffix}.csv")
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
            str(PREDICTIONS_DIR / f"test_set_predictions_cost_audio{log_suffix}.csv"),
            legacy_output_path(f"test_set_predictions_cost_audio{log_suffix}.csv"),
        ],
        "image": [
            str(PREDICTIONS_DIR / f"test_set_predictions_cost_image{log_suffix}.csv"),
            legacy_output_path(f"test_set_predictions_cost_image{log_suffix}.csv"),
        ],
        "text": [
            str(PREDICTIONS_DIR / f"test_set_predictions_cost_text{log_suffix}.csv"),
            legacy_output_path(f"test_set_predictions_cost_text{log_suffix}.csv"),
        ],
    }


def build_pipeline_prediction_file(log_target, extra_suffix):
    suffix = build_suffix(log_target, extra_suffix)
    filename = f"test_set_predictions_cost_full_with_pipeline{suffix}.csv"
    candidates = [str(PREDICTIONS_DIR / filename), legacy_output_path(filename)]
    return first_existing(candidates) or candidates[0]


def build_summary_csv(log_target, extra_suffix):
    suffix = build_suffix(log_target, extra_suffix)
    return str(SUMMARIES_DIR / f"datatype_performance_summary{suffix}.csv")


def build_report_md(log_target, extra_suffix):
    suffix = build_suffix(log_target, extra_suffix)
    return str(REPORTS_DIR / f"总体准确率汇总报告{suffix}.md")


def first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def resolve_existing_model_comparison_file(datatype, log_target, extra_suffix):
    if not extra_suffix:
        return None

    normalized = normalize_extra_suffix(extra_suffix).lstrip("_")
    if normalized not in {MODE_OVERALL_ONLY, MODE_AUDIO_IMAGE_OVERALL_TEXT_ONLY}:
        return None

    log_suffix = "_log" if log_target else ""

    filename_map = {
        "overall": f"fixed_testset_overall_predictions{log_suffix}.csv",
        "audio": f"fixed_testset_audio_predictions{log_suffix}.csv",
        "image": f"fixed_testset_image_predictions{log_suffix}.csv",
        "text": f"fixed_testset_text_predictions{log_suffix}.csv",
    }
    filename = filename_map.get(datatype)
    if filename is None:
        return None
    candidates = [
        str(EXISTING_MODEL_COMPARISON_DIR / filename),
        os.path.join(LEGACY_EXISTING_MODEL_COMPARISON_DIR, filename),
    ]
    return first_existing(candidates) or candidates[0]


def get_pred_column_for_existing_comparison(datatype, extra_suffix):
    normalized = normalize_extra_suffix(extra_suffix).lstrip("_")
    if normalized == MODE_OVERALL_ONLY:
        return "op_process_time_pred_overall"
    if normalized == MODE_AUDIO_IMAGE_OVERALL_TEXT_ONLY:
        if datatype == "overall":
            return "op_process_time_pred_overall"
        if datatype in {"audio", "image"}:
            return "op_process_time_pred_overall"
        if datatype == "text":
            return "op_process_time_pred_text"
    return None


def build_metric_dict(datatype, input_file, y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    accuracy = float(100 - mape)

    return {
        "datatype": datatype,
        "operator_file": input_file,
        "sample_count": int(len(y_true)),
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "accuracy": accuracy,
    }


def load_operator_metrics_from_pipeline_prediction(datatype, log_target, extra_suffix):
    input_file = build_pipeline_prediction_file(log_target, extra_suffix)
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"pipeline prediction file not found: {input_file}")

    df = pd.read_csv(input_file, low_memory=False)
    required_cols = [LABEL, PRED_LABEL, "ds_type"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"pipeline prediction file missing columns: {missing_cols}")

    if datatype in {"audio", "image", "text"}:
        df = df[df["ds_type"] == datatype].copy()

    df = df.dropna(subset=[LABEL, PRED_LABEL]).copy()
    df = df[df[LABEL] > 0].copy()

    y_true = pd.to_numeric(df[LABEL], errors="coerce")
    y_pred = pd.to_numeric(df[PRED_LABEL], errors="coerce")
    valid_mask = y_true.notna() & y_pred.notna() & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    source = f"{input_file}::ds_type={datatype}" if datatype != "overall" else input_file
    return build_metric_dict(datatype, source, y_true, y_pred)


def load_operator_metrics_from_standard_file(datatype, log_target, single_model=False):
    input_file = first_existing(build_operator_file_candidates(log_target, single_model)[datatype])
    if input_file is None:
        raise FileNotFoundError(
            f"no operator prediction file found for {datatype}, log_target={log_target}"
        )

    df = pd.read_csv(input_file, low_memory=False)
    if single_model and datatype in {"audio", "image", "text"}:
        if "ds_type" not in df.columns:
            raise ValueError(f"missing ds_type column in single-model operator file: {input_file}")
        df = df[df["ds_type"] == datatype].copy()

    df = df.dropna(subset=[LABEL, PRED_LABEL]).copy()
    df = df[df[LABEL] > 0].copy()

    y_true = pd.to_numeric(df[LABEL], errors="coerce")
    y_pred = pd.to_numeric(df[PRED_LABEL], errors="coerce")
    valid_mask = y_true.notna() & y_pred.notna() & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    source = f"{input_file}::ds_type={datatype}" if single_model and datatype != "overall" else input_file
    return build_metric_dict(datatype, source, y_true, y_pred)


def load_operator_metrics_from_existing_model_comparison(datatype, log_target, extra_suffix):
    input_file = resolve_existing_model_comparison_file(datatype, log_target, extra_suffix)
    pred_col = get_pred_column_for_existing_comparison(datatype, extra_suffix)
    if input_file is None or pred_col is None or not os.path.exists(input_file):
        raise FileNotFoundError(
            f"existing-model comparison file not found for datatype={datatype}, suffix={extra_suffix}"
        )

    df = pd.read_csv(input_file, low_memory=False)
    if LABEL not in df.columns or pred_col not in df.columns:
        raise ValueError(
            f"comparison file missing required columns: {input_file}, pred_col={pred_col}"
        )

    df = df.dropna(subset=[LABEL, pred_col]).copy()
    df = df[df[LABEL] > 0].copy()
    if datatype in {"audio", "image", "text"} and "ds_type" in df.columns:
        df = df[df["ds_type"] == datatype].copy()

    y_true = pd.to_numeric(df[LABEL], errors="coerce")
    y_pred = pd.to_numeric(df[pred_col], errors="coerce")
    valid_mask = y_true.notna() & y_pred.notna() & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    return build_metric_dict(datatype, f"{input_file}::{pred_col}", y_true, y_pred)


def load_operator_metrics(datatype, log_target, extra_suffix, single_model=False):
    if single_model:
        return load_operator_metrics_from_standard_file(datatype, log_target, single_model=True)

    comparison_file = resolve_existing_model_comparison_file(datatype, log_target, extra_suffix)
    pred_col = get_pred_column_for_existing_comparison(datatype, extra_suffix)
    if comparison_file and pred_col:
        return load_operator_metrics_from_existing_model_comparison(
            datatype,
            log_target,
            extra_suffix,
        )

    return load_operator_metrics_from_standard_file(datatype, log_target)


def build_pipeline_metrics(log_target, extra_suffix):
    pipeline_prediction_file = build_pipeline_prediction_file(log_target, extra_suffix)
    if not os.path.exists(pipeline_prediction_file):
        raise FileNotFoundError(f"pipeline prediction file not found: {pipeline_prediction_file}")

    df = pd.read_csv(pipeline_prediction_file, low_memory=False)
    required_cols = ["pipeline_name", "ds_type", LABEL, PRED_LABEL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"pipeline prediction file missing columns: {missing_cols}")

    df = df.dropna(subset=["pipeline_name", "ds_type", LABEL, PRED_LABEL]).copy()
    df = df[df[LABEL] > 0].copy()

    grouped = (
        df.groupby(["pipeline_name", "ds_type"], sort=True, dropna=False)
        .agg(
            total_time=(LABEL, "sum"),
            total_pred_time=(PRED_LABEL, "sum"),
        )
        .reset_index()
    )
    grouped["absolute_error"] = (grouped["total_time"] - grouped["total_pred_time"]).abs()
    grouped["relative_error"] = grouped["absolute_error"] / grouped["total_time"] * 100
    grouped["accuracy"] = 100 - grouped["relative_error"]

    metrics = {}

    def summarize(group_df, datatype):
        metrics[datatype] = {
            "pipeline_count": int(len(group_df)),
            "pipeline_accuracy_mean": float(group_df["accuracy"].mean()),
            "pipeline_accuracy_median": float(group_df["accuracy"].median()),
        }

    summarize(grouped, "overall")
    for datatype in ["audio", "image", "text"]:
        summarize(grouped[grouped["ds_type"] == datatype].copy(), datatype)

    return grouped, metrics, pipeline_prediction_file


def build_final_summary(log_target, extra_suffix, single_model=False, operator_from_pipeline_file=False):
    operator_rows = {}
    for datatype in ["overall", "audio", "image", "text"]:
        if operator_from_pipeline_file:
            operator_rows[datatype] = load_operator_metrics_from_pipeline_prediction(
                datatype,
                log_target,
                extra_suffix,
            )
        else:
            operator_rows[datatype] = load_operator_metrics(
                datatype,
                log_target,
                extra_suffix,
                single_model=single_model,
            )

    _, pipeline_metrics, pipeline_prediction_file = build_pipeline_metrics(log_target, extra_suffix)

    rows = []
    for datatype in ["overall", "audio", "image", "text"]:
        rows.append(
            {
                "datatype": datatype,
                "data_range": DISPLAY_NAME[datatype],
                "sample_count": operator_rows[datatype]["sample_count"],
                "pipeline_count": pipeline_metrics[datatype]["pipeline_count"],
                "rmse": operator_rows[datatype]["rmse"],
                "mae": operator_rows[datatype]["mae"],
                "mape": operator_rows[datatype]["mape"],
                "accuracy": operator_rows[datatype]["accuracy"],
                "overall_accuracy_pipeline_mean": pipeline_metrics[datatype]["pipeline_accuracy_mean"],
                "overall_accuracy_pipeline_median": pipeline_metrics[datatype]["pipeline_accuracy_median"],
                "operator_file": operator_rows[datatype]["operator_file"],
                "pipeline_file": pipeline_prediction_file,
            }
        )

    return pd.DataFrame(rows)


def print_table(summary_df):
    print("\n" + "=" * 110)
    print("最终结果表")
    print("=" * 110)
    print(
        "{:<20} {:>10} {:>10} {:>12} {:>12} {:>10} {:>12} {:>14}".format(
            "数据范围",
            "算子样本",
            "pipeline数",
            "RMSE (秒)",
            "MAE (秒)",
            "MAPE",
            "准确率",
            "总体准确率",
        )
    )
    print("-" * 110)

    for _, row in summary_df.iterrows():
        print(
            "{:<20} {:>10} {:>10} {:>12.2f} {:>12.2f} {:>9.2f}% {:>11.2f}% {:>13.2f}%".format(
                row["data_range"],
                int(row["sample_count"]),
                int(row["pipeline_count"]),
                row["rmse"],
                row["mae"],
                row["mape"],
                row["accuracy"],
                row["overall_accuracy_pipeline_mean"],
            )
        )


def write_report(summary_df, report_md, extra_suffix):
    mode_text = extra_suffix if extra_suffix else "(default)"

    with open(report_md, "w", encoding="utf-8") as file_obj:
        file_obj.write("# 总体准确率汇总报告\n\n")
        file_obj.write("## 结果标识\n\n")
        file_obj.write(f"- 当前结果后缀：`{mode_text}`\n\n")
        file_obj.write("## 指标口径\n\n")
        file_obj.write("- `准确率 (算子级别)`：默认来自 `prediction_cost.py` 生成的算子级预测文件；若指定了特殊 suffix（如 `overall_only` 或 `audio_image_overall_text_only`），则按 suffix 从固定测试集对比文件中读取对应预测列。\n")
        file_obj.write("- `总体准确率 (pipeline级别)`：来自 `predict_full_dataset_with_existing_models.py` 生成的全量算子预测文件，按 `pipeline_name` 汇总后取 pipeline 准确率均值。\n\n")
        file_obj.write("## 最终结果表\n\n")
        file_obj.write("| 数据范围 | 算子样本数 | pipeline数 | RMSE (秒) | MAE (秒) | MAPE | 准确率 (算子级别) | 总体准确率 (pipeline级别) |\n")
        file_obj.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")

        for _, row in summary_df.iterrows():
            file_obj.write(
                "| {0} | {1} | {2} | {3:.2f} | {4:.2f} | {5:.2f}% | {6:.2f}% | {7:.2f}% |\n".format(
                    row["data_range"],
                    int(row["sample_count"]),
                    int(row["pipeline_count"]),
                    row["rmse"],
                    row["mae"],
                    row["mape"],
                    row["accuracy"],
                    row["overall_accuracy_pipeline_mean"],
                )
            )

        file_obj.write("\n## 数据文件来源\n\n")
        for _, row in summary_df.iterrows():
            file_obj.write(f"- `{row['data_range']}` 算子级文件：`{row['operator_file']}`\n")
        file_obj.write(f"- pipeline级文件：`{summary_df['pipeline_file'].iloc[0]}`\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute final operator/pipeline accuracy summary."
    )
    parser.add_argument(
        "--log_target",
        action="store_true",
        help="Read/write *_log files generated from log-target cost prediction runs.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Extra suffix appended to pipeline-related input/output filenames, such as overall_only or audio_image_overall_text_only.",
    )
    parser.add_argument(
        "--single_model",
        action="store_true",
        help=(
            "Use one overall cost model's operator prediction file for overall/audio/image/text "
            "metrics, then filter by ds_type instead of requiring per-modality operator files."
        ),
    )
    parser.add_argument(
        "--operator_from_pipeline_file",
        action="store_true",
        help=(
            "Compute operator-level metrics from the same full pipeline prediction file "
            "used for pipeline-level metrics. Useful for external result batches such as 20260408."
        ),
    )
    args = parser.parse_args()
    ensure_output_dirs()

    effective_suffix = args.suffix
    if args.single_model and not effective_suffix:
        effective_suffix = MODE_OVERALL_ONLY

    summary_df = build_final_summary(
        args.log_target,
        effective_suffix,
        args.single_model,
        operator_from_pipeline_file=args.operator_from_pipeline_file,
    )
    summary_csv = build_summary_csv(args.log_target, effective_suffix)
    report_md = build_report_md(args.log_target, effective_suffix)

    summary_df.to_csv(summary_csv, index=False, float_format="%.4f")
    print(f"saved summary csv: {summary_csv}")
    print_table(summary_df)
    write_report(summary_df, report_md, effective_suffix)
    print(f"saved report: {report_md}")


if __name__ == "__main__":
    main()
