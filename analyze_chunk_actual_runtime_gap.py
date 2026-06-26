import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_chunk_runtime_scaling import (
    build_run_rows,
    choose_full_reference,
)
from project_paths import REPORTS_DIR, SUMMARIES_DIR, ensure_output_dirs


DEFAULT_PIPELINE_REGEX = (
    "^(audio_pipeline_1781071552228|audio_pipeline_1781071552263|audio_pipeline_1781071552218|"
    "image_pipeline_1781071560263|image_pipeline_1781071560242|image_pipeline_1781071560252|"
    "text_pipeline_1781071569701|text_pipeline_1781071569367|text_pipeline_1781071569512)$"
)


def normalize_suffix(suffix):
    if not suffix:
        return ""
    return suffix if suffix.startswith("_") else f"_{suffix}"


def format_number(value, digits=2):
    if pd.isna(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def safe_divide(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return numerator / denominator.replace(0, np.nan)


def aggregate_full(full_rows):
    return (
        full_rows.groupby(
            [
                "ds_type",
                "pipeline_base_name",
                "pipeline_category",
                "operator_index",
                "operator_name",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            full_pipeline_name=("pipeline_name", "first"),
            full_scale_token=("pipeline_scale_token", "first"),
            full_scale_value=("scale_value", "first"),
            full_actual_time=("op_process_time", "sum"),
            full_input_count=("ds_input_count", "sum"),
            full_output_count=("ds_output_count", "sum"),
        )
        .copy()
    )


def aggregate_chunks(chunk_rows):
    return (
        chunk_rows.groupby(
            [
                "ds_type",
                "pipeline_base_name",
                "pipeline_category",
                "chunk_size",
                "operator_index",
                "operator_name",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            chunk_parts=("pipeline_name", "nunique"),
            chunk_actual_time_sum=("op_process_time", "sum"),
            chunk_actual_time_mean=("op_process_time", "mean"),
            chunk_actual_time_median=("op_process_time", "median"),
            chunk_actual_time_min=("op_process_time", "min"),
            chunk_actual_time_max=("op_process_time", "max"),
            chunk_input_count_sum=("ds_input_count", "sum"),
            chunk_output_count_sum=("ds_output_count", "sum"),
            chunk_input_count_mean=("ds_input_count", "mean"),
            chunk_output_count_mean=("ds_output_count", "mean"),
        )
        .copy()
    )


def add_gap_features(merged):
    result = merged.copy()
    result["chunk_minus_full_time"] = result["chunk_actual_time_sum"] - result["full_actual_time"]
    result["chunk_sum_to_full_ratio"] = safe_divide(
        result["chunk_actual_time_sum"],
        result["full_actual_time"],
    )
    result["chunk_sum_relative_error"] = safe_divide(
        result["chunk_minus_full_time"].abs(),
        result["full_actual_time"],
    )
    result["chunk_sum_accuracy"] = 100 - result["chunk_sum_relative_error"] * 100
    result["chunk_input_coverage_ratio"] = safe_divide(
        result["chunk_input_count_sum"],
        result["full_input_count"],
    )
    result["chunk_output_coverage_ratio"] = safe_divide(
        result["chunk_output_count_sum"],
        result["full_output_count"],
    )
    result["avg_chunk_time_to_full_ratio"] = safe_divide(
        result["chunk_actual_time_mean"],
        result["full_actual_time"],
    )
    result["estimated_repeated_overhead"] = np.where(
        result["chunk_actual_time_sum"] > result["full_actual_time"],
        result["chunk_actual_time_sum"] - result["full_actual_time"],
        0.0,
    )
    result["estimated_repeated_overhead_per_chunk"] = safe_divide(
        result["estimated_repeated_overhead"],
        result["chunk_parts"],
    )
    return result


def infer_reason(row):
    ratio = row.get("chunk_sum_to_full_ratio", np.nan)
    input_cov = row.get("chunk_input_coverage_ratio", np.nan)
    output_cov = row.get("chunk_output_coverage_ratio", np.nan)
    chunk_parts = row.get("chunk_parts", np.nan)
    chunk_size = row.get("chunk_size", np.nan)
    op_name = str(row.get("operator_name", ""))

    reasons = []
    if np.isfinite(ratio) and ratio > 5:
        reasons.append("chunk_sum远大于full，重复固定开销/初始化开销很明显")
    elif np.isfinite(ratio) and ratio > 1.5:
        reasons.append("chunk_sum大于full，存在重复调度/I/O/初始化开销")
    elif np.isfinite(ratio) and ratio < 0.7:
        reasons.append("chunk_sum小于full，可能存在full端额外开销、缓存差异或chunk覆盖不足")
    else:
        reasons.append("chunk_sum与full相对接近")

    if np.isfinite(chunk_size) and chunk_size <= 200:
        reasons.append("chunk_size较小，固定开销占比更高")

    if np.isfinite(chunk_parts) and chunk_parts >= 50:
        reasons.append("chunk数量多，固定开销被重复计入多次")

    if np.isfinite(input_cov) and (input_cov < 0.9 or input_cov > 1.1):
        reasons.append(f"输入基数覆盖比例异常({input_cov:.2f})")
    if np.isfinite(output_cov) and row.get("full_output_count", np.nan) > 0 and (
        output_cov < 0.9 or output_cov > 1.1
    ):
        reasons.append(f"输出基数覆盖比例异常({output_cov:.2f})")

    heavy_keywords = [
        "tagging",
        "detection",
        "segment",
        "remove_background",
        "ffmpeg",
        "nmf",
        "perplexity",
    ]
    if any(keyword in op_name for keyword in heavy_keywords):
        reasons.append("可能包含模型加载/外部库初始化/重型计算开销")

    return "；".join(reasons)


def aggregate_summary(merged, group_cols, level):
    rows = []
    for keys, group_df in merged.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        full_sum = group_df["full_actual_time"].sum()
        chunk_sum = group_df["chunk_actual_time_sum"].sum()
        rel_error = abs(chunk_sum - full_sum) / full_sum if full_sum > 0 else np.nan
        row = {
            "level": level,
            "operator_rows": int(len(group_df)),
            "full_actual_time_sum": float(full_sum),
            "chunk_actual_time_sum": float(chunk_sum),
            "chunk_sum_to_full_ratio": float(chunk_sum / full_sum) if full_sum > 0 else np.nan,
            "chunk_sum_mape_like": float(rel_error * 100) if np.isfinite(rel_error) else np.nan,
            "chunk_sum_accuracy": float(100 - rel_error * 100) if np.isfinite(rel_error) else np.nan,
            "mean_operator_relative_error": float(group_df["chunk_sum_relative_error"].mean() * 100),
            "median_operator_relative_error": float(group_df["chunk_sum_relative_error"].median() * 100),
            "estimated_repeated_overhead_sum": float(group_df["estimated_repeated_overhead"].sum()),
        }
        for col, value in zip(group_cols, keys):
            row[col] = value
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown_table(file_obj, df, columns, limit=None):
    if df.empty:
        file_obj.write("No rows.\n\n")
        return
    table_df = df.loc[:, [col for col in columns if col in df.columns]].copy()
    if limit:
        table_df = table_df.head(limit)
    for col in table_df.columns:
        if pd.api.types.is_float_dtype(table_df[col]):
            table_df[col] = table_df[col].map(format_number)
    file_obj.write("| " + " | ".join(table_df.columns) + " |\n")
    file_obj.write("| " + " | ".join(["---"] * len(table_df.columns)) + " |\n")
    for _, row in table_df.iterrows():
        values = ["" if pd.isna(value) else str(value) for value in row.tolist()]
        file_obj.write("| " + " | ".join(values) + " |\n")
    file_obj.write("\n")


def write_report(report_path, overview, summary_df, operator_df, worst_df):
    with open(report_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("# Chunk真实耗时求和与全量真实耗时差异分析\n\n")
        file_obj.write("本报告只使用真实执行耗时，不涉及模型预测。核心问题是验证：\n\n")
        file_obj.write("`sum(chunk_actual_op_time)` 是否等于 `full_actual_op_time`。\n\n")
        file_obj.write("如果两者不相等，说明把数据切成很多小块分别执行再求和，本身就不是完整数据一次执行的等价替代；原因通常来自重复的固定开销、I/O/调度开销、模型或外部库初始化开销，以及小块规模导致的资源利用率变化。\n\n")

        file_obj.write("## 1. 实验概况\n\n")
        for key, value in overview.items():
            file_obj.write(f"- {key}: {value}\n")
        file_obj.write("\n")

        file_obj.write("## 2. 按模态和chunk size汇总\n\n")
        mod_chunk = summary_df[summary_df["level"] == "by_ds_type_chunk_size"].copy()
        write_markdown_table(
            file_obj,
            mod_chunk,
            [
                "ds_type",
                "chunk_size",
                "operator_rows",
                "full_actual_time_sum",
                "chunk_actual_time_sum",
                "chunk_sum_to_full_ratio",
                "chunk_sum_mape_like",
                "chunk_sum_accuracy",
                "estimated_repeated_overhead_sum",
            ],
        )

        file_obj.write("## 3. 按pipeline类型汇总\n\n")
        category = summary_df[summary_df["level"] == "by_pipeline_category_chunk_size"].copy()
        write_markdown_table(
            file_obj,
            category,
            [
                "ds_type",
                "pipeline_category",
                "chunk_size",
                "operator_rows",
                "chunk_sum_to_full_ratio",
                "chunk_sum_mape_like",
                "chunk_sum_accuracy",
                "estimated_repeated_overhead_sum",
            ],
        )

        file_obj.write("## 4. 差异最大的算子级记录\n\n")
        write_markdown_table(
            file_obj,
            worst_df,
            [
                "ds_type",
                "pipeline_base_name",
                "pipeline_category",
                "chunk_size",
                "operator_index",
                "operator_name",
                "chunk_parts",
                "full_actual_time",
                "chunk_actual_time_sum",
                "chunk_actual_time_mean",
                "chunk_sum_to_full_ratio",
                "chunk_sum_accuracy",
                "estimated_repeated_overhead",
                "reason",
            ],
            limit=40,
        )

        file_obj.write("## 5. 结论\n\n")
        file_obj.write("- 如果 `chunk_sum_to_full_ratio` 显著大于 1，说明小块真实执行求和已经比全量一次执行慢很多，模型预测再准也无法让“直接求和”成为等价估计。\n")
        file_obj.write("- 如果 chunk size 增大后该比例下降，说明主要问题来自小块重复固定开销，而不是 pipeline 配置本身。\n")
        file_obj.write("- 后续如果继续使用 chunk 思路，应考虑建模 `fixed_overhead + variable_cost_per_record * record_count`，而不是直接把所有 chunk 耗时简单相加。\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze why actual chunk runtime sums differ from full-scale actual runtime."
    )
    parser.add_argument("--chunk_root", default="./collect_data/runs_chunks")
    parser.add_argument("--full_root", default="./collect_data/result_20260611")
    parser.add_argument("--ds_type", choices=["audio", "image", "text"], default=None)
    parser.add_argument(
        "--base_name_regex",
        default=DEFAULT_PIPELINE_REGEX,
        help="Regex of pipeline_base_name to analyze. Defaults to the 9 representative pipelines.",
    )
    parser.add_argument("--output_suffix", default="20260611_representative")
    args = parser.parse_args()

    ensure_output_dirs()
    chunk_rows, chunk_skipped, chunk_discovered = build_run_rows(
        args.chunk_root,
        expected_kind="chunk",
        ds_type_filter=args.ds_type,
        base_name_regex=args.base_name_regex,
    )
    full_rows, full_skipped, full_discovered = build_run_rows(
        args.full_root,
        expected_kind="full",
        ds_type_filter=args.ds_type,
        base_name_regex=args.base_name_regex,
    )

    if chunk_rows.empty:
        raise ValueError("no chunk rows parsed")
    if full_rows.empty:
        raise ValueError("no full rows parsed")

    chunk_bases = set(chunk_rows["pipeline_base_name"].dropna().unique().tolist())
    full_rows = full_rows[full_rows["pipeline_base_name"].isin(chunk_bases)].copy()
    full_ref_rows = choose_full_reference(full_rows)
    if full_ref_rows.empty:
        raise ValueError("no matched full reference rows")

    full_agg = aggregate_full(full_ref_rows)
    chunk_agg = aggregate_chunks(chunk_rows)
    merged = chunk_agg.merge(
        full_agg,
        on=["ds_type", "pipeline_base_name", "pipeline_category", "operator_index", "operator_name"],
        how="inner",
    )
    merged = add_gap_features(merged)
    merged["reason"] = merged.apply(infer_reason, axis=1)

    summary_frames = [
        aggregate_summary(merged, ["ds_type", "chunk_size"], "by_ds_type_chunk_size"),
        aggregate_summary(
            merged,
            ["ds_type", "pipeline_category", "chunk_size"],
            "by_pipeline_category_chunk_size",
        ),
        aggregate_summary(
            merged,
            ["ds_type", "pipeline_base_name", "pipeline_category", "chunk_size"],
            "by_pipeline_chunk_size",
        ),
    ]
    summary_df = pd.concat(summary_frames, ignore_index=True)
    worst_df = merged.sort_values(
        ["chunk_sum_relative_error", "chunk_actual_time_sum"],
        ascending=[False, False],
    ).copy()

    suffix = normalize_suffix(args.output_suffix)
    operator_path = SUMMARIES_DIR / f"chunk_actual_runtime_gap_operator{suffix}.csv"
    summary_path = SUMMARIES_DIR / f"chunk_actual_runtime_gap_summary{suffix}.csv"
    skipped_path = SUMMARIES_DIR / f"chunk_actual_runtime_gap_skipped{suffix}.csv"
    report_path = REPORTS_DIR / f"chunk_actual_runtime_gap_report{suffix}.md"

    merged.to_csv(operator_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pd.concat(
        [
            chunk_skipped.assign(source="chunk"),
            full_skipped.assign(source="full"),
        ],
        ignore_index=True,
    ).to_csv(skipped_path, index=False)

    overview = {
        "chunk root": args.chunk_root,
        "full root": args.full_root,
        "chunk yaml discovered": chunk_discovered,
        "full yaml discovered": full_discovered,
        "chunk operator rows": len(chunk_rows),
        "full reference operator rows": len(full_ref_rows),
        "matched operator comparison rows": len(merged),
        "pipeline filter": args.base_name_regex or "(all)",
    }
    write_report(report_path, overview, summary_df, merged, worst_df)

    print(f"saved operator comparison: {operator_path}")
    print(f"saved summary csv: {summary_path}")
    print(f"saved skipped csv: {skipped_path}")
    print(f"saved report: {report_path}")

    print("\nactual chunk-sum vs full runtime summary")
    print("=" * 96)
    display = summary_df[summary_df["level"] == "by_ds_type_chunk_size"].copy()
    for _, row in display.iterrows():
        print(
            "{0:<6} chunk={1:<5} ratio={2:>8} error={3:>8}% accuracy={4:>8}% overhead={5:>10}s".format(
                row["ds_type"],
                int(row["chunk_size"]),
                format_number(row["chunk_sum_to_full_ratio"]),
                format_number(row["chunk_sum_mape_like"]),
                format_number(row["chunk_sum_accuracy"]),
                format_number(row["estimated_repeated_overhead_sum"]),
            )
        )


if __name__ == "__main__":
    main()
