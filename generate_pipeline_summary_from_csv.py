import argparse
import os

import numpy as np
import pandas as pd

from project_paths import (
    DATA_DIR,
    PREDICTIONS_DIR,
    SUMMARIES_DIR,
    ensure_output_dirs,
    legacy_output_path,
    resolve_legacy_aware_path,
)

PIPELINE_ID_COLS = [
    "pipeline_name",
    "pipeline_base_name",
    "pipeline_scale_token",
    "operator_index",
    "operator_name",
]

MODE_AUDIO_IMAGE_OVERALL_TEXT_ONLY = "audio_image_overall_text_only"


def normalize_extra_suffix(extra_suffix):
    if not extra_suffix:
        return ""
    return extra_suffix if extra_suffix.startswith("_") else f"_{extra_suffix}"


def build_suffix(log_target, extra_suffix=""):
    return ("_log" if log_target else "") + normalize_extra_suffix(extra_suffix)


def first_existing(paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def pick_actual_data_file(log_target):
    log_suffix = build_suffix(log_target)
    candidates = []
    if log_target:
        filename = f"dataset_header_for_cost_estimation_with_pipeline{log_suffix}.csv"
        candidates.extend(
            [
                str(DATA_DIR / filename),
                legacy_output_path(filename),
            ]
        )
    candidates.extend(
        [
            str(DATA_DIR / "dataset_header_for_cost_estimation_with_pipeline.csv"),
            legacy_output_path("dataset_header_for_cost_estimation_with_pipeline.csv"),
        ]
    )
    if log_target:
        filename = f"dataset_header_for_cost_estimation{log_suffix}.csv"
        candidates.extend([str(DATA_DIR / filename), legacy_output_path(filename)])
    candidates.extend(
        [
            str(DATA_DIR / "dataset_header_for_cost_estimation.csv"),
            legacy_output_path("dataset_header_for_cost_estimation.csv"),
        ]
    )
    return first_existing(candidates)


def pick_predicted_data_file(log_target, extra_suffix):
    suffix = build_suffix(log_target, extra_suffix)
    filename = f"test_set_predictions_cost_full_with_pipeline{suffix}.csv"
    full_with_pipeline = first_existing(
        [
            str(PREDICTIONS_DIR / filename),
            legacy_output_path(filename),
        ]
    )
    if full_with_pipeline:
        return full_with_pipeline

    if extra_suffix:
        return None

    filename = f"test_set_predictions_cost{build_suffix(log_target)}.csv"
    return first_existing([str(PREDICTIONS_DIR / filename), legacy_output_path(filename)])


def build_output_file(log_target, extra_suffix):
    suffix = build_suffix(log_target, extra_suffix)
    return str(SUMMARIES_DIR / f"pipeline_performance_from_csv{suffix}.csv")


def load_and_clean_data(file_path, is_predicted=False):
    if not file_path or not os.path.exists(file_path):
        print(f"file not found: {file_path}")
        return pd.DataFrame()

    print(f"loading: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)

    required_cols = ["op_process_time"]
    if is_predicted:
        required_cols.append("op_process_time_pred")

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"missing required columns: {missing_cols}")
        return pd.DataFrame()

    initial_count = len(df)
    if is_predicted:
        df = df.dropna(subset=["op_process_time", "op_process_time_pred"])
    else:
        df = df.dropna(subset=["op_process_time"])

    df = df[df["op_process_time"] > 0].copy()
    print(f"  rows: {initial_count} -> {len(df)}")
    return df


def extract_operator_type(row):
    if "operator_name" in row.index and pd.notna(row["operator_name"]):
        return str(row["operator_name"])

    for col in row.index:
        if not col.startswith("op_type_"):
            continue
        val = row[col]
        if pd.notna(val) and str(val).strip() != "":
            return col.replace("op_type_", "")
    return "unknown"


def attach_pipeline_columns(predicted_df, actual_df):
    if predicted_df.empty or "pipeline_name" in predicted_df.columns:
        return predicted_df

    if "pipeline_name" not in actual_df.columns:
        print("pipeline_name not found in actual dataset, cannot enrich predicted rows.")
        return predicted_df

    common_cols = sorted(
        [
            col
            for col in predicted_df.columns
            if col in actual_df.columns and col not in PIPELINE_ID_COLS
        ]
    )
    if not common_cols:
        print("no common columns available to enrich predicted rows.")
        return predicted_df

    mapping_cols = common_cols + [col for col in PIPELINE_ID_COLS if col in actual_df.columns]
    mapping_df = actual_df[mapping_cols].drop_duplicates().copy()

    ambiguous_mask = mapping_df.duplicated(subset=common_cols, keep=False)
    ambiguous_count = int(ambiguous_mask.sum())
    if ambiguous_count > 0:
        print(
            "warning: {0} actual rows have ambiguous keys, they will not be used for pipeline enrichment.".format(
                ambiguous_count
            )
        )
    mapping_df = mapping_df[~ambiguous_mask].copy()

    enriched_df = predicted_df.merge(mapping_df, on=common_cols, how="left")
    matched_count = int(enriched_df["pipeline_name"].notna().sum())
    print(f"pipeline enrichment for predicted rows: matched {matched_count}/{len(enriched_df)}")
    return enriched_df


def resolve_group_cols(actual_df):
    if "pipeline_name" in actual_df.columns:
        return ["pipeline_name"]
    return ["ds_input_count", "ds_type"]


def summarize_actual_groups(df, group_cols):
    rows = []
    for group_key, group_df in df.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_dict = dict(zip(group_cols, group_key))
        sorted_group = (
            group_df.sort_values("operator_index")
            if "operator_index" in group_df.columns
            else group_df
        )
        rows.append(
            {
                **group_dict,
                "pipeline_base_name": sorted_group["pipeline_base_name"].iloc[0]
                if "pipeline_base_name" in sorted_group.columns
                else np.nan,
                "pipeline_scale_token": sorted_group["pipeline_scale_token"].iloc[0]
                if "pipeline_scale_token" in sorted_group.columns
                else np.nan,
                "ds_type": sorted_group["ds_type"].iloc[0]
                if "ds_type" in sorted_group.columns
                else np.nan,
                "num_operators": len(sorted_group),
                "initial_ds_input_count": sorted_group["ds_input_count"].iloc[0]
                if "ds_input_count" in sorted_group.columns
                else np.nan,
                "final_ds_output_count": sorted_group["ds_output_count"].iloc[-1]
                if "ds_output_count" in sorted_group.columns
                else np.nan,
                "total_time": sorted_group["op_process_time"].sum(),
                "min_time": sorted_group["op_process_time"].min(),
                "max_time": sorted_group["op_process_time"].max(),
                "mean_time": sorted_group["op_process_time"].mean(),
                "operator_types": " | ".join(
                    sorted_group.apply(extract_operator_type, axis=1).tolist()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_predicted_groups(df, group_cols):
    if df.empty:
        return pd.DataFrame()

    rows = []
    for group_key, group_df in df.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_dict = dict(zip(group_cols, group_key))
        rows.append(
            {
                **group_dict,
                "predicted_num_operators": len(group_df),
                "total_pred_time": group_df["op_process_time_pred"].sum(),
                "predicted_subset_true_time": group_df["op_process_time"].sum(),
                "min_pred_time": group_df["op_process_time_pred"].min(),
                "max_pred_time": group_df["op_process_time_pred"].max(),
                "mean_pred_time": group_df["op_process_time_pred"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_pipeline_summary(actual_df, predicted_df, group_cols):
    actual_summary = summarize_actual_groups(actual_df, group_cols)
    predicted_summary = summarize_predicted_groups(predicted_df, group_cols)

    result_df = actual_summary.merge(predicted_summary, on=group_cols, how="left")
    result_df["predicted_num_operators"] = result_df["predicted_num_operators"].fillna(0).astype(int)
    result_df["has_prediction"] = result_df["predicted_num_operators"] > 0
    result_df["coverage_ratio"] = result_df["predicted_num_operators"] / result_df["num_operators"]
    result_df["has_complete_prediction"] = result_df["predicted_num_operators"] == result_df["num_operators"]

    result_df["absolute_error"] = np.nan
    result_df["relative_error"] = np.nan
    result_df["accuracy"] = np.nan

    complete_mask = result_df["has_complete_prediction"] & result_df["total_time"].gt(0)
    result_df.loc[complete_mask, "absolute_error"] = (
        result_df.loc[complete_mask, "total_time"]
        - result_df.loc[complete_mask, "total_pred_time"]
    ).abs()
    result_df.loc[complete_mask, "relative_error"] = (
        result_df.loc[complete_mask, "absolute_error"]
        / result_df.loc[complete_mask, "total_time"]
        * 100
    )
    result_df.loc[complete_mask, "accuracy"] = 100 - result_df.loc[complete_mask, "relative_error"]
    return result_df


def print_summary(result_df, group_cols):
    print("\n" + "=" * 80)
    print("Pipeline Summary")
    print("=" * 80)
    print(f"group columns: {group_cols}")
    print(f"total groups: {len(result_df)}")
    print(f"groups with any predicted operators: {int(result_df['has_prediction'].sum())}")
    print(
        "groups with complete predicted operators: {0}".format(
            int(result_df["has_complete_prediction"].sum())
        )
    )

    complete_df = result_df[result_df["has_complete_prediction"]].copy()
    if complete_df.empty:
        print("no complete pipeline predictions found.")
        return

    print(f"mean accuracy (complete only): {complete_df['accuracy'].mean():.2f}%")

    if "ds_type" in complete_df.columns:
        for ds_type, group in complete_df.groupby("ds_type"):
            print(
                "{0}: complete={1}, mean={2:.2f}%".format(
                    ds_type,
                    len(group),
                    group["accuracy"].mean(),
                )
            )


def main():
    parser = argparse.ArgumentParser(description="Generate pipeline summary from CSV.")
    parser.add_argument(
        "--log_target",
        action="store_true",
        help="Read/write *_log files generated from log-target cost prediction runs.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Extra suffix appended to predicted/output filenames, such as overall_only or audio_image_overall_text_only.",
    )
    parser.add_argument(
        "--actual_data_path",
        default="",
        help="Optional actual pipeline dataset CSV. Defaults to the current standard output files.",
    )
    parser.add_argument(
        "--predicted_data_path",
        default="",
        help="Optional predicted pipeline CSV. Defaults to files resolved from --log_target and --suffix.",
    )
    args = parser.parse_args()
    ensure_output_dirs()

    print("=" * 80)
    print("Generate pipeline summary from CSV")
    print("=" * 80)

    actual_file = (
        resolve_legacy_aware_path(args.actual_data_path, DATA_DIR)
        if args.actual_data_path
        else pick_actual_data_file(args.log_target)
    )
    actual_df = load_and_clean_data(actual_file, is_predicted=False)
    if actual_df.empty:
        print("failed to load actual data.")
        return

    predicted_file = (
        resolve_legacy_aware_path(args.predicted_data_path, PREDICTIONS_DIR)
        if args.predicted_data_path
        else pick_predicted_data_file(args.log_target, args.suffix)
    )
    predicted_df = load_and_clean_data(predicted_file, is_predicted=True)
    if predicted_df.empty:
        print("failed to load predicted data.")
        return

    predicted_df = attach_pipeline_columns(predicted_df, actual_df)
    group_cols = resolve_group_cols(actual_df)
    result_df = build_pipeline_summary(actual_df, predicted_df, group_cols)

    sort_cols = [
        col for col in ["ds_type", "pipeline_base_name", "pipeline_name", "total_time"]
        if col in result_df.columns
    ]
    ascending = [True] * len(sort_cols)
    if "total_time" in sort_cols:
        ascending[sort_cols.index("total_time")] = False
    if sort_cols:
        result_df = result_df.sort_values(sort_cols, ascending=ascending)

    output_file = build_output_file(args.log_target, args.suffix)
    result_df.to_csv(output_file, index=False, float_format="%.4f")
    print(f"\nsaved: {output_file}")
    print_summary(result_df, group_cols)


if __name__ == "__main__":
    main()
