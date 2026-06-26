import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from autogluon.tabular import TabularPredictor

from predict_full_dataset_with_existing_models import (
    MODE_AUDIO_IMAGE_OVERALL_TEXT_ONLY,
    MODE_DEFAULT,
    MODE_OVERALL_ONLY,
    get_predictor_features,
    resolve_model_dir,
)
from project_paths import PREDICTIONS_DIR, REPORTS_DIR, SUMMARIES_DIR, ensure_output_dirs

LABEL = "op_process_time"
PRED_LABEL = "op_process_time_pred"
INPUT_COUNT_COL = "ds_input_count"
OUTPUT_COUNT_COL = "ds_output_count"

REMOVE_COLS = {
    "op_audio_add_gaussian_noise_mapper_noise_level",
    "op_audio_ffmpeg_wrapped_mapper_save_dir",
    "op_audio_add_gaussian_noise_mapper_save_dir",
    "op_image_blur_mapper_save_dir",
    "op_image_face_blur_mapper_save_dir",
    "op_image_remove_background_mapper_save_dir",
}

PERCENTILE_KEYS = ["min", "max", "mean", "p5", "p10", "p25", "p50", "p75", "p90", "p95"]

PIPELINE_CATEGORY_MAP = {
    "audio_pipeline_1781071552228": "filter-heavy",
    "audio_pipeline_1781071552263": "mapper-heavy",
    "audio_pipeline_1781071552218": "mixed",
    "image_pipeline_1781071560263": "filter-heavy",
    "image_pipeline_1781071560242": "mapper-heavy",
    "image_pipeline_1781071560252": "mixed",
    "text_pipeline_1781071569701": "filter-heavy",
    "text_pipeline_1781071569367": "mapper-heavy",
    "text_pipeline_1781071569512": "mixed",
}


def normalize_feature_value(value):
    """Keep YAML nested params hashable for AutoGluon category processing."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def infer_pipeline_category(pipeline_base_name):
    return PIPELINE_CATEGORY_MAP.get(str(pipeline_base_name), "unlabeled")


def flatten_chunk_profile_features(ds_type, stats):
    features = {}
    if not isinstance(stats, dict) or stats.get("error"):
        return features

    feature_prefix = f"profile_{ds_type}"
    skip_keys = {
        "source_index",
        "search_roots",
        "scanned_records",
        "missed_files",
        "errored_files",
        "resolved_files",
    }
    for metric_name, metric_stats in stats.items():
        if metric_name in skip_keys or not isinstance(metric_stats, dict):
            continue
        for key in PERCENTILE_KEYS:
            if key in metric_stats and metric_stats[key] is not None:
                features[f"{feature_prefix}_{metric_name}_{key}"] = metric_stats[key]
        if metric_stats.get("n") is not None:
            features[f"{feature_prefix}_{metric_name}_n"] = metric_stats["n"]
    return features


def load_chunk_profile_lookup(profile_paths):
    lookup = {}
    invalid_count = 0
    loaded_files = []
    for profile_path in profile_paths or []:
        if not profile_path:
            continue
        path_obj = Path(profile_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"chunk profile stats file not found: {profile_path}")
        with open(path_obj, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        loaded_files.append(str(path_obj))
        for chunk_info in payload.get("chunks", []):
            ds_type = str(chunk_info.get("ds_type", "")).strip().lower()
            chunk_size = chunk_info.get("chunk_size")
            chunk_part = chunk_info.get("chunk_part")
            if not ds_type or chunk_size is None or chunk_part is None:
                invalid_count += 1
                continue
            features = flatten_chunk_profile_features(ds_type, chunk_info.get("stats", {}))
            if not features:
                invalid_count += 1
                continue
            lookup[(ds_type, int(chunk_size), int(chunk_part))] = features
    return lookup, loaded_files, invalid_count


def apply_chunk_profile_features(df, chunk_profile_lookup):
    if df.empty or not chunk_profile_lookup:
        result_df = df.copy()
        result_df["chunk_profile_matched"] = False
        return result_df, 0, 0

    result_df = df.copy()
    profile_rows = []
    matched = 0
    unmatched_keys = set()
    for _, row in result_df.iterrows():
        if row.get("source_kind") != "chunk":
            profile_rows.append({})
            continue
        try:
            key = (
                str(row.get("ds_type", "")).strip().lower(),
                int(row.get("chunk_size")),
                int(row.get("chunk_part")),
            )
        except Exception:
            key = None
        features = chunk_profile_lookup.get(key) if key else None
        if features:
            matched += 1
            profile_rows.append(features)
        else:
            if key:
                unmatched_keys.add(key)
            profile_rows.append({})

    profile_df = pd.DataFrame(profile_rows, index=result_df.index)
    overlap_cols = [col for col in profile_df.columns if col in result_df.columns]
    for col in overlap_cols:
        result_df[col] = profile_df[col].combine_first(result_df[col])
    new_cols = [col for col in profile_df.columns if col not in result_df.columns]
    if new_cols:
        result_df = pd.concat([result_df, profile_df[new_cols]], axis=1)
    result_df["chunk_profile_matched"] = False
    chunk_mask = result_df["source_kind"] == "chunk"
    result_df.loc[chunk_mask, "chunk_profile_matched"] = [
        bool(
            chunk_profile_lookup.get(
                (
                    str(row["ds_type"]).strip().lower(),
                    int(row["chunk_size"]),
                    int(row["chunk_part"]),
                )
            )
        )
        for _, row in result_df.loc[chunk_mask].iterrows()
    ]
    return result_df.copy(), matched, len(unmatched_keys)


def normalize_extra_suffix(extra_suffix):
    if not extra_suffix:
        return ""
    return extra_suffix if extra_suffix.startswith("_") else f"_{extra_suffix}"


def build_suffix(log_target, extra_suffix=""):
    return ("_log" if log_target else "") + normalize_extra_suffix(extra_suffix)


def infer_ds_type(data_name, operator=None):
    data_name_lower = str(data_name).lower()
    if "text" in data_name_lower:
        return "text"
    if "image" in data_name_lower:
        return "image"
    if "audio" in data_name_lower:
        return "audio"

    if operator:
        op_name = next(iter(operator.keys())).lower()
        if op_name.startswith("text_") or op_name.startswith("document_"):
            return "text"
        if op_name.startswith("image_"):
            return "image"
        if op_name.startswith("audio_"):
            return "audio"

    return "unknown"


def find_yaml_files(base_path):
    for root, _, files in os.walk(base_path):
        for file_name in files:
            if file_name.endswith(".yaml"):
                yield os.path.join(root, file_name)


def get_dataset_path_from_pipeline_dict(pipeline_dict):
    if "dataset_path" in pipeline_dict and pipeline_dict["dataset_path"]:
        return str(pipeline_dict["dataset_path"])

    dataset_info = pipeline_dict.get("dataset")
    if isinstance(dataset_info, dict):
        configs = dataset_info.get("configs") or []
        if configs and isinstance(configs[0], dict):
            return str(configs[0].get("path", ""))
    return ""


def find_log_file(root):
    log_dir = os.path.join(root, "log")
    if not os.path.isdir(log_dir):
        return ""

    candidate = ""
    for log_root, _, files in os.walk(log_dir):
        for file_name in sorted(files):
            upper_name = file_name.upper()
            if "DEBUG" in upper_name or "ERROR" in upper_name or "WARNING" in upper_name:
                continue
            candidate = os.path.join(log_root, file_name)
    return candidate


def find_monitor_file(root):
    monitor_dir = os.path.join(root, "monitor")
    if not os.path.isdir(monitor_dir):
        return ""

    monitor_file = os.path.join(monitor_dir, "monitor.json")
    if os.path.exists(monitor_file):
        return monitor_file

    root_name = os.path.basename(os.path.normpath(root))
    alt_monitor_file = os.path.join(monitor_dir, f"{root_name}_monitor.json")
    if os.path.exists(alt_monitor_file):
        return alt_monitor_file

    monitor_candidates = sorted(
        file_name for file_name in os.listdir(monitor_dir) if file_name.endswith("_monitor.json")
    )
    if monitor_candidates:
        return os.path.join(monitor_dir, monitor_candidates[0])
    return ""


def split_pipeline_name(data_name):
    data_name = str(data_name)
    chunk_match = re.match(r"^(.*)_chunk(\d+)_part(\d+)$", data_name)
    if chunk_match:
        return (
            chunk_match.group(1),
            f"chunk{chunk_match.group(2)}_part{chunk_match.group(3)}",
            int(chunk_match.group(2)),
            int(chunk_match.group(3)),
            "chunk",
        )

    per_chunk_match = re.match(r"^(.*)_per(\d+)_chunk_(\d+)$", data_name)
    if per_chunk_match:
        return (
            per_chunk_match.group(1),
            f"chunk{per_chunk_match.group(2)}_part{per_chunk_match.group(3)}",
            int(per_chunk_match.group(2)),
            int(per_chunk_match.group(3)),
            "chunk",
        )

    match = re.match(r"^(.*)_([^_]+)$", data_name)
    if match:
        return match.group(1), match.group(2), np.nan, np.nan, "full"
    return data_name, "", np.nan, np.nan, "full"


def normalize_size_label(size_mb):
    if not np.isfinite(size_mb):
        return np.nan
    if size_mb >= 1024:
        size_g = size_mb / 1024.0
        if float(size_g).is_integer():
            return f"{int(size_g)}G"
        return f"{size_g:g}G"
    if float(size_mb).is_integer():
        return f"{int(size_mb)}MB"
    return f"{size_mb:g}MB"


def extract_scale_features(scale_source_name, ds_type):
    scale_label = None
    ds_scale_mb = np.nan
    ds_scale_records = np.nan

    match = re.search(r"_([^_]+)$", str(scale_source_name))
    if match:
        scale_label = match.group(1)

    if scale_label:
        label_upper = scale_label.upper()
        if ds_type in ("audio", "image"):
            mi_match = re.fullmatch(r"(\d+(?:\.\d+)?)(MIB|MB)", label_upper)
            g_match = re.fullmatch(r"(\d+(?:\.\d+)?)G", label_upper)
            if mi_match:
                ds_scale_mb = float(mi_match.group(1))
                scale_label = normalize_size_label(ds_scale_mb)
            elif g_match:
                ds_scale_mb = float(g_match.group(1)) * 1024.0
                scale_label = normalize_size_label(ds_scale_mb)
        elif ds_type == "text" and re.fullmatch(r"\d+", scale_label):
            ds_scale_records = float(scale_label)

    return {
        "ds_scale_label": scale_label,
        "ds_scale_mb": ds_scale_mb,
        "ds_scale_records": ds_scale_records,
    }


def parse_log(log_file_path):
    ds_input_count = 0
    operator_list = []

    with open(log_file_path, "r", encoding="utf-8") as file_obj:
        for log_line in file_obj:
            if "in the original dataset" in log_line:
                log_match = re.search(
                    r"There are (.*?) sample\(s\) in the original dataset",
                    log_line,
                )
                if log_match:
                    ds_input_count += int(log_match.group(1))
                continue

            if "] OP [" in log_line:
                log_match = re.search(
                    r"OP \[(.*?)\] Done in (.*?)s\. Left (.*?) samples\.",
                    log_line,
                )
                if not log_match:
                    continue
                operator_list.append(
                    {
                        "op_name": log_match.group(1),
                        "op_process_time": float(log_match.group(2)),
                        "ds_output_count": float(log_match.group(3)),
                        "ds_input_count": float(ds_input_count),
                    }
                )
                ds_input_count = float(log_match.group(3))

    return operator_list


def parse_monitor(monitor_file_path):
    operator_env_list = []
    with open(monitor_file_path, "r", encoding="utf-8") as file_obj:
        monitor_info_list = json.load(file_obj)

    for monitor_info in monitor_info_list:
        env_info = monitor_info["resource"][0]
        gpu_util_list = env_info.get("GPU util.", [0.0])
        gpu_used_mem_list = env_info.get("GPU used mem.", [0.0])
        gpu_total_mem_list = env_info.get("GPU total mem.", [1.0])

        operator_env_list.append(
            {
                "env_cpu_count": env_info["CPU count"],
                "env_cpu_util": env_info["CPU util."],
                "env_mem_util": env_info["Mem. util."],
                "env_gpu_util": gpu_util_list[0],
                "env_gpu_mem_util": (
                    gpu_used_mem_list[0] / max(gpu_total_mem_list[0], 1)
                    if gpu_total_mem_list[0] > 0
                    else 0.0
                ),
            }
        )

    return operator_env_list


def build_run_rows(base_path, expected_kind=None, ds_type_filter=None, base_name_regex=""):
    base_name_pattern = re.compile(base_name_regex) if base_name_regex else None
    rows = []
    skipped = []
    discovered = 0

    for yaml_path in find_yaml_files(base_path):
        discovered += 1
        root = os.path.dirname(yaml_path)
        data_name = os.path.splitext(os.path.basename(yaml_path))[0]
        pipeline_base_name, pipeline_scale_token, chunk_size, chunk_part, run_kind = split_pipeline_name(
            data_name
        )
        if expected_kind and run_kind != expected_kind:
            continue
        if base_name_pattern and not base_name_pattern.search(pipeline_base_name):
            continue

        try:
            with open(yaml_path, "r", encoding="utf-8") as file_obj:
                pipeline_dict = yaml.safe_load(file_obj)
        except Exception as exc:
            skipped.append({"pipeline_name": data_name, "reason": f"yaml_parse_failed: {exc}"})
            continue

        log_file_path = find_log_file(root)
        monitor_file_path = find_monitor_file(root)
        if not log_file_path or not os.path.exists(log_file_path):
            skipped.append({"pipeline_name": data_name, "reason": "missing_log"})
            continue
        if not monitor_file_path or not os.path.exists(monitor_file_path):
            skipped.append({"pipeline_name": data_name, "reason": "missing_monitor"})
            continue

        try:
            log_info = parse_log(log_file_path)
            monitor_info = parse_monitor(monitor_file_path)
        except Exception as exc:
            skipped.append({"pipeline_name": data_name, "reason": f"parse_failed: {exc}"})
            continue

        operator_list = pipeline_dict.get("process", [])
        if not operator_list:
            skipped.append({"pipeline_name": data_name, "reason": "empty_operator_list"})
            continue
        if len(log_info) < len(operator_list) or len(monitor_info) < len(operator_list):
            skipped.append(
                {
                    "pipeline_name": data_name,
                    "reason": "incomplete_data ops={0} log={1} monitor={2}".format(
                        len(operator_list),
                        len(log_info),
                        len(monitor_info),
                    ),
                }
            )
            continue

        dataset_path = get_dataset_path_from_pipeline_dict(pipeline_dict)
        scale_source_name = data_name
        if run_kind == "chunk" and dataset_path:
            scale_source_name = os.path.splitext(os.path.basename(dataset_path))[0]
            scale_source_name = re.sub(r"_part\d+$", "", scale_source_name)
        ds_type = infer_ds_type(pipeline_base_name, operator_list[0])
        if ds_type_filter and ds_type != ds_type_filter:
            continue

        scale_features = extract_scale_features(scale_source_name, ds_type)
        source_scale_value = scale_features["ds_scale_mb"]
        if ds_type == "text":
            source_scale_value = scale_features["ds_scale_records"]

        for op_index, operator in enumerate(operator_list, start=1):
            op_name, op_params = next(iter(operator.items()))
            row = {
                "pipeline_name": data_name,
                "pipeline_base_name": pipeline_base_name,
                "pipeline_category": infer_pipeline_category(pipeline_base_name),
                "pipeline_scale_token": pipeline_scale_token,
                "chunk_size": chunk_size,
                "chunk_part": chunk_part,
                "source_kind": run_kind,
                "scale_source_name": scale_source_name,
                "scale_value": source_scale_value,
                "operator_index": op_index,
                "operator_name": op_name,
                "op_process_number": pipeline_dict.get("np", np.nan),
                "ds_input_count": log_info[op_index - 1]["ds_input_count"],
                "ds_output_count": log_info[op_index - 1]["ds_output_count"],
                "op_process_time": log_info[op_index - 1]["op_process_time"],
                "ds_type": ds_type,
            }
            row.update(scale_features)
            row.update(monitor_info[op_index - 1])
            row[f"op_type_{op_name}"] = op_name.split("_")[-1]

            if isinstance(op_params, dict):
                for param_name, param_value in op_params.items():
                    col_name = f"op_{op_name}_{param_name}"
                    if col_name in REMOVE_COLS:
                        continue
                    row[col_name] = normalize_feature_value(param_value)
            rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(skipped), discovered


def get_model_features(predictor):
    features = get_predictor_features(predictor)
    if features:
        return list(features)
    if hasattr(predictor, "feature_metadata"):
        try:
            return list(predictor.feature_metadata.get_features())
        except Exception:
            pass
    return None


def predict_dataframe(df, log_target, mode, model_suffix=""):
    if df.empty:
        return df.copy()

    predicted_df = df.copy()
    predicted_df[PRED_LABEL] = np.nan
    predicted_df["model_dir_used"] = ""

    predictor_cache = {}
    for ds_type in sorted(predicted_df["ds_type"].dropna().unique().tolist()):
        model_dir = resolve_model_dir(
            ds_type=ds_type,
            log_target=log_target,
            mode=mode,
            model_suffix=model_suffix,
        )
        if model_dir not in predictor_cache:
            predictor_cache[model_dir] = TabularPredictor.load(model_dir)
        predictor = predictor_cache[model_dir]
        feature_cols = get_model_features(predictor)
        if not feature_cols:
            raise ValueError(f"unable to resolve model features for {model_dir}")

        mask = predicted_df["ds_type"] == ds_type
        predict_df = predicted_df.loc[mask].copy()
        missing_cols = [col for col in feature_cols if col not in predict_df.columns]
        if missing_cols:
            missing_df = pd.DataFrame(np.nan, index=predict_df.index, columns=missing_cols)
            predict_df = pd.concat([predict_df, missing_df], axis=1)
        predict_df = predict_df[feature_cols].copy()

        y_pred = predictor.predict(predict_df)
        y_pred = pd.to_numeric(y_pred, errors="coerce")
        if log_target:
            y_pred = np.expm1(y_pred)
        y_pred = np.clip(y_pred, 0, None)

        predicted_df.loc[mask, PRED_LABEL] = y_pred.to_numpy()
        predicted_df.loc[mask, "model_dir_used"] = os.path.basename(model_dir)

    return predicted_df


def safe_metric_block(df, true_col, pred_col):
    if df.empty:
        return {
            "sample_count": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "accuracy": np.nan,
        }

    y_true = pd.to_numeric(df[true_col], errors="coerce")
    y_pred = pd.to_numeric(df[pred_col], errors="coerce")
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
            "sample_count": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "accuracy": np.nan,
        }

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    accuracy = float(100 - mape)
    return {
        "sample_count": int(len(y_true)),
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "accuracy": accuracy,
    }


def add_row_prediction_errors(df):
    if df.empty:
        return df.copy()

    result_df = df.copy()
    y_true = pd.to_numeric(result_df[LABEL], errors="coerce")
    y_pred = pd.to_numeric(result_df[PRED_LABEL], errors="coerce")
    valid_mask = y_true.notna() & y_pred.notna() & np.isfinite(y_true) & np.isfinite(y_pred)
    positive_mask = valid_mask & (y_true > 0)

    result_df["row_abs_error"] = np.nan
    result_df["row_relative_error"] = np.nan
    result_df["row_accuracy"] = np.nan
    result_df.loc[valid_mask, "row_abs_error"] = (y_pred[valid_mask] - y_true[valid_mask]).abs()
    result_df.loc[positive_mask, "row_relative_error"] = (
        (y_pred[positive_mask] - y_true[positive_mask]).abs() / y_true[positive_mask]
    )
    result_df.loc[positive_mask, "row_accuracy"] = 100 - result_df.loc[positive_mask, "row_relative_error"] * 100
    return result_df


def aggregate_operator_level(full_df, chunk_df):
    chunk_df = add_row_prediction_errors(chunk_df)
    full_actual = (
        full_df.groupby(
            ["ds_type", "pipeline_base_name", "pipeline_category", "operator_index", "operator_name"],
            as_index=False,
        )
        .agg(
            full_pipeline_name=("pipeline_name", "first"),
            full_scale_token=("pipeline_scale_token", "first"),
            full_scale_value=("scale_value", "first"),
            full_actual_time=(LABEL, "sum"),
            full_pred_time=(PRED_LABEL, "sum"),
        )
        .copy()
    )

    chunk_df = chunk_df.copy()
    chunk_df["chunk_row_squared_error"] = chunk_df["row_abs_error"] ** 2
    chunk_df["chunk_row_ape_percent"] = chunk_df["row_relative_error"] * 100
    chunk_actual = (
        chunk_df.groupby(
            [
                "ds_type",
                "pipeline_base_name",
                "pipeline_category",
                "chunk_size",
                "operator_index",
                "operator_name",
            ],
            as_index=False,
        )
        .agg(
            chunk_parts=("pipeline_name", "nunique"),
            chunk_actual_time=(LABEL, "sum"),
            chunk_pred_time=(PRED_LABEL, "sum"),
            chunk_actual_time_mean=(LABEL, "mean"),
            chunk_actual_time_median=(LABEL, "median"),
            chunk_actual_time_min=(LABEL, "min"),
            chunk_actual_time_max=(LABEL, "max"),
            chunk_pred_time_mean=(PRED_LABEL, "mean"),
            chunk_pred_time_median=(PRED_LABEL, "median"),
            chunk_pred_time_min=(PRED_LABEL, "min"),
            chunk_pred_time_max=(PRED_LABEL, "max"),
            chunk_self_mae=("row_abs_error", "mean"),
            chunk_self_mse=("chunk_row_squared_error", "mean"),
            chunk_self_mape=("chunk_row_ape_percent", "mean"),
            chunk_self_valid_rows=("chunk_row_ape_percent", "count"),
        )
        .copy()
    )
    chunk_actual["chunk_self_rmse"] = np.sqrt(chunk_actual["chunk_self_mse"])
    chunk_actual["chunk_self_accuracy"] = 100 - chunk_actual["chunk_self_mape"]
    chunk_actual = chunk_actual.drop(columns=["chunk_self_mse"], errors="ignore")

    merged = chunk_actual.merge(
        full_actual,
        on=["ds_type", "pipeline_base_name", "pipeline_category", "operator_index", "operator_name"],
        how="left",
    )
    merged["chunk_actual_abs_error"] = (merged["chunk_actual_time"] - merged["full_actual_time"]).abs()
    merged["chunk_pred_abs_error"] = (merged["chunk_pred_time"] - merged["full_actual_time"]).abs()
    merged["full_pred_abs_error"] = (merged["full_pred_time"] - merged["full_actual_time"]).abs()
    merged["chunk_actual_relative_error"] = merged["chunk_actual_abs_error"] / merged["full_actual_time"]
    merged["chunk_pred_relative_error"] = merged["chunk_pred_abs_error"] / merged["full_actual_time"]
    merged["full_pred_relative_error"] = merged["full_pred_abs_error"] / merged["full_actual_time"]
    merged["chunk_actual_sum_vs_full_relative_error"] = merged["chunk_actual_relative_error"]
    merged["chunk_pred_sum_vs_full_relative_error"] = merged["chunk_pred_relative_error"]
    return merged, full_actual


def aggregate_pipeline_level(operator_merged):
    pipeline_df = (
        operator_merged.groupby(
            [
                "ds_type",
                "pipeline_base_name",
                "pipeline_category",
                "chunk_size",
                "full_pipeline_name",
                "full_scale_token",
            ],
            as_index=False,
        )
        .agg(
            chunk_parts=("chunk_parts", "sum"),
            operator_count=("operator_index", "nunique"),
            full_actual_time=("full_actual_time", "sum"),
            full_pred_time=("full_pred_time", "sum"),
            chunk_actual_time=("chunk_actual_time", "sum"),
            chunk_pred_time=("chunk_pred_time", "sum"),
        )
        .copy()
    )
    pipeline_df["chunk_actual_abs_error"] = (pipeline_df["chunk_actual_time"] - pipeline_df["full_actual_time"]).abs()
    pipeline_df["chunk_pred_abs_error"] = (pipeline_df["chunk_pred_time"] - pipeline_df["full_actual_time"]).abs()
    pipeline_df["full_pred_abs_error"] = (pipeline_df["full_pred_time"] - pipeline_df["full_actual_time"]).abs()
    pipeline_df["chunk_actual_relative_error"] = pipeline_df["chunk_actual_abs_error"] / pipeline_df["full_actual_time"]
    pipeline_df["chunk_pred_relative_error"] = pipeline_df["chunk_pred_abs_error"] / pipeline_df["full_actual_time"]
    pipeline_df["full_pred_relative_error"] = pipeline_df["full_pred_abs_error"] / pipeline_df["full_actual_time"]
    pipeline_df["chunk_actual_accuracy"] = 100 - pipeline_df["chunk_actual_relative_error"] * 100
    pipeline_df["chunk_pred_accuracy"] = 100 - pipeline_df["chunk_pred_relative_error"] * 100
    pipeline_df["full_pred_accuracy"] = 100 - pipeline_df["full_pred_relative_error"] * 100
    return pipeline_df


def aggregate_modality_level(pipeline_df):
    modality_df = (
        pipeline_df.groupby(["ds_type", "chunk_size"], as_index=False)
        .agg(
            pipeline_count=("pipeline_base_name", "nunique"),
            operator_count=("operator_count", "sum"),
            chunk_parts=("chunk_parts", "sum"),
            full_actual_time=("full_actual_time", "sum"),
            full_pred_time=("full_pred_time", "sum"),
            chunk_actual_time=("chunk_actual_time", "sum"),
            chunk_pred_time=("chunk_pred_time", "sum"),
            mean_chunk_actual_relative_error=("chunk_actual_relative_error", "mean"),
            mean_chunk_pred_relative_error=("chunk_pred_relative_error", "mean"),
            mean_full_pred_relative_error=("full_pred_relative_error", "mean"),
        )
        .copy()
    )
    modality_df["chunk_actual_abs_error"] = (modality_df["chunk_actual_time"] - modality_df["full_actual_time"]).abs()
    modality_df["chunk_pred_abs_error"] = (modality_df["chunk_pred_time"] - modality_df["full_actual_time"]).abs()
    modality_df["full_pred_abs_error"] = (modality_df["full_pred_time"] - modality_df["full_actual_time"]).abs()
    modality_df["chunk_actual_mape"] = modality_df["chunk_actual_abs_error"] / modality_df["full_actual_time"] * 100
    modality_df["chunk_pred_mape"] = modality_df["chunk_pred_abs_error"] / modality_df["full_actual_time"] * 100
    modality_df["full_pred_mape"] = modality_df["full_pred_abs_error"] / modality_df["full_actual_time"] * 100
    modality_df["chunk_actual_accuracy"] = 100 - modality_df["chunk_actual_mape"]
    modality_df["chunk_pred_accuracy"] = 100 - modality_df["chunk_pred_mape"]
    modality_df["full_pred_accuracy"] = 100 - modality_df["full_pred_mape"]
    return modality_df


def aggregate_category_level(pipeline_df):
    category_df = (
        pipeline_df.groupby(["ds_type", "pipeline_category", "chunk_size"], as_index=False)
        .agg(
            pipeline_count=("pipeline_base_name", "nunique"),
            operator_count=("operator_count", "sum"),
            chunk_parts=("chunk_parts", "sum"),
            full_actual_time=("full_actual_time", "sum"),
            full_pred_time=("full_pred_time", "sum"),
            chunk_actual_time=("chunk_actual_time", "sum"),
            chunk_pred_time=("chunk_pred_time", "sum"),
        )
        .copy()
    )
    category_df["chunk_pred_abs_error"] = (
        category_df["chunk_pred_time"] - category_df["full_actual_time"]
    ).abs()
    category_df["chunk_actual_abs_error"] = (
        category_df["chunk_actual_time"] - category_df["full_actual_time"]
    ).abs()
    category_df["full_pred_abs_error"] = (
        category_df["full_pred_time"] - category_df["full_actual_time"]
    ).abs()
    category_df["chunk_pred_mape"] = (
        category_df["chunk_pred_abs_error"] / category_df["full_actual_time"] * 100
    )
    category_df["chunk_actual_mape"] = (
        category_df["chunk_actual_abs_error"] / category_df["full_actual_time"] * 100
    )
    category_df["full_pred_mape"] = (
        category_df["full_pred_abs_error"] / category_df["full_actual_time"] * 100
    )
    category_df["chunk_pred_accuracy"] = 100 - category_df["chunk_pred_mape"]
    category_df["chunk_actual_accuracy"] = 100 - category_df["chunk_actual_mape"]
    category_df["full_pred_accuracy"] = 100 - category_df["full_pred_mape"]
    return category_df


def aggregate_row_level(predicted_df, kind):
    if predicted_df.empty:
        return pd.DataFrame()

    rows = []
    for ds_type in sorted(predicted_df["ds_type"].dropna().unique().tolist()):
        subset = predicted_df[predicted_df["ds_type"] == ds_type].copy()
        metrics = safe_metric_block(subset, LABEL, PRED_LABEL)
        metrics.update(
            {
                "ds_type": ds_type,
                "source_kind": kind,
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def choose_full_reference(full_df):
    if full_df.empty:
        return full_df.copy()

    full_df = full_df.copy()
    full_df["_scale_rank"] = pd.to_numeric(full_df["scale_value"], errors="coerce")
    # Only keep the largest full-scale run for each pipeline_base_name:
    # audio/image -> 8G, text -> 10000.
    selected_pipeline_names = []
    for (ds_type, pipeline_base_name), group_df in full_df.groupby(
        ["ds_type", "pipeline_base_name"], sort=True, dropna=False
    ):
        group_df = group_df.sort_values(
            by=["_scale_rank", "pipeline_name"],
            ascending=[True, True],
            na_position="first",
        )
        selected_pipeline_names.append(group_df["pipeline_name"].iloc[-1])
    selected_df = full_df[full_df["pipeline_name"].isin(selected_pipeline_names)].copy()
    selected_df = selected_df.drop(columns=["_scale_rank"], errors="ignore")
    return selected_df.reset_index(drop=True)


def format_number(value, digits=2):
    if pd.isna(value):
        return "nan"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def write_markdown_table(file_obj, df, columns, limit=None):
    if df.empty:
        file_obj.write("无数据。\n\n")
        return

    table_df = df.loc[:, [col for col in columns if col in df.columns]].copy()
    if limit:
        table_df = table_df.head(limit)

    for col in table_df.columns:
        if pd.api.types.is_float_dtype(table_df[col]):
            table_df[col] = table_df[col].map(lambda value: format_number(value))

    file_obj.write(table_df.to_markdown(index=False))
    file_obj.write("\n\n")


def write_report(
    report_path,
    overview,
    row_summary,
    modality_summary,
    category_summary,
    pipeline_summary,
    operator_merged,
    worst_rows,
):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 算子级 chunk 耗时外推实验报告\n\n")
        f.write("## 1. 实验口径\n\n")
        f.write(
            "本报告比较同一条 pipeline 在最大完整规模数据上的真实算子耗时，"
            "与将数据切成 chunk 后逐 chunk 预测再按算子求和得到的耗时。\n\n"
        )
        f.write("核心比较对象：`chunk_pred_time = sum(predicted chunk op time)` vs `full_actual_time`。\n\n")
        for line in overview:
            f.write(f"- {line}\n")
        f.write("\n")

        f.write("## 2. 模态汇总\n\n")
        write_markdown_table(
            f,
            modality_summary,
            [
                "ds_type",
                "chunk_size",
                "pipeline_count",
                "operator_count",
                "chunk_parts",
                "full_actual_time",
                "chunk_pred_time",
                "chunk_pred_accuracy",
                "chunk_actual_time",
                "chunk_actual_accuracy",
                "full_pred_time",
                "full_pred_accuracy",
            ],
        )

        f.write("## 3. Pipeline 类别汇总\n\n")
        write_markdown_table(
            f,
            category_summary,
            [
                "ds_type",
                "pipeline_category",
                "chunk_size",
                "pipeline_count",
                "operator_count",
                "full_actual_time",
                "chunk_pred_time",
                "chunk_pred_accuracy",
                "chunk_actual_time",
                "chunk_actual_accuracy",
                "full_pred_time",
                "full_pred_accuracy",
            ],
        )

        f.write("## 4. 每条 Pipeline 对比\n\n")
        pipeline_order = (
            pipeline_summary[["ds_type", "pipeline_category", "pipeline_base_name"]]
            .drop_duplicates()
            .sort_values(["ds_type", "pipeline_category", "pipeline_base_name"])
        )
        for _, pipeline_row in pipeline_order.iterrows():
            pipeline_name = pipeline_row["pipeline_base_name"]
            pipeline_df = pipeline_summary[pipeline_summary["pipeline_base_name"] == pipeline_name].copy()
            full_name = pipeline_df["full_pipeline_name"].iloc[0]
            full_scale = pipeline_df["full_scale_token"].iloc[0]
            category = pipeline_row["pipeline_category"]
            ds_type = pipeline_row["ds_type"]

            f.write(f"### {pipeline_name}\n\n")
            f.write(f"- 模态：`{ds_type}`\n")
            f.write(f"- 类别：`{category}`\n")
            f.write(f"- 完整规模基线：`{full_name}`，scale=`{full_scale}`\n\n")
            write_markdown_table(
                f,
                pipeline_df.sort_values("chunk_size"),
                [
                    "chunk_size",
                    "chunk_parts",
                    "operator_count",
                    "full_actual_time",
                    "chunk_pred_time",
                    "chunk_pred_accuracy",
                    "chunk_actual_time",
                    "chunk_actual_accuracy",
                    "full_pred_time",
                    "full_pred_accuracy",
                ],
            )

            best_row = pipeline_df.sort_values("chunk_pred_relative_error").iloc[0]
            f.write(
                "当前 pipeline 中 chunk 预测求和最接近完整执行的是 "
                f"`chunk_size={int(best_row['chunk_size'])}`，"
                f"chunk_pred_accuracy={format_number(best_row['chunk_pred_accuracy'])}%。\n\n"
            )

            op_df = operator_merged[operator_merged["pipeline_base_name"] == pipeline_name].copy()
            op_df = op_df.sort_values(["chunk_size", "operator_index"])
            f.write("算子级明细：\n\n")
            write_markdown_table(
                f,
                op_df,
                [
                    "chunk_size",
                    "operator_index",
                    "operator_name",
                    "chunk_parts",
                    "chunk_actual_time_mean",
                    "chunk_pred_time_mean",
                    "chunk_self_mape",
                    "chunk_self_accuracy",
                    "full_actual_time",
                    "chunk_actual_time",
                    "chunk_actual_sum_vs_full_relative_error",
                    "chunk_pred_time",
                    "chunk_pred_sum_vs_full_relative_error",
                    "full_pred_time",
                    "full_pred_relative_error",
                ],
                limit=80,
            )

        f.write("## 5. 误差最大的算子组合\n\n")
        write_markdown_table(
            f,
            worst_rows,
            [
                "ds_type",
                "pipeline_category",
                "pipeline_base_name",
                "chunk_size",
                "operator_index",
                "operator_name",
                "chunk_parts",
                "chunk_actual_time_mean",
                "chunk_pred_time_mean",
                "chunk_self_mape",
                "full_actual_time",
                "chunk_actual_time",
                "chunk_actual_sum_vs_full_relative_error",
                "chunk_pred_time",
                "chunk_pred_sum_vs_full_relative_error",
            ],
            limit=30,
        )

        f.write("## 6. 行级模型预测摘要\n\n")
        f.write(
            "该部分只用于辅助理解模型在 full/chunk 单行算子样本上的基础预测误差，"
            "不是本实验的主要结论。\n\n"
        )
        write_markdown_table(
            f,
            row_summary,
            ["source_kind", "ds_type", "sample_count", "rmse", "mae", "mape", "accuracy"],
        )


def main():
    parser = argparse.ArgumentParser(
        description="Predict chunk/full runtime with existing cost models and compare summed chunk cost against full pipeline runtime."
    )
    parser.add_argument(
        "--chunk_root",
        default="./collect_data/runs_chunks",
        help="Directory containing chunk run results.",
    )
    parser.add_argument(
        "--full_root",
        default="./collect_data/result_20260611",
        help="Directory containing full pipeline results.",
    )
    parser.add_argument(
        "--ds_type",
        choices=["audio", "image", "text"],
        default=None,
        help="Optional modality filter.",
    )
    parser.add_argument(
        "--base_name_regex",
        default="",
        help="Optional regex filter on pipeline_base_name.",
    )
    parser.add_argument(
        "--log_target",
        action="store_true",
        help="Use *_log cost models and log-target prediction files.",
    )
    parser.add_argument(
        "--model_mode",
        choices=[
            MODE_DEFAULT,
            MODE_OVERALL_ONLY,
            MODE_AUDIO_IMAGE_OVERALL_TEXT_ONLY,
        ],
        default=MODE_DEFAULT,
        help="How to choose cost models by modality.",
    )
    parser.add_argument(
        "--model_suffix",
        default="",
        help="Optional extra suffix appended to model directory names.",
    )
    parser.add_argument(
        "--chunk_profile_stats",
        nargs="*",
        default=[],
        help="Optional chunk profile stats JSON files generated by collect_data/analyze_chunk_dataset_distribution.py.",
    )
    parser.add_argument(
        "--output_suffix",
        default="20260611",
        help="Suffix appended to output files.",
    )
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=100,
        help="Print a progress message every N discovered YAML runs.",
    )
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
        raise ValueError("no chunk rows were parsed; check chunk_root and filters")
    if full_rows.empty:
        raise ValueError("no full rows were parsed; check full_root and filters")

    chunk_bases = set(chunk_rows["pipeline_base_name"].dropna().unique().tolist())
    full_rows = full_rows[full_rows["pipeline_base_name"].isin(chunk_bases)].copy()
    if full_rows.empty:
        raise ValueError("no full rows matched the parsed chunk pipelines")

    full_ref_rows = choose_full_reference(full_rows)
    if full_ref_rows.empty:
        raise ValueError("unable to select full reference rows")

    chunk_profile_files = []
    chunk_profile_invalid = 0
    chunk_profile_matched = 0
    chunk_profile_unmatched_keys = 0
    if args.chunk_profile_stats:
        chunk_profile_lookup, chunk_profile_files, chunk_profile_invalid = load_chunk_profile_lookup(
            args.chunk_profile_stats
        )
        chunk_rows, chunk_profile_matched, chunk_profile_unmatched_keys = apply_chunk_profile_features(
            chunk_rows,
            chunk_profile_lookup,
        )
    else:
        chunk_rows["chunk_profile_matched"] = False

    predicted_chunk_rows = predict_dataframe(
        chunk_rows,
        log_target=args.log_target,
        mode=args.model_mode,
        model_suffix=args.model_suffix,
    )
    predicted_chunk_rows = add_row_prediction_errors(predicted_chunk_rows)
    predicted_full_rows = predict_dataframe(
        full_ref_rows,
        log_target=args.log_target,
        mode=args.model_mode,
        model_suffix=args.model_suffix,
    )
    predicted_full_rows = add_row_prediction_errors(predicted_full_rows)

    row_summary = pd.concat(
        [
            aggregate_row_level(predicted_full_rows, "full_reference"),
            aggregate_row_level(predicted_chunk_rows, "chunk"),
        ],
        ignore_index=True,
    )

    operator_merged, selected_full_actual = aggregate_operator_level(predicted_full_rows, predicted_chunk_rows)
    pipeline_summary = aggregate_pipeline_level(operator_merged)
    modality_summary = aggregate_modality_level(pipeline_summary)
    category_summary = aggregate_category_level(pipeline_summary)

    predictions_suffix = build_suffix(args.log_target, args.output_suffix)
    rows_output_path = PREDICTIONS_DIR / f"chunk_runtime_rows{predictions_suffix}.csv"
    chunk_self_output_path = SUMMARIES_DIR / f"chunk_runtime_chunk_self_prediction{predictions_suffix}.csv"
    operator_output_path = SUMMARIES_DIR / f"chunk_runtime_operator_comparison{predictions_suffix}.csv"
    pipeline_output_path = SUMMARIES_DIR / f"chunk_runtime_pipeline_summary{predictions_suffix}.csv"
    modality_output_path = SUMMARIES_DIR / f"chunk_runtime_modality_summary{predictions_suffix}.csv"
    category_output_path = SUMMARIES_DIR / f"chunk_runtime_category_summary{predictions_suffix}.csv"
    reference_output_path = SUMMARIES_DIR / f"chunk_runtime_reference_selection{predictions_suffix}.csv"
    skipped_output_path = SUMMARIES_DIR / f"chunk_runtime_skipped{predictions_suffix}.csv"
    report_output_path = REPORTS_DIR / f"chunk_runtime_scaling_report{predictions_suffix}.md"

    predicted_chunk_rows.to_csv(rows_output_path, index=False)
    chunk_self_cols = [
        "ds_type",
        "pipeline_base_name",
        "pipeline_category",
        "pipeline_name",
        "chunk_size",
        "chunk_part",
        "chunk_profile_matched",
        "operator_index",
        "operator_name",
        "ds_input_count",
        "ds_output_count",
        LABEL,
        PRED_LABEL,
        "row_abs_error",
        "row_relative_error",
        "row_accuracy",
    ]
    predicted_chunk_rows[
        [col for col in chunk_self_cols if col in predicted_chunk_rows.columns]
    ].to_csv(chunk_self_output_path, index=False)
    operator_merged.to_csv(operator_output_path, index=False)
    pipeline_summary.to_csv(pipeline_output_path, index=False)
    modality_summary.to_csv(modality_output_path, index=False)
    category_summary.to_csv(category_output_path, index=False)
    selected_full_actual.to_csv(reference_output_path, index=False)

    skipped_df = pd.concat(
        [
            chunk_skipped.assign(source_kind="chunk"),
            full_skipped.assign(source_kind="full"),
        ],
        ignore_index=True,
    )
    skipped_df.to_csv(skipped_output_path, index=False)

    overview = [
        f"chunk_root: {args.chunk_root}",
        f"full_root: {args.full_root}",
        f"parsed chunk YAMLs: {chunk_discovered}",
        f"parsed full YAMLs: {full_discovered}",
        f"chunk rows: {len(predicted_chunk_rows)}",
        f"full reference rows: {len(predicted_full_rows)}",
        f"skipped runs: {len(skipped_df)}",
        f"chunk profile files: {chunk_profile_files if chunk_profile_files else 'not used'}",
        f"chunk profile matched rows: {chunk_profile_matched}",
        f"chunk profile unmatched keys: {chunk_profile_unmatched_keys}",
        f"chunk profile invalid chunks: {chunk_profile_invalid}",
    ]

    chunk_full_metrics = safe_metric_block(operator_merged, "full_actual_time", "chunk_pred_time")
    chunk_actual_metrics = safe_metric_block(operator_merged, "full_actual_time", "chunk_actual_time")
    full_pred_metrics = safe_metric_block(operator_merged, "full_actual_time", "full_pred_time")

    overview.extend(
        [
            "operator-level chunk prediction: RMSE={0}, MAE={1}, MAPE={2}%, accuracy={3}%".format(
                format_number(chunk_full_metrics["rmse"]),
                format_number(chunk_full_metrics["mae"]),
                format_number(chunk_full_metrics["mape"]),
                format_number(chunk_full_metrics["accuracy"]),
            ),
            "operator-level chunk actual sum: RMSE={0}, MAE={1}, MAPE={2}%, accuracy={3}%".format(
                format_number(chunk_actual_metrics["rmse"]),
                format_number(chunk_actual_metrics["mae"]),
                format_number(chunk_actual_metrics["mape"]),
                format_number(chunk_actual_metrics["accuracy"]),
            ),
            "operator-level direct full prediction: RMSE={0}, MAE={1}, MAPE={2}%, accuracy={3}%".format(
                format_number(full_pred_metrics["rmse"]),
                format_number(full_pred_metrics["mae"]),
                format_number(full_pred_metrics["mape"]),
                format_number(full_pred_metrics["accuracy"]),
            ),
        ]
    )

    worst_rows = operator_merged.sort_values("chunk_pred_relative_error", ascending=False).copy()
    write_report(
        report_output_path,
        overview,
        row_summary,
        modality_summary,
        category_summary,
        pipeline_summary,
        operator_merged,
        worst_rows,
    )

    print(f"saved rows: {rows_output_path}")
    print(f"saved chunk self prediction: {chunk_self_output_path}")
    print(f"saved operator comparison: {operator_output_path}")
    print(f"saved pipeline summary: {pipeline_output_path}")
    print(f"saved modality summary: {modality_output_path}")
    print(f"saved category summary: {category_output_path}")
    print(f"saved full references: {reference_output_path}")
    print(f"saved skipped rows: {skipped_output_path}")
    print(f"saved report: {report_output_path}")
    print("\npipeline-level chunk prediction summary:")
    print(pipeline_summary[[
        "ds_type",
        "pipeline_category",
        "pipeline_base_name",
        "chunk_size",
        "full_actual_time",
        "chunk_actual_time",
        "chunk_pred_time",
        "full_pred_time",
        "chunk_pred_accuracy",
        "full_pred_accuracy",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
