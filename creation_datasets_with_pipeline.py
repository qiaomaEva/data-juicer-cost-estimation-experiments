import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import yaml

from project_paths import DATA_DIR, ensure_output_dirs


INCLUDE_PARAMS = True
BASE_PATH = "./collect_data/result_20260428"
OUTPUT_PATH = str(DATA_DIR)

FILTER_INCLUDE_LIST = [
    "image-pipeline",
    "image_pipeline",
    "text-pipeline",
    "text_pipeline",
    "audio-pipeline",
    "audio_pipeline",
]


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


def extract_scale_features(data_name, ds_type):
    scale_label = None
    ds_scale_mb = np.nan
    ds_scale_records = np.nan

    match = re.search(r"_([^_]+)$", str(data_name))
    if match:
        scale_label = match.group(1)

    if scale_label:
        label_upper = scale_label.upper()
        if ds_type in ("audio", "image"):
            mb_match = re.fullmatch(r"(\d+(?:\.\d+)?)MB", label_upper)
            g_match = re.fullmatch(r"(\d+(?:\.\d+)?)G", label_upper)
            if mb_match:
                ds_scale_mb = float(mb_match.group(1))
            elif g_match:
                ds_scale_mb = float(g_match.group(1)) * 1024.0
        elif ds_type == "text" and re.fullmatch(r"\d+", scale_label):
            ds_scale_records = float(scale_label)

    return {
        "ds_scale_label": scale_label,
        "ds_scale_mb": ds_scale_mb,
        "ds_scale_records": ds_scale_records,
    }


def split_pipeline_name(data_name):
    match = re.search(r"^(.*)_([^_]+)$", str(data_name))
    if not match:
        return data_name, ""
    return match.group(1), match.group(2)


def normalize_extra_suffix(extra_suffix):
    if not extra_suffix:
        return ""
    return extra_suffix if extra_suffix.startswith("_") else f"_{extra_suffix}"


def build_suffix(log_target, extra_suffix=""):
    return ("_log" if log_target else "") + normalize_extra_suffix(extra_suffix)


def get_datasets(base_path):
    result = []
    for root, _, files in os.walk(base_path):
        for file_name in files:
            if not file_name.endswith(".yaml"):
                continue

            data_name = os.path.splitext(file_name)[0]
            pipeline_base_name, pipeline_scale_token = split_pipeline_name(data_name)

            log_file_path = os.path.join(root, "log")
            for root2, _, files2 in os.walk(log_file_path):
                for log_file in files2:
                    if "DEBUG" in log_file or "ERROR" in log_file or "WARNING" in log_file:
                        continue
                    log_file_path = os.path.join(root2, log_file)

            monitor_dir = os.path.join(root, "monitor")
            monitor_file_path = os.path.join(monitor_dir, "monitor.json")
            if not os.path.exists(monitor_file_path):
                root_name = os.path.basename(os.path.normpath(root))
                alt_monitor_file_path = os.path.join(monitor_dir, f"{root_name}_monitor.json")
                if os.path.exists(alt_monitor_file_path):
                    monitor_file_path = alt_monitor_file_path
                elif os.path.isdir(monitor_dir):
                    monitor_candidates = [
                        file_name
                        for file_name in os.listdir(monitor_dir)
                        if file_name.endswith("_monitor.json")
                    ]
                    if monitor_candidates:
                        monitor_file_path = os.path.join(monitor_dir, monitor_candidates[0])

            result.append(
                {
                    "data_name": data_name,
                    "pipeline_name": data_name,
                    "pipeline_base_name": pipeline_base_name,
                    "pipeline_scale_token": pipeline_scale_token,
                    "log_file_path": log_file_path,
                    "monitor_file_path": monitor_file_path,
                    "pipeline_file_path": os.path.join(root, file_name),
                }
            )

    result.sort(key=lambda item: item["pipeline_name"])
    return result


def parse_log(log_file_path):
    ds_input_count = 0
    operator_list = []

    with open(log_file_path, "r", encoding="utf-8") as file_obj:
        for log_line in file_obj.readlines():
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
                    r"OP \[(.*?)\] Done in (.*?)s. Left (.*?) samples\.",
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


def build_operator_header(datasets):
    operator_set = set()
    for dataset in datasets:
        with open(dataset["pipeline_file_path"], "r", encoding="utf-8") as file_obj:
            pipeline_dict = yaml.safe_load(file_obj)
        for operator in pipeline_dict["process"]:
            for op_name, op_params in operator.items():
                operator_set.add("op_type_{0}".format(op_name))
                if INCLUDE_PARAMS and isinstance(op_params, dict):
                    for param_name in op_params.keys():
                        operator_set.add("op_{0}_{1}".format(op_name, param_name))

    data_set = set(
        [
            "pipeline_name",
            "pipeline_base_name",
            "pipeline_scale_token",
            "operator_index",
            "operator_name",
            "ds_input_count",
            "ds_output_count",
            "ds_type",
            "ds_scale_label",
            "ds_scale_mb",
            "ds_scale_records",
        ]
    )

    operator_set.add("op_process_time")
    operator_set.add("op_process_number")

    env_set = set(
        [
            "env_cpu_count",
            "env_cpu_util",
            "env_mem_util",
            "env_gpu_util",
            "env_gpu_mem_util",
        ]
    )

    remove_cols = {
        "op_audio_add_gaussian_noise_mapper_noise_level",
        "op_audio_ffmpeg_wrapped_mapper_save_dir",
        "op_audio_add_gaussian_noise_mapper_save_dir",
        "op_image_blur_mapper_save_dir",
        "op_image_face_blur_mapper_save_dir",
        "op_image_remove_background_mapper_save_dir",
    }
    operator_set = operator_set - remove_cols

    dataset_header = sorted(data_set | operator_set | env_set)
    dataset_header_for_cost = dataset_header.copy()
    dataset_header_for_cost.remove("ds_output_count")
    return dataset_header_for_cost


def main():
    parser = argparse.ArgumentParser(
        description="Generate cost dataset headers with pipeline_name columns."
    )
    parser.add_argument(
        "--log_target",
        action="store_true",
        help="Write *_log.csv outputs for log-target experiment flows.",
    )
    parser.add_argument(
        "--base_path",
        default=BASE_PATH,
        help="Collected result directory to parse. Defaults to collect_data/result_20260428.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Extra suffix appended to output filenames, e.g. 20260408.",
    )
    args = parser.parse_args()

    suffix = build_suffix(args.log_target, args.suffix)
    org_datasets = get_datasets(args.base_path)
    datasets = []
    for data_info in org_datasets:
        if any(include_name in data_info["monitor_file_path"] for include_name in FILTER_INCLUDE_LIST):
            datasets.append(data_info)

    print("matched datasets: {0}".format(len(datasets)))
    dataset_header_for_cost = build_operator_header(datasets)

    data_rows = []
    skipped_rows = []

    for data_index, dataset in enumerate(datasets, start=1):
        print("({0}/{1}) processing: {2}".format(data_index, len(datasets), dataset["data_name"]))
        try:
            log_info = parse_log(dataset["log_file_path"])
            monitor_info = parse_monitor(dataset["monitor_file_path"])
        except Exception as exc:
            skipped_rows.append(
                {
                    "pipeline_name": dataset["pipeline_name"],
                    "reason": "parse_failed: {0}".format(exc),
                }
            )
            print("[SKIP] parse failed: {0} | {1}".format(dataset["data_name"], exc))
            continue

        with open(dataset["pipeline_file_path"], "r", encoding="utf-8") as file_obj:
            pipeline_dict = yaml.safe_load(file_obj)
        operator_list = pipeline_dict["process"]

        if len(log_info) < len(operator_list) or len(monitor_info) < len(operator_list):
            skipped_rows.append(
                {
                    "pipeline_name": dataset["pipeline_name"],
                    "reason": "incomplete_data ops={0} log={1} monitor={2}".format(
                        len(operator_list),
                        len(log_info),
                        len(monitor_info),
                    ),
                }
            )
            print(
                "[SKIP] incomplete data: {0} | ops={1}, log={2}, monitor={3}".format(
                    dataset["data_name"],
                    len(operator_list),
                    len(log_info),
                    len(monitor_info),
                )
            )
            continue

        for op_index, operator in enumerate(operator_list, start=1):
            data_row = {
                "pipeline_name": dataset["pipeline_name"],
                "pipeline_base_name": dataset["pipeline_base_name"],
                "pipeline_scale_token": dataset["pipeline_scale_token"],
                "operator_index": op_index,
                "operator_name": next(iter(operator.keys())),
                "op_process_number": pipeline_dict["np"],
            }

            for op_name, op_params in operator.items():
                data_row["op_type_{0}".format(op_name)] = op_name.split("_")[-1]
                if isinstance(op_params, dict):
                    for param_name, param_value in op_params.items():
                        data_row["op_{0}_{1}".format(op_name, param_name)] = param_value

            data_row["ds_input_count"] = log_info[op_index - 1]["ds_input_count"]
            data_row["ds_output_count"] = log_info[op_index - 1]["ds_output_count"]
            data_row["op_process_time"] = log_info[op_index - 1]["op_process_time"]
            data_row["ds_type"] = infer_ds_type(dataset["data_name"], operator)
            data_row.update(extract_scale_features(dataset["data_name"], data_row["ds_type"]))
            data_row.update(monitor_info[op_index - 1])
            data_rows.append(data_row)

    ensure_output_dirs()

    cost_df = pd.DataFrame(data_rows, columns=dataset_header_for_cost)

    cost_path = os.path.join(
        OUTPUT_PATH,
        f"dataset_header_for_cost_estimation_with_pipeline{suffix}.csv",
    )
    skipped_path = os.path.join(OUTPUT_PATH, f"dataset_with_pipeline_skipped{suffix}.csv")

    cost_df.to_csv(cost_path, index=False)
    pd.DataFrame(skipped_rows).to_csv(skipped_path, index=False)

    print("saved: {0}".format(cost_path))
    print("saved: {0}".format(skipped_path))


if __name__ == "__main__":
    main()
