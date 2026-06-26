import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import DATA_DIR, ensure_output_dirs, resolve_legacy_aware_path


STATS_DEFAULT_PATH = "collect_data/dataset_stats_full.json"
PROFILE_PREFIX = "profile"
PERCENTILE_KEYS = ["min", "max", "mean", "p5", "p10", "p25", "p50", "p75", "p90", "p95"]


def normalize_scale_token(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""

    token = str(value).strip()
    if not token:
        return ""

    upper = token.upper()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(M|MB|MIB|G|GB|GIB)", upper)
    if match:
        number = float(match.group(1))
        unit = match.group(2)
        if unit in {"G", "GB", "GIB"}:
            mb = number * 1024.0
        else:
            mb = number
        if abs(mb - round(mb)) < 1e-9:
            return f"{int(round(mb))}MB"
        return f"{mb:g}MB"

    if re.fullmatch(r"\d+(?:\.0+)?", upper):
        return str(int(float(upper)))

    return upper


def scale_token_from_stats_file(ds_type, stats_file):
    name = Path(stats_file).name
    if ds_type == "audio":
        match = re.search(r"by-size-(\d+(?:\.\d+)?)(MiB|MB|G|GB|GiB)", name, re.IGNORECASE)
        if match:
            return normalize_scale_token("".join(match.groups()))
    if ds_type == "image":
        match = re.search(r"physical_(\d+(?:\.\d+)?)(M|MB|G|GB)", name, re.IGNORECASE)
        if match:
            return normalize_scale_token("".join(match.groups()))
    if ds_type == "text":
        match = re.search(r"c4_(\d+)", name, re.IGNORECASE)
        if match:
            return normalize_scale_token(match.group(1))
    return ""


def flatten_metric(prefix, metric_name, metric_stats):
    features = {}
    if not isinstance(metric_stats, dict):
        return features

    for key in PERCENTILE_KEYS:
        value = metric_stats.get(key)
        if value is not None:
            features[f"{prefix}_{metric_name}_{key}"] = value

    n_value = metric_stats.get("n")
    if n_value is not None:
        features[f"{prefix}_{metric_name}_n"] = n_value

    return features


def build_profile_lookup(stats_path):
    with open(stats_path, "r", encoding="utf-8") as file_obj:
        stats = json.load(file_obj)

    sections = {
        "audio": "audio_AudioSet",
        "image": "image_COCO2017",
        "text": "text_RedPajama_c4",
    }

    lookup = {}
    for ds_type, section_name in sections.items():
        section = stats.get(section_name, {})
        for item in section.get("by_file", []):
            stats_file = item.get("file", "")
            scale_token = scale_token_from_stats_file(ds_type, stats_file)
            if not scale_token:
                continue

            feature_prefix = f"{PROFILE_PREFIX}_{ds_type}"
            features = {}
            for metric_name, metric_stats in item.get("stats", {}).items():
                features.update(flatten_metric(feature_prefix, metric_name, metric_stats))

            lookup[(ds_type, scale_token)] = features

    return lookup


def collect_profile_columns(lookup):
    columns = set()
    for features in lookup.values():
        columns.update(features.keys())
    return sorted(columns)


def add_profile_features(df, lookup):
    profile_columns = collect_profile_columns(lookup)

    if "ds_type" not in df.columns:
        raise ValueError("input table is missing required column: ds_type")

    if "pipeline_scale_token" in df.columns:
        scale_source = "pipeline_scale_token"
    elif "ds_scale_label" in df.columns:
        scale_source = "ds_scale_label"
    else:
        raise ValueError("input table must contain pipeline_scale_token or ds_scale_label")

    profile_rows = []
    matched = 0
    unmatched_keys = set()
    for _, row in df.iterrows():
        ds_type = str(row["ds_type"]).strip().lower()
        scale_token = normalize_scale_token(row[scale_source])
        features = lookup.get((ds_type, scale_token))
        if not features:
            unmatched_keys.add((ds_type, scale_token))
            profile_rows.append({})
            continue
        matched += 1
        profile_rows.append(features)

    profile_df = pd.DataFrame(profile_rows, columns=profile_columns, index=df.index)
    enhanced_df = pd.concat([df, profile_df], axis=1).copy()
    return enhanced_df, matched, sorted(unmatched_keys)


def default_output_path(input_path, suffix):
    input_path = Path(input_path)
    return str(input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}"))


def main():
    parser = argparse.ArgumentParser(
        description="Add dataset profiling features from dataset_stats_full.json to training tables."
    )
    parser.add_argument("--input", required=True, help="Input CSV training table.")
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path. Defaults to input basename plus --suffix.",
    )
    parser.add_argument(
        "--stats_path",
        default=STATS_DEFAULT_PATH,
        help="Path to dataset_stats_full.json.",
    )
    parser.add_argument(
        "--suffix",
        default="_profile",
        help="Suffix used when --output is omitted.",
    )
    args = parser.parse_args()
    ensure_output_dirs()

    lookup = build_profile_lookup(args.stats_path)
    if not lookup:
        raise ValueError(f"no usable profile stats loaded from: {args.stats_path}")

    input_path = resolve_legacy_aware_path(args.input, DATA_DIR)
    output_path = args.output or default_output_path(input_path, args.suffix)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)
    enhanced_df, matched, unmatched_keys = add_profile_features(df, lookup)
    enhanced_df.to_csv(output_path, index=False)

    print(f"loaded profile groups: {len(lookup)}")
    print(f"input rows: {len(df)}")
    print(f"matched rows: {matched}")
    print(f"profile columns added: {len(collect_profile_columns(lookup))}")
    print(f"saved: {output_path}")
    if unmatched_keys:
        print("unmatched ds_type/scale keys:")
        for key in unmatched_keys:
            print(f"  {key}")


if __name__ == "__main__":
    main()
