import argparse
import os
import time

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from project_paths import (
    DATA_DIR,
    MODEL_DIR,
    PREDICTIONS_DIR,
    ensure_output_dirs,
    legacy_output_path,
    resolve_legacy_aware_path,
)

LABEL = "op_process_time"
INPUT_COUNT_COL = "ds_input_count"
CARDINALITY_PRED_COL = "ds_output_count_pred_from_cardinality"

MODE_DEFAULT = "per_modality_preferred"
MODE_OVERALL_ONLY = "overall_only"
MODE_AUDIO_IMAGE_OVERALL_TEXT_ONLY = "audio_image_overall_text_only"

INPUT_COUNT_ACTUAL = "actual"
INPUT_COUNT_PREDICTED_CARDINALITY = "predicted_cardinality"


def normalize_extra_suffix(extra_suffix):
    if not extra_suffix:
        return ""
    return extra_suffix if extra_suffix.startswith("_") else f"_{extra_suffix}"


def build_suffix(log_target, extra_suffix=""):
    return ("_log" if log_target else "") + normalize_extra_suffix(extra_suffix)


def resolve_input_file(log_target, data_path=""):
    if data_path:
        data_path = resolve_legacy_aware_path(data_path, DATA_DIR)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"input dataset not found: {data_path}")
        return data_path

    log_suffix = build_suffix(log_target)
    candidates = [
        str(DATA_DIR / f"dataset_header_for_cost_estimation_with_pipeline{log_suffix}.csv"),
        legacy_output_path(f"dataset_header_for_cost_estimation_with_pipeline{log_suffix}.csv"),
    ]
    if log_target:
        candidates.extend(
            [
                str(DATA_DIR / "dataset_header_for_cost_estimation_with_pipeline.csv"),
                legacy_output_path("dataset_header_for_cost_estimation_with_pipeline.csv"),
            ]
        )

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"no usable input dataset found for log_target={log_target}")


def resolve_output_file(log_target, extra_suffix):
    suffix = build_suffix(log_target, extra_suffix)
    return str(PREDICTIONS_DIR / f"test_set_predictions_cost_full_with_pipeline{suffix}.csv")


def resolve_mode(args):
    if args.model_mode:
        return args.model_mode
    if args.use_overall_model_only:
        return MODE_OVERALL_ONLY
    return MODE_DEFAULT


def choose_extra_suffix(args, mode):
    if args.suffix:
        return args.suffix
    if args.input_count_mode == INPUT_COUNT_PREDICTED_CARDINALITY:
        return INPUT_COUNT_PREDICTED_CARDINALITY
    if mode == MODE_OVERALL_ONLY:
        return MODE_OVERALL_ONLY
    if mode == MODE_AUDIO_IMAGE_OVERALL_TEXT_ONLY:
        return MODE_AUDIO_IMAGE_OVERALL_TEXT_ONLY
    return ""


def resolve_model_dir(ds_type, log_target, mode, model_suffix=""):
    log_suffix = build_suffix(log_target)
    model_extra_suffix = normalize_extra_suffix(model_suffix)
    model_root = str(MODEL_DIR)
    overall_model = os.path.join(model_root, f"cost_model{log_suffix}{model_extra_suffix}")
    modality_model = os.path.join(model_root, f"cost_model_{ds_type}{log_suffix}{model_extra_suffix}")

    if mode == MODE_OVERALL_ONLY:
        candidates = [overall_model]
    elif mode == MODE_AUDIO_IMAGE_OVERALL_TEXT_ONLY:
        if ds_type in {"audio", "image"}:
            candidates = [overall_model]
        elif ds_type == "text":
            candidates = [modality_model, overall_model]
        else:
            candidates = [overall_model]
    else:
        candidates = [modality_model, overall_model]

    for model_dir in candidates:
        if os.path.isdir(model_dir):
            return model_dir

    raise FileNotFoundError(
        "no usable model found for ds_type={0}, log_target={1}, mode={2}".format(
            ds_type,
            log_target,
            mode,
        )
    )


def resolve_cardinality_model_dir(log_target, model_dir="", model_suffix=""):
    if model_dir:
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"cardinality model dir not found: {model_dir}")
        return model_dir

    log_suffix = build_suffix(log_target)
    model_extra_suffix = normalize_extra_suffix(model_suffix)
    candidate = os.path.join(
        str(MODEL_DIR),
        f"cardinality_model{log_suffix}{model_extra_suffix}",
    )
    if os.path.isdir(candidate):
        return candidate

    raise FileNotFoundError(f"cardinality model dir not found: {candidate}")


def get_predictor_features(predictor):
    if getattr(predictor, "feature_metadata_in", None) is not None:
        return predictor.feature_metadata_in.get_features()
    if hasattr(predictor, "features"):
        return predictor.features()
    return None


def build_predict_frame(row, predictor):
    features = get_predictor_features(predictor)
    row_df = pd.DataFrame([row])
    if not features:
        return row_df

    missing_cols = [col for col in features if col not in row_df.columns]
    for col in missing_cols:
        row_df[col] = np.nan
    return row_df[features]


def predict_one(predictor, row, log_target):
    predict_df = build_predict_frame(row, predictor)
    pred = predictor.predict(predict_df)
    pred = pd.to_numeric(pred, errors="coerce").iloc[0]
    if log_target:
        pred = np.expm1(pred)
    if not np.isfinite(pred):
        return np.nan
    return max(float(pred), 0.0)


def predict_with_actual_input_counts(
    result_df,
    feature_columns,
    predictor_cache,
    log_target,
    mode,
    model_suffix,
):
    for ds_type in sorted(result_df["ds_type"].dropna().unique().tolist()):
        model_dir = resolve_model_dir(
            ds_type=ds_type,
            log_target=log_target,
            mode=mode,
            model_suffix=model_suffix,
        )
        print(
            "using cost model for {0}: {1} | log_target={2} | mode={3}".format(
                ds_type,
                model_dir,
                log_target,
                mode,
            )
        )

        if model_dir not in predictor_cache:
            predictor_cache[model_dir] = TabularPredictor.load(model_dir)
        predictor = predictor_cache[model_dir]

        mask = result_df["ds_type"] == ds_type
        predict_df = result_df.loc[mask, feature_columns].copy()
        y_pred = predictor.predict(predict_df)
        y_pred = pd.to_numeric(y_pred, errors="coerce")
        if log_target:
            y_pred = np.expm1(y_pred)
        y_pred = np.clip(y_pred, 0, None)

        result_df.loc[mask, "op_process_time_pred"] = y_pred.to_numpy()
        result_df.loc[mask, "model_dir_used"] = os.path.basename(model_dir)

    return result_df


def predict_with_propagated_cardinality(
    result_df,
    cost_predictor_cache,
    cardinality_predictor,
    cost_log_target,
    cardinality_log_target,
    mode,
    model_suffix,
    progress_interval,
):
    required_cols = ["pipeline_name", "operator_index", INPUT_COUNT_COL]
    missing_cols = [col for col in required_cols if col not in result_df.columns]
    if missing_cols:
        raise ValueError(
            "predicted_cardinality mode requires columns: {0}".format(missing_cols)
        )

    result_df["original_ds_input_count"] = result_df[INPUT_COUNT_COL]
    result_df["ds_input_count_used_for_cost"] = np.nan
    result_df[CARDINALITY_PRED_COL] = np.nan
    result_df["input_count_mode"] = INPUT_COUNT_PREDICTED_CARDINALITY

    model_dir_by_ds_type = {}
    for ds_type in sorted(result_df["ds_type"].dropna().unique().tolist()):
        model_dir_by_ds_type[ds_type] = resolve_model_dir(
            ds_type=ds_type,
            log_target=cost_log_target,
            mode=mode,
            model_suffix=model_suffix,
        )
        print(
            "using cost model for {0}: {1} | log_target={2} | mode={3}".format(
                ds_type,
                model_dir_by_ds_type[ds_type],
                cost_log_target,
                mode,
            )
        )

    total_pipelines = result_df["pipeline_name"].nunique()
    total_ops = len(result_df)
    processed_ops = 0
    start_time = time.time()

    for pipeline_idx, (pipeline_name, group_df) in enumerate(
        result_df.groupby("pipeline_name", sort=True),
        start=1,
    ):
        sorted_index = group_df.sort_values("operator_index").index.tolist()
        current_input_count = result_df.loc[sorted_index[0], INPUT_COUNT_COL]

        for row_index in sorted_index:
            row = result_df.loc[row_index].copy()
            row[INPUT_COUNT_COL] = current_input_count

            ds_type = row["ds_type"]
            model_dir = model_dir_by_ds_type[ds_type]
            if model_dir not in cost_predictor_cache:
                cost_predictor_cache[model_dir] = TabularPredictor.load(model_dir)
            cost_predictor = cost_predictor_cache[model_dir]

            cost_pred = predict_one(cost_predictor, row, cost_log_target)
            result_df.loc[row_index, "op_process_time_pred"] = cost_pred
            result_df.loc[row_index, "model_dir_used"] = os.path.basename(model_dir)
            result_df.loc[row_index, INPUT_COUNT_COL] = current_input_count
            result_df.loc[row_index, "ds_input_count_used_for_cost"] = current_input_count

            if pd.isna(current_input_count) or float(current_input_count) <= 0:
                cardinality_pred = 0.0
            else:
                cardinality_pred = predict_one(
                    cardinality_predictor,
                    row,
                    cardinality_log_target,
                )
                if pd.isna(cardinality_pred) or not np.isfinite(cardinality_pred):
                    cardinality_pred = 0.0

            result_df.loc[row_index, CARDINALITY_PRED_COL] = cardinality_pred
            current_input_count = cardinality_pred

        processed_ops += len(sorted_index)
        should_report = (
            progress_interval > 0
            and (
                pipeline_idx == 1
                or pipeline_idx % progress_interval == 0
                or pipeline_idx == total_pipelines
            )
        )
        if should_report:
            elapsed = time.time() - start_time
            ops_per_sec = processed_ops / elapsed if elapsed > 0 else 0.0
            remaining_ops = total_ops - processed_ops
            eta_sec = remaining_ops / ops_per_sec if ops_per_sec > 0 else float("nan")
            eta_text = f"{eta_sec / 60:.1f}min" if np.isfinite(eta_sec) else "unknown"
            print(
                "progress: {0}/{1} pipelines ({2:.1f}%) | ops {3}/{4} | "
                "elapsed {5:.1f}min | eta {6} | current={7}".format(
                    pipeline_idx,
                    total_pipelines,
                    pipeline_idx / total_pipelines * 100,
                    processed_ops,
                    total_ops,
                    elapsed / 60,
                    eta_text,
                    pipeline_name,
                ),
                flush=True,
            )

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Run full-dataset inference with existing cost models."
    )
    parser.add_argument(
        "--log_target",
        action="store_true",
        help="Use *_log model directories and write *_log prediction files.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Extra suffix appended to output filenames. If omitted, some modes use their own default suffix.",
    )
    parser.add_argument(
        "--use_overall_model_only",
        action="store_true",
        help="Deprecated shortcut for --model_mode overall_only.",
    )
    parser.add_argument(
        "--model_mode",
        choices=[
            MODE_DEFAULT,
            MODE_OVERALL_ONLY,
            MODE_AUDIO_IMAGE_OVERALL_TEXT_ONLY,
        ],
        default=None,
        help="How to choose models by modality.",
    )
    parser.add_argument(
        "--data_path",
        default="",
        help="Optional input CSV with pipeline columns. Defaults to output/data/dataset_header_for_cost_estimation_with_pipeline*.csv.",
    )
    parser.add_argument(
        "--model_suffix",
        default="",
        help="Extra suffix appended to model directory names, e.g. profile.",
    )
    parser.add_argument(
        "--input_count_mode",
        choices=[INPUT_COUNT_ACTUAL, INPUT_COUNT_PREDICTED_CARDINALITY],
        default=INPUT_COUNT_ACTUAL,
        help=(
            "How to provide ds_input_count to the cost model. "
            "'actual' uses collected true input counts; "
            "'predicted_cardinality' propagates cardinality model predictions along each pipeline."
        ),
    )
    parser.add_argument(
        "--cardinality_log_target",
        action="store_true",
        help="Use expm1 to restore predictions from a log1p cardinality model.",
    )
    parser.add_argument(
        "--cardinality_model_dir",
        default="",
        help="Optional explicit cardinality model directory. Defaults to output/AutogluonModels/cardinality_model[_log].",
    )
    parser.add_argument(
        "--cardinality_model_suffix",
        default="",
        help="Extra suffix appended to default cardinality model directory names.",
    )
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=50,
        help=(
            "In predicted_cardinality mode, print progress every N pipelines. "
            "Set to 0 to disable progress logs."
        ),
    )
    args = parser.parse_args()
    ensure_output_dirs()

    mode = resolve_mode(args)
    extra_suffix = choose_extra_suffix(args, mode)
    input_file = resolve_input_file(args.log_target, args.data_path)
    output_file = resolve_output_file(args.log_target, extra_suffix)

    df = pd.read_csv(input_file, low_memory=False)
    if LABEL not in df.columns:
        raise ValueError(f"missing label column: {LABEL}")
    if "ds_type" not in df.columns:
        raise ValueError("missing ds_type column")
    if args.input_count_mode == INPUT_COUNT_PREDICTED_CARDINALITY and INPUT_COUNT_COL not in df.columns:
        raise ValueError(f"missing input count column: {INPUT_COUNT_COL}")

    feature_columns = [col for col in df.columns if col != LABEL]
    predictor_cache = {}
    result_df = df.copy()
    result_df["op_process_time_pred"] = np.nan
    result_df["model_dir_used"] = ""
    result_df["model_log_target"] = args.log_target
    result_df["model_selection_mode"] = mode
    result_df["input_count_mode"] = args.input_count_mode

    if args.input_count_mode == INPUT_COUNT_PREDICTED_CARDINALITY:
        cardinality_model_dir = resolve_cardinality_model_dir(
            args.cardinality_log_target,
            args.cardinality_model_dir,
            args.cardinality_model_suffix,
        )
        print(
            "using cardinality model for propagated input counts: {0} | log_target={1}".format(
                cardinality_model_dir,
                args.cardinality_log_target,
            )
        )
        cardinality_predictor = TabularPredictor.load(cardinality_model_dir)
        result_df = predict_with_propagated_cardinality(
            result_df=result_df,
            cost_predictor_cache=predictor_cache,
            cardinality_predictor=cardinality_predictor,
            cost_log_target=args.log_target,
            cardinality_log_target=args.cardinality_log_target,
            mode=mode,
            model_suffix=args.model_suffix,
            progress_interval=args.progress_interval,
        )
    else:
        result_df = predict_with_actual_input_counts(
            result_df=result_df,
            feature_columns=feature_columns,
            predictor_cache=predictor_cache,
            log_target=args.log_target,
            mode=mode,
            model_suffix=args.model_suffix,
        )

    result_df.to_csv(output_file, index=False)
    print(f"saved: {output_file}")
    print(f"rows: {len(result_df)}")
    print(f"pipelines: {result_df['pipeline_name'].nunique()}")


if __name__ == "__main__":
    main()
