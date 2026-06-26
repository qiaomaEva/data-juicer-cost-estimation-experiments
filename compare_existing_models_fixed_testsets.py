import argparse
import json
import os

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

from project_paths import (
    DATA_DIR,
    EXISTING_MODEL_COMPARISON_DIR,
    MODEL_DIR,
    data_path,
    ensure_output_dirs,
    resolve_legacy_aware_path,
)

DATA_PATH = data_path("dataset_header_for_cost_estimation.csv")
LABEL = "op_process_time"
MODALITIES = ["audio", "image", "text"]


def build_suffix(log_target):
    return "_log" if log_target else ""


def split_721(dataframe, random_state):
    train_data, temp_data = train_test_split(
        dataframe,
        test_size=0.3,
        random_state=random_state,
    )
    val_data, test_data = train_test_split(
        temp_data,
        test_size=1 / 3,
        random_state=random_state,
    )
    return train_data.copy(), val_data.copy(), test_data.copy()


def resolve_model_dir(ds_type, log_target):
    suffix = build_suffix(log_target)
    model_root = str(MODEL_DIR)
    model_name = f"cost_model{suffix}" if ds_type is None else f"cost_model_{ds_type}{suffix}"
    model_dir = os.path.join(model_root, model_name)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    return model_dir


def load_predictor(model_dir, predictor_cache):
    if model_dir not in predictor_cache:
        predictor_cache[model_dir] = TabularPredictor.load(model_dir)
    return predictor_cache[model_dir]


def predict_values(predictor, eval_df, log_target):
    feature_df = eval_df.drop(columns=[LABEL, "source_row_id"], errors="ignore").copy()
    y_pred = predictor.predict(feature_df)
    y_pred = pd.to_numeric(y_pred, errors="coerce")

    if log_target:
        y_pred = np.expm1(y_pred)

    y_pred = np.clip(y_pred, 0, None)
    return y_pred


def calculate_metrics(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {
        "sample_count": int(len(y_true)),
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }


def evaluate_model(eval_df, predictor, pred_col, model_dir, log_target):
    result_df = eval_df.copy()
    result_df[pred_col] = predict_values(predictor, result_df, log_target)

    y_true = pd.to_numeric(result_df[LABEL], errors="coerce")
    y_pred = pd.to_numeric(result_df[pred_col], errors="coerce")
    valid_mask = (
        y_true.notna()
        & y_pred.notna()
        & np.isfinite(y_true)
        & np.isfinite(y_pred)
        & (y_true > 0)
    )

    valid_df = result_df.loc[valid_mask].copy()
    metrics = calculate_metrics(
        pd.to_numeric(valid_df[LABEL], errors="coerce"),
        pd.to_numeric(valid_df[pred_col], errors="coerce"),
    )
    metrics["model_dir"] = model_dir
    return result_df, metrics


def build_combined_overall_metrics(df):
    y_true = pd.to_numeric(df[LABEL], errors="coerce")
    y_pred = pd.to_numeric(df["op_process_time_pred_overall"], errors="coerce")
    valid_mask = (
        y_true.notna()
        & y_pred.notna()
        & np.isfinite(y_true)
        & np.isfinite(y_pred)
        & (y_true > 0)
    )
    valid_df = df.loc[valid_mask].copy()
    metrics = calculate_metrics(
        pd.to_numeric(valid_df[LABEL], errors="coerce"),
        pd.to_numeric(valid_df["op_process_time_pred_overall"], errors="coerce"),
    )
    return valid_df, metrics


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare existing overall and modality-specific cost models on the same fixed "
            "test set for each modality."
        )
    )
    parser.add_argument(
        "--data_path",
        default=DATA_PATH,
        help="Path to dataset_header_for_cost_estimation.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=str(EXISTING_MODEL_COMPARISON_DIR),
        help="Directory to save fixed test sets, merged predictions and summary files.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed used to build each modality's fixed 7:2:1 split.",
    )
    parser.add_argument(
        "--log_target",
        action="store_true",
        help="Load *_log models and compare them on the same fixed modality test sets.",
    )
    args = parser.parse_args()
    ensure_output_dirs()
    args.data_path = resolve_legacy_aware_path(args.data_path, DATA_DIR)

    df = pd.read_csv(args.data_path, low_memory=False)
    if LABEL not in df.columns:
        raise ValueError(f"missing label column: {LABEL}")
    if "ds_type" not in df.columns:
        raise ValueError("missing ds_type column")

    df = df.reset_index(drop=False).rename(columns={"index": "source_row_id"})
    os.makedirs(args.output_dir, exist_ok=True)

    suffix = build_suffix(args.log_target)
    overall_model_dir = resolve_model_dir(None, args.log_target)
    predictor_cache = {}
    overall_predictor = load_predictor(overall_model_dir, predictor_cache)

    split_info = {
        "data_path": args.data_path,
        "random_state": args.random_state,
        "log_target": args.log_target,
        "overall_model_dir": overall_model_dir,
        "modalities": {},
    }
    summary_rows = []
    combined_prediction_rows = []

    for ds_type in MODALITIES:
        modality_df = df[df["ds_type"] == ds_type].copy()
        if len(modality_df) == 0:
            print(f"[SKIP] no rows found for ds_type={ds_type}")
            continue

        train_df, val_df, test_df = split_721(modality_df, args.random_state)
        eval_df = test_df[test_df[LABEL] > 0].copy()
        if len(eval_df) == 0:
            print(f"[SKIP] no positive-label test rows for ds_type={ds_type}")
            continue

        modality_model_dir = resolve_model_dir(ds_type, args.log_target)
        modality_predictor = load_predictor(modality_model_dir, predictor_cache)

        merged_df, overall_metrics = evaluate_model(
            eval_df=eval_df,
            predictor=overall_predictor,
            pred_col="op_process_time_pred_overall",
            model_dir=overall_model_dir,
            log_target=args.log_target,
        )
        merged_df, modality_metrics = evaluate_model(
            eval_df=merged_df,
            predictor=modality_predictor,
            pred_col=f"op_process_time_pred_{ds_type}",
            model_dir=modality_model_dir,
            log_target=args.log_target,
        )

        merged_df["absolute_error_overall"] = (
            merged_df[LABEL] - merged_df["op_process_time_pred_overall"]
        ).abs()
        merged_df[f"absolute_error_{ds_type}"] = (
            merged_df[LABEL] - merged_df[f"op_process_time_pred_{ds_type}"]
        ).abs()

        fixed_test_path = os.path.join(
            args.output_dir,
            f"fixed_testset_{ds_type}{suffix}.csv",
        )
        prediction_path = os.path.join(
            args.output_dir,
            f"fixed_testset_{ds_type}_predictions{suffix}.csv",
        )
        test_df.to_csv(fixed_test_path, index=False)
        merged_df.to_csv(prediction_path, index=False)

        split_info["modalities"][ds_type] = {
            "total_rows": int(len(modality_df)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "positive_label_test_rows": int(len(eval_df)),
            "modality_model_dir": modality_model_dir,
            "fixed_testset_file": fixed_test_path,
            "prediction_file": prediction_path,
        }

        summary_rows.append(
            {
                "ds_type": ds_type,
                "model_type": "overall",
                **overall_metrics,
                "fixed_testset_file": fixed_test_path,
                "prediction_file": prediction_path,
            }
        )
        summary_rows.append(
            {
                "ds_type": ds_type,
                "model_type": f"{ds_type}_only",
                **modality_metrics,
                "fixed_testset_file": fixed_test_path,
                "prediction_file": prediction_path,
            }
        )

        combined_prediction_rows.append(merged_df.copy())

    combined_prediction_path = os.path.join(
        args.output_dir,
        f"fixed_testset_overall_predictions{suffix}.csv",
    )
    if combined_prediction_rows:
        combined_prediction_df = pd.concat(
            combined_prediction_rows,
            ignore_index=True,
            sort=False,
        )
        combined_prediction_df.to_csv(combined_prediction_path, index=False)
        _, combined_metrics = build_combined_overall_metrics(combined_prediction_df)
        combined_metrics["model_dir"] = overall_model_dir
        split_info["overall_prediction_file"] = combined_prediction_path
        summary_rows.append(
            {
                "ds_type": "overall",
                "model_type": "overall",
                **combined_metrics,
                "fixed_testset_file": combined_prediction_path,
                "prediction_file": combined_prediction_path,
            }
        )
    else:
        split_info["overall_prediction_file"] = None

    split_info_path = os.path.join(args.output_dir, f"split_info{suffix}.json")
    summary_path = os.path.join(args.output_dir, f"comparison_metrics{suffix}.csv")

    with open(split_info_path, "w", encoding="utf-8") as file_obj:
        json.dump(split_info, file_obj, ensure_ascii=False, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df.to_csv(summary_path, index=False, float_format="%.6f")

    print("=" * 100)
    print("Existing-model comparison on fixed modality test sets")
    print("=" * 100)
    print(f"overall model: {overall_model_dir}")
    print(f"saved split info: {split_info_path}")
    if summary_df.empty:
        print("no summary rows generated.")
        return

    print(summary_df.to_string(index=False))
    print(f"\nsaved metrics: {summary_path}")


if __name__ == "__main__":
    main()
