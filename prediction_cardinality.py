import argparse

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

from project_paths import (
    DATA_DIR,
    FEATURE_IMPORTANCE_DIR,
    MODEL_DIR,
    PREDICTIONS_DIR,
    data_path,
    ensure_output_dirs,
    resolve_legacy_aware_path,
)

DATA_PATH = data_path("dataset_header_for_cardinality_estimation.csv")
OUTPUT_LABEL = "ds_output_count"
INPUT_COUNT_COL = "ds_input_count"
RATIO_LABEL = "cardinality_ratio"
SMOOTH_LOG_RATIO_LABEL = "cardinality_smooth_log_ratio"
PRED_OUTPUT_LABEL = "ds_output_count_pred"
PRED_RATIO_LABEL = "cardinality_ratio_pred"
PRED_SMOOTH_LOG_RATIO_LABEL = "cardinality_smooth_log_ratio_pred"
TARGET_OUTPUT = "output"
TARGET_RATIO = "ratio"
TARGET_SMOOTH_LOG_RATIO = "smooth_log_ratio"


def normalize_extra_suffix(extra_suffix):
    if not extra_suffix:
        return ""
    return extra_suffix if extra_suffix.startswith("_") else f"_{extra_suffix}"


def build_suffix(ds_type, target_mode, log_target, extra_suffix):
    parts = []
    if ds_type:
        parts.append(ds_type)
    if target_mode == TARGET_RATIO:
        parts.append(TARGET_RATIO)
    elif target_mode == TARGET_SMOOTH_LOG_RATIO:
        parts.append(TARGET_SMOOTH_LOG_RATIO)
    if log_target:
        parts.append("log")
    suffix = "".join(f"_{part}" for part in parts)
    return f"{suffix}{normalize_extra_suffix(extra_suffix)}"


def get_label(target_mode):
    if target_mode == TARGET_RATIO:
        return RATIO_LABEL
    if target_mode == TARGET_SMOOTH_LOG_RATIO:
        return SMOOTH_LOG_RATIO_LABEL
    return OUTPUT_LABEL


def is_ratio_like_mode(target_mode):
    return target_mode in {TARGET_RATIO, TARGET_SMOOTH_LOG_RATIO}


def transform_target(dataframe: pd.DataFrame, label: str) -> pd.DataFrame:
    transformed = dataframe.copy()
    if (transformed[label] < 0).any():
        raise ValueError(f"列 '{label}' 存在负值，无法执行 log1p 变换！")
    transformed[label] = np.log1p(transformed[label])
    return transformed


def add_ratio_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    if INPUT_COUNT_COL not in dataframe.columns:
        raise ValueError(f"数据集中缺少 '{INPUT_COUNT_COL}' 列，无法构造 ratio 目标！")
    if OUTPUT_LABEL not in dataframe.columns:
        raise ValueError(f"目标列 '{OUTPUT_LABEL}' 不存在！")

    result = dataframe.copy()
    result[INPUT_COUNT_COL] = pd.to_numeric(result[INPUT_COUNT_COL], errors="coerce")
    result[OUTPUT_LABEL] = pd.to_numeric(result[OUTPUT_LABEL], errors="coerce")

    if result[INPUT_COUNT_COL].isna().any():
        raise ValueError(f"列 '{INPUT_COUNT_COL}' 存在非数值或缺失值，无法构造 ratio 目标！")
    if result[OUTPUT_LABEL].isna().any():
        raise ValueError(f"列 '{OUTPUT_LABEL}' 存在非数值或缺失值，无法构造 ratio 目标！")
    if (result[OUTPUT_LABEL] < 0).any():
        raise ValueError(f"列 '{OUTPUT_LABEL}' 存在负值，无法构造 ratio 目标！")

    non_positive_input = result[INPUT_COUNT_COL] <= 0
    invalid_rows = non_positive_input & (result[OUTPUT_LABEL] > 0)
    if invalid_rows.any():
        raise ValueError(
            "存在 ds_input_count <= 0 但 ds_output_count > 0 的样本，"
            "无法定义 ds_output_count / ds_input_count。"
        )

    result[RATIO_LABEL] = np.nan
    positive_input = result[INPUT_COUNT_COL] > 0
    result.loc[positive_input, RATIO_LABEL] = (
        result.loc[positive_input, OUTPUT_LABEL] / result.loc[positive_input, INPUT_COUNT_COL]
    )
    return result


def add_smooth_log_ratio_column(dataframe: pd.DataFrame, ratio_alpha: float) -> pd.DataFrame:
    if ratio_alpha <= 0:
        raise ValueError("--ratio_alpha 必须大于 0。")

    result = add_ratio_column(dataframe)
    result[SMOOTH_LOG_RATIO_LABEL] = np.nan
    positive_input = result[INPUT_COUNT_COL] > 0
    result.loc[positive_input, SMOOTH_LOG_RATIO_LABEL] = np.log(
        (result.loc[positive_input, OUTPUT_LABEL] + ratio_alpha)
        / (result.loc[positive_input, INPUT_COUNT_COL] + ratio_alpha)
    )
    return result


def drop_non_feature_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.drop(
        columns=[
            OUTPUT_LABEL,
            RATIO_LABEL,
            SMOOTH_LOG_RATIO_LABEL,
            PRED_OUTPUT_LABEL,
            PRED_RATIO_LABEL,
            PRED_SMOOTH_LOG_RATIO_LABEL,
            "cardinality_ratio_true",
            "cardinality_smooth_log_ratio_true",
            "cardinality_target_mode",
            "cardinality_ratio_alpha",
            "prediction_rule",
        ],
        errors="ignore",
    )


def prepare_fit_data(
    dataframe: pd.DataFrame,
    target_mode: str,
    log_target: bool,
    ratio_alpha: float,
) -> pd.DataFrame:
    label = get_label(target_mode)

    if is_ratio_like_mode(target_mode):
        if target_mode == TARGET_SMOOTH_LOG_RATIO:
            prepared = add_smooth_log_ratio_column(dataframe, ratio_alpha)
        else:
            prepared = add_ratio_column(dataframe)
        before_count = len(prepared)
        prepared = prepared[prepared[INPUT_COUNT_COL] > 0].copy()
        prepared = prepared.dropna(subset=[label]).copy()
        if len(prepared) == 0:
            raise ValueError(f"过滤 ds_input_count > 0 后没有可用于 {target_mode} 训练的数据！")
        print(f"{target_mode} 训练数据过滤：{before_count} -> {len(prepared)}")
        prepared = prepared.drop(columns=[OUTPUT_LABEL], errors="ignore")
        if target_mode == TARGET_SMOOTH_LOG_RATIO:
            prepared = prepared.drop(columns=[RATIO_LABEL], errors="ignore")
    else:
        prepared = dataframe.copy()

    return transform_target(prepared, label) if log_target else prepared


def add_prediction_metadata(
    result_df: pd.DataFrame,
    target_mode: str,
    ratio_alpha: float | None = None,
) -> pd.DataFrame:
    result_df = result_df.copy()
    result_df["cardinality_target_mode"] = target_mode
    result_df["cardinality_ratio_alpha"] = ratio_alpha if ratio_alpha is not None else np.nan
    result_df["cardinality_ratio_true"] = np.nan
    positive_input = result_df[INPUT_COUNT_COL] > 0
    result_df.loc[positive_input, "cardinality_ratio_true"] = (
        result_df.loc[positive_input, OUTPUT_LABEL] / result_df.loc[positive_input, INPUT_COUNT_COL]
    )
    result_df["cardinality_smooth_log_ratio_true"] = np.nan
    if ratio_alpha is not None:
        result_df.loc[positive_input, "cardinality_smooth_log_ratio_true"] = np.log(
            (result_df.loc[positive_input, OUTPUT_LABEL] + ratio_alpha)
            / (result_df.loc[positive_input, INPUT_COUNT_COL] + ratio_alpha)
        )
    result_df[PRED_RATIO_LABEL] = np.nan
    result_df[PRED_SMOOTH_LOG_RATIO_LABEL] = np.nan
    result_df[PRED_OUTPUT_LABEL] = np.nan
    result_df["prediction_rule"] = ""
    return result_df


def predict_output_mode(predictor, test_data: pd.DataFrame, log_target: bool) -> pd.DataFrame:
    result_df = test_data[test_data[OUTPUT_LABEL] > 0].copy()
    if len(result_df) == 0:
        raise ValueError("测试集中没有正的目标值，无法计算 MAPE！")

    result_df = add_prediction_metadata(result_df, TARGET_OUTPUT)
    predict_df = drop_non_feature_columns(result_df)
    y_pred = predictor.predict(predict_df)
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    if log_target:
        y_pred = np.expm1(y_pred)
    y_pred = np.clip(y_pred, 0, None)

    result_df[PRED_OUTPUT_LABEL] = y_pred.to_numpy()
    positive_input = result_df[INPUT_COUNT_COL] > 0
    result_df.loc[positive_input, PRED_RATIO_LABEL] = (
        result_df.loc[positive_input, PRED_OUTPUT_LABEL]
        / result_df.loc[positive_input, INPUT_COUNT_COL]
    )
    result_df["prediction_rule"] = "model"
    return result_df


def predict_ratio_mode(predictor, test_data: pd.DataFrame, log_target: bool) -> pd.DataFrame:
    result_df = add_ratio_column(test_data)
    result_df = add_prediction_metadata(result_df, TARGET_RATIO)

    zero_input_mask = result_df[INPUT_COUNT_COL] <= 0
    result_df.loc[zero_input_mask, PRED_OUTPUT_LABEL] = 0.0
    result_df.loc[zero_input_mask, "prediction_rule"] = "zero_input"

    model_mask = result_df[INPUT_COUNT_COL] > 0
    if model_mask.any():
        predict_df = drop_non_feature_columns(result_df.loc[model_mask].copy())
        ratio_pred = predictor.predict(predict_df)
        ratio_pred = pd.to_numeric(ratio_pred, errors="coerce")
        if log_target:
            ratio_pred = np.expm1(ratio_pred)
        ratio_pred = np.clip(ratio_pred, 0, None)

        result_df.loc[model_mask, PRED_RATIO_LABEL] = ratio_pred.to_numpy()
        result_df.loc[model_mask, PRED_OUTPUT_LABEL] = (
            result_df.loc[model_mask, INPUT_COUNT_COL].to_numpy() * ratio_pred.to_numpy()
        )
        result_df.loc[model_mask, "prediction_rule"] = "model"

    print(f"ratio 测试集 zero_input 规则样本数：{int(zero_input_mask.sum())}")
    return result_df


def predict_smooth_log_ratio_mode(
    predictor,
    test_data: pd.DataFrame,
    ratio_alpha: float,
) -> pd.DataFrame:
    result_df = add_smooth_log_ratio_column(test_data, ratio_alpha)
    result_df = add_prediction_metadata(result_df, TARGET_SMOOTH_LOG_RATIO, ratio_alpha)

    zero_input_mask = result_df[INPUT_COUNT_COL] <= 0
    result_df.loc[zero_input_mask, PRED_OUTPUT_LABEL] = 0.0
    result_df.loc[zero_input_mask, "prediction_rule"] = "zero_input"

    model_mask = result_df[INPUT_COUNT_COL] > 0
    if model_mask.any():
        predict_df = drop_non_feature_columns(result_df.loc[model_mask].copy())
        smooth_log_ratio_pred = predictor.predict(predict_df)
        smooth_log_ratio_pred = pd.to_numeric(smooth_log_ratio_pred, errors="coerce")

        input_count = result_df.loc[model_mask, INPUT_COUNT_COL].to_numpy()
        output_pred = np.exp(smooth_log_ratio_pred.to_numpy()) * (input_count + ratio_alpha) - ratio_alpha
        output_pred = np.clip(output_pred, 0, None)
        ratio_pred = output_pred / input_count

        result_df.loc[model_mask, PRED_SMOOTH_LOG_RATIO_LABEL] = smooth_log_ratio_pred.to_numpy()
        result_df.loc[model_mask, PRED_RATIO_LABEL] = ratio_pred
        result_df.loc[model_mask, PRED_OUTPUT_LABEL] = output_pred
        result_df.loc[model_mask, "prediction_rule"] = "model"

    print(f"smooth_log_ratio 测试集 zero_input 规则样本数：{int(zero_input_mask.sum())}")
    return result_df


def calculate_output_metrics(result_df: pd.DataFrame):
    metric_df = result_df.dropna(subset=[OUTPUT_LABEL, PRED_OUTPUT_LABEL]).copy()
    metric_df = metric_df[metric_df[OUTPUT_LABEL] > 0].copy()
    if len(metric_df) == 0:
        raise ValueError("测试集中没有正的目标值，无法计算 MAPE！")

    y_true = pd.to_numeric(metric_df[OUTPUT_LABEL], errors="coerce")
    y_pred = pd.to_numeric(metric_df[PRED_OUTPUT_LABEL], errors="coerce")
    valid_mask = y_true.notna() & y_pred.notna() & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    approx_accuracy = 100 - mape
    return len(y_true), rmse, mae, mape, approx_accuracy


def predict_by_target_mode(
    predictor,
    data: pd.DataFrame,
    target_mode: str,
    log_target: bool,
    ratio_alpha: float,
) -> pd.DataFrame:
    if target_mode == TARGET_RATIO:
        return predict_ratio_mode(predictor, data, log_target)
    if target_mode == TARGET_SMOOTH_LOG_RATIO:
        return predict_smooth_log_ratio_mode(predictor, data, ratio_alpha)
    return predict_output_mode(predictor, data, log_target)


def print_metric_block(title: str, sample_label: str, metrics):
    sample_count, rmse, mae, mape, approx_accuracy = metrics
    print("\n" + "=" * 60 + "\n")
    print(title)
    print("\n" + "=" * 60 + "\n")
    print(f"{sample_label}：{sample_count}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"近似预测准确率：{approx_accuracy:.2f}%")
    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train cardinality prediction model.")
    parser.add_argument(
        "--ds_type",
        choices=["audio", "image", "text"],
        default=None,
        help="Only train/evaluate on one modality.",
    )
    parser.add_argument(
        "--data_path",
        default=DATA_PATH,
        help="Training CSV path. Defaults to output/data/dataset_header_for_cardinality_estimation.csv.",
    )
    parser.add_argument(
        "--log_target",
        action="store_true",
        help="Train on log1p(target) and restore predictions with expm1.",
    )
    parser.add_argument(
        "--target_mode",
        choices=[TARGET_OUTPUT, TARGET_RATIO, TARGET_SMOOTH_LOG_RATIO],
        default=TARGET_OUTPUT,
        help=(
            "Prediction target. 'output' predicts ds_output_count directly; "
            "'ratio' predicts ds_output_count / ds_input_count; "
            "'smooth_log_ratio' predicts log((ds_output_count + alpha) / (ds_input_count + alpha))."
        ),
    )
    parser.add_argument(
        "--ratio_alpha",
        type=float,
        default=1.0,
        help="Smoothing alpha for --target_mode smooth_log_ratio. Defaults to 1.0.",
    )
    parser.add_argument(
        "--model_suffix",
        default="",
        help="Extra suffix appended to model and output filenames, e.g. profile.",
    )
    args = parser.parse_args()
    ensure_output_dirs()
    args.data_path = resolve_legacy_aware_path(args.data_path, DATA_DIR)

    if args.ratio_alpha <= 0:
        raise ValueError("--ratio_alpha 必须大于 0。")
    if args.target_mode == TARGET_SMOOTH_LOG_RATIO and args.log_target:
        raise ValueError("smooth_log_ratio 模式本身已经是 log 目标，不要再叠加 --log_target。")

    label = get_label(args.target_mode)
    suffix = build_suffix(args.ds_type, args.target_mode, args.log_target, args.model_suffix)
    model_save_dir = str(MODEL_DIR / f"cardinality_model{suffix}")
    validation_output_path = str(PREDICTIONS_DIR / f"validation_set_predictions_cardinality{suffix}.csv")
    prediction_output_path = str(PREDICTIONS_DIR / f"test_set_predictions_cardinality{suffix}.csv")
    feature_importance_path = str(FEATURE_IMPORTANCE_DIR / f"feature_importance_cardinality{suffix}.txt")

    df = pd.read_csv(args.data_path, low_memory=False)
    if OUTPUT_LABEL not in df.columns:
        raise ValueError(f"目标列 '{OUTPUT_LABEL}' 不存在！")
    if INPUT_COUNT_COL not in df.columns:
        raise ValueError(f"数据集中缺少 '{INPUT_COUNT_COL}' 列！")

    if args.ds_type:
        if "ds_type" not in df.columns:
            raise ValueError("数据集中缺少 'ds_type' 列，无法按模态过滤！")
        df = df[df["ds_type"] == args.ds_type].copy()
        if len(df) == 0:
            raise ValueError(f"过滤 ds_type='{args.ds_type}' 后没有可用数据！")
        print(f"按 ds_type='{args.ds_type}' 过滤后数据量：{len(df)}")
    else:
        print(f"原始数据量：{len(df)}")
    print(f"训练数据：{args.data_path}")
    print(f"目标模式：{args.target_mode} | 训练标签：{label}")

    train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=1 / 3, random_state=42)

    print("划分完成:")
    print(f"   - 训练集：{len(train_data)} ({len(train_data) / len(df):.1%})")
    print(f"   - 验证集：{len(val_data)} ({len(val_data) / len(df):.1%})")
    print(f"   - 测试集：{len(test_data)} ({len(test_data) / len(df):.1%})")

    train_fit = prepare_fit_data(train_data, args.target_mode, args.log_target, args.ratio_alpha)
    val_fit = prepare_fit_data(val_data, args.target_mode, args.log_target, args.ratio_alpha)
    if args.log_target:
        print(f"\n当前模式：训练目标为 log1p({label})，预测后使用 expm1 还原。")
    if args.target_mode == TARGET_RATIO:
        print(
            "ratio 模式：训练/验证仅使用 ds_input_count > 0 样本；"
            "ds_input_count <= 0 的测试样本直接预测 ds_output_count_pred=0。"
        )
    if args.target_mode == TARGET_SMOOTH_LOG_RATIO:
        print(
            "smooth_log_ratio 模式：训练目标为 "
            f"log((ds_output_count + {args.ratio_alpha}) / (ds_input_count + {args.ratio_alpha}))；"
            "训练/验证仅使用 ds_input_count > 0 样本；"
            "ds_input_count <= 0 的测试样本直接预测 ds_output_count_pred=0。"
        )

    print("\n使用训练集训练 AutoGluon 模型...")
    predictor = TabularPredictor(
        label=label,
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        path=model_save_dir,
    )

    predictor.fit(
        train_data=train_fit,      # 训练集：用于拟合各个候选模型的参数。
        tuning_data=val_fit,       # 验证集：用于模型选择、早停和加权集成，不参与最终测试集评估。
        presets="medium_quality",  # AutoGluon 预设：在训练时间和模型效果之间做中等强度折中。
        verbosity=2,               # 日志级别：输出主要训练过程和每个模型的验证集分数。
    )

    fi = predictor.feature_importance(data=train_fit)
    with open(feature_importance_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(fi.to_string())

    best_model = predictor.model_best
    print(f"\n最佳模型：{best_model}")

    print("\n在验证集上评估（用于模型选择和手动调参参考）...")
    val_result_df = predict_by_target_mode(
        predictor,
        val_data,
        args.target_mode,
        args.log_target,
        args.ratio_alpha,
    )
    val_metrics = calculate_output_metrics(val_result_df)
    print_metric_block(
        "验证集性能（用于模型选择和调参参考）",
        "验证样本数",
        val_metrics,
    )
    val_result_df.to_csv(validation_output_path, index=False)

    print("\n在完全未见过的测试集上评估...")
    result_df = predict_by_target_mode(
        predictor,
        test_data,
        args.target_mode,
        args.log_target,
        args.ratio_alpha,
    )
    test_metrics = calculate_output_metrics(result_df)
    print_metric_block(
        "真实泛化性能（测试集，仅用于最终评估）",
        "测试样本数",
        test_metrics,
    )

    print(f"\n验证集预测结果已保存至：{validation_output_path}")
    result_df.to_csv(prediction_output_path, index=False)
    print(f"\n测试集预测结果已保存至：{prediction_output_path}")
    print(f"特征重要性已保存至：{feature_importance_path}")
    print(f"模型已保存至：{model_save_dir}")


if __name__ == "__main__":
    main()
