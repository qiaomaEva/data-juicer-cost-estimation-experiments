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

# ==============================
# 1. 配置参数定义
# ==============================
# 成本估计数据集文件路径
DATA_PATH = data_path("dataset_header_for_cost_estimation.csv")
# 目标变量列名：操作处理时间
LABEL = "op_process_time"


def transform_target(dataframe: pd.DataFrame) -> pd.DataFrame:
    transformed = dataframe.copy()
    if (transformed[LABEL] < 0).any():
        raise ValueError(f"列 '{LABEL}' 存在负值，无法执行 log1p 变换！")
    transformed[LABEL] = np.log1p(transformed[LABEL])
    return transformed


def predict_cost(predictor, data: pd.DataFrame, log_target: bool) -> pd.DataFrame:
    result_df = data[data[LABEL] > 0].copy()
    if len(result_df) == 0:
        raise ValueError("评估集中没有正的目标值，无法计算 MAPE！")

    predict_df = transform_target(result_df) if log_target else result_df
    y_pred = predictor.predict(predict_df)
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    if log_target:
        y_pred = np.expm1(y_pred)
        y_pred = np.clip(y_pred, 0, None)

    result_df["op_process_time_pred"] = y_pred.to_numpy()
    return result_df


def calculate_metrics(result_df: pd.DataFrame):
    y_true = pd.to_numeric(result_df[LABEL], errors="coerce")
    y_pred = pd.to_numeric(result_df["op_process_time_pred"], errors="coerce")
    valid_mask = y_true.notna() & y_pred.notna() & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    approx_accuracy = 100 - mape
    return len(y_true), rmse, mae, mape, approx_accuracy


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
    parser = argparse.ArgumentParser(description="Train cost prediction model.")
    parser.add_argument(
        "--ds_type",
        choices=["audio", "image", "text"],
        default=None,
        help="Only train/evaluate on one modality.",
    )
    parser.add_argument(
        "--log_target",
        action="store_true",
        help="Train on log1p(op_process_time) and restore predictions with expm1.",
    )
    parser.add_argument(
        "--data_path",
        default=DATA_PATH,
        help="Training CSV path. Defaults to output/data/dataset_header_for_cost_estimation.csv.",
    )
    parser.add_argument(
        "--model_suffix",
        default="",
        help="Extra suffix appended to model and output filenames, e.g. profile.",
    )
    args = parser.parse_args()
    ensure_output_dirs()
    args.data_path = resolve_legacy_aware_path(args.data_path, DATA_DIR)

    extra_suffix = args.model_suffix
    if extra_suffix and not extra_suffix.startswith("_"):
        extra_suffix = f"_{extra_suffix}"
    suffix = f"_{args.ds_type}" if args.ds_type else ""
    if args.log_target:
        suffix = f"{suffix}_log"
    suffix = f"{suffix}{extra_suffix}"
    model_save_dir = str(MODEL_DIR / f"cost_model{suffix}")
    feature_importance_path = str(FEATURE_IMPORTANCE_DIR / f"feature_importance_cost{suffix}.txt")
    validation_output_path = str(PREDICTIONS_DIR / f"validation_set_predictions_cost{suffix}.csv")
    prediction_output_path = str(PREDICTIONS_DIR / f"test_set_predictions_cost{suffix}.csv")

    # ==============================
    # 2. 加载数据并按 7:2:1 比例划分
    # ==============================
    # 加载 CSV 数据
    df = pd.read_csv(args.data_path, low_memory=False)

    # 验证目标列是否存在
    if LABEL not in df.columns:
        raise ValueError(f"目标列 '{LABEL}' 不存在！")

    if args.ds_type:
        if 'ds_type' not in df.columns:
            raise ValueError("数据集中缺少 'ds_type' 列，无法按模态过滤！")
        df = df[df['ds_type'] == args.ds_type].copy()
        if len(df) == 0:
            raise ValueError(f"过滤 ds_type='{args.ds_type}' 后没有可用数据！")
        print(f"按 ds_type='{args.ds_type}' 过滤后数据量：{len(df)}")
    else:
        print(f"原始数据量：{len(df)}")

    # 第一步：将数据划分为 70% 训练集和 30% 临时集（用于后续划分验证集和测试集）
    train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)

    # 第二步：将 30% 临时集再划分为 2/3 验证集（约占总数据的 20%）和 1/3 测试集（约占 10%）
    val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)  # 1/3 of 30% ≈ 10%

    # 输出各数据集的划分结果
    print("划分完成:")
    print(f"   - 训练集：{len(train_data)} ({len(train_data)/len(df):.1%})")
    print(f"   - 验证集：{len(val_data)} ({len(val_data)/len(df):.1%})")
    print(f"   - 测试集：{len(test_data)} ({len(test_data)/len(df):.1%})")

    train_fit = transform_target(train_data) if args.log_target else train_data
    val_fit = transform_target(val_data) if args.log_target else val_data
    if args.log_target:
        print("\n当前模式：训练目标为 log1p(op_process_time)，预测后使用 expm1 还原。")

    # ==============================
    # 3. 使用训练集训练 AutoGluon 模型
    # ==============================
    print("\n使用训练集训练 AutoGluon 模型...")
    predictor = TabularPredictor(
        label=LABEL,                            # 目标变量：op_process_time
        problem_type="regression",              # 问题类型：回归
        eval_metric="root_mean_squared_error",  # 评估指标：均方根误差
        path=model_save_dir                     # 模型保存路径
    )

    predictor.fit(
        train_data=train_fit,       # 训练集：用于拟合各个候选模型的参数。
        tuning_data=val_fit,        # 验证集：用于模型选择、早停和加权集成，不参与最终测试集评估。
        presets="medium_quality",   # AutoGluon 预设：在训练时间和模型效果之间做中等强度折中。
        verbosity=2,                # 日志级别：输出主要训练过程和每个模型的验证集分数。
    )

    fi = predictor.feature_importance(data=train_fit)
    with open(feature_importance_path, "w", encoding="utf-8") as f:
        f.write(fi.to_string())

    best_model = predictor.model_best
    print(f"\n最佳模型：{best_model}")

    print("\n在验证集上评估（用于模型选择和手动调参参考）...")
    val_result_df = predict_cost(predictor, val_data, args.log_target)
    val_metrics = calculate_metrics(val_result_df)
    print_metric_block(
        "验证集性能（用于模型选择和调参参考）",
        "验证样本数",
        val_metrics,
    )
    val_result_df.to_csv(validation_output_path, index=False)

    # ==============================
    # 4. 在独立测试集上评估模型性能（关键步骤：避免数据泄露）
    # ==============================
    print("\n在完全未见过的测试集上评估...")

    test_result_df = predict_cost(predictor, test_data, args.log_target)
    test_metrics = calculate_metrics(test_result_df)
    print_metric_block(
        "真实泛化性能（测试集，仅用于最终评估）",
        "测试样本数",
        test_metrics,
    )

    print(f"\n验证集预测结果已保存至：{validation_output_path}")
    test_result_df.to_csv(prediction_output_path, index=False)
    print(f"\n测试集预测结果已保存至：{prediction_output_path}")
    print(f"特征重要性已保存至：{feature_importance_path}")
    print(f"模型已保存至：{model_save_dir}")


if __name__ == "__main__":
    main()
