# T3 实验项目说明

> 面向新接手项目的开发者或 Agent，用于快速理解当前实验结构、主流程脚本和最新模型表现。

---

## 一、项目目标

T3（Time & Throughput Trading）方案：给定一条 Data-Juicer 数据处理 pipeline 配置，在**不实际完整运行用户 pipeline** 的情况下，预测：

1. `op_process_time`：每个算子的执行耗时，单位为秒。
2. `ds_output_count`：算子执行后的输出样本数。

当前实验分为两条线：

- **代价估计 cost estimation**：预测 `op_process_time`，再按 `pipeline_name` 聚合得到端到端 pipeline 预测时间。
- **基数估计 cardinality estimation**：预测 `ds_output_count`，用于后续在真实 pipeline 推理时逐算子传递输入/输出基数。

当前主结果：

- Cost：**dataset profile 特征 + `log1p(op_process_time)` 目标变换**。
- Cardinality：**dataset profile 特征 + `smooth_log_ratio`（α=0.1）目标变换**。

---

## 二、目录结构

```text
D:\Test-T3\
├─ assets/                                  # pipeline 配置定义：算子池、stage、参数范围
│  ├─ audio.json
│  ├─ image.json
│  └─ text.json
├─ collect_data/                            # 数据采集相关文件与历史采集结果
│  ├─ pipeline_yaml/                        # 本地生成后上传到服务器执行的 YAML
│  ├─ result_20260611/                      # 当前主要训练与评估使用的数据批次
│  ├─ runs_chunks/                          # chunk 外推实验运行结果
│  ├─ prepare_pipeline_runs_multiscale.py   # 服务器端多规模展开脚本
│  ├─ analyze_dataset_distribution.py       # 数据集画像统计脚本
│  ├─ analyze_chunk_dataset_distribution.py # chunk 数据画像统计脚本
│  ├─ dataset_stats_full.json               # profile 特征来源
│  ├─ chunk_profile_stats_*.json            # chunk 数据画像统计
│  └─ 104 数据采集实验环境配置.md / 105 图像实验环境.md
├─ output/                                  # 当前工作输出目录
│  ├─ data/                                 # 训练表、pipeline 回放大表、skipped 记录
│  │  ├─ dataset_header_for_cost_estimation*.csv
│  │  ├─ dataset_header_for_cost_estimation_with_pipeline*.csv
│  │  ├─ dataset_header_for_cardinality_estimation*.csv
│  │  └─ dataset_with_pipeline_skipped*.csv
│  ├─ predictions/                          # 验证集/测试集/全量 pipeline 预测明细
│  │  ├─ validation_set_predictions_*.csv
│  │  ├─ test_set_predictions_*.csv
│  │  └─ existing_model_fixed_testset_comparison/
│  ├─ summaries/                            # 汇总 CSV
│  │  ├─ pipeline_performance_from_csv*.csv
│  │  ├─ datatype_performance_summary*.csv
│  │  ├─ cardinality_performance_summary*.csv
│  │  └─ processed_op_performance*.csv
│  ├─ reports/                              # Markdown 汇总报告
│  │  ├─ 总体准确率汇总报告*.md
│  │  └─ 基数估计准确率汇总报告*.md
│  ├─ feature_importance/                   # AutoGluon 特征重要性
│  │  └─ feature_importance_*.txt
│  ├─ figures/                              # 图表
│  │  └─ operator_accuracy_distribution*.png
│  └─ AutogluonModels/                      # AutoGluon 模型目录（本地保留，不进 git）
├─ generator_audio_pipeline.py              # 生成 audio pipeline YAML
├─ generator_image_pipeline.py              # 生成 image pipeline YAML
├─ generator_text_pipeline.py               # 生成 text pipeline YAML
├─ update_assets.py                         # 根据数据分布重写 assets/*.json 的辅助脚本
├─ creation_datasets.py                     # 生成算子级 cost/cardinality 训练表
├─ creation_datasets_with_pipeline.py       # 生成 cost pipeline 级回放用大表
├─ add_dataset_profile_features.py          # 将 dataset_stats_full.json 画像特征加入大表
├─ prediction_cost.py                       # 训练 cost 模型
├─ prediction_cardinality.py                # 训练 cardinality 模型
├─ prediction_accracy_summary.py            # 汇总算子级 cost 预测分布
├─ project_paths.py                         # 统一管理 output 子目录和旧路径兼容
├─ compare_existing_models_fixed_testsets.py # 固定测试集比较总模型与分模态模型
├─ predict_full_dataset_with_existing_models.py # 使用已有 cost 模型做全量推理
├─ generate_pipeline_summary_from_csv.py    # 按 pipeline_name 汇总预测时间
├─ compute_overall_accuracy.py              # 汇总 cost 算子级与 pipeline 级准确率
├─ compute_cardinality_accuracy.py          # 汇总 cardinality 算子级准确率
├─ compute_cost_operator_accuracy.py        # 算子级 cost 准确率细分
├─ compute_chunk_operator_accuracy.py       # chunk 级算子准确率
├─ analyze_chunk_actual_runtime_gap.py      # chunk 实测耗时与完整执行差距分析
├─ analyze_chunk_runtime_scaling.py         # chunk 外推 vs 完整执行的耗时缩放分析
├─ op_database.csv                          # 算子参数数据库，只读参考
├─ requirements.txt
└─ readme.md
```

说明：

- `output/` 是当前工作目录，核心数据来自 `collect_data/result_20260611`。
- `output/` 已按文件类型拆分：训练表在 `output/data/`，预测明细在 `output/predictions/`，汇总 CSV 在 `output/summaries/`，Markdown 报告在 `output/reports/`，特征重要性在 `output/feature_importance/`，图表在 `output/figures/`。
- `output/AutogluonModels/` 不进 git（单个 `.pkl` 经常 >100MB）。需要时按本文第四节命令在本地重训。
- 核心脚本通过 `project_paths.py` 统一管理输出路径；多数读取逻辑会优先查找新目录，并对旧的 `output/*.csv` 历史路径做兼容兜底。
- `dataset_stats_full.json` 中的数据画像统计特征通过 `add_dataset_profile_features.py` 加入训练表，形成 `*_profile.csv`。
- `creation_datasets_with_pipeline.py` 只服务 cost 的端到端 pipeline 回放；基数估计当前只做算子级评估，不再生成 `dataset_header_for_cardinality_estimation_with_pipeline*.csv`。

---

## 三、完整工作流程

```text
[本地] 修改 assets/*.json
    |
[本地] python generator_*_pipeline.py --num_pipelines N
    |
[手动] 上传 collect_data/pipeline_yaml/ 到服务器 Data-Juicer 容器
    |
[服务器] prepare_pipeline_runs_multiscale.py 展开多规模数据运行
    |
[服务器] 批量执行 pipeline，保存日志、monitor、YAML
    |
[手动] 下载结果到 collect_data/result_YYYYMMDD/
    |
[本地] creation_datasets.py 生成算子级训练表
    |
[本地] add_dataset_profile_features.py 加入 dataset profile 特征
    |
[本地] prediction_cost.py / prediction_cardinality.py 训练模型
    |
[本地] predict_full_dataset_with_existing_models.py + generate_pipeline_summary_from_csv.py + compute_overall_accuracy.py
```

---

## 四、各脚本说明

> 本节脚本命令统一对齐当前主方案：
> - Cost：`profile + log1p(op_process_time)`
> - Cardinality：`profile + smooth_log_ratio(α=0.1)`
>
> 其它历史方案（profile-only、原始 ratio、`log1p(ratio)`、`log1p(ds_output_count)` 等）的对比结果保留在第七节。

### 4.1 Pipeline 生成与资产配置

| 脚本 | 作用 |
|---|---|
| `generator_audio_pipeline.py` | 根据 `assets/audio.json` 生成 audio pipeline YAML。 |
| `generator_image_pipeline.py` | 根据 `assets/image.json` 生成 image pipeline YAML。 |
| `generator_text_pipeline.py` | 根据 `assets/text.json` 生成 text pipeline YAML。 |
| `update_assets.py` | 根据前期数据分布统计重写 `assets/*.json`，用于调整算子参数范围。 |

示例：

```powershell
python generator_audio_pipeline.py --num_pipelines 30 --json_path .\assets\audio.json --output_dir .\collect_data\pipeline_yaml
python generator_image_pipeline.py --num_pipelines 40 --json_path .\assets\image.json --output_dir .\collect_data\pipeline_yaml
python generator_text_pipeline.py  --num_pipelines 50 --json_path .\assets\text.json  --output_dir .\collect_data\pipeline_yaml
```

生成逻辑要点：

- 连续参数使用 LHS（Latin Hypercube Sampling）覆盖参数空间。
- 离散参数从候选集合中随机选择。
- `stage` 决定执行顺序。
- text 生成器显式按 `(stage, layer)` 做互斥抽样，audio/image 当前主要按 stage 池抽取。

### 4.2 特征表生成

```powershell
python creation_datasets.py
python creation_datasets_with_pipeline.py --log_target
```

| 脚本 | 当前用途 | 主要输出 |
|---|---|---|
| `creation_datasets.py` | 从采集结果中提取算子级特征，生成 cost/cardinality 训练表。 | `output/data/dataset_header_for_cost_estimation.csv`、`output/data/dataset_header_for_cardinality_estimation.csv` |
| `creation_datasets_with_pipeline.py --log_target` | 额外保留 `pipeline_name / pipeline_base_name / pipeline_scale_token / operator_index / operator_name`，用于 cost 端到端回放，并生成 `_log` 后缀文件名。 | `output/data/dataset_header_for_cost_estimation_with_pipeline_log.csv` |

注意：

- `creation_datasets_with_pipeline.py --log_target` 只是生成带 `_log` 后缀的文件名，方便后续命令链区分实验结果；它不会在这里对目标值做 log 变换。
- 基数估计目前不做 pipeline 级回放，因此不生成 cardinality with-pipeline 大表。

### 4.3 数据画像特征增强

`add_dataset_profile_features.py` 读取 `collect_data/dataset_stats_full.json`，将数据集画像统计特征加入训练大表。新增列统一以 `profile_` 开头。

当前主方案需要生成下面三张带 profile 的表：

```powershell
python add_dataset_profile_features.py --input .\output\data\dataset_header_for_cost_estimation.csv --output .\output\data\dataset_header_for_cost_estimation_profile.csv
python add_dataset_profile_features.py --input .\output\data\dataset_header_for_cardinality_estimation.csv --output .\output\data\dataset_header_for_cardinality_estimation_profile.csv
python add_dataset_profile_features.py --input .\output\data\dataset_header_for_cost_estimation_with_pipeline_log.csv --output .\output\data\dataset_header_for_cost_estimation_with_pipeline_log_profile.csv
```

`profile_` 字段示例：

- audio：`profile_audio_duration_sec_mean`、`profile_audio_sample_rate_hz_mean`、`profile_audio_channels_mean`
- image：`profile_image_width_px_mean`、`profile_image_height_px_p95`、`profile_image_file_size_kb_mean`
- text：`profile_text_char_length_mean`、`profile_text_word_count_p95`、`profile_text_token_count_approx_chars_div_4_p50`

### 4.4 Cost 模型训练（profile + log1p）

主方案训练目标为 `log1p(op_process_time)`，特征表为 `dataset_header_for_cost_estimation_profile.csv`：

```powershell
python prediction_cost.py --data_path .\output\data\dataset_header_for_cost_estimation_profile.csv --log_target
python prediction_cost.py --ds_type audio --data_path .\output\data\dataset_header_for_cost_estimation_profile.csv --log_target
python prediction_cost.py --ds_type image --data_path .\output\data\dataset_header_for_cost_estimation_profile.csv --log_target
python prediction_cost.py --ds_type text  --data_path .\output\data\dataset_header_for_cost_estimation_profile.csv --log_target
```

参数说明：

- `--data_path`：指定训练表。
- `--ds_type audio|image|text`：只训练单一模态模型。
- `--log_target`：训练 `log1p(op_process_time)`，预测后用 `expm1` 还原。
- `--model_suffix`：给模型目录和输出文件追加额外后缀。

训练脚本会同时输出验证集和测试集预测结果：

- `output/predictions/validation_set_predictions_cost*.csv`：用于模型选择和参数探索时查看。
- `output/predictions/test_set_predictions_cost*.csv`：用于最终测试集评估。

模型目录位于 `output/AutogluonModels/`，不进 git，需要时通过上面命令本地重训。

### 4.5 Cost 端到端 pipeline 回放（profile + log1p）

```powershell
python creation_datasets_with_pipeline.py --log_target
python add_dataset_profile_features.py --input .\output\data\dataset_header_for_cost_estimation_with_pipeline_log.csv --output .\output\data\dataset_header_for_cost_estimation_with_pipeline_log_profile.csv
python predict_full_dataset_with_existing_models.py --data_path .\output\data\dataset_header_for_cost_estimation_with_pipeline_log_profile.csv --log_target --suffix profile
python generate_pipeline_summary_from_csv.py --log_target --suffix profile
python compute_overall_accuracy.py --log_target --suffix profile
```

输出：

- `output/predictions/test_set_predictions_cost_full_with_pipeline_log_profile.csv`
- `output/summaries/pipeline_performance_from_csv_log_profile.csv`
- `output/summaries/datatype_performance_summary_log_profile.csv`
- `output/reports/总体准确率汇总报告_log_profile.md`

### 4.6 Cardinality 模型训练与评估（smooth_log_ratio, α=0.1）

主方案训练目标为：

```text
cardinality_smooth_log_ratio = log((ds_output_count + alpha) / (ds_input_count + alpha))
```

其中 α 由 `--ratio_alpha` 指定，主结果取 `alpha=0.1`。预测后还原：

```text
ds_output_count_pred = exp(pred) * (ds_input_count + alpha) - alpha
```

再裁剪到不小于 0。注意：`smooth_log_ratio` 本身已经是 log 目标，**不要**再叠加 `--log_target`。

只训练一个总体模型，再按模态切分评估：

```powershell
python prediction_cardinality.py --target_mode smooth_log_ratio --ratio_alpha 0.1 --model_suffix alpha0p1 --data_path .\output\data\dataset_header_for_cardinality_estimation_profile.csv
python compute_cardinality_accuracy.py --target_mode smooth_log_ratio --input_suffix alpha0p1 --single_model
```

`--single_model` 不会重新训练模型，也不会读取分模态预测文件；它只读取总体预测文件，再根据 `ds_type` 过滤统计各模态指标。

输出：

- `output/predictions/validation_set_predictions_cardinality_smooth_log_ratio_alpha0p1.csv`
- `output/predictions/test_set_predictions_cardinality_smooth_log_ratio_alpha0p1.csv`
- `output/summaries/cardinality_performance_summary_smooth_log_ratio_alpha0p1_single_model.csv`
- `output/reports/基数估计准确率汇总报告_smooth_log_ratio_alpha0p1_single_model.md`

说明：

- 训练脚本会同时打印并保存验证集和测试集预测结果。
- 验证集结果用于模型选择、`ratio_alpha` 等参数的探索。
- 测试集结果只用于最终评估，不建议反复根据测试集结果调整方案。

### 4.7 辅助分析脚本

| 脚本 | 作用 |
|---|---|
| `prediction_accracy_summary.py` | 基于 `test_set_predictions_cost*.csv` 汇总算子预测准确率分布并画图。 |
| `compare_existing_models_fixed_testsets.py` | 在同一批固定测试样本上比较总模型与分模态模型。 |
| `predict_full_dataset_with_existing_models.py` | 加载已有 cost 模型，对带 pipeline 信息的大表做全量推理。 |
| `generate_pipeline_summary_from_csv.py` | 按 `pipeline_name` 聚合真实耗时与预测耗时。 |
| `compute_overall_accuracy.py` | 汇总 cost 算子级与 pipeline 级 RMSE / MAE / MAPE / 准确率；支持 `--single_model` 从总体预测文件按 `ds_type` 切分统计分模态指标。 |
| `compute_cardinality_accuracy.py` | 汇总基数估计测试集算子级 RMSE / MAE / MAPE / 准确率；支持 `--target_mode smooth_log_ratio` 和 `--single_model`。 |
| `compute_cost_operator_accuracy.py` | 算子级 cost 准确率细分到 `operator_name`。 |
| `compute_chunk_operator_accuracy.py` | chunk 级算子准确率统计。 |
| `analyze_chunk_actual_runtime_gap.py` | 比较 chunk 实测求和耗时与完整执行真实耗时之间的差距。 |
| `analyze_chunk_runtime_scaling.py` | chunk 预测求和与完整执行真实耗时之间的耗时缩放分析。 |

---

## 五、assets/*.json 配置结构

```json
{
  "stages": [
    {
      "id": 0,
      "description": "阶段描述",
      "min_operator_count": 1,
      "max_operator_count": 3,
      "operator_total": 5
    }
  ],
  "operators": [
    {
      "name": "operator_name",
      "type": "filter | mapper | deduplicator",
      "stage": 0,
      "layer": 0,
      "params": [
        {
          "name": "param_name",
          "type": "continuous | categorical",
          "range": [0, 1],
          "values": [1, 2, 3]
        }
      ]
    }
  ]
}
```

关键约束：

- `stage` 决定执行顺序，值小的先执行。
- `layer` 用于表示互斥分组。
- 连续参数可带 `gt` 约束，用于保证一个参数严格大于另一个参数。

---

## 六、实验输出时间线

| 轮次 | 输出目录 | 时间 | 主要调整 / 特点 | 代表性结果 |
|---|---|---|---|---|
| 第 1 轮 | `output_20260409` | 2026-04-09 | 初始基线；统一训练。 | 算子级总体约 77.72%，pipeline 级总体约 79.53%。 |
| 第 2 轮 | `output_20260422` | 2026-04-22 | 扩充大批量样本后重训；短时算子占比偏高。 | 算子级总体约 69.52%，pipeline 级总体约 79.24%。 |
| 第 3 轮 | `output_20260424` | 2026-04-24 | 引入 LHS 参数采样，降低 `ds_type` 依赖。 | 算子级总体约 56.65%，pipeline 级总体约 65.10%。 |
| 第 4 轮 | `output_20260425` | 2026-04-25 | 修正部分特征表达，开始分模态分析。 | 算子级总体约 60.88%，pipeline 级总体约 66.65%。 |
| 第 5 轮 | `output_20260428` | 2026-04-28 | 不加 log、不加 profile 的历史主结果；包含全量、分模态、固定测试集和混合模式分析。 | 混合模式 pipeline 级总体达到 95.10%。 |
| 第 6 轮 | `output_20260429` | 2026-04-29 | 首次使用 `log1p(op_process_time)`。 | 算子级总体 92.33%，pipeline 级总体 97.69%。 |
| 第 7 轮 | `output` | 2026-05-11 | 加入 dataset profile 特征。 | profile-only：算子级总体 82.47%，pipeline 级总体 90.67%。 |
| 第 8 轮 | `output` | 2026-05-27 | 使用 `profile + log1p(op_process_time)`。 | 当前主结果：算子级总体 91.66%，pipeline 级总体 97.98%。 |
| 第 9 轮 | `output` | 2026-05-14 至 2026-05-23 | 基数估计使用 `profile + log1p(ds_output_count)`。 | 基数估计总体准确率 91.08%。 |
| 第 10 轮 | `output` | 2026-06-05 | 基数估计尝试 `smooth_log_ratio`。 | α=0.1 时总体准确率 93.90%，图像模态 91.07%。 |

说明：

- `output_20*` 历史快照仅作为对照保留在本地，未推到 git。
- 当前工作目录 `output/` 中的主结果以 cost = `profile + log1p`、cardinality = `profile + smooth_log_ratio(α=0.1)` 为准。

---

## 七、当前模型表现

### 7.1 当前主结果：Cost estimation，`profile + log1p(op_process_time)`

数据来源：

- 训练表：`output/data/dataset_header_for_cost_estimation_profile.csv`
- pipeline 回放表：`output/data/dataset_header_for_cost_estimation_with_pipeline_log_profile.csv`
- 汇总报告：`output/reports/总体准确率汇总报告_log_profile.md`

| 数据范围 | 算子样本 | pipeline数 | RMSE(秒) | MAE(秒) | MAPE | 算子级准确率 | pipeline级准确率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 总和采集数据 | 784 | 950 | 14.68 | 2.36 | 8.34% | 91.66% | 97.98% |
| 音频采集数据 | 104 | 300 | 38.73 | 12.32 | 9.84% | 90.16% | 98.14% |
| 图像采集数据 | 277 | 400 | 23.00 | 2.72 | 9.99% | 90.01% | 97.72% |
| 文本采集数据 | 403 | 250 | 0.57 | 0.24 | 7.71% | 92.29% | 98.21% |

结论：

- `profile + log1p` 是当前最推荐保留的 cost 方案。
- 相比 profile-only，`log1p` 对 MAPE 和准确率改善更明显。
- 三个模态的算子级准确率都达到约 90%，pipeline 级准确率接近 98%。

### 7.2 对照结果：Profile-only cost estimation

数据来源：`output/reports/总体准确率汇总报告_profile.md`

| 数据范围 | 算子样本 | pipeline数 | RMSE(秒) | MAE(秒) | MAPE | 算子级准确率 | pipeline级准确率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 总和采集数据 | 784 | 950 | 15.88 | 2.29 | 17.53% | 82.47% | 90.67% |
| 音频采集数据 | 104 | 300 | 20.58 | 8.78 | 38.35% | 61.65% | 91.25% |
| 图像采集数据 | 277 | 400 | 16.95 | 3.91 | 38.23% | 61.77% | 85.71% |
| 文本采集数据 | 403 | 250 | 0.51 | 0.25 | 9.60% | 90.40% | 97.90% |

对比分析：

- profile-only 已经把 pipeline 级总体准确率推到 90% 以上，但音频和图像的算子级 MAPE 仍然较高。
- 加入 `log1p(op_process_time)` 后，总体 MAPE 从 17.53% 降到 8.34%，算子级准确率从 82.47% 提升到 91.66%。
- 因此当前提升的主要来源是目标变换，profile 特征更多是补充数据集画像信息。

### 7.3 Cardinality estimation，`profile + smooth_log_ratio`

当前基数估计推荐结果使用 `smooth_log_ratio` 目标，α=0.1：

```text
log((ds_output_count + alpha) / (ds_input_count + alpha))
```

该方案仍然保留 ratio 建模对输入规模的归一化作用，同时比 `log1p(ratio)` 更适合处理极小 ratio 和输出为 0 的强过滤场景。

数据来源：`output/reports/基数估计准确率汇总报告_smooth_log_ratio_alpha0p1_single_model.md`

| 数据范围 | 样本数 | RMSE | MAE | MAPE | 基数预测准确率 | ratio_RMSE | ratio_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 总和采集数据 | 630 | 760.35 | 157.26 | 6.10% | 93.90% | 0.0935 | 0.0316 |
| 音频采集数据 | 123 | 345.19 | 116.07 | 3.55% | 96.45% | 0.0458 | 0.0233 |
| 图像采集数据 | 183 | 1328.50 | 309.98 | 8.93% | 91.07% | 0.1030 | 0.0376 |
| 文本采集数据 | 324 | 286.49 | 86.63 | 5.47% | 94.53% | 0.1003 | 0.0312 |

对照：`smooth_log_ratio` 默认 α=1.0

数据来源：`output/reports/基数估计准确率汇总报告_smooth_log_ratio_single_model.md`

| 数据范围 | 样本数 | RMSE | MAE | MAPE | 基数预测准确率 | ratio_RMSE | ratio_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 总和采集数据 | 630 | 773.57 | 164.69 | 6.54% | 93.46% | 0.0879 | 0.0312 |
| 音频采集数据 | 123 | 417.77 | 136.87 | 4.13% | 95.87% | 0.0537 | 0.0274 |
| 图像采集数据 | 183 | 1347.26 | 339.39 | 9.69% | 90.31% | 0.0982 | 0.0386 |
| 文本采集数据 | 324 | 268.55 | 76.59 | 5.68% | 94.32% | 0.0914 | 0.0282 |

对照：`profile + log1p(ds_output_count)`

数据来源：`output/reports/基数估计准确率汇总报告_log.md`

| 数据范围 | 样本数 | RMSE | MAE | MAPE | 基数预测准确率 |
|---|---:|---:|---:|---:|---:|
| 总和采集数据 | 630 | 1070.53 | 257.18 | 8.92% | 91.08% |
| 音频采集数据 | 104 | 628.81 | 261.14 | 5.26% | 94.74% |
| 图像采集数据 | 193 | 1159.55 | 405.21 | 14.19% | 85.81% |
| 文本采集数据 | 322 | 321.35 | 145.96 | 10.46% | 89.54% |

目标方案对比：

| 方案 | 总体 RMSE | 总体 MAE | 总体 MAPE | 总体准确率 | 主要观察 |
|---|---:|---:|---:|---:|---|
| `ratio` | 547.32 | 101.95 | 29.49% | 70.51% | RMSE/MAE 较低，但图像强过滤算子导致 MAPE 很差。 |
| `log1p(ratio)` | 312.23 | 73.03 | 16.58% | 83.42% | 进一步降低 RMSE/MAE，但仍无法处理极小 ratio 的相对误差。 |
| `log1p(ds_output_count)` | 1070.53 | 257.18 | 8.92% | 91.08% | 长尾问题得到缓解，是此前基数估计主方案。 |
| `smooth_log_ratio, alpha=1.0` | 773.57 | 164.69 | 6.54% | 93.46% | 与 alpha=0.1 接近，部分 ratio 侧指标略稳。 |
| `smooth_log_ratio, alpha=0.1` | 760.35 | 157.26 | 6.10% | 93.90% | 输出基数 MAPE/准确率略优，图像模态提升到 91.07%。**当前主结果。** |

结论：

- `smooth_log_ratio, α=0.1` 相比 `log1p(ds_output_count)` 进一步降低了总体 MAPE，总体准确率从 91.08% 提升到 93.90%。
- α=1.0 与 α=0.1 的结果差距不大，说明当前方案对平滑项不算特别敏感；其中 α=0.1 在输出基数的总体 MAPE、MAE 和准确率上略优。
- 分模态来看，α=0.1 在音频和图像上提升更明确，文本的 MAPE/准确率略优但 RMSE/MAE 略差，因此不能简单认为 α 越小越好。
- ratio 侧指标与最终输出基数指标不完全一致，例如 α=1.0 的总体 ratio_RMSE 略低，但最终评估仍以还原后的 `ds_output_count` 误差为主。
- 直接预测 `ratio` 或 `log1p(ratio)` 虽然 RMSE/MAE 较低，但 MAPE 明显受图像强过滤算子影响，不适合作为当前主结果。
- 因此当前基数估计保留 `smooth_log_ratio, α=0.1` 作为主方案，同时记录 α=1.0 作为敏感性对照。

### 7.4 为什么使用 `log1p` / `smooth_log_ratio`

`op_process_time` 和 `ds_output_count` 都是非负、长尾分布明显的目标。直接预测原始值时，模型容易被少量极大值主导；使用对数类变换后：

- 大值被压缩，中小样本的学习权重相对提高。
- 模型更关注相对变化关系，有利于降低 MAPE。
- `log1p(x) = log(1 + x)` 可以处理接近 0 的值，比直接 `log(x)` 更稳。
- `smooth_log_ratio` 在 ratio 上叠加平滑项 α，避免极小 ratio 或输出为 0 的强过滤算子在还原阶段被放大相对误差。

预测完成后通过 `expm1(pred)`（cost）或 `exp(pred) * (ds_input_count + α) - α`（cardinality）还原到原始尺度。

### 7.5 当前仍需注意的问题

- 当前主结果主要说明在 `result_20260611` 这批数据上效果稳定，不能直接等价于跨批次泛化已经解决。
- pipeline 级准确率较高时，仍可能存在算子级误差互相抵消的情况，因此仍需要关注算子级分模态误差。
- 基数估计在真实 pipeline 中会逐算子传递，前序算子的输出基数误差会影响后续算子的输入基数，因此后续仍需要继续优化稳定性。

---

## 八、本地环境说明

- 操作系统：Windows 11 Home
- Python 环境：Conda 环境 `t3-ag` 或 `T3-ag`
- 工作目录：`D:\Test-T3\`
- 依赖安装：`pip install -r requirements.txt`
- 主要依赖：`PyYAML`、`pandas`、`numpy`、`scikit-learn`、`autogluon`、`matplotlib`

---

## 九、服务器 Docker 环境摘要

详见 `collect_data/104 数据采集实验环境配置.md` 和 `collect_data/105 图像实验环境.md`。要点如下：

- 容器：`yyw_dj_test`（104 文本/音频）、`dj_lab`（105 图像）
- 镜像：`xy_ray_env:v1`，Ubuntu 22.04 + Ray 2.51.1
- GPU：RTX A6000 48GB
- 工作目录：`/home/yyw/data-juicer`（104）、`/data-juicer`（105）
- 离线模式：`HF_HUB_OFFLINE=1`
- 模型缓存挂载：`/data/data-juicer_models`
- 已知风险：
  - `open_tracer=true` 时，`text_chunk_mapper` / `image_segment_mapper` 易触发 tracer 断言错误。
  - 多人共用 GPU 时，`image_segment_mapper` 可能 OOM。
  - `document_minhash_deduplicator` 偶现空数组报错。
