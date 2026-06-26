"""
数据处理管道特征提取脚本

此脚本用于从数据处理管道的日志、监控和配置文件中提取特征，
用于训练机器学习模型来预测算子执行时间和输出数据量。

主要功能：
- 解析数据集目录结构
- 提取算子类型和参数
- 解析执行日志获取时间和数据量信息
- 解析监控文件获取环境资源信息
- 生成用于基数估计和成本估计的CSV数据集 (在 OUTPUT_PATH 目录输出两个csv文件)
"""

import json
import os
import re
import yaml
import pandas as pd
import numpy as np

from project_paths import DATA_DIR, ensure_output_dirs


# 生成的训练数据集是否包含参数
INCLUDE_PARAMS = True

# 数据集原始目录路径
BASE_PATH = './collect_data/result_20260428'

# 输出目录路径
OUTPUT_PATH = str(DATA_DIR)

# 过滤数据（数据名字中包含以下字符串，则加载数据）

filter_include_list = [
    # 图像数据
    'image-pipeline', 'image_pipeline',
    # 文本数据
    'text-pipeline', 'text_pipeline',
    # 音频数据
    'audio-pipeline', 'audio_pipeline'
]


def infer_ds_type(data_name: str, operator: dict | None = None):
    """
    根据 pipeline 名称推断数据类型，必要时回退到算子名。
    """
    data_name_lower = str(data_name).lower()
    if 'text' in data_name_lower:
        return 'text'
    if 'image' in data_name_lower:
        return 'image'
    if 'audio' in data_name_lower:
        return 'audio'

    if operator:
        op_name = next(iter(operator.keys())).lower()
        if op_name.startswith('text_') or op_name.startswith('document_'):
            return 'text'
        if op_name.startswith('image_'):
            return 'image'
        if op_name.startswith('audio_'):
            return 'audio'

    return 'unknown'


def extract_scale_features(data_name: str, ds_type: str):
    """
    从 data_name 最后一个 '_' 后面的 token 提取数据规模特征。

    示例:
    - audio_pipeline_xxx_128MB -> ds_scale_label=128MB, ds_scale_mb=128
    - image_pipeline_xxx_1.5G  -> ds_scale_label=1.5G, ds_scale_mb=1536
    - text_pipeline_xxx_5000   -> ds_scale_label=5000, ds_scale_records=5000
    """
    scale_label = None
    ds_scale_mb = np.nan
    ds_scale_records = np.nan

    match = re.search(r'_([^_]+)$', str(data_name))
    if match:
        scale_label = match.group(1)

    if scale_label:
        label_upper = scale_label.upper()
        if ds_type in ('audio', 'image'):
            mb_match = re.fullmatch(r'(\d+(?:\.\d+)?)MB', label_upper)
            g_match = re.fullmatch(r'(\d+(?:\.\d+)?)G', label_upper)
            if mb_match:
                ds_scale_mb = float(mb_match.group(1))
            elif g_match:
                ds_scale_mb = float(g_match.group(1)) * 1024.0
        elif ds_type == 'text':
            if re.fullmatch(r'\d+', scale_label):
                ds_scale_records = float(scale_label)

    return {
        'ds_scale_label': scale_label,
        'ds_scale_mb': ds_scale_mb,
        'ds_scale_records': ds_scale_records,
    }


def get_datasets():
    '''
    获取 BASE_PATH 下的所有数据集的信息

    返回值格式：
    [
        {
            'data_name': '数据集名称',
            'log_file_path': '日志文件路径',
            'monitor_file_path': '监控文件路径',
            'pipeline_file_path': '管道配置文件路径',
        },
        ...
    ]

    该函数遍历dataset_original目录下的所有子目录，查找YAML配置文件、
    日志文件和监控文件，为每个数据集构建信息字典。
    '''
    result = []
    # 遍历 BASE_PATH 下的 yaml 文件
    for root, dirs, files in os.walk(BASE_PATH):
        for file in files:
            if file.endswith('.yaml'):
                # 只去掉最终的 .yaml 后缀，保留 1.5G 这类带小数点的规模标记
                data_name = os.path.splitext(file)[0]
                # 获取 log 文件路径
                log_file_path = os.path.join(root, 'log')
                for root2, dirs2, files2 in os.walk(log_file_path):
                    for log_file in files2:
                        if 'DEBUG' not in log_file and 'ERROR' not in log_file and 'WARNING' not in log_file:
                            log_file_path = os.path.join(root2, log_file)
                # 获取 monitor 文件路径
                monitor_dir = os.path.join(root, 'monitor')
                monitor_file_path = os.path.join(monitor_dir, 'monitor.json')
                # 兼容 <pipeline_name>_monitor.json 命名
                if not os.path.exists(monitor_file_path):
                    root_name = os.path.basename(os.path.normpath(root))
                    alt_monitor_file_path = os.path.join(monitor_dir, f"{root_name}_monitor.json")
                    if os.path.exists(alt_monitor_file_path):
                        monitor_file_path = alt_monitor_file_path
                    elif os.path.isdir(monitor_dir):
                        monitor_candidates = [f for f in os.listdir(monitor_dir) if f.endswith('_monitor.json')]
                        if monitor_candidates:
                            monitor_file_path = os.path.join(monitor_dir, monitor_candidates[0])
                # 获取 pipeline 文件路径
                pipeline_file_path = os.path.join(root, file)
                # 添加数据集信息
                result.append({
                    'data_name': data_name,
                    'log_file_path': log_file_path,
                    'monitor_file_path': monitor_file_path,
                    'pipeline_file_path': pipeline_file_path,
                })
    return result

# 解析log中的txt文件，获取op_process_time和ds_output_count
def parse_log(log_file_path: str):
    '''
    解析日志文件，提取算子执行信息

    参数：
    log_file_path (str): 日志文件的路径

    返回值：
    list: 包含每个算子执行信息的字典列表
    [
        {
            'op_name': '算子名称',
            'op_process_time': '执行时间（秒）',
            'ds_output_count': '输出数据样本数量',
            'ds_input_count': '输入数据样本数量'
        },
        ...
    ]

    解析逻辑：
    - 匹配包含"] OP ["的行，提取算子名称、执行时间和剩余样本数
    - 提取原始数据集的样本数量作为初始输入
    - 按顺序记录每个算子的输入输出信息
    '''
    ds_input_count = 0

    operator_list = []
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_list = f.readlines()
        for log_line in log_list:
            # 获取一开始输入数据数量
            # There are 12677 sample(s) in the original dataset
            if 'in the original dataset' in log_line:
                pattern = r"There are (.*?) sample\(s\) in the original dataset"
                log_match = re.search(pattern, log_line)
                ds_input_count += int(log_match.group(1))
                continue
            # 获取算子信息
            if '] OP [' in log_line:
                pattern = r"OP \[(.*?)\] Done in (.*?)s. Left (.*?) samples."
                log_match = re.search(pattern, log_line)
                operator_dict = {
                    'op_name': log_match.group(1),
                    'op_process_time': log_match.group(2),
                    'ds_output_count': log_match.group(3),
                    'ds_input_count': ds_input_count,
                }
                operator_list.append(operator_dict)
                ds_input_count = log_match.group(3)
    return operator_list


# 解析monitor.json文件，获取每个算子执行时刻的环境信息
def parse_monitor(monitor_file_path: str):
    '''
    解析监控文件，提取环境资源使用信息

    参数：
    monitor_file_path (str): 监控JSON文件的路径

    返回值：
    list: 包含每个算子执行时环境信息的字典列表
    [
        {
            'env_cpu_count': 'CPU核心数量',
            'env_cpu_util': 'CPU使用率',
            'env_mem_util': '内存使用率',
            'env_gpu_util': 'GPU使用率（当前为空）',
            'env_gpu_mem_util': 'GPU内存使用率（当前为空）'
        },
        ...
    ]

    注意：当前实现中GPU相关字段为空，可能是因为监控数据中未包含GPU信息
    '''
    operator_env_list = []
    monitor_info_list = json.load(open(monitor_file_path, 'r', encoding='utf-8'))
    for monitor_info in monitor_info_list:
        env_info = monitor_info["resource"][0]

        # 改进版本（避免除零和冗余判断）
        gpu_util_list = env_info.get('GPU util.', [0.0])
        gpu_used_mem_list = env_info.get('GPU used mem.', [0])
        gpu_total_mem_list = env_info.get('GPU total mem.', [1])
        
        env_dict = {
            'env_cpu_count': env_info['CPU count'],
            'env_cpu_util': env_info['CPU util.'],
            'env_mem_util': env_info['Mem. util.'],
            'env_gpu_util': gpu_util_list[0],
            'env_gpu_mem_util': gpu_used_mem_list[0] / max(gpu_total_mem_list[0], 1) if gpu_total_mem_list[0] > 0 else 0.0,
            
            # ''' BUG 空值注意
            # 'env_gpu_util': '',
            # 'env_gpu_mem_util': '',
        }
        operator_env_list.append(env_dict)
    return operator_env_list


if __name__ == '__main__':
    # 获取所有数据集的信息
    org_datasets = get_datasets()

    datasets = []

    # 过滤数据集实现
    for data_info in org_datasets:
        for include_name in filter_include_list:
            if include_name in data_info['monitor_file_path']:
                datasets.append(data_info)
                continue

    # 测试用的算子列表（用于特征选择）
    # test = [
    #     'maximum_line_length_filter',
    #     'language_id_score_filter',
    #     'clean_email_mapper',
    #     'word_repetition_filter',
    #     'words_num_filter',
    #     'fix_unicode_mapper',
    #     'clean_links_mapper'
    # ]

    '''
    获取pipeline中的算子和参数 (保持唯一性)
    算子名 op_type_<op_name>
    参数名 op_<op_name>_<op_param_name>

    输入示例：
        {'clean_email_mapper': None}
        {'alphanumeric_filter': {'tokenization': False, 'min_ratio': 0.5, 'max_ratio': 0.8}}

    输出示例：
        op_type_clean_email_mapper, 
        op_type_alphanumeric_filter, op_alphanumeric_filter_tokenization, op_alphanumeric_filter_min_ratio, op_alphanumeric_filter_max_ratio
    '''
    # 收集所有算子特征
    operator_set = set()
    for dataset in datasets:
        pipeline_file = dataset['pipeline_file_path']
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            pipeline_dict = yaml.safe_load(f)
            operator_list = pipeline_dict['process']
            for operator in operator_list:
                for op_name, op_params in operator.items():
                    # if op_name in test:
                    operator_set.add(f'op_type_{op_name}')
                    # 对照实验，不需要算子参数的情况，请设置为 False
                    if INCLUDE_PARAMS:
                        if isinstance(op_params, dict):
                            for param_name, param_value in op_params.items():
                                operator_set.add(f'op_{op_name}_{param_name}')

    '''
    生成数据集CSV头部
    '''
    # 数据集相关特征
    data_set = set()
    data_set.add('ds_input_count')  # 输入数据样本数量
    data_set.add('ds_output_count')  # 输出数据样本数量（预测目标）
    data_set.add('ds_type')  # 数据集类型（当前为空）
    data_set.add('ds_scale_label')  # 原始规模标签，如 128MB / 1.5G / 5000
    data_set.add('ds_scale_mb')  # 音频/图像统一换算为 MB
    data_set.add('ds_scale_records')  # 文本统一换算为记录数

    # 算子相关特征
    operator_set.add("op_process_time")  # 算子执行时间（预测目标）
    operator_set.add('op_process_number')  # 并发处理数量

    # 环境相关特征
    env_set = set()
    env_set.add('env_cpu_count')  # CPU核心数量
    env_set.add('env_cpu_util')  # CPU使用率
    env_set.add('env_mem_util')  # 内存使用率
    env_set.add('env_gpu_util')  # GPU使用率
    env_set.add('env_gpu_mem_util')  # GPU内存使用率

    # 移除无关的字段，比如数据保存路径（如果存在）PS: 这里可以改为一个 remove_list 列表
    if 'op_audio_add_gaussian_noise_mapper_noise_level' in operator_set:
        operator_set.remove('op_audio_add_gaussian_noise_mapper_noise_level')
    if 'op_audio_ffmpeg_wrapped_mapper_save_dir' in operator_set:
        operator_set.remove('op_audio_ffmpeg_wrapped_mapper_save_dir')
    if 'op_audio_add_gaussian_noise_mapper_save_dir' in operator_set:
        operator_set.remove('op_audio_add_gaussian_noise_mapper_save_dir')
    if 'op_image_blur_mapper_save_dir' in operator_set:
        operator_set.remove('op_image_blur_mapper_save_dir')
    if 'op_image_face_blur_mapper_save_dir' in operator_set:
        operator_set.remove('op_image_face_blur_mapper_save_dir')
    if 'op_image_remove_background_mapper_save_dir' in operator_set:
        operator_set.remove('op_image_remove_background_mapper_save_dir')

    # 合并所有特征并排序
    dataset_header = sorted(data_set | operator_set | env_set)

    # 用于预测算子输出数据量模型的特征头部（移除执行时间）
    dataset_header_for_cardinality_estimation = dataset_header.copy()
    dataset_header_for_cardinality_estimation.remove('op_process_time')

    # 用于预测算子执行时间模型的特征头部（移除输出数据量）
    dataset_header_for_cost_estimation = dataset_header.copy()
    dataset_header_for_cost_estimation.remove('ds_output_count')

    # 示例解析
    # parse_log(f'{BASE_PATH}/000003/log/export_result.jsonl_time_20260126040701.txt')
    # parse_monitor(f'{BASE_PATH}/000003/monitor/monitor.json')

    '''
    正式开始数据转换

    转换思路：
    遍历pipeline中的算子，设置index，与遍历monitor.json中的index对应
    （目标是获取第一次采集的环境特征），同时，对应log中的index
    （目标是获取执行时间和输出数据量）
    '''
    data_rows = []  # 存储所有数据行的列表
    data_index = 0  # 当前处理的数据集索引
    data_len = len(datasets)  # 总数据集数量

    # 遍历所有数据集
    for dataset in datasets:
        # 只处理前N%的数据（当前设置为100%）
        if data_index >= int(data_len * 1.0):
            break
        data_index += 1
        print(f'({data_index}/{data_len}) 正在处理：{dataset["data_name"]}')

        op_index = 0  # 算子索引

        # 解析日志和监控文件
        try:
            log_info = parse_log(dataset['log_file_path'])
            monitor_info = parse_monitor(dataset['monitor_file_path'])
        except Exception as e:
            print(f"[SKIP] 解析失败：{dataset['data_name']} | {e}")
            continue

        # 读取管道配置文件
        pipeline_file_path = dataset['pipeline_file_path']
        with open(pipeline_file_path, 'r', encoding='utf-8') as f:
            pipeline_dict = yaml.safe_load(f)
            operator_list = pipeline_dict['process']

            # 跳过日志/监控记录不完整的样本，避免索引越界
            operator_count = len(operator_list)
            if len(log_info) < operator_count or len(monitor_info) < operator_count:
                print(
                    f"[SKIP] 数据不完整：{dataset['data_name']} | "
                    f"ops={operator_count}, log={len(log_info)}, monitor={len(monitor_info)}"
                )
                continue

            # 遍历管道中的每个算子
            for operator in operator_list:
                op_index += 1
                data_row = {}  # 当前数据行

                # 获取算子特征值
                data_row['op_process_number'] = pipeline_dict['np']  # 并发处理数量
                for op_name, op_params in operator.items():
                    data_row[f'op_type_{op_name}'] = op_name.split('_')[-1]  # 算子类型
                    if isinstance(op_params, dict):
                        # 添加算子参数（如果启用）
                        for param_name, param_value in op_params.items():
                            data_row[f'op_{op_name}_{param_name}'] = param_value

                # 获取数据特征值
                data_row['ds_input_count'] = log_info[op_index - 1]['ds_input_count']
                data_row['ds_output_count'] = log_info[op_index - 1]['ds_output_count']
                data_row['op_process_time'] = log_info[op_index - 1]['op_process_time']
                data_row['ds_type'] = infer_ds_type(dataset['data_name'], operator)
                data_row.update(extract_scale_features(dataset['data_name'], data_row['ds_type']))

                # 获取环境特征值
                data_row.update(monitor_info[op_index - 1])

                # 添加数据行到列表
                data_rows.append(data_row)

    # 判断输出路径是否存在
    ensure_output_dirs()

    # 生成用于基数估计的CSV文件
    df = pd.DataFrame(data_rows, columns=dataset_header_for_cardinality_estimation)
    df.to_csv(f'{OUTPUT_PATH}/dataset_header_for_cardinality_estimation.csv', index=False)

    # 生成用于成本估计的CSV文件
    df = pd.DataFrame(data_rows, columns=dataset_header_for_cost_estimation)
    df.to_csv(f'{OUTPUT_PATH}/dataset_header_for_cost_estimation.csv', index=False)
