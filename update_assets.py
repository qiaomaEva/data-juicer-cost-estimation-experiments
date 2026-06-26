"""
用实际数据分布重写 assets/audio.json, image.json, text.json。
参数格式：
  continuous: {"name":..., "type":"continuous", "range":[lo,hi], "int":true/false, "unit":"kb"(可选), "gt":"other_param"(可选)}
  categorical: {"name":..., "type":"categorical", "values":[...]}
算子级别新增 occurrence_prob 字段（audio_duration_filter=0.2，因为 AudioSet 全是 10s 固定时长）。
"""
import json, os

ASSETS = os.path.join(os.path.dirname(__file__), "assets")

# ─────────────────────────── audio.json ───────────────────────────
audio = {
    "stages": [
        {"id": 0, "description": "数据清洗",
         "min_operator_count": 2, "max_operator_count": 3, "operator_total": 3},
        {"id": 1, "description": "数据增强",
         "min_operator_count": 1, "max_operator_count": 1, "operator_total": 2}
    ],
    "operators": [
        {
            "name": "audio_nmf_snr_filter", "type": "filter", "stage": 0, "layer": 0,
            "description": "保持音频信噪比（SNR）在指定范围内的数据样本。",
            "params": [
                {"name": "min_snr",    "type": "continuous", "range": [-5, 5],  "int": True},
                {"name": "snr_max",    "type": "continuous", "range": [5, 30],  "int": True, "gt": "min_snr"},
                {"name": "any_or_all", "type": "categorical", "values": ["any", "all"]}
            ]
        },
        {
            "name": "audio_size_filter", "type": "filter", "stage": 0, "layer": 1,
            "description": "根据音频文件的大小保留数据样本。",
            "params": [
                {"name": "min_size",   "type": "continuous", "range": [100, 400], "int": True, "unit": "kb"},
                {"name": "max_size",   "type": "continuous", "range": [400, 700], "int": True, "unit": "kb", "gt": "min_size"},
                {"name": "any_or_all", "type": "categorical", "values": ["any", "all"]}
            ]
        },
        {
            "name": "audio_duration_filter", "type": "filter", "stage": 0, "layer": 2,
            "description": "保留音频持续时间在指定范围内的数据样本。",
            "occurrence_prob": 0.2,
            "params": [
                {"name": "min_duration", "type": "continuous", "range": [1.0, 9.0]},
                {"name": "max_duration", "type": "continuous", "range": [9.5, 11.0], "gt": "min_duration"},
                {"name": "any_or_all",   "type": "categorical", "values": ["any", "all"]}
            ]
        },
        {
            "name": "audio_ffmpeg_wrapped_mapper", "type": "mapper", "stage": 1, "layer": 3,
            "description": "包装FFmpeg音频过滤器，用于处理数据集中的音频文件。",
            "params": [
                {"name": "filter_name",   "type": "categorical", "values": ["aresample"]},
                {"name": "filter_kwargs", "type": "categorical", "values": [
                    {"sample_rate": 8000}, {"sample_rate": 16000},
                    {"sample_rate": 22050}, {"sample_rate": 44100}
                ]},
                {"name": "save_dir", "type": "categorical",
                 "values": ["/tmp/data-juicer/output/audio/ffmpeg_wrapped"]}
            ]
        },
        {
            "name": "audio_add_gaussian_noise_mapper", "type": "mapper", "stage": 1, "layer": 4,
            "description": "映射器将高斯噪声添加到音频样本。",
            "params": [
                {"name": "save_dir", "type": "categorical",
                 "values": ["/tmp/data-juicer/output/audio/gaussian_noise"]}
            ]
        }
    ]
}

# ─────────────────────────── image.json ───────────────────────────
image = {
    "stages": [
        {"id": 0, "description": "预处理（去重与基础大小过滤）",
         "min_operator_count": 1, "max_operator_count": 2, "operator_total": 2},
        {"id": 1, "description": "基础过滤（形状、长宽比等）",
         "min_operator_count": 1, "max_operator_count": 2, "operator_total": 2},
        {"id": 2, "description": "高级过滤（美学、NSFW、人脸、水印等）",
         "min_operator_count": 0, "max_operator_count": 3, "operator_total": 5},
        {"id": 3, "description": "图像转换与增强（模糊、背景移除、检测等）",
         "min_operator_count": 0, "max_operator_count": 3, "operator_total": 5},
        {"id": 4, "description": "元数据生成（打标签等）",
         "min_operator_count": 0, "max_operator_count": 1, "operator_total": 1}
    ],
    "operators": [
        {
            "name": "image_deduplicator", "type": "filter", "stage": 0, "layer": 0,
            "description": "通过图像的精确匹配在文档级别删除重复的样本。",
            "params": []
        },
        {
            "name": "image_size_filter", "type": "filter", "stage": 0, "layer": 1,
            "description": "保留图像大小在特定范围内的数据样本。",
            "params": [
                {"name": "min_size", "type": "continuous", "range": [50, 150],  "int": True, "unit": "kb"},
                {"name": "max_size", "type": "continuous", "range": [150, 300], "int": True, "unit": "kb", "gt": "min_size"}
            ]
        },
        {
            "name": "image_aspect_ratio_filter", "type": "filter", "stage": 1, "layer": 2,
            "description": "保持样本的图像纵横比在特定范围内。",
            "params": [
                {"name": "min_ratio", "type": "continuous", "range": [0.65, 1.05]},
                {"name": "max_ratio", "type": "continuous", "range": [1.35, 1.65], "gt": "min_ratio"}
            ]
        },
        {
            "name": "image_shape_filter", "type": "filter", "stage": 1, "layer": 3,
            "description": "保持样本的图像形状（宽度，高度）在特定的范围内。",
            "params": [
                {"name": "min_width",  "type": "continuous", "range": [400, 550], "int": True},
                {"name": "max_width",  "type": "continuous", "range": [550, 640], "int": True, "gt": "min_width"},
                {"name": "min_height", "type": "continuous", "range": [320, 480], "int": True},
                {"name": "max_height", "type": "continuous", "range": [480, 640], "int": True, "gt": "min_height"}
            ]
        },
        {
            "name": "image_aesthetics_filter", "type": "filter", "stage": 2, "layer": 4,
            "description": "过滤以保持美学分数在特定范围内的样品。",
            "params": [
                {"name": "hf_scorer_model", "type": "categorical",
                 "values": ["shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"]},
                {"name": "min_score", "type": "continuous", "range": [0.3, 0.6]},
                {"name": "max_score", "type": "categorical", "values": [1.0]}
            ]
        },
        {
            "name": "image_nsfw_filter", "type": "filter", "stage": 2, "layer": 5,
            "description": "过滤器保留其图像的nsfw分数在指定范围内的样本。",
            "params": [
                {"name": "hf_nsfw_model", "type": "categorical",
                 "values": ["/data/coco2017/weights/nsfw_model"]}
            ]
        },
        {
            "name": "image_face_count_filter", "type": "filter", "stage": 2, "layer": 6,
            "description": "过滤以保持样本的面数在特定范围内。",
            "params": []
        },
        {
            "name": "image_face_ratio_filter", "type": "filter", "stage": 2, "layer": 7,
            "description": "过滤以保持面面积比在特定范围内的样本。",
            "params": [
                {"name": "min_ratio", "type": "continuous", "range": [0.0, 0.4]},
                {"name": "max_ratio", "type": "continuous", "range": [0.5, 1.0], "gt": "min_ratio"}
            ]
        },
        {
            "name": "image_watermark_filter", "type": "filter", "stage": 2, "layer": 8,
            "description": "过滤器以保持其图像没有水印的样本具有高概率。",
            "params": [
                {"name": "hf_watermark_model", "type": "categorical",
                 "values": ["/data/coco2017/weights/watermark_model"]}
            ]
        },
        {
            "name": "image_blur_mapper", "type": "mapper", "stage": 3, "layer": 9,
            "description": "使用指定的概率和模糊类型对数据集中的图像进行模糊处理。",
            "params": [
                {"name": "save_dir", "type": "categorical",
                 "values": ["/tmp/data-juicer/output/image/blur"]}
            ]
        },
        {
            "name": "image_detection_yolo_mapper", "type": "mapper", "stage": 3, "layer": 10,
            "description": "使用YOLO对图像执行对象检测，并返回边界框和类标签。",
            "params": []
        },
        {
            "name": "image_face_blur_mapper", "type": "mapper", "stage": 3, "layer": 11,
            "description": "映射器模糊图像中检测到的人脸。",
            "params": [
                {"name": "save_dir", "type": "categorical",
                 "values": ["/tmp/data-juicer/output/image/face_blur"]}
            ]
        },
        {
            "name": "image_remove_background_mapper", "type": "mapper", "stage": 3, "layer": 12,
            "description": "映射器删除图像的背景。",
            "params": [
                {"name": "save_dir", "type": "categorical",
                 "values": ["/tmp/data-juicer/output/image/remove_background"]}
            ]
        },
        {
            "name": "image_segment_mapper", "type": "mapper", "stage": 3, "layer": 13,
            "description": "图像执行segment-任何操作并返回边界框。",
            "params": []
        },
        {
            "name": "image_tagging_mapper", "type": "mapper", "stage": 4, "layer": 14,
            "description": "为样本中的每个图像生成图像标记。",
            "params": []
        }
    ]
}

# ─────────────────────────── text.json ───────────────────────────
# 只改与数据分布直接相关的数值型参数；比例/布尔/语言代码等保持 categorical
text_ops_updated = {
    # stage 0 — 文本规范化
    "fix_unicode_mapper": [
        {"name": "normalization", "type": "categorical", "values": ["NFC", "NFKC", "NFD", "NFKD"]}
    ],
    "whitespace_normalization_mapper": [],
    "punctuation_normalization_mapper": [],
    # stage 1 — 噪声与隐私信息清理
    "clean_copyright_mapper": [],
    "clean_email_mapper": [],
    "clean_ip_mapper": [],
    "clean_links_mapper": [],
    "remove_table_text_mapper": [],
    "remove_specific_chars_mapper": [],
    "remove_long_words_mapper": [
        {"name": "min_len", "type": "continuous", "range": [3, 5],   "int": True},
        {"name": "max_len", "type": "continuous", "range": [10, 20], "int": True, "gt": "min_len"}
    ],
    "remove_words_with_incorrect_substrings_mapper": [
        {"name": "lang",        "type": "categorical", "values": ["en"]},
        {"name": "tokenization","type": "categorical", "values": [True, False]},
        {"name": "substrings",  "type": "categorical", "values": [
            ["http", "www", ".com", "href", "//"],
            ["<", ">", "=", "&", "?"]
        ]}
    ],
    "replace_content_mapper": [
        {"name": "pattern", "type": "categorical", "values": ["●■", "\\d+(?:,\\d+)*"]},
        {"name": "repl",    "type": "categorical", "values": ["", "<hide>"]}
    ],
    "remove_repeat_sentences_mapper": [
        {"name": "lowercase", "type": "categorical", "values": [True, False]}
    ],
    "sentence_split_mapper": [
        {"name": "lang", "type": "categorical", "values": ["en"]}
    ],
    # stage 2 — 基于质量的过滤
    "alphanumeric_filter": [
        {"name": "tokenization", "type": "categorical", "values": [True, False]},
        {"name": "min_ratio",    "type": "categorical", "values": [0.1, 0.2, 0.3]},
        {"name": "max_ratio",    "type": "categorical", "values": [0.8, 0.9]},
        {"name": "batch_size",   "type": "categorical", "values": [2, 3]},
        {"name": "num_proc",     "type": "categorical", "values": [1, 2, 4]}
    ],
    "average_line_length_filter": [
        {"name": "min_len",    "type": "continuous", "range": [10, 100],   "int": True},
        {"name": "max_len",    "type": "continuous", "range": [2000, 8000],"int": True, "gt": "min_len"},
        {"name": "batch_size", "type": "categorical", "values": [2, 3]}
    ],
    "character_repetition_filter": [
        {"name": "rep_len",    "type": "categorical", "values": [3, 5, 10, 15]},
        {"name": "min_ratio",  "type": "categorical", "values": [0.0, 0.1]},
        {"name": "max_ratio",  "type": "categorical", "values": [0.3, 0.4, 0.5, 0.6]},
        {"name": "batch_size", "type": "categorical", "values": [2, 3]}
    ],
    "flagged_words_filter": [
        {"name": "lang",          "type": "categorical", "values": ["en"]},
        {"name": "tokenization",  "type": "categorical", "values": [True, False]},
        {"name": "max_ratio",     "type": "categorical", "values": [0.045, 0.05]},
        {"name": "use_words_aug", "type": "categorical", "values": [True, False]}
    ],
    "language_id_score_filter": [
        {"name": "lang",      "type": "categorical", "values": ["en", ["en", "zh"]]},
        {"name": "min_score", "type": "categorical", "values": [0.7, 0.8]}
    ],
    "maximum_line_length_filter": [
        {"name": "min_len",    "type": "continuous", "range": [100, 400],  "int": True},
        {"name": "max_len",    "type": "continuous", "range": [400, 1200], "int": True, "gt": "min_len"},
        {"name": "batch_size", "type": "categorical", "values": [2, 3]}
    ],
    "perplexity_filter": [
        {"name": "lang",       "type": "categorical", "values": ["en"]},
        {"name": "max_ppl",    "type": "continuous",  "range": [500, 3000], "int": True},
        {"name": "batch_size", "type": "categorical", "values": [2, 3]}
    ],
    "special_characters_filter": [
        {"name": "min_ratio",  "type": "categorical", "values": [0.0, 0.1]},
        {"name": "max_ratio",  "type": "categorical", "values": [0.20, 0.25]},
        {"name": "batch_size", "type": "categorical", "values": [2, 3]}
    ],
    "stopwords_filter": [
        {"name": "lang",          "type": "categorical", "values": ["en"]},
        {"name": "tokenization",  "type": "categorical", "values": [True, False]},
        {"name": "min_ratio",     "type": "categorical", "values": [0.3, 0.4]},
        {"name": "use_words_aug", "type": "categorical", "values": [True, False]}
    ],
    "text_action_filter": [
        {"name": "lang", "type": "categorical", "values": ["en"]}
    ],
    "text_entity_dependency_filter": [
        {"name": "lang",       "type": "categorical", "values": ["en"]},
        {"name": "any_or_all", "type": "categorical", "values": ["any", "all"]}
    ],
    "text_length_filter": [
        {"name": "min_len", "type": "continuous", "range": [200, 1200],  "int": True},
        {"name": "max_len", "type": "continuous", "range": [1200, 7500], "int": True, "gt": "min_len"}
    ],
    "token_num_filter": [
        {"name": "hf_tokenizer", "type": "categorical", "values": ["EleutherAI/pythia-14m"]},
        {"name": "min_num", "type": "continuous", "range": [50, 300],   "int": True},
        {"name": "max_num", "type": "continuous", "range": [300, 1800], "int": True, "gt": "min_num"}
    ],
    "word_repetition_filter": [
        {"name": "lang",         "type": "categorical", "values": ["en"]},
        {"name": "tokenization", "type": "categorical", "values": [True, False]},
        {"name": "rep_len",      "type": "categorical", "values": [3, 5, 10, 15, 20]},
        {"name": "min_ratio",    "type": "categorical", "values": [0.0, 0.1]},
        {"name": "max_ratio",    "type": "categorical", "values": [0.15, 0.2, 0.3, 0.5]},
        {"name": "batch_size",   "type": "categorical", "values": [2, 3]}
    ],
    "words_num_filter": [
        {"name": "lang",         "type": "categorical", "values": ["en"]},
        {"name": "tokenization", "type": "categorical", "values": [True, False]},
        {"name": "min_num", "type": "continuous", "range": [35, 200],   "int": True},
        {"name": "max_num", "type": "continuous", "range": [200, 1200], "int": True, "gt": "min_num"},
        {"name": "batch_size",   "type": "categorical", "values": [2, 3]}
    ],
    "text_chunk_mapper": [
        {"name": "max_len",       "type": "continuous",  "range": [256, 2000], "int": True},
        {"name": "split_pattern", "type": "categorical", "values": ["\\n", "\\n\\n", ". "]}
    ],
    # stage 3 — 文档级去重
    "document_deduplicator": [
        {"name": "lowercase",             "type": "categorical", "values": [True, False]},
        {"name": "ignore_non_character",  "type": "categorical", "values": [True, False]}
    ],
    "document_minhash_deduplicator": [
        {"name": "tokenization",   "type": "categorical", "values": ["space", "character"]},
        {"name": "lowercase",      "type": "categorical", "values": [True, False]},
        {"name": "ignore_pattern", "type": "categorical", "values": ["\\p{P}"]}
    ],
    "document_simhash_deduplicator": [
        {"name": "tokenization",   "type": "categorical", "values": ["space", "character"]},
        {"name": "lowercase",      "type": "categorical", "values": [True, False]},
        {"name": "ignore_pattern", "type": "categorical", "values": ["\\p{P}"]}
    ],
    # stage 4 — 大语言模型处理
    **{name: [
        {"name": "api_model",    "type": "categorical", "values": ["qwen3.5:9b"]},
        {"name": "api_endpoint", "type": "categorical",
         "values": ["https://open.h104.xclab.brisen.top/v1/chat/completions"]}
    ] for name in [
        "extract_entity_attribute_mapper", "extract_entity_relation_mapper",
        "extract_event_mapper", "extract_keyword_mapper", "extract_nickname_mapper",
        "extract_support_text_mapper", "pair_preference_mapper",
        "relation_identity_mapper", "nested_aggregator"
    ]}
}

# ── rebuild text.json by patching original operator list ──────────
import copy

with open(os.path.join(ASSETS, "text.json"), "r", encoding="utf-8") as f:
    text_orig = json.load(f)

text = copy.deepcopy(text_orig)
for op in text["operators"]:
    if op["name"] in text_ops_updated:
        op["params"] = text_ops_updated[op["name"]]

# ── write all three files ─────────────────────────────────────────
for name, data in [("audio.json", audio), ("image.json", image), ("text.json", text)]:
    path = os.path.join(ASSETS, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Written: {path}")
