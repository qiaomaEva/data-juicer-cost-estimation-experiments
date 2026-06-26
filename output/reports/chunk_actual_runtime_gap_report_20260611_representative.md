# Chunk真实耗时求和与全量真实耗时差异分析

本报告只使用真实执行耗时，不涉及模型预测。核心问题是验证：

`sum(chunk_actual_op_time)` 是否等于 `full_actual_op_time`。

如果两者不相等，说明把数据切成很多小块分别执行再求和，本身就不是完整数据一次执行的等价替代；原因通常来自重复的固定开销、I/O/调度开销、模型或外部库初始化开销，以及小块规模导致的资源利用率变化。

## 1. 实验概况

- chunk root: .\collect_data\runs_chunks
- full root: .\collect_data\result_20260611
- chunk yaml discovered: 4539
- full yaml discovered: 950
- chunk operator rows: 36659
- full reference operator rows: 93
- matched operator comparison rows: 372
- pipeline filter: ^(audio_pipeline_1781071552228|audio_pipeline_1781071552263|audio_pipeline_1781071552218|image_pipeline_1781071560263|image_pipeline_1781071560242|image_pipeline_1781071560252|text_pipeline_1781071569701|text_pipeline_1781071569367|text_pipeline_1781071569512)$

## 2. 按模态和chunk size汇总

| ds_type | chunk_size | operator_rows | full_actual_time_sum | chunk_actual_time_sum | chunk_sum_to_full_ratio | chunk_sum_mape_like | chunk_sum_accuracy | estimated_repeated_overhead_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| audio | 100 | 11 | 11776.48 | 17664.31 | 1.50 | 50.00 | 50.00 | 5887.83 |
| audio | 200 | 11 | 11776.48 | 14963.08 | 1.27 | 27.06 | 72.94 | 3186.60 |
| audio | 500 | 11 | 11776.48 | 13401.37 | 1.14 | 13.80 | 86.20 | 1624.89 |
| audio | 1000 | 11 | 11776.48 | 12898.98 | 1.10 | 9.53 | 90.47 | 1122.50 |
| image | 100 | 23 | 340.45 | 22884.23 | 67.22 | 6621.84 | -6521.84 | 22543.79 |
| image | 200 | 23 | 340.45 | 13924.29 | 40.90 | 3990.01 | -3890.01 | 13583.84 |
| image | 500 | 23 | 340.45 | 7057.96 | 20.73 | 1973.15 | -1873.15 | 6717.52 |
| image | 1000 | 23 | 340.45 | 4275.06 | 12.56 | 1155.72 | -1055.72 | 3934.62 |
| text | 100 | 59 | 145.18 | 10414.15 | 71.73 | 7073.12 | -6973.12 | 10268.96 |
| text | 200 | 59 | 145.18 | 5397.78 | 37.18 | 3617.92 | -3517.92 | 5252.60 |
| text | 500 | 59 | 145.18 | 2195.35 | 15.12 | 1412.12 | -1312.12 | 2050.17 |
| text | 1000 | 59 | 145.18 | 1106.79 | 7.62 | 662.34 | -562.34 | 961.61 |

## 3. 按pipeline类型汇总

| ds_type | pipeline_category | chunk_size | operator_rows | chunk_sum_to_full_ratio | chunk_sum_mape_like | chunk_sum_accuracy | estimated_repeated_overhead_sum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| audio | filter-heavy | 100 | 4 | 1.63 | 63.31 | 36.69 | 1830.81 |
| audio | filter-heavy | 200 | 4 | 1.31 | 31.24 | 68.76 | 903.59 |
| audio | filter-heavy | 500 | 4 | 1.16 | 16.02 | 83.98 | 463.17 |
| audio | filter-heavy | 1000 | 4 | 1.08 | 8.16 | 91.84 | 235.98 |
| audio | mapper-heavy | 100 | 3 | 1.36 | 36.07 | 63.93 | 1810.83 |
| audio | mapper-heavy | 200 | 3 | 1.21 | 21.06 | 78.94 | 1057.15 |
| audio | mapper-heavy | 500 | 3 | 1.12 | 11.86 | 88.14 | 595.42 |
| audio | mapper-heavy | 1000 | 3 | 1.09 | 9.25 | 90.75 | 464.16 |
| audio | mixed | 100 | 4 | 1.58 | 58.13 | 41.87 | 2246.19 |
| audio | mixed | 200 | 4 | 1.32 | 31.72 | 68.28 | 1225.87 |
| audio | mixed | 500 | 4 | 1.15 | 14.66 | 85.34 | 566.31 |
| audio | mixed | 1000 | 4 | 1.11 | 10.93 | 89.07 | 422.37 |
| image | filter-heavy | 100 | 8 | 88.25 | 8725.26 | -8625.26 | 4848.28 |
| image | filter-heavy | 200 | 8 | 43.92 | 4291.53 | -4191.53 | 2384.63 |
| image | filter-heavy | 500 | 8 | 18.89 | 1789.24 | -1689.24 | 994.21 |
| image | filter-heavy | 1000 | 8 | 11.19 | 1018.89 | -918.89 | 566.16 |
| image | mapper-heavy | 100 | 7 | 52.66 | 5165.92 | -5065.92 | 10457.90 |
| image | mapper-heavy | 200 | 7 | 37.06 | 3606.35 | -3506.35 | 7300.69 |
| image | mapper-heavy | 500 | 7 | 21.49 | 2048.64 | -1948.64 | 4147.26 |
| image | mapper-heavy | 1000 | 7 | 13.13 | 1212.97 | -1112.97 | 2455.53 |
| image | mixed | 100 | 8 | 88.79 | 8779.25 | -8679.25 | 7237.61 |
| image | mixed | 200 | 8 | 48.29 | 4728.92 | -4628.92 | 3898.52 |
| image | mixed | 500 | 8 | 20.12 | 1911.74 | -1811.74 | 1576.04 |
| image | mixed | 1000 | 8 | 12.07 | 1107.38 | -1007.38 | 912.93 |
| text | filter-heavy | 100 | 23 | 86.28 | 8527.66 | -8427.66 | 4790.41 |
| text | filter-heavy | 200 | 23 | 44.11 | 4310.84 | -4210.84 | 2421.61 |
| text | filter-heavy | 500 | 23 | 17.71 | 1671.21 | -1571.21 | 938.80 |
| text | filter-heavy | 1000 | 23 | 8.78 | 777.92 | -677.92 | 437.00 |
| text | mapper-heavy | 100 | 16 | 73.06 | 7206.02 | -7106.02 | 2770.93 |
| text | mapper-heavy | 200 | 16 | 40.06 | 3906.20 | -3806.20 | 1502.05 |
| text | mapper-heavy | 500 | 16 | 16.45 | 1544.67 | -1444.67 | 593.97 |
| text | mapper-heavy | 1000 | 16 | 8.26 | 725.91 | -625.91 | 279.14 |
| text | mixed | 100 | 20 | 54.56 | 5355.79 | -5255.79 | 2707.62 |
| text | mixed | 200 | 20 | 27.29 | 2628.70 | -2528.70 | 1328.94 |
| text | mixed | 500 | 20 | 11.23 | 1023.42 | -923.42 | 517.39 |
| text | mixed | 1000 | 20 | 5.86 | 485.57 | -385.57 | 245.48 |

## 4. 差异最大的算子级记录

| ds_type | pipeline_base_name | pipeline_category | chunk_size | operator_index | operator_name | chunk_parts | full_actual_time | chunk_actual_time_sum | chunk_actual_time_mean | chunk_sum_to_full_ratio | chunk_sum_accuracy | estimated_repeated_overhead | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image | image_pipeline_1781071560263 | filter-heavy | 100 | 8 | image_segment_mapper | 526 | 0.74 | 413.58 | 0.79 | 555.15 | -55314.77 | 412.84 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次；可能包含模型加载/外部库初始化/重型计算开销 |
| image | image_pipeline_1781071560252 | mixed | 100 | 6 | image_detection_yolo_mapper | 526 | 0.87 | 479.25 | 0.91 | 550.23 | -54823.19 | 478.38 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次；可能包含模型加载/外部库初始化/重型计算开销 |
| image | image_pipeline_1781071560263 | filter-heavy | 100 | 7 | image_face_count_filter | 526 | 0.77 | 419.50 | 0.80 | 541.99 | -53998.97 | 418.73 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| image | image_pipeline_1781071560263 | filter-heavy | 100 | 1 | image_size_filter | 526 | 2.70 | 1177.83 | 2.24 | 436.72 | -43471.86 | 1175.13 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| image | image_pipeline_1781071560252 | mixed | 100 | 1 | image_size_filter | 526 | 4.06 | 1735.89 | 3.30 | 427.14 | -42513.95 | 1731.83 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| image | image_pipeline_1781071560252 | mixed | 200 | 6 | image_detection_yolo_mapper | 263 | 0.87 | 241.28 | 0.92 | 277.01 | -27501.15 | 240.41 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次；可能包含模型加载/外部库初始化/重型计算开销 |
| image | image_pipeline_1781071560263 | filter-heavy | 200 | 8 | image_segment_mapper | 263 | 0.74 | 203.71 | 0.77 | 273.43 | -27143.22 | 202.96 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次；可能包含模型加载/外部库初始化/重型计算开销 |
| image | image_pipeline_1781071560263 | filter-heavy | 200 | 7 | image_face_count_filter | 263 | 0.77 | 204.27 | 0.78 | 263.91 | -26191.21 | 203.49 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| image | image_pipeline_1781071560252 | mixed | 200 | 1 | image_size_filter | 263 | 4.06 | 957.16 | 3.64 | 235.52 | -23352.14 | 953.10 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| image | image_pipeline_1781071560263 | filter-heavy | 200 | 1 | image_size_filter | 263 | 2.70 | 570.26 | 2.17 | 211.44 | -20944.12 | 567.56 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| image | image_pipeline_1781071560263 | filter-heavy | 100 | 4 | image_aspect_ratio_filter | 526 | 2.36 | 423.16 | 0.80 | 179.53 | -17753.37 | 420.80 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| image | image_pipeline_1781071560242 | mapper-heavy | 100 | 6 | image_blur_mapper | 526 | 2.87 | 488.94 | 0.93 | 170.36 | -16836.38 | 486.07 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| image | image_pipeline_1781071560252 | mixed | 100 | 4 | image_shape_filter | 526 | 7.76 | 1312.73 | 2.50 | 169.21 | -16721.04 | 1304.98 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| image | image_pipeline_1781071560252 | mixed | 100 | 3 | image_aspect_ratio_filter | 526 | 11.38 | 1338.91 | 2.55 | 117.62 | -11562.41 | 1327.53 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| image | image_pipeline_1781071560252 | mixed | 500 | 6 | image_detection_yolo_mapper | 106 | 0.87 | 96.85 | 0.91 | 111.20 | -10919.86 | 95.98 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk数量多，固定开销被重复计入多次；可能包含模型加载/外部库初始化/重型计算开销 |
| image | image_pipeline_1781071560263 | filter-heavy | 100 | 3 | image_shape_filter | 526 | 7.39 | 821.81 | 1.56 | 111.19 | -10919.10 | 814.42 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| image | image_pipeline_1781071560263 | filter-heavy | 500 | 8 | image_segment_mapper | 106 | 0.74 | 82.16 | 0.78 | 110.28 | -10827.79 | 81.41 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk数量多，固定开销被重复计入多次；可能包含模型加载/外部库初始化/重型计算开销 |
| text | text_pipeline_1781071569512 | mixed | 100 | 17 | language_id_score_filter | 100 | 0.72 | 78.87 | 0.79 | 109.39 | -10739.25 | 78.15 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569367 | mapper-heavy | 100 | 12 | text_length_filter | 100 | 3.45 | 373.17 | 3.73 | 108.23 | -10622.68 | 369.72 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| image | image_pipeline_1781071560242 | mapper-heavy | 100 | 7 | image_tagging_mapper | 526 | 30.18 | 3248.55 | 6.18 | 107.64 | -10563.93 | 3218.38 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次；可能包含模型加载/外部库初始化/重型计算开销 |
| image | image_pipeline_1781071560263 | filter-heavy | 500 | 7 | image_face_count_filter | 106 | 0.77 | 82.61 | 0.78 | 106.72 | -10472.48 | 81.83 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569701 | filter-heavy | 100 | 13 | text_length_filter | 100 | 4.89 | 521.48 | 5.21 | 106.53 | -10453.22 | 516.58 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| image | image_pipeline_1781071560252 | mixed | 100 | 7 | image_remove_background_mapper | 526 | 4.78 | 503.62 | 0.96 | 105.27 | -10327.26 | 498.84 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次；可能包含模型加载/外部库初始化/重型计算开销 |
| text | text_pipeline_1781071569701 | filter-heavy | 100 | 22 | flagged_words_filter | 100 | 0.78 | 81.46 | 0.81 | 105.11 | -10310.97 | 80.68 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569701 | filter-heavy | 100 | 20 | text_entity_dependency_filter | 100 | 0.78 | 81.24 | 0.81 | 104.70 | -10269.59 | 80.47 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569512 | mixed | 100 | 7 | remove_repeat_sentences_mapper | 100 | 1.47 | 152.66 | 1.53 | 103.99 | -10198.84 | 151.19 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569701 | filter-heavy | 100 | 23 | document_simhash_deduplicator | 100 | 0.78 | 81.20 | 0.81 | 103.58 | -10157.65 | 80.42 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569701 | filter-heavy | 100 | 11 | remove_repeat_sentences_mapper | 100 | 2.13 | 219.74 | 2.20 | 103.21 | -10121.32 | 217.61 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569701 | filter-heavy | 100 | 4 | clean_links_mapper | 100 | 2.14 | 219.75 | 2.20 | 102.78 | -10078.39 | 217.61 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569701 | filter-heavy | 100 | 1 | fix_unicode_mapper | 100 | 2.20 | 224.93 | 2.25 | 102.29 | -10028.83 | 222.73 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569512 | mixed | 100 | 20 | document_simhash_deduplicator | 100 | 0.77 | 78.81 | 0.79 | 102.22 | -10021.53 | 78.04 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569701 | filter-heavy | 100 | 2 | punctuation_normalization_mapper | 100 | 2.14 | 218.22 | 2.18 | 102.17 | -10016.53 | 216.09 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569367 | mapper-heavy | 100 | 1 | punctuation_normalization_mapper | 100 | 1.52 | 154.73 | 1.55 | 101.80 | -9979.67 | 153.21 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569701 | filter-heavy | 100 | 5 | sentence_split_mapper | 100 | 2.17 | 220.65 | 2.21 | 101.45 | -9944.83 | 218.47 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569512 | mixed | 100 | 4 | remove_specific_chars_mapper | 100 | 1.50 | 151.58 | 1.52 | 101.32 | -9932.49 | 150.09 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569512 | mixed | 100 | 13 | token_num_filter | 100 | 0.79 | 80.05 | 0.80 | 100.94 | -9893.95 | 79.25 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569701 | filter-heavy | 100 | 6 | remove_specific_chars_mapper | 100 | 2.19 | 219.81 | 2.20 | 100.19 | -9818.82 | 217.62 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569367 | mapper-heavy | 100 | 3 | fix_unicode_mapper | 100 | 1.52 | 152.78 | 1.53 | 100.18 | -9818.49 | 151.26 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569701 | filter-heavy | 100 | 3 | whitespace_normalization_mapper | 100 | 2.20 | 219.76 | 2.20 | 99.80 | -9780.15 | 217.56 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |
| text | text_pipeline_1781071569367 | mapper-heavy | 100 | 4 | sentence_split_mapper | 100 | 1.52 | 151.59 | 1.52 | 99.60 | -9759.92 | 150.07 | chunk_sum远大于full，重复固定开销/初始化开销很明显；chunk_size较小，固定开销占比更高；chunk数量多，固定开销被重复计入多次 |

## 5. 结论

- 如果 `chunk_sum_to_full_ratio` 显著大于 1，说明小块真实执行求和已经比全量一次执行慢很多，模型预测再准也无法让“直接求和”成为等价估计。
- 如果 chunk size 增大后该比例下降，说明主要问题来自小块重复固定开销，而不是 pipeline 配置本身。
- 后续如果继续使用 chunk 思路，应考虑建模 `fixed_overhead + variable_cost_per_record * record_count`，而不是直接把所有 chunk 耗时简单相加。
