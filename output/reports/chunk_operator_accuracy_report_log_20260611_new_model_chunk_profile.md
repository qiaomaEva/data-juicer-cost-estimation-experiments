# Chunk Operator Cost Accuracy Report

This report evaluates each chunk operator row independently: `op_process_time` vs `op_process_time_pred`. It does not sum chunks and does not compare against the full-scale run.

Metric definition is consistent with the operator-level cost reports: `MAPE = mean(abs(y_true - y_pred) / y_true) * 100`, `accuracy = 100 - MAPE`; rows with `op_process_time <= 0` are skipped.

## Overall

| sample_count | rmse | mae | mape | accuracy | actual_time_sum | pred_time_sum |
| --- | --- | --- | --- | --- | --- | --- |
| 36659 | 4.18 | 1.15 | 24.26 | 75.74 | 126183.37 | 149614.41 |


## By Modality

| ds_type | sample_count | rmse | mae | mape | accuracy |
| --- | --- | --- | --- | --- | --- |
| audio | 4235 | 12.08 | 6.56 | 57.45 | 42.55 |
| image | 21804 | 0.97 | 0.54 | 23.33 | 76.67 |
| text | 10620 | 0.37 | 0.24 | 12.94 | 87.06 |


## By Modality And Chunk Size

| ds_type | chunk_size | sample_count | rmse | mae | mape | accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| audio | 100.00 | 2343 | 14.31 | 8.19 | 76.65 | 23.35 |
| audio | 200.00 | 1177 | 8.72 | 5.04 | 42.53 | 57.47 |
| audio | 500.00 | 473 | 5.14 | 2.66 | 21.23 | 78.77 |
| audio | 1000.00 | 242 | 12.20 | 5.75 | 14.86 | 85.14 |
| image | 100.00 | 12098 | 0.87 | 0.47 | 23.24 | 76.76 |
| image | 200.00 | 6049 | 0.90 | 0.58 | 24.10 | 75.90 |
| image | 500.00 | 2438 | 1.01 | 0.62 | 22.50 | 77.50 |
| image | 1000.00 | 1219 | 1.81 | 0.85 | 22.03 | 77.97 |
| text | 100.00 | 5900 | 0.38 | 0.24 | 13.41 | 86.59 |
| text | 200.00 | 2950 | 0.37 | 0.23 | 12.29 | 87.71 |
| text | 500.00 | 1180 | 0.35 | 0.23 | 12.45 | 87.55 |
| text | 1000.00 | 590 | 0.33 | 0.23 | 12.52 | 87.48 |


## By Pipeline Category And Chunk Size

| ds_type | pipeline_category | chunk_size | sample_count | rmse | mae | mape | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| audio | filter-heavy | 100.00 | 852 | 15.28 | 8.15 | 81.52 | 18.48 |
| audio | filter-heavy | 200.00 | 428 | 9.98 | 5.45 | 48.01 | 51.99 |
| audio | filter-heavy | 500.00 | 172 | 5.43 | 2.78 | 21.78 | 78.22 |
| audio | filter-heavy | 1000.00 | 88 | 5.00 | 2.30 | 10.65 | 89.35 |
| audio | mapper-heavy | 100.00 | 639 | 13.42 | 9.26 | 102.47 | -2.47 |
| audio | mapper-heavy | 200.00 | 321 | 9.12 | 6.23 | 56.35 | 43.65 |
| audio | mapper-heavy | 500.00 | 129 | 6.00 | 3.50 | 26.10 | 73.90 |
| audio | mapper-heavy | 1000.00 | 66 | 18.31 | 10.98 | 24.80 | 75.20 |
| audio | mixed | 100.00 | 852 | 13.95 | 7.43 | 52.42 | 47.58 |
| audio | mixed | 200.00 | 428 | 6.85 | 3.75 | 26.68 | 73.32 |
| audio | mixed | 500.00 | 172 | 4.02 | 1.92 | 17.03 | 82.97 |
| audio | mixed | 1000.00 | 88 | 11.54 | 5.28 | 11.61 | 88.39 |
| image | filter-heavy | 100.00 | 4208 | 0.43 | 0.26 | 20.00 | 80.00 |
| image | filter-heavy | 200.00 | 2104 | 0.40 | 0.25 | 17.98 | 82.02 |
| image | filter-heavy | 500.00 | 848 | 0.46 | 0.25 | 17.19 | 82.81 |
| image | filter-heavy | 1000.00 | 424 | 0.68 | 0.34 | 21.05 | 78.95 |
| image | mapper-heavy | 100.00 | 3682 | 1.34 | 0.79 | 28.29 | 71.71 |
| image | mapper-heavy | 200.00 | 1841 | 1.34 | 0.97 | 28.29 | 71.71 |
| image | mapper-heavy | 500.00 | 742 | 1.57 | 1.17 | 27.17 | 72.83 |
| image | mapper-heavy | 1000.00 | 371 | 2.88 | 1.72 | 22.73 | 77.27 |
| image | mixed | 100.00 | 4208 | 0.65 | 0.41 | 22.07 | 77.93 |
| image | mixed | 200.00 | 2104 | 0.76 | 0.57 | 26.55 | 73.45 |
| image | mixed | 500.00 | 848 | 0.73 | 0.52 | 23.73 | 76.27 |
| image | mixed | 1000.00 | 424 | 1.32 | 0.60 | 22.40 | 77.60 |
| text | filter-heavy | 100.00 | 2300 | 0.32 | 0.19 | 7.99 | 92.01 |
| text | filter-heavy | 200.00 | 1150 | 0.29 | 0.17 | 7.60 | 92.40 |
| text | filter-heavy | 500.00 | 460 | 0.25 | 0.16 | 7.24 | 92.76 |
| text | filter-heavy | 1000.00 | 230 | 0.23 | 0.15 | 7.49 | 92.51 |
| text | mapper-heavy | 100.00 | 1600 | 0.39 | 0.33 | 20.89 | 79.11 |
| text | mapper-heavy | 200.00 | 800 | 0.36 | 0.29 | 17.17 | 82.83 |
| text | mapper-heavy | 500.00 | 320 | 0.41 | 0.34 | 19.22 | 80.78 |
| text | mapper-heavy | 1000.00 | 160 | 0.43 | 0.36 | 20.10 | 79.90 |
| text | mixed | 100.00 | 2000 | 0.42 | 0.24 | 13.66 | 86.34 |
| text | mixed | 200.00 | 1000 | 0.44 | 0.25 | 13.78 | 86.22 |
| text | mixed | 500.00 | 400 | 0.39 | 0.23 | 13.02 | 86.98 |
| text | mixed | 1000.00 | 200 | 0.33 | 0.21 | 12.24 | 87.76 |


## Worst Groups

| level | ds_type | pipeline_base_name | pipeline_category | operator_name | chunk_size | sample_count | mape | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| by_pipeline_chunk_size | audio | audio_pipeline_1781071552263 | mapper-heavy |  | 100.00 | 639 | 102.47 | -2.47 |
| by_operator | audio |  |  | audio_ffmpeg_wrapped_mapper | nan | 770 | 98.20 | 1.80 |
| by_operator | audio |  |  | audio_nmf_snr_filter | nan | 1155 | 93.43 | 6.57 |
| by_pipeline_chunk_size | audio | audio_pipeline_1781071552228 | filter-heavy |  | 100.00 | 852 | 81.52 | 18.48 |
| by_operator | image |  |  | image_deduplicator | nan | 2844 | 78.40 | 21.60 |
| by_pipeline_chunk_size | audio | audio_pipeline_1781071552263 | mapper-heavy |  | 200.00 | 321 | 56.35 | 43.65 |
| by_pipeline_chunk_size | audio | audio_pipeline_1781071552218 | mixed |  | 100.00 | 852 | 52.42 | 47.58 |
| by_pipeline_chunk_size | audio | audio_pipeline_1781071552228 | filter-heavy |  | 200.00 | 428 | 48.01 | 51.99 |
| by_operator | audio |  |  | audio_add_gaussian_noise_mapper | nan | 385 | 35.68 | 64.32 |
| by_operator | text |  |  | alphanumeric_filter | nan | 360 | 32.24 | 67.76 |
| by_operator | audio |  |  | audio_duration_filter | nan | 1155 | 30.75 | 69.25 |
| by_operator | text |  |  | remove_table_text_mapper | nan | 360 | 30.46 | 69.54 |
| by_pipeline_chunk_size | image | image_pipeline_1781071560242 | mapper-heavy |  | 200.00 | 1841 | 28.29 | 71.71 |
| by_pipeline_chunk_size | image | image_pipeline_1781071560242 | mapper-heavy |  | 100.00 | 3682 | 28.29 | 71.71 |
| by_operator | text |  |  | replace_content_mapper | nan | 180 | 27.98 | 72.02 |
| by_pipeline_chunk_size | image | image_pipeline_1781071560242 | mapper-heavy |  | 500.00 | 742 | 27.17 | 72.83 |
| by_operator | text |  |  | clean_email_mapper | nan | 360 | 27.17 | 72.83 |
| by_pipeline_chunk_size | audio | audio_pipeline_1781071552218 | mixed |  | 200.00 | 428 | 26.68 | 73.32 |
| by_pipeline_chunk_size | image | image_pipeline_1781071560252 | mixed |  | 200.00 | 2104 | 26.55 | 73.45 |
| by_pipeline_chunk_size | audio | audio_pipeline_1781071552263 | mapper-heavy |  | 500.00 | 129 | 26.10 | 73.90 |
| by_operator | text |  |  | clean_copyright_mapper | nan | 540 | 25.42 | 74.58 |
| by_operator | image |  |  | image_detection_yolo_mapper | nan | 948 | 25.27 | 74.73 |
| by_pipeline_chunk_size | audio | audio_pipeline_1781071552263 | mapper-heavy |  | 1000.00 | 66 | 24.80 | 75.20 |
| by_pipeline_chunk_size | image | image_pipeline_1781071560252 | mixed |  | 500.00 | 848 | 23.73 | 76.27 |
| by_pipeline_chunk_size | image | image_pipeline_1781071560242 | mapper-heavy |  | 1000.00 | 371 | 22.73 | 77.27 |
| by_pipeline_chunk_size | image | image_pipeline_1781071560252 | mixed |  | 1000.00 | 424 | 22.40 | 77.60 |
| by_pipeline_chunk_size | image | image_pipeline_1781071560252 | mixed |  | 100.00 | 4208 | 22.07 | 77.93 |
| by_pipeline_chunk_size | audio | audio_pipeline_1781071552228 | filter-heavy |  | 500.00 | 172 | 21.78 | 78.22 |
| by_pipeline_chunk_size | image | image_pipeline_1781071560263 | filter-heavy |  | 1000.00 | 424 | 21.05 | 78.95 |
| by_pipeline_chunk_size | text | text_pipeline_1781071569367 | mapper-heavy |  | 100.00 | 1600 | 20.89 | 79.11 |


## Worst Rows

| ds_type | pipeline_name | chunk_size | chunk_part | operator_index | operator_name | op_process_time | op_process_time_pred | row_ape | row_accuracy | chunk_profile_matched |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| audio | audio_pipeline_1781071552228_chunk100_part0150 | 100 | 150 | 2 | audio_nmf_snr_filter | 11.04 | 48.19 | 336.57 | -236.57 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0044 | 100 | 44 | 2 | audio_nmf_snr_filter | 12.33 | 50.03 | 305.91 | -205.91 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0102 | 100 | 102 | 2 | audio_nmf_snr_filter | 12.25 | 49.71 | 305.71 | -205.71 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0121 | 100 | 121 | 2 | audio_nmf_snr_filter | 11.08 | 43.71 | 294.55 | -194.55 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0190 | 100 | 190 | 2 | audio_nmf_snr_filter | 11.47 | 45.09 | 293.08 | -193.08 | True |
| audio | audio_pipeline_1781071552263_chunk100_part0111 | 100 | 111 | 3 | audio_ffmpeg_wrapped_mapper | 1.54 | 5.95 | 285.15 | -185.15 | True |
| audio | audio_pipeline_1781071552263_chunk100_part0013 | 100 | 13 | 3 | audio_ffmpeg_wrapped_mapper | 1.58 | 6.04 | 282.82 | -182.82 | True |
| audio | audio_pipeline_1781071552263_chunk100_part0088 | 100 | 88 | 3 | audio_ffmpeg_wrapped_mapper | 1.55 | 5.94 | 282.08 | -182.08 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0198 | 100 | 198 | 2 | audio_nmf_snr_filter | 13.01 | 49.45 | 280.23 | -180.23 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0087 | 100 | 87 | 2 | audio_nmf_snr_filter | 13.78 | 52.27 | 279.35 | -179.35 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0162 | 100 | 162 | 2 | audio_nmf_snr_filter | 11.84 | 44.65 | 276.98 | -176.98 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0001 | 100 | 1 | 2 | audio_nmf_snr_filter | 12.85 | 48.36 | 276.33 | -176.33 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0058 | 100 | 58 | 2 | audio_nmf_snr_filter | 11.65 | 43.84 | 276.30 | -176.30 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0152 | 100 | 152 | 2 | audio_nmf_snr_filter | 11.41 | 42.87 | 275.59 | -175.59 | True |
| audio | audio_pipeline_1781071552263_chunk100_part0052 | 100 | 52 | 3 | audio_ffmpeg_wrapped_mapper | 1.54 | 5.72 | 272.20 | -172.20 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0138 | 100 | 138 | 2 | audio_nmf_snr_filter | 11.74 | 43.71 | 272.13 | -172.13 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0210 | 100 | 210 | 2 | audio_nmf_snr_filter | 13.36 | 49.69 | 271.98 | -171.98 | True |
| audio | audio_pipeline_1781071552263_chunk100_part0063 | 100 | 63 | 3 | audio_ffmpeg_wrapped_mapper | 1.61 | 5.99 | 271.01 | -171.01 | True |
| audio | audio_pipeline_1781071552263_chunk100_part0118 | 100 | 118 | 3 | audio_ffmpeg_wrapped_mapper | 1.60 | 5.91 | 269.39 | -169.39 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0041 | 100 | 41 | 2 | audio_nmf_snr_filter | 12.16 | 44.70 | 267.57 | -167.57 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0176 | 100 | 176 | 2 | audio_nmf_snr_filter | 12.17 | 44.64 | 266.71 | -166.71 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0027 | 100 | 27 | 2 | audio_nmf_snr_filter | 13.33 | 48.60 | 264.52 | -164.52 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0046 | 100 | 46 | 2 | audio_nmf_snr_filter | 11.18 | 40.74 | 264.50 | -164.50 | True |
| audio | audio_pipeline_1781071552263_chunk100_part0157 | 100 | 157 | 3 | audio_ffmpeg_wrapped_mapper | 1.58 | 5.75 | 263.26 | -163.26 | True |
| audio | audio_pipeline_1781071552263_chunk100_part0189 | 100 | 189 | 3 | audio_ffmpeg_wrapped_mapper | 1.57 | 5.68 | 262.85 | -162.85 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0206 | 100 | 206 | 2 | audio_nmf_snr_filter | 11.86 | 43.02 | 262.83 | -162.83 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0136 | 100 | 136 | 2 | audio_nmf_snr_filter | 12.73 | 46.14 | 262.32 | -162.32 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0104 | 100 | 104 | 2 | audio_nmf_snr_filter | 12.99 | 47.06 | 262.16 | -162.16 | True |
| audio | audio_pipeline_1781071552263_chunk100_part0203 | 100 | 203 | 3 | audio_ffmpeg_wrapped_mapper | 1.63 | 5.90 | 261.74 | -161.74 | True |
| audio | audio_pipeline_1781071552228_chunk100_part0037 | 100 | 37 | 2 | audio_nmf_snr_filter | 12.40 | 44.80 | 261.20 | -161.20 | True |


