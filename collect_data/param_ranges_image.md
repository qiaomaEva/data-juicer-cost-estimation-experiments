# Image 参数范围设计（基于 COCO2017，n=173,710）

| 参数 | 旧候选值 | 新范围 | 依据（分位数） |
|------|---------|--------|--------------|
| image_size_filter.min_size | 16/32/64/128 kb | [50, 150] kb | p5=69 → p50=152 |
| image_size_filter.max_size | 200/256/320/512 kb | [150, 300] kb | p50=152 → p95=276 |
| image_shape_filter.min_width | 200/280/320/360 px | [400, 550] px | p5=418 → p25=500 |
| image_shape_filter.max_width | 420/480/520/640/800 px | [550, 640] px | p25=500 → max=640 |
| image_shape_filter.min_height | 200/280/320/360 px | [320, 480] px | p5=341 → p50=480 |
| image_shape_filter.max_height | 420/480/520/640/800 px | [480, 640] px | p50=480 → max=640 |
| image_aspect_ratio_filter.min_ratio | 0.4/0.6/0.8/1.0 | [0.65, 1.05] | p5=0.67 → p25=1.0 |
| image_aspect_ratio_filter.max_ratio | 1.4/1.6/1.8/2.5 | [1.35, 1.65] | p50=1.33 → p95=1.65 |

## 关键分位数（merged，全量）

| 指标 | min | p5 | p25 | p50 | p75 | p95 | max |
|------|-----|----|-----|-----|-----|-----|-----|
| file_size_kb | 5.9 | 69 | 112 | 152 | 198 | 276 | 799 |
| width_px | 72 | 418 | 500 | 640 | 640 | 640 | 640 |
| height_px | 51 | 341 | 426 | 480 | 512 | 640 | 640 |
| aspect_ratio_w_over_h | 0.30 | 0.67 | 1.00 | 1.33 | 1.50 | 1.65 | 6.15 |
