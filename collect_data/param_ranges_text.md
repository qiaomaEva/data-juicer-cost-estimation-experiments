# Text 参数范围设计（基于 RedPajama c4，n=26,000）

| 参数 | 旧候选值 | 新范围 | 依据（分位数） |
|------|---------|--------|--------------|
| text_length_filter.min_len | 20/40/60/80 | [200, 1200] | p5=211 → p50=1157 |
| text_length_filter.max_len | 3000/4000/5000/6000 | [1200, 7500] | p50=1157 → p95=6954 |
| words_num_filter.min_num | 20/40/60/80 | [35, 200] | p5=36 → p50=191 |
| words_num_filter.max_num | 3000/4000/5000/6000 | [200, 1200] | p50=191 → p95=1177 |
| token_num_filter.min_num | 20/40/60/80 | [50, 300] | p5=53 → p50=289 |
| token_num_filter.max_num | 3000/4000/5000/6000 | [300, 1800] | p50=289 → p95=1739 |
| maximum_line_length_filter.min_len | 20/40/60/80 | [100, 400] | p5=108 → p50=405 |
| maximum_line_length_filter.max_len | 3000/4000/5000/6000 | [400, 1200] | p50=405 → p95=1131 |
| average_line_length_filter.min_len | 10/20/40/60/80/100 | [40, 400] | 覆盖 p5~p50 |
| average_line_length_filter.max_len | 2000~8000 | [400, 1200] | p50=405 → p95=1131 |

## 关键分位数（merged，全量）

| 指标 | min | p5 | p25 | p50 | p75 | p95 | max |
|------|-----|----|-----|-----|-----|-----|-----|
| char_length | 33 | 211 | 529 | 1157 | 2480 | 6954 | 134046 |
| word_count | 6 | 36 | 88 | 191 | 415 | 1177 | 23099 |
| token_count_approx | 8 | 53 | 132 | 289 | 620 | 1739 | 33512 |
| max_line_length | 26 | 108 | 257 | 405 | 610 | 1131 | 21420 |
