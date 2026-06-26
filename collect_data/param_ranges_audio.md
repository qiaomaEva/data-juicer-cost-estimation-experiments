# Audio 参数范围设计（基于 AudioSet balanced，n=70,195）

| 参数 | 旧候选值 | 新范围 | 依据（分位数） |
|------|---------|--------|--------------|
| audio_size_filter.min_size | 64/128/256/512 kb | [100, 400] kb | p5=122 → p50=396 |
| audio_size_filter.max_size | 512/640/1024/2048 kb | [400, 700] kb | p50=396 → p95=631 |
| audio_duration_filter.min_duration | 2/4/5/8 s | [1, 9] s | p5=6.8，min=0.5 |
| audio_duration_filter.max_duration | 8/10/15/20/30 s | [9.5, 11] s | ⚠ p25~p95 全是 10.0s，AudioSet 几乎全是 10s 定长，建议降低出现概率到 20% |
| audio_ffmpeg_wrapped_mapper.sample_rate | 8000/16000/22050/44100 | 保留（8000/16000 会真实重采样，原始 p50=48000Hz） | — |

## 关键分位数（merged，全量）

| 指标 | min | p5 | p25 | p50 | p75 | p95 | max |
|------|-----|----|-----|-----|-----|-----|-----|
| file_size_kb | 1.4 | 122 | 299 | 396 | 496 | 631 | 886 |
| duration_sec | 0.5 | 6.8 | 10.0 | 10.0 | 10.0 | 10.0 | 32.5 |
| sample_rate_hz | 16000 | 16000 | 48000 | 48000 | 48000 | 48000 | 48000 |
