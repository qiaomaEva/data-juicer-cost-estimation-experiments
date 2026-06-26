#!/usr/bin/env python3
"""
分析 coco2017 / AudioSet / RedPajama-c4 三个数据集的物理分布，
输出关键指标的分位数（p5/p10/p25/p50/p75/p90/p95），供 LHS 参数范围设计使用。

运行环境：104 服务器上的 data-juicer 容器
    docker exec -it yyw_dj_test bash
    cd /home/yyw
    # 全扫三个数据集（推荐）
    python analyze_dataset_distribution.py \
        --scan_audio --scan_image --scan_text \
        --output ./dataset_stats_full.json

    # 单文件快速验证
    python analyze_dataset_distribution.py \
        --audio_index audio/indices/test-by-size-128MiB.jsonl \
        --image_index coco2017/indices/coco_physical_128M.jsonl \
        --text_index RedPajama/indices/c4_1000.jsonl \
        --output ./dataset_stats.json

用 soundfile.info / PIL lazy-open 只读 header，不解码数据。
全扫模式下每个文件独立输出分位数，并在 merged 字段给出跨文件合并后的总体分位数。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("[fatal] numpy required", file=sys.stderr)
    sys.exit(1)

PCTS = [5, 10, 25, 50, 75, 90, 95]


def percentiles(arr):
    if not arr:
        return {"n": 0}
    a = np.asarray(arr, dtype=float)
    out = {
        "n": int(len(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }
    for p in PCTS:
        out[f"p{p}"] = float(np.percentile(a, p))
    return out


def extract_path_field(rec, candidates):
    """从 jsonl 记录里找出可能的路径字段。"""
    for k in candidates:
        if k in rec:
            v = rec[k]
            if isinstance(v, str):
                return [v]
            if isinstance(v, list):
                return [x for x in v if isinstance(x, str)]
    return []


def resolve_path(raw_path, search_roots):
    """
    尝试把 jsonl 里的路径解析为真实文件：
    1) 如果是绝对路径且文件存在，直接用
    2) 否则依次和 search_roots 拼接，返回第一个存在的
    3) 还不行：尝试去掉 raw_path 前面若干级，用 basename/倒数两级/倒数三级去各 root 下找
    """
    if not raw_path:
        return None
    if os.path.isabs(raw_path) and os.path.isfile(raw_path):
        return raw_path
    rel = raw_path.lstrip("/").lstrip("\\")
    for r in search_roots:
        cand = os.path.join(r, rel)
        if os.path.isfile(cand):
            return cand
    # 按路径末尾若干段 fuzzy 匹配
    parts = raw_path.replace("\\", "/").split("/")
    for depth in (1, 2, 3):
        tail = "/".join(parts[-depth:])
        for r in search_roots:
            cand = os.path.join(r, tail)
            if os.path.isfile(cand):
                return cand
    return None


def peek_record_structure(index_file, max_records=3):
    """打印 jsonl 前几条记录的键结构，便于诊断字段名。"""
    print(f"  [peek] first records of {index_file}:")
    with open(index_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_records:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"    #{i}: <bad json>")
                continue
            keys = list(rec.keys())
            preview = {}
            for k in keys[:8]:
                v = rec[k]
                if isinstance(v, str):
                    preview[k] = v[:80] + ("..." if len(v) > 80 else "")
                elif isinstance(v, list):
                    preview[k] = f"list(len={len(v)}, head={v[:1]})"
                else:
                    preview[k] = type(v).__name__
            print(f"    #{i} keys={keys}")
            print(f"        preview={preview}")


def analyze_audio(index_file, max_samples, search_roots, verbose=True, return_raw=False):
    """
    分析 audio 索引文件。
    max_samples: 最大扫描行数，None 或 0 表示无限制。
    return_raw: 是否在结果中包含原始数据数组（用于合并统计）。
    """
    try:
        import soundfile as sf
    except ImportError:
        print("[warn] soundfile missing — skip audio", file=sys.stderr)
        return {"error": "soundfile not installed"}

    peek_record_structure(index_file)

    sizes_kb, durations, srs, channels = [], [], [], []
    missed, errored = 0, 0
    missed_samples = []
    t0 = time.time()

    with open(index_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and max_samples > 0 and i >= max_samples:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            paths = extract_path_field(rec, ["audios", "audio", "audio_path"])
            for raw in paths:
                p = resolve_path(raw, search_roots)
                if p is None:
                    missed += 1
                    if len(missed_samples) < 5:
                        missed_samples.append(raw)
                    continue
                try:
                    sizes_kb.append(os.path.getsize(p) / 1024.0)
                    info = sf.info(p)
                    durations.append(float(info.duration))
                    srs.append(int(info.samplerate))
                    channels.append(int(info.channels))
                except Exception:
                    errored += 1

            if verbose and (i + 1) % 200 == 0:
                print(f"  [audio] scanned {i+1} records, ok={len(sizes_kb)}, "
                      f"missed={missed}, err={errored}, t={time.time()-t0:.1f}s")

    if missed_samples:
        print(f"  [audio] 示例 missed 原始路径: {missed_samples}")

    result = {
        "source_index": str(index_file),
        "search_roots": search_roots,
        "scanned_records": i + 1 if 'i' in locals() else 0,
        "missed_files": missed,
        "errored_files": errored,
        "resolved_files": len(sizes_kb),
        "file_size_kb": percentiles(sizes_kb),
        "duration_sec": percentiles(durations),
        "sample_rate_hz": percentiles(srs),
        "channels": percentiles(channels),
    }
    if return_raw:
        result["raw_data"] = {
            "file_size_kb": sizes_kb,
            "duration_sec": durations,
            "sample_rate_hz": srs,
            "channels": channels,
        }
    return result


def analyze_image(index_file, max_samples, search_roots, verbose=True, return_raw=False):
    """
    分析 image 索引文件。
    max_samples: 最大扫描行数，None 或 0 表示无限制。
    return_raw: 是否在结果中包含原始数据数组（用于合并统计）。
    """
    try:
        from PIL import Image
    except ImportError:
        print("[warn] PIL missing — skip image", file=sys.stderr)
        return {"error": "PIL not installed"}

    peek_record_structure(index_file)

    sizes_kb, widths, heights, aspects = [], [], [], []
    missed, errored = 0, 0
    missed_samples = []
    t0 = time.time()

    with open(index_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and max_samples > 0 and i >= max_samples:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            paths = extract_path_field(rec, ["images", "image", "image_path"])
            for raw in paths:
                p = resolve_path(raw, search_roots)
                if p is None:
                    missed += 1
                    if len(missed_samples) < 5:
                        missed_samples.append(raw)
                    continue
                try:
                    sizes_kb.append(os.path.getsize(p) / 1024.0)
                    with Image.open(p) as im:
                        w, h = im.size
                    widths.append(int(w))
                    heights.append(int(h))
                    aspects.append(float(w) / max(int(h), 1))
                except Exception:
                    errored += 1

            if verbose and (i + 1) % 200 == 0:
                print(f"  [image] scanned {i+1} records, ok={len(sizes_kb)}, "
                      f"missed={missed}, err={errored}, t={time.time()-t0:.1f}s")

    if missed_samples:
        print(f"  [image] 示例 missed 原始路径: {missed_samples}")

    result = {
        "source_index": str(index_file),
        "search_roots": search_roots,
        "scanned_records": i + 1 if 'i' in locals() else 0,
        "missed_files": missed,
        "errored_files": errored,
        "resolved_files": len(sizes_kb),
        "file_size_kb": percentiles(sizes_kb),
        "width_px": percentiles(widths),
        "height_px": percentiles(heights),
        "aspect_ratio_w_over_h": percentiles(aspects),
    }
    if return_raw:
        result["raw_data"] = {
            "file_size_kb": sizes_kb,
            "width_px": widths,
            "height_px": heights,
            "aspect_ratio_w_over_h": aspects,
        }
    return result


def analyze_text(index_file, max_samples, verbose=True, return_raw=False):
    """
    分析 text 索引文件。
    max_samples: 最大扫描行数，None 或 0 表示无限制。
    return_raw: 是否在结果中包含原始数据数组（用于合并统计）。
    """
    char_lens, word_counts, line_max_lens, sent_counts = [], [], [], []
    t0 = time.time()

    with open(index_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and max_samples > 0 and i >= max_samples:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text") or rec.get("content") or ""
            if not isinstance(text, str):
                text = json.dumps(text, ensure_ascii=False)
            char_lens.append(len(text))
            word_counts.append(len(text.split()))
            lines = text.splitlines() or [text]
            line_max_lens.append(max(len(l) for l in lines))
            sent_counts.append(text.count(".") + text.count("!") + text.count("?") + 1)

            if verbose and (i + 1) % 1000 == 0:
                print(f"  [text] scanned {i+1} records, t={time.time()-t0:.1f}s")

    token_approx = [c / 4.0 for c in char_lens]
    result = {
        "source_index": str(index_file),
        "scanned_records": i + 1 if 'i' in locals() else 0,
        "char_length": percentiles(char_lens),
        "word_count": percentiles(word_counts),
        "token_count_approx_chars_div_4": percentiles(token_approx),
        "max_line_length": percentiles(line_max_lens),
        "sentence_count": percentiles(sent_counts),
    }
    if return_raw:
        result["raw_data"] = {
            "char_length": char_lens,
            "word_count": word_counts,
            "token_count_approx_chars_div_4": token_approx,
            "max_line_length": line_max_lens,
            "sentence_count": sent_counts,
        }
    return result


def merge_raw_data(all_raw_data_list, metric_keys):
    """
    合并多个文件的原始数据，计算整体分位数。
    all_raw_data_list: list of dict，每个 dict 包含 metric_keys 对应的数组。
    metric_keys: 要合并的指标键名列表。
    返回一个 dict，包含每个指标的合并分位数。
    """
    merged = {}
    for key in metric_keys:
        all_values = []
        for raw_data in all_raw_data_list:
            if key in raw_data:
                all_values.extend(raw_data[key])
        if all_values:
            merged[key] = percentiles(all_values)
        else:
            merged[key] = {"n": 0}
    return merged


def find_all_indices(dataset_root, subdir_pattern):
    """在 dataset_root/subdir_pattern 下查找所有 .jsonl 文件，返回排序后的路径列表。"""
    import glob
    pattern = os.path.join(dataset_root, subdir_pattern, "*.jsonl")
    files = sorted(glob.glob(pattern, recursive=False))
    # 也尝试匹配子目录下的 .jsonl（如 indices/ 下的子目录）
    pattern2 = os.path.join(dataset_root, subdir_pattern, "**", "*.jsonl")
    files2 = sorted(glob.glob(pattern2, recursive=True))
    all_files = sorted(set(files + files2))
    return [Path(f) for f in all_files]


def main():
    ap = argparse.ArgumentParser(
        description="分析数据集的物理分布（支持多规模全扫）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset_root", default="/data/data-juicer_dataset")

    # 单文件模式（向后兼容）
    ap.add_argument(
        "--audio_index",
        help="相对 dataset_root 的单个 audio indices 文件（如果设置，则不用 --scan_audio）"
    )
    ap.add_argument(
        "--image_index",
        help="相对 dataset_root 的单个 image indices 文件"
    )
    ap.add_argument(
        "--text_index",
        help="相对 dataset_root 的单个 text indices 文件"
    )

    # 全扫模式
    ap.add_argument(
        "--scan_audio", action="store_true",
        help="扫描 audio/indices/ 下所有 .jsonl 文件"
    )
    ap.add_argument(
        "--scan_image", action="store_true",
        help="扫描 coco2017/indices/ 下所有 .jsonl 文件"
    )
    ap.add_argument(
        "--scan_text", action="store_true",
        help="扫描 RedPajama/indices/ 下所有 .jsonl 文件"
    )

    ap.add_argument(
        "--audio_search_roots",
        nargs="*",
        default=[
            "/data/data-juicer_dataset/audio/audio_set/exports_agkphysics_AudioSet_balanced/audio",
            "/data/data-juicer_dataset/audio/audio_set/exports_agkphysics_AudioSet_balanced/audio/test",
            "/data/data-juicer_dataset/audio/audio_set/exports_agkphysics_AudioSet_balanced/audio/train",
            "/data/data-juicer_dataset",
        ],
        help="audio 文件搜索根目录列表；jsonl 里的路径会依次尝试拼接。",
    )
    ap.add_argument(
        "--image_search_roots",
        nargs="*",
        default=[
            "/data/data-juicer_dataset/coco2017",
            "/data/data-juicer_dataset/coco2017/train2017",
            "/data/data-juicer_dataset/coco2017/val2017",
            "/data/data-juicer_dataset",
        ],
    )
    ap.add_argument("--max_samples", type=int, default=0,
                    help="每个文件最多扫描的 jsonl 行数，0 表示无限制（全扫）")
    ap.add_argument("--output", default="./dataset_stats_full.json")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    stats = {
        "meta": {
            "dataset_root": str(root),
            "max_samples_per_file": args.max_samples,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scan_mode": {
                "audio": "single" if args.audio_index else ("full" if args.scan_audio else "none"),
                "image": "single" if args.image_index else ("full" if args.scan_image else "none"),
                "text": "single" if args.text_index else ("full" if args.scan_text else "none"),
            }
        }
    }

    # Audio
    if args.audio_index:
        # 单文件模式
        audio_f = root / args.audio_index
        if audio_f.exists():
            print(f"\n[audio] 单文件模式: {audio_f}")
            print(f"  search_roots={args.audio_search_roots}")
            stats["audio_AudioSet"] = analyze_audio(
                audio_f, args.max_samples if args.max_samples > 0 else None,
                args.audio_search_roots
            )
        else:
            print(f"[skip] audio index not found: {audio_f}")
    elif args.scan_audio:
        # 全扫模式
        indices_dir = "audio/indices"
        all_files = find_all_indices(str(root), indices_dir)
        if not all_files:
            print(f"[warn] 未找到 audio indices 文件: {root}/{indices_dir}/*.jsonl")
        else:
            print(f"\n[audio] 全扫模式，找到 {len(all_files)} 个索引文件:")
            for f in all_files:
                print(f"  - {f.relative_to(root)}")
            stats["audio_AudioSet"] = {}
            stats["audio_AudioSet"]["by_file"] = []
            all_raw_data = []
            for idx_file in all_files:
                print(f"\n  --- 处理 {idx_file.name} ---")
                file_stats = analyze_audio(
                    idx_file, args.max_samples if args.max_samples > 0 else None,
                    args.audio_search_roots, verbose=True, return_raw=True
                )
                stats["audio_AudioSet"]["by_file"].append({
                    "file": str(idx_file.relative_to(root)),
                    "stats": {k: v for k, v in file_stats.items() if k != "raw_data"}
                })
                if "raw_data" in file_stats:
                    all_raw_data.append(file_stats["raw_data"])
            # 合并统计
            if all_raw_data:
                merged = merge_raw_data(all_raw_data,
                    ["file_size_kb", "duration_sec", "sample_rate_hz", "channels"])
                stats["audio_AudioSet"]["merged"] = merged
            else:
                stats["audio_AudioSet"]["merged"] = {"note": "无原始数据可合并"}
    else:
        print("[skip] audio 未指定（请用 --audio_index 或 --scan_audio）")

    # Image
    if args.image_index:
        image_f = root / args.image_index
        if image_f.exists():
            print(f"\n[image] 单文件模式: {image_f}")
            print(f"  search_roots={args.image_search_roots}")
            stats["image_COCO2017"] = analyze_image(
                image_f, args.max_samples if args.max_samples > 0 else None,
                args.image_search_roots
            )
        else:
            print(f"[skip] image index not found: {image_f}")
    elif args.scan_image:
        indices_dir = "coco2017/indices"
        all_files = find_all_indices(str(root), indices_dir)
        if not all_files:
            print(f"[warn] 未找到 image indices 文件: {root}/{indices_dir}/*.jsonl")
        else:
            print(f"\n[image] 全扫模式，找到 {len(all_files)} 个索引文件:")
            for f in all_files:
                print(f"  - {f.relative_to(root)}")
            stats["image_COCO2017"] = {}
            stats["image_COCO2017"]["by_file"] = []
            all_raw_data = []
            for idx_file in all_files:
                print(f"\n  --- 处理 {idx_file.name} ---")
                file_stats = analyze_image(
                    idx_file, args.max_samples if args.max_samples > 0 else None,
                    args.image_search_roots, verbose=True, return_raw=True
                )
                stats["image_COCO2017"]["by_file"].append({
                    "file": str(idx_file.relative_to(root)),
                    "stats": {k: v for k, v in file_stats.items() if k != "raw_data"}
                })
                if "raw_data" in file_stats:
                    all_raw_data.append(file_stats["raw_data"])
            if all_raw_data:
                merged = merge_raw_data(all_raw_data,
                    ["file_size_kb", "width_px", "height_px", "aspect_ratio_w_over_h"])
                stats["image_COCO2017"]["merged"] = merged
            else:
                stats["image_COCO2017"]["merged"] = {"note": "无原始数据可合并"}
    else:
        print("[skip] image 未指定（请用 --image_index 或 --scan_image）")

    # Text
    if args.text_index:
        text_f = root / args.text_index
        if text_f.exists():
            print(f"\n[text] 单文件模式: {text_f}")
            stats["text_RedPajama_c4"] = analyze_text(
                text_f, args.max_samples if args.max_samples > 0 else None
            )
        else:
            print(f"[skip] text index not found: {text_f}")
    elif args.scan_text:
        indices_dir = "RedPajama/indices"
        all_files = find_all_indices(str(root), indices_dir)
        if not all_files:
            print(f"[warn] 未找到 text indices 文件: {root}/{indices_dir}/*.jsonl")
        else:
            print(f"\n[text] 全扫模式，找到 {len(all_files)} 个索引文件:")
            for f in all_files:
                print(f"  - {f.relative_to(root)}")
            stats["text_RedPajama_c4"] = {}
            stats["text_RedPajama_c4"]["by_file"] = []
            all_raw_data = []
            for idx_file in all_files:
                print(f"\n  --- 处理 {idx_file.name} ---")
                file_stats = analyze_text(
                    idx_file, args.max_samples if args.max_samples > 0 else None,
                    verbose=True, return_raw=True
                )
                stats["text_RedPajama_c4"]["by_file"].append({
                    "file": str(idx_file.relative_to(root)),
                    "stats": {k: v for k, v in file_stats.items() if k != "raw_data"}
                })
                if "raw_data" in file_stats:
                    all_raw_data.append(file_stats["raw_data"])
            if all_raw_data:
                merged = merge_raw_data(all_raw_data,
                    ["char_length", "word_count", "token_count_approx_chars_div_4",
                     "max_line_length", "sentence_count"])
                stats["text_RedPajama_c4"]["merged"] = merged
            else:
                stats["text_RedPajama_c4"]["merged"] = {"note": "无原始数据可合并"}
    else:
        print("[skip] text 未指定（请用 --text_index 或 --scan_text）")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n[ok] saved to {args.output}")
    print("--- preview ---")
    print(json.dumps(stats, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
