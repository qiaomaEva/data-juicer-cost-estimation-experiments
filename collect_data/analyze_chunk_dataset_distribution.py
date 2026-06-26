#!/usr/bin/env python3
"""
Analyze per-chunk dataset profile features for the chunk runtime experiment.

This script is intended to run on the servers where the chunk index files and
source media/text files are available. It mirrors the metric definitions in
analyze_dataset_distribution.py, but the output is keyed by each chunk so the
local cost replay can attach profile_* features per chunk.

Example commands:

Image server 105:
    python analyze_chunk_dataset_distribution.py \
        --scan_image_chunks \
        --image_chunk_root /data/coco2017/indices/chunks_by_count \
        --image_search_roots /data/coco2017 /data/coco2017/train2017 /data/coco2017/val2017 \
        --output ./chunk_profile_stats_image.json

Audio/text server 104:
    python analyze_chunk_dataset_distribution.py \
        --scan_audio_chunks --scan_text_chunks \
        --audio_chunk_root /data/data-juicer_dataset/audio/indices/chunks \
        --text_chunk_root /data/data-juicer_dataset/RedPajama/indices/chunks \
        --dataset_root /data/data-juicer_dataset \
        --output ./chunk_profile_stats_audio_text.json

The output JSON contains:
    meta
    chunks: [
      {
        ds_type,
        chunk_size,
        chunk_part,
        chunk_key,
        source_index,
        stats
      }
    ]
"""

import argparse
import json
import os
import re
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
    for key in candidates:
        if key not in rec:
            continue
        value = rec[key]
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str)]
    return []


def resolve_path(raw_path, search_roots):
    if not raw_path:
        return None
    if os.path.isabs(raw_path) and os.path.isfile(raw_path):
        return raw_path

    rel = raw_path.lstrip("/").lstrip("\\")
    for root in search_roots:
        candidate = os.path.join(root, rel)
        if os.path.isfile(candidate):
            return candidate

    parts = raw_path.replace("\\", "/").split("/")
    for depth in (1, 2, 3):
        tail = "/".join(parts[-depth:])
        for root in search_roots:
            candidate = os.path.join(root, tail)
            if os.path.isfile(candidate):
                return candidate
    return None


def analyze_audio(index_file, max_samples, search_roots, verbose=True):
    try:
        import soundfile as sf
    except ImportError:
        return {"error": "soundfile not installed"}

    sizes_kb, durations, sample_rates, channels = [], [], [], []
    missed, errored = 0, 0
    with open(index_file, "r", encoding="utf-8") as file_obj:
        for line_index, line in enumerate(file_obj):
            if max_samples and line_index >= max_samples:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for raw_path in extract_path_field(rec, ["audios", "audio", "audio_path"]):
                path = resolve_path(raw_path, search_roots)
                if path is None:
                    missed += 1
                    continue
                try:
                    sizes_kb.append(os.path.getsize(path) / 1024.0)
                    info = sf.info(path)
                    durations.append(float(info.duration))
                    sample_rates.append(int(info.samplerate))
                    channels.append(int(info.channels))
                except Exception:
                    errored += 1
            if verbose and (line_index + 1) % 500 == 0:
                print(f"    [audio] {index_file.name}: scanned {line_index + 1}")

    return {
        "source_index": str(index_file),
        "scanned_records": line_index + 1 if "line_index" in locals() else 0,
        "missed_files": missed,
        "errored_files": errored,
        "resolved_files": len(sizes_kb),
        "file_size_kb": percentiles(sizes_kb),
        "duration_sec": percentiles(durations),
        "sample_rate_hz": percentiles(sample_rates),
        "channels": percentiles(channels),
    }


def analyze_image(index_file, max_samples, search_roots, verbose=True):
    try:
        from PIL import Image
    except ImportError:
        return {"error": "PIL not installed"}

    sizes_kb, widths, heights, aspects = [], [], [], []
    missed, errored = 0, 0
    with open(index_file, "r", encoding="utf-8") as file_obj:
        for line_index, line in enumerate(file_obj):
            if max_samples and line_index >= max_samples:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for raw_path in extract_path_field(rec, ["images", "image", "image_path"]):
                path = resolve_path(raw_path, search_roots)
                if path is None:
                    missed += 1
                    continue
                try:
                    sizes_kb.append(os.path.getsize(path) / 1024.0)
                    with Image.open(path) as image_obj:
                        width, height = image_obj.size
                    widths.append(int(width))
                    heights.append(int(height))
                    aspects.append(float(width) / max(int(height), 1))
                except Exception:
                    errored += 1
            if verbose and (line_index + 1) % 500 == 0:
                print(f"    [image] {index_file.name}: scanned {line_index + 1}")

    return {
        "source_index": str(index_file),
        "scanned_records": line_index + 1 if "line_index" in locals() else 0,
        "missed_files": missed,
        "errored_files": errored,
        "resolved_files": len(sizes_kb),
        "file_size_kb": percentiles(sizes_kb),
        "width_px": percentiles(widths),
        "height_px": percentiles(heights),
        "aspect_ratio_w_over_h": percentiles(aspects),
    }


def analyze_text(index_file, max_samples, verbose=True):
    char_lengths, word_counts, max_line_lengths, sentence_counts = [], [], [], []
    with open(index_file, "r", encoding="utf-8") as file_obj:
        for line_index, line in enumerate(file_obj):
            if max_samples and line_index >= max_samples:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text") or rec.get("content") or ""
            if not isinstance(text, str):
                text = json.dumps(text, ensure_ascii=False)
            char_lengths.append(len(text))
            word_counts.append(len(text.split()))
            lines = text.splitlines() or [text]
            max_line_lengths.append(max(len(item) for item in lines))
            sentence_counts.append(text.count(".") + text.count("!") + text.count("?") + 1)
            if verbose and (line_index + 1) % 1000 == 0:
                print(f"    [text] {index_file.name}: scanned {line_index + 1}")

    token_approx = [value / 4.0 for value in char_lengths]
    return {
        "source_index": str(index_file),
        "scanned_records": line_index + 1 if "line_index" in locals() else 0,
        "char_length": percentiles(char_lengths),
        "word_count": percentiles(word_counts),
        "token_count_approx_chars_div_4": percentiles(token_approx),
        "max_line_length": percentiles(max_line_lengths),
        "sentence_count": percentiles(sentence_counts),
    }


def parse_chunk_identity(index_file, ds_type):
    path_text = str(index_file).replace("\\", "/")
    name = index_file.name

    if ds_type in {"audio", "text"}:
        size_match = re.search(r"chunk[-_]?(\d+)", path_text)
        part_match = re.search(r"_part(\d+)\.jsonl$", name)
        if part_match:
            base_name = re.sub(r"_part\d+\.jsonl$", "", name)
        else:
            base_name = index_file.stem
    else:
        size_match = re.search(r"_per_(\d+)", path_text)
        part_match = re.search(r"chunk_(\d+)\.jsonl$", name)
        base_match = re.search(r"([^/]+)_per_\d+/chunk_\d+\.jsonl$", path_text)
        base_name = base_match.group(1) if base_match else index_file.stem

    chunk_size = int(size_match.group(1)) if size_match else None
    chunk_part = int(part_match.group(1)) if part_match else None
    chunk_key = f"{ds_type}:{base_name}:chunk{chunk_size}:part{chunk_part}"
    return {
        "base_name": base_name,
        "chunk_size": chunk_size,
        "chunk_part": chunk_part,
        "chunk_key": chunk_key,
    }


def find_jsonl_files(root):
    root_path = Path(root)
    if not root_path.exists():
        return []
    files = []
    for path in root_path.rglob("*.jsonl"):
        if not path.is_file():
            continue
        path_text = str(path).replace("\\", "/")
        if "/trace/" in path_text:
            continue
        if path.name.endswith("_stats.jsonl"):
            continue
        files.append(path)
    return sorted(files)


def analyze_chunk_files(ds_type, files, max_samples, search_roots, verbose=True):
    rows = []
    for file_index, index_file in enumerate(files, start=1):
        identity = parse_chunk_identity(index_file, ds_type)
        print(
            f"[{ds_type}] ({file_index}/{len(files)}) chunk_size={identity['chunk_size']} "
            f"part={identity['chunk_part']} file={index_file}"
        )
        if ds_type == "audio":
            stats = analyze_audio(index_file, max_samples, search_roots, verbose=verbose)
        elif ds_type == "image":
            stats = analyze_image(index_file, max_samples, search_roots, verbose=verbose)
        elif ds_type == "text":
            stats = analyze_text(index_file, max_samples, verbose=verbose)
        else:
            raise ValueError(f"unknown ds_type: {ds_type}")

        rows.append(
            {
                "ds_type": ds_type,
                **identity,
                "source_index": str(index_file),
                "stats": stats,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Analyze profile statistics for every chunk index file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_root", default="/data/data-juicer_dataset")
    parser.add_argument("--audio_chunk_root", default="/data/data-juicer_dataset/audio/indices/chunks")
    parser.add_argument("--image_chunk_root", default="/data/coco2017/indices/chunks_by_count")
    parser.add_argument("--text_chunk_root", default="/data/data-juicer_dataset/RedPajama/indices/chunks")
    parser.add_argument("--scan_audio_chunks", action="store_true")
    parser.add_argument("--scan_image_chunks", action="store_true")
    parser.add_argument("--scan_text_chunks", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0, help="Max records per chunk; 0 means all.")
    parser.add_argument("--output", default="./chunk_profile_stats.json")
    parser.add_argument(
        "--audio_search_roots",
        nargs="*",
        default=[
            "/data/data-juicer_dataset/audio/audio_set/exports_agkphysics_AudioSet_balanced/audio",
            "/data/data-juicer_dataset/audio/audio_set/exports_agkphysics_AudioSet_balanced/audio/test",
            "/data/data-juicer_dataset/audio/audio_set/exports_agkphysics_AudioSet_balanced/audio/train",
            "/data/data-juicer_dataset",
        ],
    )
    parser.add_argument(
        "--image_search_roots",
        nargs="*",
        default=[
            "/data/coco2017",
            "/data/coco2017/train2017",
            "/data/coco2017/val2017",
            "/data/data-juicer_dataset/coco2017",
            "/data/data-juicer_dataset/coco2017/train2017",
            "/data/data-juicer_dataset/coco2017/val2017",
        ],
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    max_samples = args.max_samples if args.max_samples > 0 else None
    all_chunks = []
    started_at = time.time()

    if args.scan_audio_chunks:
        files = find_jsonl_files(args.audio_chunk_root)
        print(f"[audio] found {len(files)} chunk jsonl files under {args.audio_chunk_root}")
        all_chunks.extend(
            analyze_chunk_files("audio", files, max_samples, args.audio_search_roots, verbose=not args.quiet)
        )

    if args.scan_image_chunks:
        files = find_jsonl_files(args.image_chunk_root)
        print(f"[image] found {len(files)} chunk jsonl files under {args.image_chunk_root}")
        all_chunks.extend(
            analyze_chunk_files("image", files, max_samples, args.image_search_roots, verbose=not args.quiet)
        )

    if args.scan_text_chunks:
        files = find_jsonl_files(args.text_chunk_root)
        print(f"[text] found {len(files)} chunk jsonl files under {args.text_chunk_root}")
        all_chunks.extend(analyze_chunk_files("text", files, max_samples, [], verbose=not args.quiet))

    output = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(time.time() - started_at, 3),
            "max_samples_per_chunk": args.max_samples,
            "audio_chunk_root": args.audio_chunk_root,
            "image_chunk_root": args.image_chunk_root,
            "text_chunk_root": args.text_chunk_root,
            "chunk_count": len(all_chunks),
        },
        "chunks": all_chunks,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path(".") else None
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(output, file_obj, indent=2, ensure_ascii=False)

    print(f"[ok] saved chunk profile stats: {output_path}")
    print(f"[ok] chunks: {len(all_chunks)}")


if __name__ == "__main__":
    main()
