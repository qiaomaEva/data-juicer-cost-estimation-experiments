#!/usr/bin/env python3
"""
Generate runnable Data-Juicer YAMLs from template YAMLs.

Compared with prepare_pipeline_runs.py:
- keeps the same output structure (configs + manifests/all_jobs.tsv)
- supports text multi-scale expansion (e.g. *_1000, *_5000, *_10000)
- keeps audio/image size expansion behavior
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


AUDIO_SIZE_MAP: List[Tuple[str, str]] = [
    ("128MB", "test-by-size-128MiB.jsonl"),
    ("256MB", "test-by-size-256MiB.jsonl"),
    ("512MB", "test-by-size-512MiB.jsonl"),
    ("1G", "test-by-size-1024MiB.jsonl"),
    ("1.5G", "test-by-size-1536MiB.jsonl"),
    ("2G", "test-by-size-2048MiB.jsonl"),
    ("3G", "test-by-size-3072MiB.jsonl"),
    ("4G", "test-by-size-4096MiB.jsonl"),
    ("6G", "test-by-size-6144MiB.jsonl"),
    ("8G", "test-by-size-8192MiB.jsonl"),
]

IMAGE_SIZE_MAP: List[Tuple[str, str]] = [
    ("128MB", "coco_physical_128M.jsonl"),
    ("256MB", "coco_physical_256M.jsonl"),
    ("512MB", "coco_physical_512M.jsonl"),
    ("1G", "coco_physical_1G.jsonl"),
    ("1.5G", "coco_physical_1.5G.jsonl"),
    ("2G", "coco_physical_2G.jsonl"),
    ("3G", "coco_physical_3G.jsonl"),
    ("4G", "coco_physical_4G.jsonl"),
    ("6G", "coco_physical_6G.jsonl"),
    ("8G", "coco_physical_8G.jsonl"),
]

DEFAULT_TEXT_SIZE_MAP: List[Tuple[str, str]] = [
    ("1000", "c4_1000.jsonl"),
    ("2000", "c4_2000.jsonl"),
    ("5000", "c4_5000.jsonl"),
    ("8000", "c4_8000.jsonl"),
    ("10000", "c4_10000.jsonl"),
]


def yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def to_posix_path(value: str) -> str:
    return value.replace("\\", "/")


def join_posix(base: str, name: str) -> str:
    return to_posix_path(base).rstrip("/") + "/" + name


def parse_label_file_map(text: str, default_map: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    raw = text.strip()
    if not raw:
        return list(default_map)

    pairs: List[Tuple[str, str]] = []
    for token in [x.strip() for x in raw.split(",") if x.strip()]:
        if ":" not in token:
            raise ValueError(f"Invalid token '{token}'. Expect format label:file")
        label, file_name = token.split(":", 1)
        label = label.strip()
        file_name = file_name.strip()
        if not label or not file_name:
            raise ValueError(f"Invalid token '{token}'. Label and file must both be non-empty.")
        pairs.append((label, file_name))
    return pairs


def replace_project_name(content: str, project_name: str) -> str:
    new_line = f"project_name: {project_name}"
    if re.search(r"(?m)^project_name:\s*.*$", content):
        return re.sub(r"(?m)^project_name:\s*.*$", new_line, content, count=1)
    return new_line + "\n" + content


def replace_export_path(content: str, export_path: str) -> str:
    quoted = yaml_quote(export_path)
    if "<placeholder_export_path>" in content:
        return content.replace("<placeholder_export_path>", export_path)
    if re.search(r"(?m)^export_path:\s*.*$", content):
        return re.sub(r"(?m)^export_path:\s*.*$", f"export_path: {quoted}", content, count=1)
    return content + f"\nexport_path: {quoted}\n"


def replace_dataset_path(content: str, dataset_path: str) -> str:
    if "<placeholder_dataset_path>" in content:
        return content.replace("<placeholder_dataset_path>", dataset_path)

    quoted = yaml_quote(dataset_path)
    if re.search(r"(?m)^dataset_path:\s*.*$", content):
        return re.sub(r"(?m)^dataset_path:\s*.*$", f"dataset_path: {quoted}", content, count=1)
    if re.search(r"(?m)^\s*path:\s*.*$", content):
        return re.sub(r"(?m)^(\s*path:\s*).*$", rf"\1{quoted}", content, count=1)

    raise ValueError("No dataset path field found in template.")


def append_run_name_to_save_dirs(content: str, run_name: str) -> str:
    pattern = re.compile(r'(?m)^(\s*save_dir:\s*)(?:"([^"]+)"|\'([^\']+)\'|(\S+))\s*$')

    def repl(match: re.Match[str]) -> str:
        prefix = match.group(1)
        raw_path = match.group(2) or match.group(3) or match.group(4) or ""
        if raw_path.endswith("/" + run_name):
            new_path = raw_path
        else:
            new_path = raw_path.rstrip("/") + "/" + run_name
        return f"{prefix}{yaml_quote(new_path)}"

    return pattern.sub(repl, content)


def build_config(template: str, run_name: str, dataset_path: str, export_path: str, update_save_dirs: bool) -> str:
    content = template
    content = replace_project_name(content, run_name)
    content = replace_dataset_path(content, dataset_path)
    content = replace_export_path(content, export_path)
    if update_save_dirs:
        content = append_run_name_to_save_dirs(content, run_name)
    return content


def parse_text_map(
    text_files: Sequence[Path],
    text_map_file: Path | None,
    text_glob_paths: Sequence[Path] | None,
    text_default_path: str,
) -> Dict[str, str]:
    names = [p.name for p in text_files]

    if text_map_file:
        if not text_map_file.exists():
            raise FileNotFoundError(f"text map file not found: {text_map_file}")
        lines = [
            line.strip()
            for line in text_map_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not lines:
            raise ValueError("text map file is empty.")

        one_col: List[str] = []
        two_col: Dict[str, str] = {}
        for line in lines:
            if "\t" in line:
                parts = [p.strip() for p in line.split("\t") if p.strip()]
            elif "," in line:
                parts = [p.strip() for p in line.split(",") if p.strip()]
            else:
                parts = line.split(maxsplit=1)
                parts = [p.strip() for p in parts if p.strip()]

            if len(parts) == 1:
                one_col.append(parts[0])
            elif len(parts) >= 2:
                two_col[parts[0]] = parts[1]
            else:
                raise ValueError(f"Invalid line in text map: {line}")

        if one_col and two_col:
            raise ValueError("text map file mixes 1-column and 2-column rows; please use one format only.")

        if one_col:
            if len(one_col) != len(text_files):
                raise ValueError(
                    f"text map has {len(one_col)} paths, but found {len(text_files)} text YAML files."
                )
            return {name: one_col[idx] for idx, name in enumerate(names)}

        missing = [name for name in names if name not in two_col]
        if missing:
            raise ValueError(
                "text map is missing YAML names: " + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            )
        return {name: two_col[name] for name in names}

    if text_glob_paths:
        if len(text_glob_paths) < len(text_files):
            raise ValueError(
                f"text_glob matched {len(text_glob_paths)} files, fewer than text YAML count {len(text_files)}."
            )
        return {name: str(text_glob_paths[idx]) for idx, name in enumerate(names)}

    return {name: text_default_path for name in names}


def apply_text_placeholders(path_value: str, size_label: str, index_file: str) -> str:
    return (
        path_value
        .replace("{size}", size_label)
        .replace("{label}", size_label)
        .replace("{index_file}", index_file)
    )


def write_manifests(manifest_dir: Path, rows: Sequence[Tuple[str, str, str, str, str, str]]) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for name in ("all_jobs.tsv", "all_jobs_per_run.tsv"):
        path = manifest_dir / name
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("#config_path\tmodality\tsize_label\trun_name\tdataset_path\texport_path\n")
            for row in rows:
                f.write("\t".join(row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare multi-scale pipeline YAMLs for batch experiments.")
    parser.add_argument("--pipeline-dir", default="./pipeline", help="Directory containing template YAML files.")
    parser.add_argument("--out-dir", default="./prepared_ai", help="Output directory for generated YAMLs/manifests.")
    parser.add_argument("--export-root", default="/home/yyw/pipeline_full/results", help="Root export path.")
    parser.add_argument(
        "--audio-indices-dir",
        default="/data/data-juicer_dataset/audio/indices",
        help="Audio indices directory.",
    )
    parser.add_argument(
        "--image-indices-dir",
        default="/data/data-juicer_dataset/coco2017/indices",
        help="Image indices directory.",
    )
    parser.add_argument(
        "--text-indices-dir",
        default="/data/data-juicer_dataset/RedPajama/indices",
        help="Text indices directory for multi-scale runs.",
    )
    parser.add_argument(
        "--text-default-path",
        default="/data/data-juicer_dataset/RedPajama/wikipedia/wiki.jsonl",
        help="Fallback text dataset path when text-size-map is empty.",
    )
    parser.add_argument(
        "--text-map",
        default="",
        help="Optional map file for per-text-yaml dataset path. Supports placeholders: {size},{label},{index_file}.",
    )
    parser.add_argument(
        "--text-glob",
        default="",
        help="Optional glob for text dataset files; supports placeholders too after assignment.",
    )
    parser.add_argument(
        "--audio-size-map",
        default="",
        help="Override audio size map: '128MB:test-by-size-128MiB.jsonl,1G:test-by-size-1024MiB.jsonl'",
    )
    parser.add_argument(
        "--image-size-map",
        default="",
        help="Override image size map: '128MB:coco_physical_128M.jsonl,1G:coco_physical_1G.jsonl'",
    )
    parser.add_argument(
        "--text-size-map",
        default="1000:c4_1000.jsonl,2000:c4_2000.jsonl,5000:c4_5000.jsonl,8000:c4_8000.jsonl,10000:c4_10000.jsonl",
        help="Text scale map. Use empty string to disable text expansion and keep one run per text YAML.",
    )
    parser.add_argument(
        "--keep-shared-save-dir",
        action="store_true",
        help="Do not append run name to mapper save_dir fields.",
    )
    parser.add_argument(
        "--modalities",
        default="audio,image,text",
        help="Comma-separated modalities to generate: audio,image,text",
    )
    args = parser.parse_args()

    selected_modalities = {part.strip() for part in args.modalities.split(",") if part.strip()}
    valid_modalities = {"audio", "image", "text"}
    unknown_modalities = selected_modalities - valid_modalities
    if unknown_modalities:
        print(
            f"[ERROR] invalid modalities: {sorted(unknown_modalities)}; valid values: audio,image,text",
            file=sys.stderr,
        )
        return 1

    try:
        audio_size_map = parse_label_file_map(args.audio_size_map, AUDIO_SIZE_MAP)
        image_size_map = parse_label_file_map(args.image_size_map, IMAGE_SIZE_MAP)
        text_size_map = parse_label_file_map(args.text_size_map, DEFAULT_TEXT_SIZE_MAP) if args.text_size_map else []
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    pipeline_dir = Path(args.pipeline_dir)
    out_dir = Path(args.out_dir)
    configs_dir = out_dir / "configs"
    manifest_dir = out_dir / "manifests"
    if not pipeline_dir.exists():
        print(f"[ERROR] pipeline dir not found: {pipeline_dir}", file=sys.stderr)
        return 1

    audio_files = sorted(pipeline_dir.glob("audio_pipeline_*.yaml"))
    image_files = sorted(pipeline_dir.glob("image_pipeline_*.yaml"))
    text_files = sorted(pipeline_dir.glob("text_pipeline_*.yaml"))
    print(
        "[INFO] found templates: "
        f"audio={len(audio_files)} image={len(image_files)} text={len(text_files)}"
    )

    text_dataset_by_yaml: Dict[str, str] = {}
    has_text_custom_map = bool(args.text_map or args.text_glob)
    if "text" in selected_modalities:
        text_map_file = Path(args.text_map) if args.text_map else None
        text_glob_paths = [Path(p) for p in sorted(glob.glob(args.text_glob))] if args.text_glob else None
        text_dataset_by_yaml = parse_text_map(
            text_files=text_files,
            text_map_file=text_map_file,
            text_glob_paths=text_glob_paths,
            text_default_path=args.text_default_path,
        )

    configs_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[Tuple[str, str, str, str, str, str]] = []
    update_save_dirs = not args.keep_shared_save_dir

    if "audio" in selected_modalities:
        for template_file in audio_files:
            template = template_file.read_text(encoding="utf-8")
            stem = template_file.stem
            for size_label, index_file in audio_size_map:
                run_name = f"{stem}_{size_label}"
                dataset_path = join_posix(args.audio_indices_dir, index_file)
                export_path = f"{args.export_root}/audio/{run_name}.jsonl"
                out_path = configs_dir / f"{run_name}.yaml"
                out_path.write_text(
                    build_config(template, run_name, dataset_path, export_path, update_save_dirs),
                    encoding="utf-8",
                )
                jobs.append((str(out_path.resolve()), "audio", size_label, run_name, dataset_path, export_path))

    if "image" in selected_modalities:
        for template_file in image_files:
            template = template_file.read_text(encoding="utf-8")
            stem = template_file.stem
            for size_label, index_file in image_size_map:
                run_name = f"{stem}_{size_label}"
                dataset_path = join_posix(args.image_indices_dir, index_file)
                export_path = f"{args.export_root}/image/{run_name}.jsonl"
                out_path = configs_dir / f"{run_name}.yaml"
                out_path.write_text(
                    build_config(template, run_name, dataset_path, export_path, update_save_dirs),
                    encoding="utf-8",
                )
                jobs.append((str(out_path.resolve()), "image", size_label, run_name, dataset_path, export_path))

    if "text" in selected_modalities:
        for template_file in text_files:
            template = template_file.read_text(encoding="utf-8")
            stem = template_file.stem

            if text_size_map:
                for size_label, index_file in text_size_map:
                    run_name = f"{stem}_{size_label}"
                    if has_text_custom_map:
                        base_path = to_posix_path(text_dataset_by_yaml[template_file.name])
                        dataset_path = apply_text_placeholders(base_path, size_label, index_file)
                    else:
                        dataset_path = join_posix(args.text_indices_dir, index_file)
                    export_path = f"{args.export_root}/text/{run_name}.jsonl"
                    out_path = configs_dir / f"{run_name}.yaml"
                    out_path.write_text(
                        build_config(template, run_name, dataset_path, export_path, update_save_dirs),
                        encoding="utf-8",
                    )
                    jobs.append((str(out_path.resolve()), "text", size_label, run_name, dataset_path, export_path))
            else:
                run_name = stem
                dataset_path = to_posix_path(text_dataset_by_yaml[template_file.name])
                export_path = f"{args.export_root}/text/{run_name}.jsonl"
                out_path = configs_dir / f"{run_name}.yaml"
                out_path.write_text(
                    build_config(template, run_name, dataset_path, export_path, update_save_dirs),
                    encoding="utf-8",
                )
                jobs.append((str(out_path.resolve()), "text", "-", run_name, dataset_path, export_path))

    write_manifests(manifest_dir, jobs)

    audio_jobs = sum(1 for row in jobs if row[1] == "audio")
    image_jobs = sum(1 for row in jobs if row[1] == "image")
    text_jobs = sum(1 for row in jobs if row[1] == "text")
    print(f"[OK] Generated configs: {len(jobs)}")
    print(f"     audio={audio_jobs}, image={image_jobs}, text={text_jobs}")
    print(f"     configs dir: {configs_dir}")
    print(f"     manifests:   {manifest_dir / 'all_jobs.tsv'}")
    print(f"                  {manifest_dir / 'all_jobs_per_run.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
