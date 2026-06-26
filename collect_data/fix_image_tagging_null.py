#!/usr/bin/env python3
"""
Fix image_tagging_mapper: null in generated YAML configs.

Replaces:
    - image_tagging_mapper: null

With:
    - image_tagging_mapper:
        pretrained_model_name_or_path: /data/coco2017/weights/ram_plus_swin_large_14m.pth
        text_encoder_type: /data/coco2017/weights/bert-base-uncased
        local_files_only: true

Usage (inside 105 container):
    python3 fix_image_tagging_null.py /data-juicer/pipeline_full/prepared_image/configs
    python3 fix_image_tagging_null.py /data-juicer/pipeline_full/prepared_image/configs --dry-run
"""
import argparse
import re
from pathlib import Path

REPLACEMENT = """\
- image_tagging_mapper:
        pretrained_model_name_or_path: /data/coco2017/weights/ram_plus_swin_large_14m.pth
        text_encoder_type: /data/coco2017/weights/bert-base-uncased
        local_files_only: true"""

# Pattern matches "- image_tagging_mapper: null" with any leading whitespace
PATTERN = re.compile(r'^(\s*)- image_tagging_mapper:\s*null\s*$', re.MULTILINE)


def fix_file(path: Path, dry_run: bool = False) -> bool:
    """Fix a single YAML file. Returns True if modified."""
    content = path.read_text(encoding='utf-8')

    match = PATTERN.search(content)
    if not match:
        return False

    indent = match.group(1)  # preserve original indentation
    # Build replacement with correct indentation
    lines = REPLACEMENT.split('\n')
    indented_lines = [indent + line for line in lines]
    replacement = '\n'.join(indented_lines)

    new_content = PATTERN.sub(replacement, content)

    if not dry_run:
        path.write_text(new_content, encoding='utf-8')
    return True


def main():
    parser = argparse.ArgumentParser(description='Fix image_tagging_mapper: null in YAML configs')
    parser.add_argument('configs_dir', help='Directory containing YAML config files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without modifying files')
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    if not configs_dir.is_dir():
        print(f'[ERROR] Not a directory: {configs_dir}')
        return 1

    yaml_files = sorted(configs_dir.glob('image_pipeline_*.yaml'))
    print(f'[INFO] Scanning {len(yaml_files)} image pipeline YAML files...')

    fixed = []
    for f in yaml_files:
        if fix_file(f, dry_run=args.dry_run):
            fixed.append(f.name)

    action = 'Would fix' if args.dry_run else 'Fixed'
    print(f'[OK] {action} {len(fixed)} / {len(yaml_files)} files')

    if fixed and len(fixed) <= 20:
        for name in fixed:
            print(f'  - {name}')
    elif fixed:
        print(f'  First 5: {", ".join(fixed[:5])}')
        print(f'  Last  5: {", ".join(fixed[-5:])}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
