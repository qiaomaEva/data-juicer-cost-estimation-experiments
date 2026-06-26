"""
数据内容文件路径：./assets/text.json
生成 pipeline 文件夹路径：./collect_data/pipeline_yaml/
pipeline 的 np 可选值为 [30, 40, 50]。
参数格式支持 continuous（LHS 采样）和 categorical（随机选一个）。
用户使用方法：python generator_text_pipeline.py --num_pipelines 5
"""

import os, json, random, argparse, time
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats.qmc import LatinHypercube
from collections import defaultdict

DATASET_CONFIG = {
    "configs": [{"type": "local", "path": "<placeholder_dataset_path>", "format": "jsonl"}]
}
EXPORT_PATH = "<placeholder_export_path>"


def load_config(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_lhs_samples(operators, n):
    lhs_samples = {}
    for op in operators:
        cont = [p for p in op.get("params", []) if p.get("type") == "continuous"]
        if not cont:
            lhs_samples[op["name"]] = {}
            continue
        raw = LatinHypercube(d=len(cont)).random(n=n)
        op_s = {}
        for j, p in enumerate(cont):
            lo, hi = p["range"]
            vals = lo + raw[:, j] * (hi - lo)
            if p.get("int"):
                vals = np.round(vals).astype(int).tolist()
            else:
                vals = vals.tolist()
            op_s[p["name"]] = vals
        lhs_samples[op["name"]] = op_s
    return lhs_samples


def format_value(val, param):
    unit = param.get("unit", "")
    return f"{val}{unit}" if unit else val


def sample_op_params(op, lhs_samples, idx):
    params = {}
    for p in op.get("params", []):
        name = p["name"]
        if p.get("type") == "continuous":
            raw_val = lhs_samples[op["name"]][name][idx]
            params[name] = format_value(raw_val, p)
        else:
            params[name] = random.choice(p["values"])

    for p in op.get("params", []):
        if "gt" in p and p.get("type") == "continuous":
            ref = params.get(p["gt"])
            cur = params.get(p["name"])
            if ref is not None and cur is not None:
                unit = p.get("unit", "")
                ref_num = int(str(ref).replace(unit, "")) if unit else ref
                cur_num = int(str(cur).replace(unit, "")) if unit else cur
                if cur_num <= ref_num:
                    lo, hi = p["range"]
                    new_val = min(hi, ref_num + (1 if p.get("int") else 0.01))
                    params[p["name"]] = format_value(new_val, p)

    return params or None


def select_operators(stages, operators, n_pipelines):
    # group by (stage, layer) — same layer means mutually exclusive
    by_stage = defaultdict(lambda: defaultdict(list))
    for op in operators:
        by_stage[op["stage"]][op["layer"]].append(op)

    pipelines_ops = [[] for _ in range(n_pipelines)]
    for stage in stages:
        sid = stage["id"]
        lo, hi = stage["min_operator_count"], stage["max_operator_count"]
        layers = by_stage[sid]
        if not layers:
            continue
        # flatten: one candidate per layer (pick randomly if multiple share a layer)
        pool = [random.choice(ops) for ops in layers.values()]
        for i in range(n_pipelines):
            count = random.randint(lo, min(hi, len(pool)))
            chosen = random.sample(pool, count)
            filtered = [o for o in chosen if random.random() < o.get("occurrence_prob", 1.0)]
            while len(filtered) < lo and len(chosen) > len(filtered):
                missing = [o for o in chosen if o not in filtered]
                filtered.append(random.choice(missing))
            pipelines_ops[i].extend(filtered)

    for ops in pipelines_ops:
        ops.sort(key=lambda x: x["stage"])
    return pipelines_ops


def generate_pipelines(config, n_pipelines, np_choices):
    stages = config["stages"]
    operators = config["operators"]

    lhs_samples = build_lhs_samples(operators, n_pipelines)
    pipelines_ops = select_operators(stages, operators, n_pipelines)

    results = []
    for i in range(n_pipelines):
        process = []
        for op in pipelines_ops[i]:
            params = sample_op_params(op, lhs_samples, i)
            process.append({op["name"]: params})

        ts = str(int(datetime.now().timestamp() * 1000))
        name = f"text_pipeline_{ts}"
        pipeline = {
            "project_name": name,
            "dataset": DATASET_CONFIG,
            "export_path": EXPORT_PATH,
            "np": random.choice(np_choices),
            "use_cache": False,
            "open_tracer": True,
            "process": process,
        }
        results.append((pipeline, f"{name}.yaml"))
        time.sleep(0.01)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pipelines", type=int, default=1)
    parser.add_argument("--json_path",  default="./assets/text.json")
    parser.add_argument("--output_dir", default="./collect_data/pipeline_yaml/")
    args = parser.parse_args()

    config = load_config(args.json_path)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = generate_pipelines(config, args.num_pipelines, [30, 40, 50])
    for pipeline, filename in results:
        path = out / filename
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(pipeline, f, allow_unicode=True, sort_keys=False, indent=2)
        print(f"Generated: {path}")


if __name__ == "__main__":
    main()
