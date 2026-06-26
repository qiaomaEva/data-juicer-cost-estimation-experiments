from pathlib import Path


OUTPUT_ROOT = Path("./output")
DATA_DIR = OUTPUT_ROOT / "data"
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"
SUMMARIES_DIR = OUTPUT_ROOT / "summaries"
REPORTS_DIR = OUTPUT_ROOT / "reports"
FEATURE_IMPORTANCE_DIR = OUTPUT_ROOT / "feature_importance"
FIGURES_DIR = OUTPUT_ROOT / "figures"
MODEL_DIR = OUTPUT_ROOT / "AutogluonModels"
EXISTING_MODEL_COMPARISON_DIR = PREDICTIONS_DIR / "existing_model_fixed_testset_comparison"


def ensure_output_dirs():
    for path in [
        OUTPUT_ROOT,
        DATA_DIR,
        PREDICTIONS_DIR,
        SUMMARIES_DIR,
        REPORTS_DIR,
        FEATURE_IMPORTANCE_DIR,
        FIGURES_DIR,
        MODEL_DIR,
        EXISTING_MODEL_COMPARISON_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def first_existing(*paths):
    for path in paths:
        if path and Path(path).exists():
            return str(path)
    return None


def first_existing_or_default(default_path, *fallback_paths):
    existing = first_existing(default_path, *fallback_paths)
    return existing if existing is not None else str(default_path)


def data_path(filename):
    return str(DATA_DIR / filename)


def prediction_path(filename):
    return str(PREDICTIONS_DIR / filename)


def summary_path(filename):
    return str(SUMMARIES_DIR / filename)


def report_path(filename):
    return str(REPORTS_DIR / filename)


def feature_importance_path(filename):
    return str(FEATURE_IMPORTANCE_DIR / filename)


def figure_path(filename):
    return str(FIGURES_DIR / filename)


def model_path(dirname):
    return str(MODEL_DIR / dirname)


def legacy_output_path(filename):
    return str(OUTPUT_ROOT / filename)


def resolve_data_path(filename):
    return first_existing_or_default(DATA_DIR / filename, OUTPUT_ROOT / filename)


def resolve_prediction_path(filename):
    return first_existing_or_default(PREDICTIONS_DIR / filename, OUTPUT_ROOT / filename)


def resolve_summary_path(filename):
    return first_existing_or_default(SUMMARIES_DIR / filename, OUTPUT_ROOT / filename)


def resolve_report_path(filename):
    return first_existing_or_default(REPORTS_DIR / filename, OUTPUT_ROOT / filename)


def resolve_feature_importance_path(filename):
    return first_existing_or_default(FEATURE_IMPORTANCE_DIR / filename, OUTPUT_ROOT / filename)


def resolve_legacy_aware_path(path, new_parent):
    """Resolve an explicit old output path to the new categorized folder when possible."""
    path = Path(path)
    if path.exists():
        return str(path)
    candidate = Path(new_parent) / path.name
    if candidate.exists():
        return str(candidate)
    return str(path)
