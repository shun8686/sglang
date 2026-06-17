"""Validate features.json against features_schema.md format contract.

Usage:
    python .claude/skills/npu-risk-graph/scripts/validate_features.py [path/to/features.json]

If no path given, defaults to .sglang-risk/baselines/latest/features.json
"""

import json
import sys
from pathlib import Path

VALID_CATS = {
    # NOTE: This is the DEFINITION of valid categories — it is not derived
    # from features.json because it validates features.json. If new categories
    # are needed, update this set BEFORE adding them to features.json.
    "attention",
    "inference_engine",
    "memory",
    "quantization",
    "distributed",
    "parallelism",
    "interface",
    "platform",
}
VALID_LEVELS = {"strong", "medium", "weak", "platform_agnostic", "not_supported"}
VALID_MODELS_SENTINELS = {
    "All models",
    "All models on NPU",
    "Not applicable (NPU unsupported)",
}
REQUIRED_FIELDS = {
    "name",
    "category",
    "source_files",
    "complexity",
    "npu_participation",
    "description",
    "models_using",
    "fingerprint",
    "last_modified",
    "source",
}


def validate(features_path: str) -> int:
    with open(features_path, encoding="utf-8") as f:
        data = json.load(f)

    errors = []

    # --- top-level ---
    if not isinstance(data, dict):
        errors.append("[root]: not a dict")
        return _fail(errors)
    if data.get("total_features") != len(data.get("features", [])):
        errors.append(
            f"total_features={data.get('total_features')} != features[] len={len(data.get('features', []))}"
        )
    # --- per-feature ---
    names = []
    for f in data.get("features", []):
        name = f.get("name", "?")

        # required fields
        missing = REQUIRED_FIELDS - set(f.keys())
        if missing:
            errors.append(f"{name}: missing fields {sorted(missing)}")

        # category
        if f.get("category") not in VALID_CATS:
            errors.append(f"{name}: invalid category '{f.get('category')}'")

        # npu_participation
        if f.get("npu_participation") not in VALID_LEVELS:
            errors.append(
                f"{name}: invalid npu_participation '{f.get('npu_participation')}'"
            )

        # complexity
        c = f.get("complexity")
        if not isinstance(c, int) or not 1 <= c <= 5:
            errors.append(f"{name}: complexity={c} (need int 1-5)")

        # failure_mode (optional, 1-5)
        fm = f.get("failure_mode")
        if fm is not None and (not isinstance(fm, int) or not 1 <= fm <= 5):
            errors.append(f"{name}: failure_mode={fm} (need int 1-5 or omit)")

        # source_files non-empty
        if not f.get("source_files"):
            errors.append(f"{name}: empty source_files")

        # models_using: must be sentinel or list of model names, not mixed
        models_using = f.get("models_using", [])
        if not models_using:
            errors.append(f"{name}: empty models_using")
        else:
            sentinels_in_list = [m for m in models_using if m in VALID_MODELS_SENTINELS]
            model_names_in_list = [
                m for m in models_using if m not in VALID_MODELS_SENTINELS
            ]
            if sentinels_in_list and model_names_in_list:
                errors.append(
                    f"{name}: models_using mixes sentinels {sentinels_in_list} with model names — use one or the other"
                )
            if len(sentinels_in_list) > 1:
                errors.append(
                    f"{name}: models_using has multiple sentinels {sentinels_in_list} — use exactly one"
                )
            # Consistency with npu_participation
            npu = f.get("npu_participation", "")
            if (
                npu == "not_supported"
                and "Not applicable (NPU unsupported)" not in sentinels_in_list
            ):
                errors.append(
                    f"{name}: npu_participation=not_supported requires models_using=['Not applicable (NPU unsupported)']"
                )
            if (
                npu == "platform_agnostic"
                and sentinels_in_list
                and "All models" not in sentinels_in_list
            ):
                print(
                    f"  [WARN] {name}: platform_agnostic but models_using={models_using} — consider ['All models'] instead",
                    file=sys.stderr,
                )

        # fingerprint is valid SHA256 hex
        fp = f.get("fingerprint", "")
        if len(fp) != 64 or not all(c in "0123456789abcdef" for c in fp):
            errors.append(f"{name}: fingerprint not valid SHA256 hex")

        # source_file paths
        for p in f.get("source_files", []):
            if not (
                p.startswith("python/sglang/srt/")
                or p.startswith("sgl-model-gateway/")
                or p.startswith("python/sglang/multimodal_gen/")
            ):
                errors.append(f"{name}: unexpected path '{p}'")

        names.append(name)

    # duplicate names
    seen = set()
    for n in names:
        if n in seen:
            errors.append(f"Duplicate feature name: '{n}'")
        seen.add(n)

    return _fail(errors) if errors else _pass(data)


def _fail(errors):
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    return 1


def _pass(data):
    n = len(data["features"])
    cats = {}
    for f in data["features"]:
        cats[f["npu_participation"]] = cats.get(f["npu_participation"], 0) + 1
    print(
        f"OK: {n} features, 0 errors  ({', '.join(f'{v} {k}' for k, v in sorted(cats.items()))})"
    )
    return 0


if __name__ == "__main__":
    # Find repo root to locate .sglang-risk/baselines/
    _p = Path(__file__).resolve().parent
    while not (_p / ".git").exists() and _p != _p.parent:
        _p = _p.parent
    default = _p / ".sglang-risk" / "baselines" / "latest" / "features.json"
    path = sys.argv[1] if len(sys.argv) > 1 else str(default)
    sys.exit(validate(path))
