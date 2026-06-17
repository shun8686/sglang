#!/usr/bin/env python3
"""
Detect sglang advanced feature docs not yet tracked in features.json.

Filters out:
  - overview pages (index/table-of-contents)
  - tuning/best-practice guides
  - comparison guides (e.g. dp_dpa_smg_guide)
  - .rst duplicates of .mdx files
  - .ipynb duplicates of .mdx files
  - sub-guides embedded in a parent doc (hicache_design, hicache_best_practices, etc.)

Output: JSON list of {doc, reason, suggested_action} for Agent analysis.
"""

import json
import os
import sys
from pathlib import Path

# Find repo root by walking up to .git
_repo_root = Path(__file__).resolve().parent
while not (_repo_root / ".git").exists() and _repo_root != _repo_root.parent:
    _repo_root = _repo_root.parent
REPO_ROOT = _repo_root
DOCS_DIR = REPO_ROOT / "docs_new" / "docs" / "advanced_features"
FEATURES_PATH = REPO_ROOT / ".sglang-risk" / "baselines" / "latest" / "features.json"

# --- Known skip patterns ---
SKIP_FILENAMES = {
    "overview.mdx",  # index page
    "dp_dpa_smg_guide.mdx",  # comparison guide, not a feature
    "hyperparameter_tuning.mdx",  # tuning guide
    "hicache_best_practices.mdx",  # sub-guide of hicache
    "hicache_design.mdx",  # design doc of hicache
    "hicache_storage_runtime_attach_detach.mdx",  # sub-guide of hicache
    "hicache.rst",  # .rst duplicate of hicache.mdx
}

# Docs that are covered by a parent feature (not standalone)
SUBSUMED_BY = {
    "vlm_query.mdx": "multimodal",
    "vlm_query.ipynb": "multimodal",
    "breakable_cuda_graph.mdx": "graph_compilation",
    "piecewise_cuda_graph.mdx": "graph_compilation",
    "epd_disaggregation.mdx": "pd_disaggregation",
    "adaptive_speculative_decoding.mdx": "speculative_decoding",
    "cuda_graph_for_multi_modal_encoder.mdx": "multimodal",
    "structured_outputs_for_reasoning_models.mdx": "structured_outputs",
    "sglang_for_rl.mdx": None,  # integration guide — skip
}


def load_features():
    with open(FEATURES_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_doc_feature_names(features):
    """Extract the set of doc names that each feature comes from.
    Maps from sglang doc stem → feature name.

    All feature names are validated against features.json at runtime.
    """
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import validate_feature_name

    valid_names = {f["name"] for f in features}

    # Known direct mappings: doc_stem → feature_name
    DIRECT_MAP = {
        "attention_backend": "attention_backend",
        "quantization": "quantization",
        "speculative_decoding": "speculative_decoding",
        "pd_disaggregation": "pd_disaggregation",
        "lora": "lora",
        "hicache": "hicache",
        "forward_hooks": "forward_hooks",
        "structured_outputs": "structured_outputs",
        "tool_parser": "tool_parser",
        "separate_reasoning": "separate_reasoning",
        "observability": "observability",
        "server_arguments": "server_arguments",
        "pipeline_parallelism": "pipeline_parallelism",
        "expert_parallelism": "expert_parallelism",
        "dp_for_multi_modal_encoder": "dp_for_multi_modal_encoder",
        "object_storage": "object_storage",
        "sgl_model_gateway": "sgl_model_gateway",
        "checkpoint_engine": "checkpoint_engine",
        "deterministic_inference": "deterministic_inference",
        "hisparse": "hisparse",
        "rfork": "rfork",
        "quantized_kv_cache": "quantized_kv_cache",
        "hisparse_guide": "hisparse",
    }

    # Validate all feature name values against features.json
    for stem, fname in DIRECT_MAP.items():
        if fname not in valid_names:
            print(
                f"  [WARN] DIRECT_MAP: doc='{stem}' maps to '{fname}' not in features.json — update detect_new_docs.py",
                file=_sys.stderr,
            )

    return DIRECT_MAP


def scan_docs():
    """Scan docs directory, return list of .mdx files not yet covered."""
    if not DOCS_DIR.exists():
        print(f"ERROR: Docs directory not found: {DOCS_DIR}", file=sys.stderr)
        sys.exit(1)

    all_mdx = sorted(f.name for f in DOCS_DIR.glob("*.mdx"))
    return all_mdx


def classify_doc(filename):
    """Classify a doc file: 'covered' | 'skip' | 'subsumed' | 'candidate'"""
    if filename in SKIP_FILENAMES:
        return "skip", "filtered by SKIP_FILENAMES"
    if filename in SUBSUMED_BY:
        parent = SUBSUMED_BY[filename]
        if parent is None:
            return "skip", "integration guide, not a feature"
        return "subsumed", f"merged into '{parent}'"
    return "candidate", None


def main():
    features_data = load_features()
    feature_names = {f["name"] for f in features_data["features"]}
    doc_map = get_doc_feature_names(features_data["features"])
    covered_stems = set(doc_map.keys())

    # Validate SUBSUMED_BY feature names against features.json
    for doc, parent in SUBSUMED_BY.items():
        if parent is not None and parent not in feature_names:
            print(
                f"  [WARN] SUBSUMED_BY: doc='{doc}' maps to '{parent}' not in features.json — update detect_new_docs.py",
                file=sys.stderr,
            )

    all_mdx = scan_docs()

    results = {"covered": [], "skip": [], "subsumed": [], "candidate": []}

    for fname in all_mdx:
        stem = fname.replace(".mdx", "")
        classification, reason = classify_doc(fname)

        if classification == "skip":
            results["skip"].append({"doc": fname, "reason": reason})
        elif classification == "subsumed":
            results["subsumed"].append({"doc": fname, "reason": reason})
        elif stem in covered_stems:
            results["covered"].append(
                {
                    "doc": fname,
                    "feature": doc_map[stem],
                }
            )
        else:
            results["candidate"].append(
                {
                    "doc": fname,
                    "reason": "not found in features.json doc map",
                }
            )

    # Also check .ipynb files (duplicate detection)
    ipynb_files = sorted(f.name for f in DOCS_DIR.glob("*.ipynb"))
    for ipynb in ipynb_files:
        mdx_counterpart = ipynb.replace(".ipynb", ".mdx")
        if mdx_counterpart not in all_mdx:
            # .ipynb without .mdx counterpart — unusual, flag it
            results["candidate"].append(
                {
                    "doc": ipynb,
                    "reason": "ipynb without matching .mdx — may be standalone",
                }
            )

    # Print summary
    print(f"Docs directory: {DOCS_DIR}")
    print(f"Total .mdx files: {len(all_mdx)}")
    print(f"Total .ipynb files: {len(ipynb_files)}")
    print()

    print(f"[OK] Covered (in features.json):  {len(results['covered'])}")
    for item in results["covered"]:
        print(f"  {item['doc']} → {item['feature']}")

    print(f"\n--- Skipped (guide/overview):    {len(results['skip'])}")
    for item in results["skip"]:
        print(f"  {item['doc']} — {item['reason']}")

    print(f"\n--> Subsumed (in parent feature): {len(results['subsumed'])}")
    for item in results["subsumed"]:
        print(f"  {item['doc']} — {item['reason']}")

    print(f"\n[!!] Candidate (needs analysis):  {len(results['candidate'])}")
    for item in results["candidate"]:
        print(f"  {item['doc']} — {item['reason']}")

    if results["candidate"]:
        print("\nAction: Read each candidate doc, determine NPU participation, then:")
        print("  1. ADD as new feature, or")
        print("  2. MERGE into existing feature, or")
        print("  3. SKIP (add to SKIP_FILENAMES or SUBSUMED_BY in this script)")

    # Output JSON for programmatic use
    json_output = json.dumps(results, indent=2, ensure_ascii=False)
    out_path = REPO_ROOT / ".sglang-risk" / "deltas" / "new_docs_candidates.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_output)
    print(f"\nDetailed results written to: {out_path}")

    return 0 if not results["candidate"] else len(results["candidate"])


if __name__ == "__main__":
    sys.exit(main())
