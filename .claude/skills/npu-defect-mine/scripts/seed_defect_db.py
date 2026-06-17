#!/usr/bin/env python3
"""Seed defect_db.json from raw candidates with heuristic classification."""

import json
import re
from collections import Counter

from common import DB_DIR, get_current_commit, now_iso, save_json


def classify_heuristic(msg):
    m = msg.lower()
    if any(kw in m for kw in ("precision", "accuracy", "quantization", "quant")):
        return "precision_loss", 3, True
    if any(kw in m for kw in ("incorrect", "wrong output")):
        return "precision_loss", 3, True
    if any(kw in m for kw in ("ci", "build", "compile", "docker", "test ")):
        return "compile_error", 1, False
    if any(
        kw in m for kw in ("crash", "error", "timeout", "hang", "oom", "memory leak")
    ):
        return "crash", 10, False
    if any(kw in m for kw in ("performance", "slow", "throughput", "perf", "latency")):
        return "perf_regression", 1, False
    if any(kw in m for kw in ("memory", "mem ", "oom")):
        return "crash", 10, False
    return "precision_loss", 3, True


def infer_features(msg, files_changed):
    combined = (msg + " " + " ".join(files_changed)).lower()
    feats = []
    if any(kw in combined for kw in ("deepep", "fuseep", "token_dispatcher", "ep_moe")):
        feats.append("ascend_fuseep")
    if any(
        kw in combined for kw in ("disaggregation", "pd separation", "transfer_engine")
    ):
        feats.append("npu_disaggregation")
    if any(kw in combined for kw in ("dual.stream", "multi.stream", "multi_stream")):
        feats.append("dual_stream_moe")
    if any(
        kw in combined for kw in ("quant", "w4a", "w8a", "modelslim", "dynamic quant")
    ):
        feats.append("dynamic_quant")
    if any(kw in combined for kw in ("fractal", "nz", "acl format", "acl_format")):
        feats.append("fractal_nz_format")
    if any(
        kw in combined for kw in ("graph", "cuda.graph", "npu.graph", "compilation")
    ):
        feats.append("npu_graph_capture")
    if any(kw in combined for kw in ("mla", "mlapo", "fia")):
        feats.append("mla_preprocess")
    if any(kw in combined for kw in ("gdn", "mamba", "hybrid.linear")):
        feats.append("gdn_hybrid_linear_attn")
    if any(kw in combined for kw in ("hicache", "hierarchical.cache", "radix.cache")):
        feats.append("hicache_kernel_ascend")
    if any(kw in combined for kw in ("eagle", "speculative", "spec.dec")):
        feats.append("speculative_decoding")
    if any(kw in combined for kw in ("lora", "lora_ascend")):
        feats.append("lora_ascend")
    if any(kw in combined for kw in ("sampl", "top.k", "top.p")):
        feats.append("ascend_sampling")
    if any(
        kw in combined for kw in ("hccl", "communicator", "all.reduce", "all_reduce")
    ):
        feats.append("npu_communicator")
    if any(kw in combined for kw in ("llada", "dllm")):
        feats.append("dllm")
    if any(kw in combined for kw in ("piecewise", "piecewise_graph")):
        feats.append("piecewise_graph_prefill")
    if any(kw in combined for kw in ("attention", "ascend_backend", "ascend.attn")):
        feats.append("ascend_attention_backend")
    if any(
        kw in combined
        for kw in ("vlm", "vision", "multimodal", "qwen2.5.vl", "qwen3.vl")
    ):
        feats.append("vlm_npu")
    return feats if feats else ["basic_tp_inference"]


def infer_root_cause(msg, files):
    m = msg.lower()
    if any(kw in m for kw in ("quant", "scale", "w4a", "w8a")):
        return "quant_scale"
    if any(kw in m for kw in ("fractal", "nz", "acl format")):
        return "format_cast"
    if any(kw in m for kw in ("stream", "sync", "dual")):
        return "stream_sync"
    if any(kw in m for kw in ("graph", "capture", "replay")):
        return "graph_capture"
    if any(kw in m for kw in ("hccl", "timeout", "communicat")):
        return "hccl_timeout"
    if any(kw in m for kw in ("cann", "version", "compat")):
        return "cann_api"
    if any(kw in m for kw in ("layout", "stride", "shape")):
        return "memory_layout"
    if any(kw in m for kw in ("rout", "dispatch", "top.k", "moe")):
        return "token_routing"
    return "unknown"


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Seed defect_db.json from raw candidates or PR data"
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use Agent classification via Workflow (instead of heuristic)",
    )
    parser.add_argument(
        "--from-prs",
        action="store_true",
        help="Load from pr_defects.json (PR-first extraction) instead of raw_candidates.json",
    )
    args_cli = parser.parse_args()

    if args_cli.from_prs:
        # PR-first path: load pre-enriched PR data from extract_prs.py
        prs_path = DB_DIR / "pr_defects.json"
        data = json.loads(prs_path.read_text(encoding="utf-8"))
        pr_defects = data["defects"]
        print(f"Loading {len(pr_defects)} PR-first defects from pr_defects.json...")

        # Write directly to defect_db.json (PR data already embedded)
        db = {
            "_meta": {
                "last_scan_commit": get_current_commit(),
                "last_scan_at": now_iso(),
                "total_defects": len(pr_defects),
                "schema_version": "1.1",
                "extraction_method": "pr-first",
            },
            "defects": pr_defects,
            "near_misses": [],
        }
        save_json(DB_DIR / "defect_db.json", db)

        print(f"\ndefect_db.json seeded from {len(pr_defects)} PR-first defects")
        print(
            f"All defects have PR body: {sum(1 for d in pr_defects if d.get('pr_body') and len(str(d.get('pr_body','')))>50)}/{len(pr_defects)}"
        )
        print(f"Defects with reviews: {sum(1 for d in pr_defects if d.get('reviews'))}")

        # Auto-prepare Agent batch
        if args_cli.agent:
            from classify_pr import format_pr_batches as fmt_pr

            pr_defects_for_agent = [d for d in pr_defects if d.get("source") == "pr"]
            pr_batches = fmt_pr(pr_defects_for_agent, batch_size=15)
            save_json(
                DB_DIR / "pending_agent_batch.json",
                {
                    "generated_at": now_iso(),
                    "mode": "pr_first",
                    "pr_batches": pr_batches,
                    "commit_batch": {"count": 0, "defects": []},
                },
            )
            print(
                f"\nAgent batch auto-prepared: {len(pr_defects_for_agent)} defects, {len(pr_batches)} batches"
            )
            print("Next: Run Agent classification (Workflow)")

        return

    # Original commit-first path
    # Load candidates
    raw_path = DB_DIR / "raw_candidates.json"
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    candidates = data["candidates"]
    print(f"Processing {len(candidates)} candidates...")

    defects = []
    for i, c in enumerate(candidates):
        msg = c["message"]
        files = c.get("files_changed", [])

        pr_match = re.search(r"#(\d+)", msg)
        pr_num = int(pr_match.group(1)) if pr_match else None
        date = c["date"][:10]

        if args_cli.agent:
            # Placeholder classification — Workflow will fill in
            defects.append(
                {
                    "bug_id": f"BUG-{date[:4]}-{i+1:03d}",
                    "commit_sha": c["sha"],
                    "date_fixed": date,
                    "title": msg[:150],
                    "description": f"Extracted from commit: {msg[:200]}",
                    "category": "unknown",
                    "severity": 0,
                    "audit_count": 0,
                    "files_fixed": files,
                    "root_cause": "unknown",
                    "tags": [],
                    "confidence": 0.0,
                    "agent_version": "pending_agent_v1",
                    "source": "pr" if pr_num else "commit",
                    "pr_number": pr_num,
                    "needs_review": True,
                }
            )
        else:
            # Original heuristic classification
            category, severity, is_precision = classify_heuristic(msg)
            features = infer_features(msg, files)
            root_cause = infer_root_cause(msg, files)

            defects.append(
                {
                    "bug_id": f"BUG-{date[:4]}-{i+1:03d}",
                    "commit_sha": c["sha"],
                    "date_fixed": date,
                    "title": msg[:150],
                    "description": f"Extracted from commit: {msg[:200]}",
                    "category": category,
                    "severity": severity,
                    "files_fixed": files,
                    "root_cause": root_cause,
                    "tags": [],
                    "confidence": 0.6,
                    "agent_version": "heuristic_v2",
                    "source": "commit",
                    "pr_number": pr_num,
                    "needs_review": True,
                }
            )

    db = {
        "_meta": {
            "last_scan_commit": get_current_commit(),
            "last_scan_at": now_iso(),
            "total_defects": len(defects),
            "schema_version": "1.1",
        },
        "defects": defects,
        "near_misses": [],
    }

    save_json(DB_DIR / "defect_db.json", db)

    # Summary
    if args_cli.agent:
        print(f"\ndefect_db.json seeded with {len(defects)} placeholder records")
        print(f"Agent classification pending — run the Workflow next.")

        # Prepare Agent batch
        from classify_commit import format_commit_batch as fmt_commit
        from classify_pr import format_pr_batches as fmt_pr

        pr_defects = [d for d in defects if d.get("source") == "pr"]
        commit_defects = [d for d in defects if d.get("source") != "pr"]

        pr_batches = fmt_pr(pr_defects, batch_size=15) if pr_defects else []
        commit_batch = (
            fmt_commit(commit_defects)
            if commit_defects
            else {"count": 0, "defects": []}
        )

        batch = {
            "generated_at": now_iso(),
            "mode": "full_seed",
            "pr_batches": pr_batches,
            "commit_batch": commit_batch,
            "summary": {
                "total_pr_defects": len(pr_defects),
                "total_commit_defects": len(commit_defects),
                "pr_batches": len(pr_batches),
                "commit_batches": 1 if commit_defects else 0,
            },
        }
        save_json(DB_DIR / "pending_agent_batch.json", batch)

        print(f"\nAgent batch prepared:")
        print(f"  PR defects:     {len(pr_defects)} in {len(pr_batches)} batches")
        print(f"  Commit defects: {len(commit_defects)} in 1 batch")
        print(f"  Batch saved to: {DB_DIR / 'pending_agent_batch.json'}")
        print(f"\nNext steps:")
        print(f"  1. Workflow: classify defects")
        print(f"     (Claude reads pending_agent_batch.json, runs agent() calls)")
        print(
            f"  2. Apply:    python .claude/skills/npu-defect-mine/scripts/apply_agent_results.py"
        )
    else:
        cats = Counter(d["category"] for d in defects)
        print(f"\ndefect_db.json created: {len(defects)} records")
        print(f"Categories: {dict(cats)}")


if __name__ == "__main__":
    main()
