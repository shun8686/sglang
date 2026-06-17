"""VC checks for Phase 1 test points. Usage: python vc_check_phase1.py <test-points.json>"""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "test-out/phase1/test-points.json"
data = json.load(open(path, "r", encoding="utf-8"))
pts = data.get("test_points", data) if isinstance(data, dict) else data

# Split: active TPs that need validation vs. merged TPs that are audit-only
active = [p for p in pts if isinstance(p, dict) and "merged_into" not in p]
merged = [p for p in pts if isinstance(p, dict) and "merged_into" in p]
if merged:
    merged_ids = [p.get("id", "?") for p in merged]
    targets = [p.get("merged_into", "?") for p in merged]
    print(f"Skipping {len(merged)} merged TP(s): {merged_ids} -> {targets}")

errors = []

# VC-1: ID uniqueness, priority validity
# (source_location format NOT validated here — meaningfulness is checked by SC factual_error)
all_ids = [p["id"] for p in pts if isinstance(p, dict) and "id" in p]
dups = [x for x in all_ids if all_ids.count(x) > 1]
if set(dups):
    print(f"FAIL: duplicate IDs: {list(set(dups))}")
    errors.append({"tau": "vc_id_uniqueness", "location": "multiple", "s": 0.8,
                   "description": f"Duplicate IDs: {list(set(dups))}"})

bad_prio = [p["id"] for p in active if p.get("priority") not in {"P0", "P1", "P2"}]
if bad_prio:
    print(f"FAIL: bad priority values: {bad_prio}")
    errors.append({"tau": "vc_id_uniqueness", "location": str(bad_prio), "s": 0.8,
                   "description": f"Invalid priority values in: {bad_prio}"})

# VC-2: Required fields (active TPs only; merged TPs only need id + merged_into)
required = {"id", "feature", "source_location", "key_factors", "precondition", "expected_behavior", "priority"}
missing = [(p.get("id", "?"), [k for k in required if k not in p])
           for p in active if isinstance(p, dict) and not required.issubset(p.keys())]
if missing:
    print(f"FAIL: missing required fields: {missing}")
    errors.append({"tau": "vc_json_schema", "location": str([m[0] for m in missing]), "s": 0.5,
                   "description": f"Missing required fields: {missing}"})

if not errors:
    print(f"VC PHASE 1 OK: {len(all_ids)} IDs total ({len(active)} active, {len(merged)} merged), priorities valid, fields complete")
else:
    print(f"VC PHASE 1: {len(errors)} error(s) found")
    print(json.dumps(errors, indent=2))

sys.exit(1 if errors else 0)
