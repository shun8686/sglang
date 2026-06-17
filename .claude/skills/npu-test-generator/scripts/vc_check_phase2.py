"""VC checks for Phase 2 test cases. Usage: python vc_check_phase2.py <test-cases.json>"""
import json, re, sys

path = sys.argv[1] if len(sys.argv) > 1 else "test-out/phase2/test-cases.json"
data = json.load(open(path, "r", encoding="utf-8"))
tcs = data.get("test_cases", data) if isinstance(data, dict) else data

errors = []

# VC-1: Required fields per TC
required = {"id", "test_point_id", "title", "description", "preconditions",
            "test_data", "steps", "expected_results", "teardown"}
missing = [(tc.get("id", "?"), [k for k in required if k not in tc])
           for tc in tcs if isinstance(tc, dict) and not required.issubset(tc.keys())]
if missing:
    print(f"FAIL: missing required fields: {missing}")
    errors.append({"tau": "vc_json_schema", "location": str([m[0] for m in missing]), "s": 0.5,
                   "description": f"Missing required fields: {missing}"})

# VC-2: Trace crosscheck — orphaned/unused variables
found_trace = False
for tc in tcs:
    if not isinstance(tc, dict):
        continue
    produced = set()
    for s in tc.get("steps", []):
        m = re.search(r"→\s*produces\s+(.+)", s)
        if m:
            text = re.sub(r"\([^)]*\)", "", m.group(1))
            for var in re.split(r",\s*", text.strip()):
                if var:
                    produced.add(var)
    referenced = set(tc.get("expected_results", {}).keys())
    orphaned = referenced - produced
    unused = produced - referenced - {"side_effects"}
    if orphaned:
        print(f"ORPHANED {tc['id']}: {list(orphaned)} (in expected_results but no step produces)")
        errors.append({"tau": "vc_trace_crosscheck", "location": tc["id"], "s": 0.8,
                       "description": f"Orphaned keys in expected_results: {list(orphaned)}"})
        found_trace = True
    if unused:
        print(f"UNUSED {tc['id']}: {list(unused)} (step produces but never referenced)")
        errors.append({"tau": "vc_trace_crosscheck", "location": tc["id"], "s": 0.8,
                       "description": f"Unused step output variables: {list(unused)}"})
        found_trace = True

if not found_trace:
    print("VC-2 OK: no trace breaks")

if not errors:
    print(f"VC PHASE 2 OK: {len(tcs)} TCs schema valid, no trace breaks")
else:
    print(f"VC PHASE 2: {len(errors)} error(s) found")
    print(json.dumps(errors, indent=2))

sys.exit(1 if errors else 0)
