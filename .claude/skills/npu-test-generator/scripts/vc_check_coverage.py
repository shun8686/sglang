"""Coverage check: report APIs with no matching test point. Exit 1 if gaps found.
Usage: python vc_check_coverage.py <test-points.json> <source_dir> [source_dir2...]
       python vc_check_coverage.py <test-points.json> <source_dir> --filter eagle,worker
       python vc_check_coverage.py <test-points.json> <source_dir> --exclude dflash,ngram,standalone
       python vc_check_coverage.py --list-apis <source> [...]"""
import json, re, sys, os
from pathlib import Path
from collections import defaultdict


def extract_apis_from_source(source_dirs: list[str]) -> dict:
    """Scan source dirs for def/class, return {full_name: (file, line)}.

    full_name is 'ClassName.method' or 'function_name' or 'ClassName'.
    """
    def scan_file(fpath: str) -> dict:
        """Scan a single .py file and return {full_name: (file, line)}."""
        result = {}
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                lines = fp.readlines()
        except Exception:
            return result

        current_class = None
        for i, line in enumerate(lines, 1):
            class_m = re.match(r"^\s*class\s+(\w+)", line)
            if class_m:
                current_class = class_m.group(1)
                full = current_class
                if full not in result:
                    result[full] = (fpath, i)
                continue

            if current_class and re.match(r"^\S", line) and not line.startswith(" "):
                current_class = None

            def_m = re.match(r"^\s+def\s+(\w+)", line)
            if def_m and current_class:
                full = f"{current_class}.{def_m.group(1)}"
                if full not in result:
                    result[full] = (fpath, i)

            func_m = re.match(r"^def\s+(\w+)", line)
            if func_m and not func_m.group(1).startswith("_"):
                full = func_m.group(1)
                if full not in result:
                    result[full] = (fpath, i)

        return result

    apis = {}
    for src in source_dirs:
        if os.path.isfile(src) and src.endswith(".py"):
            apis.update(scan_file(src))
        elif os.path.isdir(src):
            for root, _, files in os.walk(src):
                for f in files:
                    if f.endswith(".py"):
                        apis.update(scan_file(os.path.join(root, f)))

    return apis


def extract_covered_from_test_points(json_path: str) -> set:
    """Extract API names from test points. If any method of a class is covered,
    the class itself is also considered covered (__init__ is implicit)."""
    data = json.load(open(json_path, "r", encoding="utf-8"))
    pts = data.get("test_points", data) if isinstance(data, dict) else data
    covered = set()
    sym_pat = re.compile(r"\(([^)]+)\)")

    for p in pts:
        if not isinstance(p, dict):
            continue
        loc = p.get("source_location", "")
        matches = sym_pat.findall(loc)
        for m in matches:
            for part in re.split(r"[,/]\s*", m):
                part = part.strip()
                if not part:
                    continue
                covered.add(part)
                # X.__init__ or X.method -> X is also covered
                if "." in part:
                    cls = part.split(".")[0]
                    covered.add(cls)

    return covered


def main():
    if "--list-apis" in sys.argv:
        # Just print the API list from source files, no coverage check
        filter_args, exclude_args, src_args = _parse_args()
        if not src_args:
            print("Usage: vc_check_coverage.py --list-apis <source> [...]")
            sys.exit(2)
        apis = extract_apis_from_source(src_args)
        dunder_skip = {"__init__", "__repr__", "__str__", "__hash__", "__eq__", "__ge__", "__gt__",
                       "__le__", "__lt__", "__ne__", "__del__", "__setattr__", "__getattr__",
                       "__delattr__", "__copy__", "__deepcopy__", "__reduce__", "__reduce_ex__",
                       "__sizeof__", "__subclasshook__", "__init_subclass__",
                       "__class_getitem__", "__format__", "__dir__", "__abstractmethods__"}
        for full_name, (fpath, line) in sorted(apis.items()):
            method_name = full_name.split(".")[-1] if "." in full_name else full_name
            if method_name in dunder_skip or method_name.startswith("_"):
                continue
            if not _file_in_scope(fpath, filter_args, exclude_args):
                continue
            print(f"{full_name}  ({os.path.basename(fpath)}:{line})")
        sys.exit(0)

    if len(sys.argv) < 3:
        print("Usage: vc_check_coverage.py <test-points.json> <source_dir> [...]")
        print("       vc_check_coverage.py <test-points.json> <source_dir> --filter <pat,...>")
        print("       vc_check_coverage.py <test-points.json> <source_dir> --exclude <pat,...>")
        print("       vc_check_coverage.py --list-apis <source> [...]")
        sys.exit(2)

    filter_args, exclude_args, src_args = _parse_args()

    json_path = src_args[0]
    source_dirs = src_args[1:]

    if filter_args:
        print(f"[filter: {filter_args}]")
    if exclude_args:
        print(f"[exclude: {exclude_args}]")

    apis = extract_apis_from_source(source_dirs)
    covered = extract_covered_from_test_points(json_path)

    # Filter out private/deprecated APIs. __init__ is implicit — testing any
    # method tests construction. Private methods (_*) are internal.
    dunder_skip = {"__init__", "__repr__", "__str__", "__hash__", "__eq__", "__ge__", "__gt__",
                   "__le__", "__lt__", "__ne__", "__del__", "__setattr__", "__getattr__",
                   "__delattr__", "__copy__", "__deepcopy__", "__reduce__", "__reduce_ex__",
                   "__sizeof__", "__subclasshook__", "__init_subclass__",
                   "__class_getitem__", "__format__", "__dir__", "__abstractmethods__"}

    uncovered = {}
    for full_name, (fpath, line) in sorted(apis.items()):
        if full_name in covered:
            continue
        method_name = full_name.split(".")[-1] if "." in full_name else full_name
        # Skip dunder methods (including __init__ — tested implicitly)
        if method_name in dunder_skip:
            continue
        # Skip private methods (leading underscore, e.g. _replay, _capture_graph)
        if method_name.startswith("_"):
            continue
        # Skip APIs from files excluded by --filter / --exclude scope
        if not _file_in_scope(fpath, filter_args, exclude_args):
            continue
        uncovered[full_name] = (fpath, line)

    if uncovered:
        print(f"UNCOVERED APIs ({len(uncovered)}):")
        for name in sorted(uncovered):
            fpath, line = uncovered[name]
            print(f"  {name}  ({os.path.basename(fpath)}:{line})")
        print()
        print(json.dumps([{"tau": "missing_coverage", "location": name,
                           "s": 0.8, "description": f"API not covered by any test point: {name}"}
                          for name in sorted(uncovered)], indent=2))
        sys.exit(1)
    else:
        print(f"COVERAGE OK: {len(apis)} APIs covered by {len(covered)} test point references")
        sys.exit(0)


def _parse_args():
    """Parse --filter and --exclude from sys.argv. Returns (filter, exclude, remaining)."""
    filter_args = []
    exclude_args = []
    src_args = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--filter" and i + 1 < len(sys.argv):
            filter_args = [p.strip().lower() for p in sys.argv[i + 1].split(",") if p.strip()]
            i += 2
        elif sys.argv[i] == "--exclude" and i + 1 < len(sys.argv):
            exclude_args = [p.strip().lower() for p in sys.argv[i + 1].split(",") if p.strip()]
            i += 2
        elif sys.argv[i] == "--list-apis":
            i += 1  # handled by caller
        else:
            src_args.append(sys.argv[i])
            i += 1
    return filter_args, exclude_args, src_args


def _file_in_scope(fpath, filter_args, exclude_args):
    """Return True if the file passes the filter/exclude scope check."""
    if not filter_args and not exclude_args:
        return True
    bn = os.path.basename(fpath).lower()
    if filter_args:
        return any(pat in bn for pat in filter_args)
    if exclude_args:
        return not any(pat in bn for pat in exclude_args)
    return True


if __name__ == "__main__":
    main()
