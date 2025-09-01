import ast
import pathlib
import re
import json

SRC = pathlib.Path(r"d:\BBB\src")
TARGET_GLOBALS = SRC / "bbb" / "dummy.py"

def find_ww_vars(py_path: pathlib.Path):
    src = py_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(py_path))
    except SyntaxError:
        return {}
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
                value_node = node.value
            else:  # AnnAssign
                targets = [node.target]
                value_node = node.value
            for t in targets:
                if isinstance(t, ast.Name) and t.id.startswith("ww_"):
                    name = t.id
                    val_repr = None
                    # try literal eval first
                    try:
                        val = ast.literal_eval(value_node)
                        val_repr = repr(val)
                    except Exception:
                        # fall back to source segment (expression text)
                        seg = ast.get_source_segment(src, value_node)
                        val_repr = seg.strip() if seg is not None else "None"
                    found[name] = val_repr
    return found

def main():
    all_vars = {}
    for py in SRC.rglob("*.py"):
        if py.name == "globals.py":
            continue
        found = find_ww_vars(py)
        if found:
            all_vars.update(found)

    if not all_vars:
        print("No ww_ variables found.")
        return

    # Build dict text (sorted for stable output)
    items = ",\n    ".join(f"{json.dumps(k)}: {v}" for k, v in sorted(all_vars.items()))
    dict_text = "WW_VARS = {\n    " + items + "\n}\n"

    # Insert or replace in globals.py
    if TARGET_GLOBALS.exists():
        content = TARGET_GLOBALS.read_text(encoding="utf-8")
        if re.search(r"^WW_VARS\s*=", content, flags=re.M):
            new_content = re.sub(r"^WW_VARS\s*=.*?(?=^\S|\Z)", dict_text, content, flags=re.S | re.M)
        else:
            new_content = content.rstrip() + "\n\n" + dict_text
    else:
        TARGET_GLOBALS.parent.mkdir(parents=True, exist_ok=True)
        new_content = dict_text

    TARGET_GLOBALS.write_text(new_content, encoding="utf-8")
    print(f"Wrote {len(all_vars)} ww_ entries into {TARGET_GLOBALS}")

if __name__ == "__main__":
    main()