from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute plain-Python notebook cells in order")
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    path = args.notebook.resolve()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__notebook__"}
    try:
        from IPython.display import display
    except ImportError:
        display = print
    namespace["display"] = display
    count = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        print(f"[notebook] executing cell {index}", flush=True)
        exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)
        count += 1
    print(f"[notebook] completed {count} code cells", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
