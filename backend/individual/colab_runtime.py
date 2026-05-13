from __future__ import annotations
import ast
from pathlib import Path
from types import ModuleType
import sys

COLAB_SOURCE = Path(__file__).resolve().parent.parent.parent / "codex_inputs" / "colab_photography_pipeline (2).py"

BLOCKED_IMPORT_PREFIXES = ("google.colab",)


def load_colab_module() -> ModuleType:
    src = COLAB_SOURCE.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(src, filename=str(COLAB_SOURCE))

    safe_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            safe_nodes.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            if mod == "__future__":
                continue
            names = [a.name for a in getattr(node, "names", [])]
            txt = mod + " " + " ".join(names)
            if any(b in txt for b in BLOCKED_IMPORT_PREFIXES):
                continue
            safe_nodes.append(node)
        elif isinstance(node, ast.Assign):
            # keep only constant/simple assignments used by functions
            if isinstance(node.value, (ast.Constant, ast.Dict, ast.List, ast.Tuple, ast.Set, ast.Name)):
                safe_nodes.append(node)

    mod = ModuleType("colab_pipeline_runtime")
    sys.modules[mod.__name__] = mod
    code = compile(ast.Module(body=safe_nodes, type_ignores=[]), filename=str(COLAB_SOURCE), mode="exec")
    exec(code, mod.__dict__)
    return mod
