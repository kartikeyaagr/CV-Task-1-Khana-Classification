"""Path bootstrap — import this before any 'from khana' imports.

Walks up from this file's location until it finds a directory containing
src/khana/__init__.py, then inserts that src/ into sys.path.
Works regardless of CWD, virtual environment state, or platform.
"""
import sys
from pathlib import Path

def _add_src_to_path() -> None:
    here = Path(__file__).resolve().parent  # scripts/
    # Walk upward: scripts/ -> khana_classifier/ -> (stop)
    for candidate in [here.parent / "src", here / "src"]:
        if (candidate / "khana" / "__init__.py").exists():
            src_str = str(candidate)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            return
    raise RuntimeError(
        f"Could not find src/khana/__init__.py relative to {here}. "
        "Make sure you're running from within the khana_classifier directory."
    )

_add_src_to_path()
