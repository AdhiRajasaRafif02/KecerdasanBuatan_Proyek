"""Convenience wrapper to run error analysis from the project root.

Usage:
  python error_analysis.py --model results/models/lstm.h5
  python error_analysis.py --model results/models/lstm.h5 --analyze_confidence

This forwards execution to src/error_analysis.py while ensuring imports work.
"""

from __future__ import annotations

import os
import runpy
import sys


def main() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(project_root, "src")

    # Ensure `import eda_preprocessing` works (module lives in src/)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    target = os.path.join(src_dir, "error_analysis.py")
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
