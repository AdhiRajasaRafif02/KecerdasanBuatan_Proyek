"""Convenience wrapper to run training from the project root.

Usage:
  python train_lstm.py

This forwards execution to src/train_lstm.py while ensuring imports work.
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

    target = os.path.join(src_dir, "train_lstm.py")
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
