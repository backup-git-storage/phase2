"""Compatibility CLI for the canonical validation-based model comparison.

The former script independently built features and ranked models using final
test R². Comparisons now use the exact feature schema, rolling-origin folds
with available-label cutoffs, and validation-MAE selection from :mod:`src.train`.
"""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import main


if __name__ == "__main__":
    raise SystemExit(main())
