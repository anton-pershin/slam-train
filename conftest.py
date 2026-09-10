"""Ensure the repo root is on sys.path so that test modules can use absolute
`tests.` imports regardless of the directory pytest is invoked from."""

import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
