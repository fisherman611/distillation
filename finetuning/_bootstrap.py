from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_project_paths() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)
    return project_root
