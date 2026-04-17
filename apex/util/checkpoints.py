"""Pipeline checkpoint save/load helpers."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from apex.config import OUTPUT_DIR
from apex.logging_util import log


def save_checkpoint(name, data, output_dir=None):
    """Save pipeline checkpoint as JSON to *output_dir*."""
    od = Path(output_dir) if output_dir else OUTPUT_DIR
    od.mkdir(parents=True, exist_ok=True)
    cp_dir = od / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    path = cp_dir / f"{name}.json"

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp, datetime)):
            return str(o)
        if isinstance(o, Path):
            return str(o)
        return str(o)

    with open(path, "w") as f:
        json.dump(data, f, default=_default, indent=2)
    log(f"Checkpoint saved: {path}")


def load_checkpoint(name, output_dir=None):
    """Load a checkpoint by name. Returns dict or None if not found."""
    od = Path(output_dir) if output_dir else OUTPUT_DIR
    path = od / "checkpoints" / f"{name}.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        data = json.load(f)
    log(f"Checkpoint loaded: {path}")
    return data
