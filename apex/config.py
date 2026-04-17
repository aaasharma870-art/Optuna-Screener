"""Pipeline configuration loading and global config constants."""

import json
import os
import sys
from pathlib import Path


def load_config(path="apex_config.json"):
    """Load pipeline configuration from JSON file."""
    script_dir = Path(__file__).resolve().parent.parent
    full_path = script_dir / path
    if not full_path.exists():
        full_path = Path(path)
    if not full_path.exists():
        print(f"[ERROR] Config file not found: {path}")
        sys.exit(1)
    with open(full_path, "r") as f:
        cfg = json.load(f)
    # Environment variable overrides
    env_polygon = os.environ.get("POLYGON_API_KEY")
    if env_polygon:
        cfg["polygon_api_key"] = env_polygon
    env_fred = os.environ.get("FRED_API_KEY")
    if env_fred:
        cfg["fred_api_key"] = env_fred
    return cfg


CFG = load_config()
POLYGON_KEY = CFG["polygon_api_key"]
CACHE_DIR = Path(CFG.get("cache_dir", "apex_cache"))
OUTPUT_DIR = Path(CFG.get("output_dir", "apex_results"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RL = CFG.get("polygon_rate_limit", {})
POLYGON_SLEEP = RL.get("sleep_between_calls", 0.12)
MAX_RETRIES = RL.get("max_retries", 3)
RETRY_WAIT = RL.get("retry_wait", 10)

POLYGON_BASE = "https://api.polygon.io"

FORCED_SYMBOLS = ["SPY", "QQQ"]
