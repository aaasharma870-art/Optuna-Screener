"""Tests for config loading with env-var overrides."""

import json
import sys
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def test_polygon_key_env_override(monkeypatch, tmp_path):
    """POLYGON_API_KEY env var overrides JSON value."""
    cfg_file = tmp_path / "test_cfg.json"
    cfg_file.write_text(json.dumps({"polygon_api_key": "FROM_JSON"}))
    monkeypatch.setenv("POLYGON_API_KEY", "FROM_ENV")
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    import apex
    cfg = apex.load_config(str(cfg_file))
    assert cfg["polygon_api_key"] == "FROM_ENV"


def test_fred_key_env_override(monkeypatch, tmp_path):
    """FRED_API_KEY env var overrides JSON value."""
    cfg_file = tmp_path / "test_cfg.json"
    cfg_file.write_text(json.dumps({"polygon_api_key": "X", "fred_api_key": "FROM_JSON"}))
    monkeypatch.setenv("FRED_API_KEY", "FROM_ENV_FRED")
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)

    import apex
    cfg = apex.load_config(str(cfg_file))
    assert cfg["fred_api_key"] == "FROM_ENV_FRED"


def test_no_env_keeps_json(monkeypatch, tmp_path):
    """Without env vars, JSON values are preserved."""
    cfg_file = tmp_path / "test_cfg.json"
    cfg_file.write_text(json.dumps({"polygon_api_key": "KEEP_ME"}))
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    import apex
    cfg = apex.load_config(str(cfg_file))
    assert cfg["polygon_api_key"] == "KEEP_ME"
