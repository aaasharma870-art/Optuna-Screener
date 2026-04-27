"""Verify the tested/ vs untested/ strategy manifest split is consistent.

Guards against three regressions:
  1. A strategy is in apex/strategies/ but missing from BOTH manifest folders
     (someone added a strategy without classifying it).
  2. An untested manifest claims to be deployable.
  3. The default ensemble in apex_config.json includes anything from
     strategies/untested/.
"""
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
TESTED = REPO / "strategies" / "tested"
UNTESTED = REPO / "strategies" / "untested"


def _manifests(folder: Path) -> set[str]:
    return {
        p.stem for p in folder.glob("*.py")
        if p.name not in ("__init__.py",)
    }


def test_tested_and_untested_folders_exist():
    assert TESTED.is_dir(), "strategies/tested/ must exist"
    assert UNTESTED.is_dir(), "strategies/untested/ must exist"
    assert (TESTED / "__init__.py").is_file()
    assert (UNTESTED / "__init__.py").is_file()


def test_every_registered_strategy_is_classified():
    """Every apex/strategies/*.py must appear in tested/ or untested/.

    A manifest can either share the module name exactly (advanced_compounder_v11)
    or strip a trailing version suffix (vix_term_structure_v1 -> vix_term_structure).
    """
    import re
    registered = {
        p.stem for p in (REPO / "apex" / "strategies").glob("*.py")
        if p.name not in ("__init__.py", "base.py")
    }
    tested = _manifests(TESTED)
    untested = _manifests(UNTESTED)
    classified = tested | untested

    def _candidates(name: str) -> set[str]:
        out = {name}
        stripped = re.sub(r"_v\d+$", "", name)
        if stripped != name:
            out.add(stripped)
        return out

    classified_modules: set[str] = set()
    for m in classified:
        classified_modules |= _candidates(m)

    missing = registered - classified_modules
    assert not missing, (
        f"These registered strategies have no tested/ or untested/ "
        f"manifest: {sorted(missing)}"
    )


def test_tested_manifests_have_status_marker():
    for path in TESTED.glob("*.py"):
        if path.name == "__init__.py":
            continue
        src = path.read_text()
        assert "STATUS" in src, f"{path.name} missing STATUS marker"
        assert "TESTED" in src, f"{path.name} STATUS must include TESTED"


def test_untested_manifests_are_not_deployable():
    """Every untested manifest must declare DEPLOYABLE = False."""
    for path in UNTESTED.glob("*.py"):
        if path.name == "__init__.py":
            continue
        src = path.read_text()
        assert "DEPLOYABLE = False" in src, (
            f"{path.name} must declare DEPLOYABLE = False"
        )
        assert "STATUS" in src, f"{path.name} missing STATUS marker"


def test_untested_make_strategy_raises():
    """Every untested manifest's make_strategy() must refuse to instantiate."""
    import importlib
    for path in UNTESTED.glob("*.py"):
        if path.name == "__init__.py":
            continue
        mod = importlib.import_module(f"strategies.untested.{path.stem}")
        with pytest.raises(RuntimeError):
            mod.make_strategy()


def test_default_ensemble_excludes_untested():
    cfg = json.loads((REPO / "apex_config.json").read_text())
    enabled = set(cfg.get("ensemble", {}).get("strategies", []))
    untested = _manifests(UNTESTED)
    leak = enabled & untested
    assert not leak, (
        f"Default ensemble must not enable untested strategies — found: {leak}"
    )


def test_tested_vix_term_structure_reexports_production_preset():
    """The tested/ manifest must point at the production-validated preset."""
    from strategies.tested.vix_term_structure_v1 import (
        TUNED_PARAMS,
        VALIDATION_METADATA,
        make_strategy,
    )
    from strategies.production.vix_term_structure_v1 import (
        TUNED_PARAMS as PROD_PARAMS,
    )
    assert TUNED_PARAMS is PROD_PARAMS  # same object — single source of truth
    assert "holdout_true_oos" in VALIDATION_METADATA
    assert callable(make_strategy)
