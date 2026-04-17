"""Utility logging functions."""

from datetime import datetime


def log(msg, level="INFO"):
    """Print a timestamped log line."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def eta_str(remaining, rate_per_sec):
    """Return a human-readable ETA string given items remaining and rate."""
    if rate_per_sec <= 0:
        return "???"
    secs = remaining / rate_per_sec
    if secs < 60:
        return f"{secs:.0f}s"
    elif secs < 3600:
        return f"{secs / 60:.1f}min"
    else:
        return f"{secs / 3600:.1f}hr"
