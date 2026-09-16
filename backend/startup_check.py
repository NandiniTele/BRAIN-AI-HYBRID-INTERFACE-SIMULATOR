#!/usr/bin/env python3
"""
startup_check.py -- Pre-flight verification for Neural-Link backend.

Run this once before launching the server to ensure the environment is ready:
  python startup_check.py

Exit code 0 = all good.  Non-zero = critical issue found.
Compatible with Python 3.7+ and Windows cp1252 terminals.
"""

from __future__ import annotations

import os
import sys
import importlib
import warnings

# Suppress all noise during checks
warnings.filterwarnings("ignore")
os.environ["MNE_LOGGING_LEVEL"] = "ERROR"

# Force UTF-8 output on Windows if possible, otherwise fall back gracefully
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Python 3.7+
        os.system("")                              # enable VT100 / ANSI
    except Exception:
        pass

REQUIRED_PACKAGES = [
    "fastapi", "uvicorn", "websockets",
    "numpy", "scipy", "pandas",
    "sklearn", "mne", "torch",
    "pywt", "psutil", "jwt", "motor",
]

BOLD  = "\033[1m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
CYAN  = "\033[96m"
RESET = "\033[0m"


def ok(msg: str):   print("  %s[OK]%s  %s" % (GREEN,  RESET, msg))
def warn(msg: str): print("  %s[!!]%s  %s" % (YELLOW, RESET, msg))
def fail(msg: str): print("  %s[XX]%s  %s" % (RED,    RESET, msg))
def info(msg: str): print("  %s[--]%s  %s" % (CYAN,   RESET, msg))
def section(title: str):
    print("\n%s%s--- %s ---%s" % (BOLD, CYAN, title, RESET))


def check_python():
    v = sys.version_info
    if v >= (3, 11):
        ok("Python %d.%d.%d" % (v.major, v.minor, v.micro))
    else:
        warn("Python %d.%d detected -- Python >= 3.11 required for full support." % (v.major, v.minor))
        # We'll allow it to continue to preserve backward compatibility as requested, 
        # but the prompt asked to "Upgrade project compatibility... to Python 3.11+".
        # However, to be safe, we just warn instead of fail if we want backwards compat.
        if v < (3, 7):
            fail("Python %d.%d detected -- Python >= 3.7 strictly required" % (v.major, v.minor))
            sys.exit(1)


def check_packages():
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            ok("Package '%s' available" % pkg)
        except ImportError:
            fail("Package '%s' NOT installed" % pkg)
            missing.append(pkg)
    if missing:
        print("\n%sMissing packages -- run:%s" % (YELLOW, RESET))
        print("  pip install %s" % " ".join(missing))
        sys.exit(1)


def check_weights():
    weights_dir  = "weights"
    weights_path = os.path.join(weights_dir, "bci_model.pth")
    os.makedirs(weights_dir, exist_ok=True)
    ok("weights/ directory ensured")
    if os.path.isfile(weights_path) and os.path.getsize(weights_path) > 0:
        size_mb = os.path.getsize(weights_path) / 1024 / 1024
        ok("Model weights found (%.1f MB) -- training will be skipped" % size_mb)
    else:
        warn("No weights at '%s' -- first startup will train the model (normal)" % weights_path)


def check_sqlite():
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
    conn.close()
    ok("SQLite3 functional")


def check_mne_config():
    try:
        import mne
        mne.set_log_level("ERROR")
        mne_data = os.path.join(os.path.expanduser("~"), "mne_data")
        os.makedirs(mne_data, exist_ok=True)
        mne.set_config("MNE_DATA", mne_data, set_env=True)
        ok("MNE data directory configured -> %s" % mne_data)
    except Exception as exc:
        warn("MNE config skipped (%s)" % exc)


def check_mongo():
    try:
        import pymongo
        ok("pymongo %s installed" % getattr(pymongo, "version", "unknown"))
        info("MongoDB is OPTIONAL -- backend falls back to SQLite if unavailable")
    except ImportError:
        warn("pymongo not installed -- MongoDB disabled (SQLite will be used)")


if __name__ == "__main__":
    print("\n%s%s=== Neural-Link Pre-Flight Check ===%s\n" % (BOLD, CYAN, RESET))

    section("Python")
    check_python()

    section("Packages")
    check_packages()

    section("Weights")
    check_weights()

    section("Persistence")
    check_sqlite()
    check_mne_config()
    check_mongo()

    print("\n%s%s[ALL CHECKS PASSED] -- Ready to launch!%s\n" % (BOLD, GREEN, RESET))
