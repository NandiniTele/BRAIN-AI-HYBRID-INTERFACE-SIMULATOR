"""
utils.py — Colorized logging utility for Neural-Link backend.
Uses ANSI escape codes; works on Windows 10+ terminals and all Unix systems.
Compatible with Python 3.7+.
"""

from __future__ import annotations

import logging
import sys
import os

# ── ANSI color palette ────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_MAGENTA= "\033[95m"
_GRAY   = "\033[90m"

# Enable ANSI on Windows
if sys.platform == "win32":
    os.system("")  # triggers VT100 mode on modern Windows terminals

_LEVEL_COLORS = {
    "DEBUG":    _GRAY   + "DEBUG"   + _RESET,
    "INFO":     _CYAN   + "INFO"    + _RESET,
    "SUCCESS":  _GREEN  + "SUCCESS" + _RESET,
    "WARNING":  _YELLOW + "WARN"    + _RESET,
    "ERROR":    _RED    + "ERROR"   + _RESET,
    "CRITICAL": _RED    + _BOLD + "CRITICAL" + _RESET,
}

# Custom SUCCESS level (between INFO=20 and WARNING=30)
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


class ColorFormatter(logging.Formatter):
    """Formats log lines with ANSI color codes."""

    FMT = "{gray}{asctime}{reset}  {level}  {magenta}{name:<22}{reset}  {msg}"

    def format(self, record: logging.LogRecord) -> str:
        level_str = _LEVEL_COLORS.get(record.levelname, record.levelname)
        msg = record.getMessage()
        asctime = self.formatTime(record, "%H:%M:%S")
        return self.FMT.format(
            gray=_GRAY, asctime=asctime, reset=_RESET,
            level=level_str,
            magenta=_MAGENTA, name=record.name,
            msg=msg,
        )


def setup_logger(name: str) -> logging.Logger:
    """
    Returns a colorized logger.  Safe to call multiple times — handlers
    are only attached once per logger name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)
        logger.propagate = False   # avoid duplicate root-logger output

    # Attach a convenience 'success' method
    def _success(msg: str, *args, **kwargs):
        logger.log(SUCCESS_LEVEL, msg, *args, **kwargs)

    logger.success = _success  # type: ignore[attr-defined]
    return logger


# ── Module-level logger (used by startup script) ──────────────────────────────
log = setup_logger("NeuralLink")
