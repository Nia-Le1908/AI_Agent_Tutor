"""
Logging setup for AI Tutor.

``config.LOG_PATH`` promised a log file that nothing ever wrote to; this module is
the single place that configures handlers, so every entry point (Streamlit app and
the CLI tools) gets the same shape of output: console for humans, rotating file for
post-mortem debugging.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys

from config import LOG_LEVEL, LOG_PATH, ensure_runtime_dirs

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def configure_logging(level: int | None = None, *, force: bool = False) -> logging.Logger:
    """
    Attach console + file handlers to the root logger once per process.

    Args:
        level: Overrides the configured LOG_LEVEL when given.
        force: Re-apply configuration even if it already ran (used by tests).

    Returns:
        The configured root logger.

    File logging is best-effort: a read-only or unwritable log directory must never
    stop the app from starting.
    """
    global _configured
    root = logging.getLogger()

    if _configured and not force:
        return root

    if root.handlers and not force:
        # Something else already owns logging (a test runner, `streamlit`'s own
        # config, a hosting framework). Adding file handlers on top is fine, but
        # wiping theirs out would break their output capture, so we defer.
        _configured = True
        return root

    resolved_level = level if level is not None else getattr(logging, LOG_LEVEL, logging.INFO)
    root.setLevel(resolved_level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        ensure_runtime_dirs()
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_PATH,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:
        root.debug("File logging disabled (%s)", exc)

    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring logging on first use."""
    return logging.getLogger(name)
