"""
shot_mapping.py
----------------
ShuttleSet stroke type -> analyst shot type mapping (finalised Nov 2026).

Shared by format_shots.py (label preparation) and the shot-type training
script, so both always use the same classes.

Three levels of label:
    shuttleset type   e.g. "切球" (drop)
    analyst label     e.g. "BH dropshot"   -- what the tagging app shows
    base type         e.g. "dropshot"      -- racket face removed; what the
                                              10-class model is trained on
"""

from __future__ import annotations

import re

import numpy as np

# ShuttleSet type -> (forehand label, backhand label). None = drop the stroke.
SHUTTLESET_TO_ANALYST: dict[str, tuple[str | None, str | None]] = {
    "放小球":     ("FH net",        "BH net"),          # net shot
    "勾球":       ("FH net",        "BH net"),          # cross-court net shot
    "擋小球":     ("Block",         "Block"),           # return net
    "防守回挑":   ("Block",         "Block"),           # defensive return lob
    "殺球":       ("FH smash",      None),              # smash (BH dropped)
    "點扣":       ("FH smash",      None),              # wrist smash (BH dropped)
    "挑球":       ("FH net lift",   "BH net lift"),     # lob
    "長球":       ("FH clear",      "BH clear"),        # clear
    "平球":       ("FH drive",      "BH drive"),        # drive
    "小平球":     ("FH drive",      "BH drive"),        # driven flight
    "推球":       ("FH net lift",   "BH net lift"),     # push
    "防守回抽":   ("FH drive",      "BH drive"),        # defensive return drive
    "後場抽平球": ("FH dropshot",   "BH dropshot"),     # back-court drive
    "切球":       ("FH dropshot",   "BH dropshot"),     # drop
    "過度切球":   ("FH dropshot",   "BH dropshot"),     # passive drop
    "撲球":       ("FH net kill",   "BH net kill"),     # rush
    "發短球":     ("FH low serve",  "BH low serve"),    # short service
    "發長球":     ("FH high serve", "BH flick serve"),  # long service
    "未知球種":   (None,            None),              # unknown type (annotator) -- dropped
}

SERVE_TYPES = {"發短球": "short", "發長球": "long"}

# Base types for the rally-shot classifier (serves handled separately).
RALLY_BASE_TYPES = [
    "net", "net lift", "net kill", "block", "drive",
    "clear", "smash", "dropshot",
]


def normalise_type(shuttleset_type) -> str:
    """Strip all whitespace, e.g. "擋小 球" -> "擋小球", "過度切球 " -> "過度切球"."""
    return re.sub(r"\s+", "", str(shuttleset_type))


def is_backhand(value) -> bool:
    """ShuttleSet's backhand column: 1/1.0 = backhand, blank/NaN/0 = forehand.

    Also accepts an already-converted bool, so calling it twice is safe.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() not in ("", "nan", "none", "0", "0.0", "false")


def map_shot(shuttleset_type, backhand) -> str | None:
    """ShuttleSet stroke -> analyst label (e.g. "BH dropshot"); None = drop.

    Raises:
        KeyError: for a type not in the mapping, so new or misspelled
            labels fail loudly instead of being silently dropped.
    """
    key = normalise_type(shuttleset_type)
    if key not in SHUTTLESET_TO_ANALYST:
        raise KeyError(f"Unmapped ShuttleSet type: {shuttleset_type!r}")
    fh, bh = SHUTTLESET_TO_ANALYST[key]
    return bh if is_backhand(backhand) else fh


def base_type(analyst_label: str | None) -> str | None:
    """Analyst label -> base type with racket face removed.

    Serves collapse to "low serve" / "long serve" (high and flick are the
    same long serve, named by racket face).
    """
    if analyst_label is None:
        return None
    if analyst_label in ("FH high serve", "BH flick serve"):
        return "long serve"
    return re.sub(r"^(FH|BH) ", "", analyst_label).lower()