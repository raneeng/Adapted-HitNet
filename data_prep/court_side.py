"""
court_side.py
--------------
Resolve which player (A/B, per ShuttleSet's convention -- A is the match
winner, B the match loser) is physically on the BOTTOM side of the court
for a given rally, accounting for badminton's side-switching rules:

  Set 1: bottom player = A if downcourt==0, else B   ("initial" arrangement)
  Set 2: bottom player = the OPPOSITE of set 1, for the whole set
  Set 3 (if played):
      - starts at the SAME arrangement as set 1
      - switches to the set-2 arrangement once either player's score
        first reaches 11
      - stays switched for the rest of the set

This is a genuine prerequisite for correctly mapping ShuttleSet's `player`
column (A/B, tied to match outcome) onto physical bottom/top identity,
which is what the pose CSVs (_player_bottom.csv / _player_top.csv) and any
player-attributed hit label need to agree on.
"""

from __future__ import annotations

MID_SET_SWITCH_SCORE = 11


def initial_bottom_player(downcourt: int) -> str:
    """Which player is on the bottom side in set 1 (and set 3, before the 11-point switch).

    Args:
        downcourt: From match.csv -- 0 if the match winner (player A)
            starts on the bottom side in set 1, 1 if the loser (player B)
            does.

    Returns:
        "A" or "B".

    Raises:
        ValueError: If downcourt isn't 0 or 1.
    """
    if downcourt == 0:
        return "A"
    if downcourt == 1:
        return "B"
    raise ValueError(f"downcourt must be 0 or 1, got {downcourt!r}")


def switched_bottom_player(downcourt: int) -> str:
    """Which player is on the bottom side in set 2 (and set 3, after the 11-point switch).

    Args:
        downcourt: Same as initial_bottom_player().

    Returns:
        "A" or "B" -- whichever player initial_bottom_player() does NOT return.
    """
    return "B" if initial_bottom_player(downcourt) == "A" else "A"


def resolve_bottom_player(
    downcourt: int,
    set_num: int,
    prior_roundscore_a: int,
    prior_roundscore_b: int,
) -> str:
    """Resolve which player (A/B) is on the bottom side for one specific rally.

    IMPORTANT -- prior_roundscore_a/b must be the score BEFORE this rally
    started (i.e. the score as it stood after the PREVIOUS rally
    completed, or 0-0 for the very first rally of a set), NOT the score
    this rally itself produced. This matters specifically for the set-3
    mid-set switch: in real play, a score of 11 is only reached at the
    moment a rally ENDS, so the rally that produces the 11 is itself
    still played on the pre-switch side -- the switch only takes effect
    starting with the rally AFTER that one. Passing this rally's own
    resulting score (rather than the prior rally's) would switch one
    rally too early.

    If you have a chronological list of a set's rally scores and don't
    want to handle this offset yourself, use
    resolve_bottom_players_for_set() instead, which does it for you.

    Args:
        downcourt:             From match.csv (0 or 1) -- see initial_bottom_player().
        set_num:               Which set this rally belongs to (1, 2, or 3).
        prior_roundscore_a:    Player A's score BEFORE this rally started.
            Only used when set_num == 3, to detect the mid-set switch point.
        prior_roundscore_b:    Player B's score BEFORE this rally started.

    Returns:
        "A" or "B" -- whichever player is physically on the bottom side of
        the court for this rally.

    Raises:
        ValueError: If set_num isn't 1, 2, or 3, or downcourt isn't 0/1.
    """
    if set_num == 1:
        return initial_bottom_player(downcourt)
    if set_num == 2:
        return switched_bottom_player(downcourt)
    if set_num == 3:
        switch_has_happened = (prior_roundscore_a >= MID_SET_SWITCH_SCORE
                                or prior_roundscore_b >= MID_SET_SWITCH_SCORE)
        if switch_has_happened:
            return switched_bottom_player(downcourt)
        return initial_bottom_player(downcourt)
    raise ValueError(f"set_num must be 1, 2, or 3, got {set_num!r}")


def resolve_bottom_players_for_set(
    downcourt: int,
    set_num: int,
    roundscores: list[tuple[int, int]],
) -> list[str]:
    """Resolve the bottom player for every rally in a set, in chronological order.

    Handles the "use the PRIOR rally's score, not this rally's own
    resulting score" offset internally (see resolve_bottom_player()'s
    docstring for why that distinction matters) -- so callers can just
    pass the roundscore values exactly as they appear in the ShuttleSet
    CSV, in rally order, with no manual shifting.

    Args:
        downcourt:    From match.csv (0 or 1).
        set_num:      Which set these rallies belong to (1, 2, or 3).
        roundscores:  List of (roundscore_a, roundscore_b) as they appear
            directly in the ShuttleSet CSV, one tuple per rally, in
            chronological rally order -- i.e. the score AFTER each rally
            completed.

    Returns:
        List of "A"/"B", same length and order as roundscores -- one
        bottom-player resolution per rally.
    """
    results = []
    prior_a, prior_b = 0, 0
    for score_a, score_b in roundscores:
        results.append(resolve_bottom_player(downcourt, set_num, prior_a, prior_b))
        prior_a, prior_b = score_a, score_b
    return results


def player_to_int_label(player: str, bottom_player: str) -> int:
    """Map a ShuttleSet `player` value (A/B) to physical bottom/top, given who's on bottom.

    Args:
        player:        The stroke's hitting player, "A" or "B" (ShuttleSet's
            `player` column value for this row).
        bottom_player: Which player is on the bottom side for this rally
            (from resolve_bottom_player()).

    Returns:
        1 if `player` matches `bottom_player`, else 2.
    """
    return 1 if player == bottom_player else 2 