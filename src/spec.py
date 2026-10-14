# -*- coding: utf-8 -*-
"""Decide, per round, whether a speculative round is worth taking.

A round pays a verification pass over the whole draft plus a refeed when the draft
is only partly accepted, so a losing draft costs more than the tokens it returns.
Rounds are therefore bucketed by draft length and each bucket's rate is measured
against plain decoding; a bucket at or below the plain rate is dropped for a
cooldown and then re-probed, with the cooldown doubling on each further failure.

CC_SPEC_GOV=0        disable the gate (speculate whenever a draft exists)
CC_SPEC_MIN=12       rounds of evidence on each side before judging
CC_SPEC_TAU=8.0      EWMA time constant, rounds
CC_SPEC_MARGIN=1.0   a bucket is dropped at or below this ratio
CC_SPEC_COOLDOWN=64  rounds before a dropped bucket is re-probed
"""
from __future__ import annotations

import math
import os
import threading
from collections import deque

# Units are rounds, not seconds: a round is milliseconds long.
_TAU = float(os.environ.get("CC_SPEC_TAU", "8.0"))
_MIN = float(os.environ.get("CC_SPEC_MIN", "12"))
_MARGIN = float(os.environ.get("CC_SPEC_MARGIN", "1.0"))
_COOLDOWN = float(os.environ.get("CC_SPEC_COOLDOWN", "64"))
_ON = os.environ.get("CC_SPEC_GOV", "1") != "0"

# The repetition detector must keep running while DDC is stood down, or it can
# never see content turn repetitive.  It therefore scores a rolling n-gram over
# the generated tokens: 1 if that n-gram was seen earlier inside the window.
_REP_N = int(os.environ.get("CC_SPEC_REP_N", "32"))
_REP_WINDOW = int(os.environ.get("CC_SPEC_REP_WINDOW", "4096"))
# Fraction of recent tokens whose n-gram had been seen before.
_REP_MIN = float(os.environ.get("CC_SPEC_REP_MIN", "0.15"))
_REP_EVIDENCE = float(os.environ.get("CC_SPEC_REP_EVIDENCE", "64"))
_REP_TAU = float(os.environ.get("CC_SPEC_REP_TAU", "32"))
# Tokens of grace before judging starts: a repetitive request BEGINS novel, so
# DDC must index the first copy before the repetition is visible at all.
_REP_GRACE = int(os.environ.get("CC_SPEC_REP_GRACE", "128"))

# Draft lengths are bucketed: neighbouring lengths behave alike, and a single
# length can go many requests without recurring, so it never gathers evidence.
BUCKETS = ((2, 3), (4, 7), (8, 15), (16, 1 << 30))

_lock = threading.Lock()
_seq = [0.0, 0.0]                      # EWMA tokens/second, rounds of evidence
# rate, evidence, retry_at (0 = live), strikes
_buckets = [[0.0, 0.0, 0, 0] for _ in BUCKETS]
# The repetition detector's own state: EWMA of the recent n-gram hit rate,
# evidence in tokens, plus the ring and index it is computed from.
_rep = [0.0, 0.0]
_ring: deque = deque(maxlen=_REP_N)
_seen: dict = {}                       # n-gram -> last position it was seen at
_pos = 0                               # generated tokens seen, the clock here
_req = [0]                             # tokens generated in the current request
_offer = [0.0, 0.0]                    # EWMA of offered draft length (diagnostic)
_round = 0                             # global round counter, the clock here


def _bucket_of(n: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= n <= hi:
            return i
    return len(BUCKETS) - 1


def _ewma(slot, value: float, tau: float = None):
    """Fold one observation into ``slot``; seeded with the first observation
    rather than zero, because zero takes ~TAU rounds to converge."""
    if slot[1] == 0:
        slot[0] = value
    else:
        alpha = 1.0 - math.exp(-1.0 / (tau if tau is not None else _TAU))
        slot[0] += alpha * (value - slot[0])
    slot[1] += 1


def _note(bucket, tokens: int, seconds: float):
    """One observation: tokens per second of decode time, evidence in rounds."""
    global _round
    _round += 1
    if tokens <= 0 or seconds <= 0:
        return
    _ewma(bucket, tokens / seconds)


def note_sequential(tokens: int, seconds: float):
    """Report tokens decoded one at a time.  This is the yardstick."""
    if not _ON:
        return
    with _lock:
        _note(_seq, tokens, seconds)


def begin_request():
    """Start of a generation.  Restarts the per-request grace counter."""
    if not _ON:
        return
    with _lock:
        _req[0] = 0


def note_token(token: int):
    """Feed one generated token to the repetition detector: scores 1 when its
    n-gram was seen before inside the window, else 0.  Called for every committed
    token, DDC indexing or not.
    """
    global _pos
    if not _ON:
        return
    with _lock:
        _pos += 1
        _req[0] += 1
        _ring.append(token)
        if len(_ring) < _REP_N:
            return
        key = hash(tuple(_ring))
        prev = _seen.get(key)
        _seen[key] = _pos
        _ewma(_rep, 1.0 if (prev is not None and _pos - prev <= _REP_WINDOW) else 0.0,
              _REP_TAU)
        # Bound the index: entries older than the window can no longer hit, and a
        # batch scan is cheaper than one per token.
        if len(_seen) > _REP_WINDOW:
            cutoff = _pos - _REP_WINDOW
            for k in [k for k, v in _seen.items() if v < cutoff]:
                del _seen[k]


def note_offer(draft_len: int):
    """Report the draft a round was OFFERED, 0 when the lookup found none.

    Diagnostic only: it is unavailable exactly when DDC is stood down.
    """
    if not _ON:
        return
    with _lock:
        _ewma(_offer, float(draft_len))


def paying() -> bool:
    """Whether the content being generated is repetitive enough to index for.

    ``allow()`` judges draft length; this judges whether to build the index at
    all, instantly and with no cooldown, since the signal keeps running while
    DDC is stood down.
    """
    if not _ON:
        return True
    with _lock:
        if _req[0] < _REP_GRACE:
            return True
        if _rep[1] < _REP_EVIDENCE:
            return True
        return _rep[0] >= _REP_MIN


def allow(draft_len: int) -> bool:
    """Whether a round of this draft length is worth taking, judged so far."""
    if not _ON:
        return True
    with _lock:
        b = _buckets[_bucket_of(draft_len)]
        if b[2]:
            if b[2] > _round:
                return False
            # Cooldown elapsed: forget the verdict and re-measure, waiting longer
            # each time it fails again.
            b[0] = b[1] = 0.0
            b[2] = 0
        # No plain-decoding sample to compare against: speculate, since refusing
        # until evidence arrives would never gather any.
        if _seq[1] < _MIN or b[1] < _MIN or _seq[0] <= 0:
            return True
        if b[0] <= _seq[0] * _MARGIN:
            b[3] += 1
            b[2] = _round + int(_COOLDOWN * (2 ** (b[3] - 1)))
            return False
        return True


def note_round(draft_len: int, committed: int, seconds: float):
    """Report a speculative round: what it returned against what it cost.

    ``seconds`` must cover the whole round, state copy and refeed included.
    """
    if not _ON:
        return
    with _lock:
        _note(_buckets[_bucket_of(draft_len)], committed, seconds)


def buckets() -> list:
    """Per-bucket (low, high, rate, evidence, retry_at, strikes) for reporting."""
    with _lock:
        return [(BUCKETS[i][0], BUCKETS[i][1], _buckets[i][0], _buckets[i][1],
                 _buckets[i][2], _buckets[i][3]) for i in range(len(BUCKETS))]


def line() -> str:
    parts = []
    for lo, hi, rate, evid, retry_at, _ in buckets():
        name = f"{lo}+" if hi > 1000 else (f"{lo}" if lo == hi else f"{lo}-{hi}")
        if retry_at > _round:
            parts.append(f"{name}:off({retry_at - _round})")
        elif evid >= _MIN:
            parts.append(f"{name}:{rate:.0f}")
        else:
            parts.append(f"{name}:?")
    if _rep[1] >= _REP_EVIDENCE:
        rep = f"rep:{_rep[0]:.2f}{'' if _rep[0] >= _REP_MIN else '*off'}"
    else:
        rep = "rep:?"
    return (f"spec_gov={'off' if not _ON else 'on'} plain={_seq[0]:.0f}tok/s "
            f"round={_round} " + " ".join(parts) + f" {rep} offer:{_offer[0]:.1f}")


def reset():
    """Forget everything; for tests and runs that need a clean start."""
    with _lock:
        global _round, _pos
        _round = 0
        _pos = 0
        _seq[0] = _seq[1] = 0.0
        _offer[0] = _offer[1] = 0.0
        _rep[0] = _rep[1] = 0.0
        _req[0] = 0
        _ring.clear()
        _seen.clear()
        for b in _buckets:
            b[0] = b[1] = 0.0
            b[2] = b[3] = 0

def consumed():
    return None