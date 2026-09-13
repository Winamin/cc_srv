# -*- coding: utf-8 -*-
"""Decide, per round, whether a speculative round is worth taking.

What went wrong before
----------------------
The first version of this switched speculation off globally once it measured
slower than plain decoding.  Measured against real traffic, that is the wrong
shape of decision, because speculation is not uniformly good or bad -- it is
good for some drafts and bad for others:

    prompt tok   ms/token   spec rounds   full accept   draft/round
     18590        17.20          0            --            --
     18803         5.64          4           100%          9.75
     19534         5.68          3           100%         12.33
     22631        18.00          0            --            --

When a draft lands, a round is ~3x faster than decoding one token at a time.
When it does not, the round costs MORE than the tokens it returns, and a global
switch would have thrown away the 3x along with the loss.

Where the loss comes from
-------------------------
A round pays a verification pass over the whole draft, and then, unless the
draft was accepted in full, a second pass to refeed the accepted prefix.  So a
partially-accepted round costs two forward passes and returns its accepted
count.  Against ~15.9 ms per token decoded one at a time:

    draft   full accept   partial accept (1 kept)
        2      1.8x              0.47x
        4      2.5x              0.62x
       16      5.5x              0.70x

Short drafts are the problem: at m=2 a partial acceptance can never pay, and the
recorded k2 corpus averaged 1.32 draft tokens per round with 19.8% full
acceptance -- mostly the losing case, which is the whole of the measured 0.712x.
k4, which barely speculates at all (0.47 verified per round), measured 0.962x.

What this does instead
----------------------
Buckets rounds by draft length and measures each bucket's own rate against the
plain decoding rate, which is measured from the rounds that do not speculate.
A bucket whose rate falls to the plain rate stops being taken; the others carry
on.  Nothing has to be predicted, because the engine is already timing both.

A dropped bucket is not dropped for good
----------------------------------------
Conditions change: a session that was editing fresh code turns into one that
re-reads the same file twenty times, and short drafts become the best kind
again.  So a dropped bucket waits out a cooldown and is then re-measured from
scratch, with the cooldown doubling each time it fails again -- a bucket that
is genuinely bad costs one probe per doubling, and one that has merely hit a
bad patch comes back on its own.

    CC_SPEC_GOV=0        disable the gate (speculate whenever a draft exists)
    CC_SPEC_MIN=12       ROUNDS of evidence on each side before judging
    CC_SPEC_TAU=8.0      EWMA time constant, rounds
    CC_SPEC_MARGIN=1.0   a bucket is dropped at or below this ratio
    CC_SPEC_COOLDOWN=64  rounds before a dropped bucket is re-probed
"""
from __future__ import annotations

import math
import os
import threading
from collections import deque

# Both in ROUNDS, not seconds.  The decision is made per round, and a round is
# milliseconds long -- 91 speculative rounds in a 16-request run is under a
# second of evidence in total, which split across four buckets never reaches a
# seconds-based threshold.  Measured: with a two-second threshold the gate never
# fired once.
_TAU = float(os.environ.get("CC_SPEC_TAU", "8.0"))
_MIN = float(os.environ.get("CC_SPEC_MIN", "12"))
_MARGIN = float(os.environ.get("CC_SPEC_MARGIN", "1.0"))
_COOLDOWN = float(os.environ.get("CC_SPEC_COOLDOWN", "64"))
_ON = os.environ.get("CC_SPEC_GOV", "1") != "0"

# The repetition detector.  The bucket gate below judges draft LENGTHS, and the
# evidence it counts comes only from rounds where a draft was actually taken --
# so on content that offers no drafts at all it never accumulates evidence and
# never arms.  That is precisely the case it is needed for.
#
# Why it watches the TOKEN STREAM and not DDC's own output.  The first attempt
# scored each round by the draft DDC was offered, and it failed outright: the
# full-vocabulary scan in ddc.features is not only the lookup, it is also what
# feeds cache.append, so standing down to save the scan froze the index, offers
# stayed at zero, and the detector confirmed itself. Measured: 0 speculative
# rounds on every request, and the 2x repetitive win (128.8 tok/s) collapsed to
# 66.1. The evidence and the cost were the same work.
#
# This breaks the circle by measuring repetition where it is cheap -- in the
# token stream itself. A rolling n-gram over the last few hundred generated
# tokens costs a tuple hash per token against a 248k-element argpartition
# (measured at 7.9% of generation time), so it can gate the expensive index
# without depending on it, and it keeps running while DDC is stood down.
#
# The signal is deliberately binary per token: an n-gram that occurred before
# inside the window scores 1, otherwise 0. Novel prose scores near zero; a
# repeated structure -- the case that pays -- scores near one once the first
# copy has gone by.
_REP_N = int(os.environ.get("CC_SPEC_REP_N", "32"))
_REP_WINDOW = int(os.environ.get("CC_SPEC_REP_WINDOW", "4096"))
# Fraction of recent tokens whose n-gram had been seen before.
_REP_MIN = float(os.environ.get("CC_SPEC_REP_MIN", "0.15"))
_REP_EVIDENCE = float(os.environ.get("CC_SPEC_REP_EVIDENCE", "64"))
_REP_TAU = float(os.environ.get("CC_SPEC_REP_TAU", "32"))
# Tokens of grace at the start of every generation, during which DDC is never
# stood down.  A repetitive request always BEGINS novel: DDC drafts the second
# copy of a structure from the first, so the first copy has to be indexed before
# the repetition is visible at all.  Measured on the dataclass prompt, the rate
# runs 0.00 at the start and reaches 0.35 only once a full class has gone by;
# judging during that window stands down precisely the requests that pay.
_REP_GRACE = int(os.environ.get("CC_SPEC_REP_GRACE", "128"))

# Draft lengths are bucketed rather than tracked individually: neighbouring
# lengths behave alike, and a bucket needs a dozen rounds of evidence before it
# can be judged -- which an individual length would never accumulate, since a
# single length can go many requests without recurring.
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
    """Fold one observation into ``slot``.  Seeded with the first observation
    rather than zero: starting from zero takes ~TAU rounds to converge, and a
    verdict is allowed after a fixed number of rounds, so a zero start means the
    estimate is still far too low when the first verdict is reached."""
    if slot[1] == 0:
        slot[0] = value
    else:
        alpha = 1.0 - math.exp(-1.0 / (tau if tau is not None else _TAU))
        slot[0] += alpha * (value - slot[0])
    slot[1] += 1


def _note(bucket, tokens: int, seconds: float):
    """One observation.  The rate is tokens per second of decode time; the
    evidence is counted in rounds, because that is the unit being decided."""
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
    """Feed one generated token to the repetition detector.

    Called for every committed token whether or not DDC is currently indexing,
    which is the point: the signal has to keep working while DDC is stood down,
    or it could never notice the content turning repetitive and switch back on.

    Scores 1 when this token's n-gram was seen before inside the window, else 0.
    Inside a repeat every token scores, so a repeated structure drives the rate
    to ~1 while novel prose leaves it near 0.
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
        # Bound the index: anything older than the window can no longer produce
        # a hit, so drop it.  Done in a batch because a full scan per token would
        # cost more than the hash it is cleaning up after.
        if len(_seen) > _REP_WINDOW:
            cutoff = _pos - _REP_WINDOW
            for k in [k for k, v in _seen.items() if v < cutoff]:
                del _seen[k]


def note_offer(draft_len: int):
    """Report the draft a round was OFFERED, 0 when the lookup found none.

    Diagnostic only -- the verdict below no longer rests on it, for the reason
    given at the top: it is unavailable exactly when DDC is stood down, and it
    costs the scan to produce.
    """
    if not _ON:
        return
    with _lock:
        _ewma(_offer, float(draft_len))


def paying() -> bool:
    """Whether the content being generated is repetitive enough to index for.

    The bucket gate in ``allow()`` decides per draft length; this decides whether
    to build the draft index at all.  Indexing costs a full-vocabulary scan per
    committed token (measured 7.9% of generation time) and pays off only where
    drafts land, so on content that never repeats it is pure cost.

    Judged instantaneously rather than through a cooldown: the signal is cheap
    and keeps running while DDC is stood down, so it can simply say "repetitive
    again" the moment that becomes true.  A cooldown would only delay the
    recovery it exists to allow.
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
            # Cooldown elapsed: forget the old verdict and measure again.  A
            # bucket that fails again gets a longer wait; one that was only
            # passing through a bad patch comes back.
            b[0] = b[1] = 0.0
            b[2] = 0
        # With no plain-decoding sample there is nothing to compare against, so
        # speculation runs -- refusing everything until evidence arrives would
        # mean never gathering any.
        if _seq[1] < _MIN or b[1] < _MIN or _seq[0] <= 0:
            return True
        if b[0] <= _seq[0] * _MARGIN:
            b[3] += 1
            b[2] = _round + int(_COOLDOWN * (2 ** (b[3] - 1)))
            return False
        return True


def note_round(draft_len: int, committed: int, seconds: float):
    """Report a speculative round: what it returned against what it cost.

    ``seconds`` should cover the whole round -- the state copy, the verification
    pass, any refeed, and the feature extraction -- because all of it is paid for
    this round's committed tokens and nothing else.
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
    """Forget everything -- for tests and for benchmarks needing a clean run."""
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