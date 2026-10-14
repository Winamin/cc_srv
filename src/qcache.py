# -*- coding: utf-8 -*-
"""Turn-keyed reuse of an earlier reply, optionally replayed from cached logits.

``Eng.lgc`` keys on the whole prompt, which almost never repeats inside a Claude
Code dialog; the key here is the trailing user turn, the same question.

CC_QREUSE   0 off, 1 the same turn again in the same dialog, 2 also similar turns
CC_QEDIT    replay the cached top-k trace and rewrite the reply; needs CC_QREUSE=2
"""
from __future__ import annotations

import difflib
import hashlib
import re

import numpy as np

import utext

# Replaying is argmax over the stored row, so a trace reproduces its own tokens;
# the width only bounds which edits count as in-distribution.
DEFAULT_TOPK = 16

# Claude Code injects volatile <system-reminder> blocks into the user turn; they
# differ between a first ask and a repeat, so leaving them in the key would stop
# a question from ever matching itself.
_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)

# Only this much turn text is kept, so the level-2 similarity test stays bounded.
QTEXT_MAX = 2000


def turn_key(msgs: list) -> tuple | None:
    """Key of the trailing user turn as ``(kind, digest, text)``; None if there is none.

    Scans backwards past trailing non-user turns: a Claude Code request ends with a
    ``role: system`` token count, so ``msgs[-1]`` is almost never the question.
    """
    for role, content in reversed(msgs):
        if role != "user" or not content:
            continue
        text = _REMINDER.sub("", content).strip()
        if not text:
            continue
        kind = "tool" if text.startswith("<tool_result>") else "text"
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
        return kind, digest, text[:QTEXT_MAX]
    return None


def _chunks(s: str) -> list:
    """Split into word-ish chunks so a diff does not glue unrelated words."""
    return re.findall(r"\s+|\w+|[^\w\s]", s)


def substitutions(old: str, new: str, min_chars: int = 2) -> list:
    """The ``[(old_span, new_span)]`` substitutions turning ``old`` into ``new``.

    Pure insertions are dropped: they have no anchor in the old reply to rewrite.
    """
    a, b = _chunks(old), _chunks(new)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        o, n = "".join(a[i1:i2]), "".join(b[j1:j2])
        if tag == "insert":
            continue
        if len(o) >= min_chars:
            out.append((o, n))
    return out


def apply_substitutions(text: str, subs: list) -> tuple:
    """Rewrite ``text`` in place; returns ``(edited, applied, skipped)``, longest span first."""
    edited = text
    applied = []
    for old, new in sorted(subs, key=lambda p: -len(p[0])):
        if old and old in edited:
            edited = edited.replace(old, new)
            applied.append((old, new))
    skipped = [p for p in subs if p not in applied]
    return edited, applied, skipped


def _trim(text: str, stopk) -> str:
    """Cut text at the earliest stop marker.

    Needed when text is rebuilt from tokens: a reply that ended on its own carries
    the stop token, which the stored text had already had trimmed off.
    """
    cut = len(text)
    for sp in (stopk or ()):
        j = text.find(sp)
        if 0 <= j < cut:
            cut = j
    return text[:cut]


def similarity(a: str, b: str) -> float:
    """Ratio in [0, 1] between two turn texts, for the level-2 match gate."""
    return difflib.SequenceMatcher(a=_chunks(a), b=_chunks(b), autojunk=False).ratio()


class Trace:
    """Top-k logits behind each generated token, one row per token.

    A row is ``(ids, logits)`` descending, or ``None`` where the logits were not
    captured; ``None`` rows replay verbatim and cannot validate an edit.
    """

    __slots__ = ("rows",)

    def __init__(self, rows=None):
        self.rows = rows if rows is not None else []

    def __len__(self):
        return len(self.rows)

    def append(self, ids, logits):
        self.rows.append(None if ids is None else (np.asarray(ids, dtype=np.int32),
                                                   np.asarray(logits, dtype=np.float32)))

    def supports(self, position: int, token: int) -> bool:
        """Whether the model had ``token`` among its top-k at ``position``."""
        if position < 0 or position >= len(self.rows):
            return False
        row = self.rows[position]
        if row is None:
            return False
        return bool((row[0] == token).any())

    def argmax(self, position: int) -> int | None:
        """The token the model actually ranked first at ``position``."""
        if position < 0 or position >= len(self.rows):
            return None
        row = self.rows[position]
        if row is None:
            return None
        return int(row[0][int(np.argmax(row[1]))])

    def replay(self, tokens: list) -> list:
        """Emit the stored token sequence through each row's argmax; an identity on an unedited trace."""
        out = []
        for i, token in enumerate(tokens):
            chosen = self.argmax(i)
            out.append(token if chosen is None else chosen)
        return out


class Entry:
    """One recorded (turn -> reply) pair, with the prompt it was recorded at."""

    __slots__ = ("qkey", "qtext", "kind", "pkey", "plen", "toks",
                 "text", "stopk", "trace", "trunc", "has_call")

    def __init__(self, qkey, qtext, kind, pkey, plen, toks, text, stopk,
                 trunc=False, trace=None, has_call=False):
        self.qkey, self.qtext, self.kind = qkey, qtext, kind
        self.pkey, self.plen = pkey, plen
        self.toks, self.text, self.stopk = toks, text, stopk
        # ``trunc``: the reply ran into the token budget rather than ending on
        # its own, and only then may it be shortened to a smaller budget.
        self.trunc = trunc
        self.trace = trace
        # Whether the reply contains a tool call: replaying one that does
        # re-issues the call that produced the tool result, and the dialog loops.
        self.has_call = has_call


class Match:
    """A reuse decision, with everything the caller needs to justify it."""

    __slots__ = ("entry", "level", "reason", "saved_tokens")

    def __init__(self, entry, level, reason, saved_tokens):
        self.entry, self.level, self.reason = entry, level, reason
        self.saved_tokens = saved_tokens


class QCache:
    """Turn-keyed store of earlier replies.

    ``level`` 1 requires the cached prompt to be a prefix of the incoming one;
    ``level`` 2 additionally accepts a similar turn from any dialog, which needs
    the edit path.
    """

    def __init__(self, level: int = 0, edit: bool = False, topk: int = DEFAULT_TOPK,
                 sim: float = 0.9, capacity: int = 512, skip_tool: bool = True):
        if level not in (0, 1, 2):
            raise ValueError("CC_QREUSE must be 0, 1 or 2")
        if topk < 2:
            raise ValueError("CC_QEDIT_TOPK must be at least 2")
        if not 0.0 < sim <= 1.0:
            raise ValueError("CC_QEDIT_SIM must be in (0, 1]")
        self.level, self.edit, self.topk = level, edit, topk
        # Reusing a reply to a tool result re-issues the very call that produced it,
        # which is how a dialog starts repeating itself; CC_QREUSE_TOOL=1 opts in.
        self.skip_tool = skip_tool
        self.sim, self.capacity = sim, capacity
        self.clear()

    def clear(self):
        self.entries: dict = {}
        self.order: list = []
        self.st = {"store": 0, "hit1": 0, "hit2": 0, "miss": 0, "evict": 0,
                   "skip_tool": 0, "loop_risk": 0, "replay_tok": 0,
                   "edited": 0, "edit_applied": 0,
                   "edit_rejected": 0, "saved_tok": 0}

    # ---------------- store ----------------
    def store(self, key, prompt, toks, text, stopk, trunc=False, trace=None,
              has_call=False):
        """Record a reply.  ``key`` is a :func:`turn_key` result.

        Prompts are not retained: only the digest for the prefix test and the text.
        """
        if key is None or not toks:
            return
        kind, digest, qtext = key
        pkey = _prompt_key(prompt)
        entry = Entry(digest, qtext, kind, pkey, len(prompt),
                      list(toks), text, tuple(stopk), trunc, trace, has_call)
        self.entries.setdefault(digest, []).append(entry)
        self.order.append((digest, entry))
        self.st["store"] += 1
        while len(self.order) > self.capacity:
            d, e = self.order.pop(0)
            bucket = [x for x in self.entries.get(d, ()) if x is not e]
            if bucket:
                self.entries[d] = bucket
            else:
                self.entries.pop(d, None)
            self.st["evict"] += 1

    # ---------------- lookup ----------------
    def match(self, key, prompt, stopk, max_new):
        """Find a reusable entry for this request, or ``None``.

        A candidate is only usable when its stop set matches and it produced enough
        tokens to satisfy the caller's budget.
        """
        if self.level == 0 or key is None:
            return None
        kind, digest, qtext = key
        if self.skip_tool and kind == "tool":
            self.st["skip_tool"] += 1
            return None
        best = None
        for entry in reversed(self.entries.get(digest, ())):
            if entry.kind == "tool" and entry.has_call:
                # Past the skip_tool gate, but this reply itself calls a tool, so replaying
                # it is the loop noted on Entry.
                self.st["loop_risk"] += 1
                continue
            if entry.kind == kind and self._usable(entry, prompt, stopk, max_new):
                best = Match(entry, 1, "same-turn-same-dialog", len(entry.toks))
                break
        if best is None and self.level >= 2 and kind == "text":
            best = self._fuzzy(key, prompt, stopk, max_new)
        if best is None:
            self.st["miss"] += 1
            return None
        self.st["hit1" if best.level == 1 else "hit2"] += 1
        return best

    def _usable(self, entry, prompt, stopk, max_new) -> bool:
        if tuple(stopk) != entry.stopk:
            return False
        if entry.plen > len(prompt):
            return False
        if not self._budget_ok(entry, max_new):
            return False
        if entry.plen == len(prompt):
            # A byte-identical prompt is the exact-prompt logits cache's job, not this layer's.
            return False
        # The same dialog, continued: the recorded prompt must be a prefix.
        return _prompt_key(prompt[:entry.plen]) == entry.pkey

    @staticmethod
    def _budget_ok(entry, max_new: int) -> bool:
        """Whether a recorded reply can answer a request with this token budget.

        A reply cut off by its budget never produced its ending, so it is only
        reusable where the caller wants no more than was recorded.
        """
        return (not entry.trunc) or max_new <= len(entry.toks)

    def _fuzzy(self, key, prompt, stopk, max_new):
        """Level 2: a similar turn from any dialog, if the edit path is on.

        Two gates: above the similarity floor, and with at least one substitution --
        no rewrite possible would just copy an answer to a different question.
        """
        if not self.edit:
            return None
        kind, _, qtext = key
        cands = []
        for entries in self.entries.values():
            for entry in reversed(entries):
                if entry.kind != kind or entry.trace is None or len(entry.trace) == 0:
                    continue
                if not entry.toks or not self._budget_ok(entry, max_new):
                    continue
                if tuple(stopk) != entry.stopk:
                    continue
                s = similarity(entry.qtext, qtext)
                if s >= self.sim:
                    cands.append((s, entry))
        for s, entry in sorted(cands, key=lambda p: -p[0])[:8]:
            if not substitutions(entry.qtext, qtext):
                continue
            return Match(entry, 2, f"similar-turn({s:.2f})", len(entry.toks))
        return None

    # ---------------- replay ----------------
    def realize(self, match, new_turn: str, max_new: int, encode, decode):
        """Turn a match into ``(tokens, text, info)``, using ``encode`` and ``decode``.

        Tokens and text always describe the same reply: a shortened reply rebuilds
        its text from the surviving tokens and re-trims at a stop marker.
        """
        entry = match.entry
        cut = min(max_new, len(entry.toks))
        shortened = cut < len(entry.toks)
        toks = list(entry.toks[:cut])
        text = _trim(utext.whole(b"".join(decode(t) for t in toks)), entry.stopk) if shortened else entry.text
        info = {"qreuse": match.level, "why": match.reason, "saved": 0,
                "replayed": 0, "edited": 0, "edit_applied": 0, "edit_rejected": 0}

        if not self.edit or entry.trace is None:
            info["saved"] = len(toks)
            self.st["saved_tok"] += len(toks)
            info["text"] = text
            return toks, text, info

        toks = entry.trace.replay(toks)
        info["replayed"] = len(toks)
        self.st["replay_tok"] += len(toks)

        subs = substitutions(entry.qtext, new_turn) if entry.qtext != new_turn else []
        edited, applied, _ = apply_substitutions(text, subs) if subs else (text, [], [])
        if edited != text:
            candidate = encode(edited)
            toks, accepted, rejected = self._validate(entry.trace, toks, candidate)
            info.update(edited=1, edit_applied=accepted, edit_rejected=rejected)
            self.st["edited"] += 1
            self.st["edit_applied"] += accepted
            self.st["edit_rejected"] += rejected
            # Always rebuild from the tokens once an edit applied: a refused substitution
            # falls back to the model's argmax, so only the tokens are authoritative.
            text = _trim(utext.whole(b"".join(decode(t) for t in toks)), entry.stopk)
        toks = toks[:max_new]
        info["saved"] = len(toks)
        self.st["saved_tok"] += len(toks)
        info["text"] = text
        return toks, text, info

    def _validate(self, trace: Trace, base: list, candidate: list):
        """Align an edited token stream against the trace; returns ``(tokens, accepted, rejected)``.

        Only substitutions are checked against the cached top-k: deletions, insertions
        and a grown replacement's tail have no cached row governing them.
        """
        sm = difflib.SequenceMatcher(a=base, b=candidate, autojunk=False)
        out, accepted, rejected = [], 0, 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                out.extend(base[i1:i2])
            elif tag == "delete":
                accepted += i2 - i1
            elif tag == "insert":
                out.extend(candidate[j1:j2])
                accepted += j2 - j1
            else:
                width = min(i2 - i1, j2 - j1)
                for n in range(width):
                    replacement = candidate[j1 + n]
                    if trace.supports(i1 + n, replacement):
                        out.append(replacement)
                        accepted += 1
                    else:
                        out.append(base[i1 + n])
                        rejected += 1
                # Only the candidate tail is carried over: re-appending a base tail
                # would splice tokens the edit deleted back into the reply.
                if j2 - j1 > width:
                    out.extend(candidate[j1 + width:j2])
                    accepted += (j2 - j1) - width
        return out, accepted, rejected

    def line(self) -> str:
        s = self.st
        return (f"qstore={s['store']} qhit1={s['hit1']} qhit2={s['hit2']} "
                f"qmiss={s['miss']} qskip={s['skip_tool']} qloop={s['loop_risk']} "
                f"qevict={s['evict']} "
                f"qreplay={s['replay_tok']} qedit={s['edited']}/"
                f"{s['edit_applied']}ok/{s['edit_rejected']}rej "
                f"qemit={s['saved_tok']}")


def _prompt_key(ids: list) -> tuple:
    """Same shape as :meth:`Eng.key`, so a prefix can be checked cheaply."""
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(ids, dtype=np.int32).tobytes())
    return (len(ids), h.digest())
