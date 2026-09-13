# -*- coding: utf-8 -*-
"""Turn-keyed reuse of an earlier reply, optionally replayed from cached logits.

Why this exists
---------------
``Eng.lgc`` keys on the whole prompt, so it only fires when a request repeats
byte for byte.  Inside a Claude Code dialog that almost never happens: every new
turn appends the previous exchange, so the exact-prompt cache only pays off when
a whole conversation is replayed.  The key here is instead the *trailing user
turn*, which is what actually identifies "the same question".

Switch 1 -- ``CC_QREUSE``
    0   off (the server behaves as before)
    1   reuse when the same turn is asked again inside the same dialog, i.e. the
        cached prompt is a prefix of the incoming prompt
    2   also reuse across dialogs and for merely *similar* turns

Switch 2 -- ``CC_QEDIT``
    Keeps a top-k trace of the logits behind every generated token, replays that
    trace through argmax, and lets a server-side edit rewrite the reply.  The
    replay needs no forward pass, so the replayed tokens cost nothing to emit.
    Level-2 reuse requires this switch: a similar question cannot be answered by
    copying the old text, only by editing it.

What this is not
----------------
Reuse is a semantic choice, not a correctness proof.  The reply is genuinely the
model's, but it was produced under the context that was current when the
question was first asked; nothing here re-derives it under the new context.
Substitutions are validated against the cached top-k, which bounds how far a
replacement may drift from what the model had in distribution.  Deletions,
insertions, and the tail of a replacement that grew have no cached row to answer
to and are written unchecked; see _validate.  None of this is a guarantee that
the result equals a fresh generation, and the two are not claimed equal.
"""
from __future__ import annotations

import difflib
import hashlib
import re

import numpy as np

# Replayed tokens are emitted by argmax over the stored row, so a stored trace
# reproduces exactly the tokens it was recorded from.  The width only bounds
# which edits count as "in distribution" for the model at that position.
DEFAULT_TOPK = 16

# Claude Code injects volatile reminders into the user turn -- the attribution
# block, the environment block.  They are not part of the question, and they
# differ between the first ask and a repeat, so leaving them in the key would
# stop the same question from ever matching itself.
_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)

# A tool result can be tens of thousands of tokens.  The digest covers all of
# it; only this much text is kept, and only so that a level-2 similarity test
# has something bounded to compare.
QTEXT_MAX = 2000


def turn_key(msgs: list) -> tuple | None:
    """Key identifying the trailing user turn of a request.

    ``msgs`` is the flattened ``[(role, content)]`` list the renderer builds.
    Returns ``(kind, digest, text)`` or ``None`` when the request carries no
    user turn at all.

    The scan walks *backwards* past any trailing non-user turn.  In a real
    Claude Code request the array ends with a ``role: system`` message holding
    ``<total_tokens>N tokens left</total_tokens>``, so the last message is
    almost never the question; keying on ``msgs[-1]`` would never match.

    The trailing turn is used rather than the whole message list because the
    list grows every turn by construction, while the question is what the user
    repeats.
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
    """Word-level span substitutions that turn ``old`` into ``new``.

    Returns ``[(old_span, new_span)]``.  Pure insertions are dropped: they have
    no anchor inside the old reply, so there is nothing to rewrite in place.
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
    """Rewrite ``text`` in place; returns ``(edited, applied, skipped)``.

    Spans are applied longest-first so that a short span cannot pre-empt a
    longer, more specific one that contains it.
    """
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

    Needed whenever text is rebuilt from a token stream: a reply that ended on
    its own carries the stop token, while the stored text was already trimmed at
    the text level, so a naive rebuild puts the marker back.
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

    A row is ``(ids, logits)`` sorted by descending logit, or ``None`` for a
    position whose logits were never captured (a token served from the exact
    cache rather than computed).  ``None`` rows are replayed verbatim and cannot
    validate an edit.
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
        """Emit the stored token sequence through argmax of each stored row.

        Rows reproduce their own top-1 by construction, so this is an identity
        on an unedited trace; it exists so that an edit has somewhere to be
        applied and so that a trace/​token mismatch is detected rather than
        silently copied.
        """
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
        # ``trunc`` means the recorded reply ran into the token budget rather
        # than ending on its own.  Only then may it be shortened to a smaller
        # budget: a naturally finished reply has no stop token inside ``text``
        # precisely because the trimming happened at the text level, so
        # re-deriving its text from tokens would put the stop marker back.
        self.trunc = trunc
        self.trace = trace
        # Whether the recorded reply contains a tool call.  This is what makes
        # reusing a reply to a TOOL RESULT safe or not: replay it, and if the
        # reply itself calls a tool, Claude Code runs that tool, gets the same
        # result, asks the same trailing turn again -- and the cache answers
        # again.  The prompt keeps growing so the prefix test keeps passing and
        # the loop never breaks on its own.  A reply with no tool call in it has
        # no such edge.
        self.has_call = has_call


class Match:
    """A reuse decision, with everything the caller needs to justify it."""

    __slots__ = ("entry", "level", "reason", "saved_tokens")

    def __init__(self, entry, level, reason, saved_tokens):
        self.entry, self.level, self.reason = entry, level, reason
        self.saved_tokens = saved_tokens


class QCache:
    """Turn-keyed store of earlier replies.

    ``level`` 1 requires the cached prompt to be a prefix of the incoming prompt
    -- the same dialog, continued.  ``level`` 2 additionally accepts a similar
    turn from any dialog, which is only sound together with the edit path.
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
        # Reusing the reply to a *tool result* is how a dialog starts repeating
        # itself: the cached reply usually contains the very tool call that
        # produced the result, so replaying it re-issues that call.  Off by
        # default; CC_QREUSE_TOOL=1 allows it.
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

        Prompts are not retained in full: only the digest needed to check the
        prefix relation, plus the turn text used for similarity.
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

        A candidate is only usable when its stop set matches and it produced
        enough tokens to satisfy the caller's budget; otherwise the reply would
        be truncated relative to what was asked for.
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
                # Allowed past the skip_tool gate, but this specific reply calls
                # a tool -- replaying it is the loop described on Entry.
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
            # Byte-identical prompt: that is the exact-prompt logits cache's job,
            # not this layer's, and the caller has already tried it.
            return False
        # The same dialog, continued: the recorded prompt must be a prefix.
        return _prompt_key(prompt[:entry.plen]) == entry.pkey

    @staticmethod
    def _budget_ok(entry, max_new: int) -> bool:
        """Whether a recorded reply can answer a request with this token budget.

        A reply that ended on its own fits any budget: max_new is a cap, and a
        reply shorter than the cap is ordinary.  One that ran into its budget is
        different -- its ending was never produced, so reusing it hands back a
        reply that stops mid-thought.  That is only acceptable when the caller
        allows no more than was recorded, which is also the only case in which
        realize() can shorten the token stream to fit.
        """
        return (not entry.trunc) or max_new <= len(entry.toks)

    def _fuzzy(self, key, prompt, stopk, max_new):
        """Level 2: a similar turn from any dialog, if the edit path is on.

        Two gates, and both were tightened after measuring a real dialog rather
        than by reasoning about one:

        * The similarity floor.  Claude Code questions are long and share a lot
          of boilerplate ("Read src/... and list the ..."), so two genuinely
          different questions score around 0.65.  At the original floor of 0.6
          this fired on 10 of the 18 requests of a live dialog, answering one
          question with a copy of the answer to another, and the dialog
          degenerated.  The floor now defaults to 0.9.
        * The edit must be non-empty.  Level 2 exists to *rewrite* a reply; if
          the difference between the two turns admits no substitution at all,
          there is nothing to rewrite and what is left is copying an answer to a
          different question.  That is refused.

        Only the candidates above the floor are scored for substitutions, so the
        expensive part stays bounded.
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
        """Turn a match into ``(tokens, text, info)``.

        ``encode`` maps text to token ids and ``decode`` maps a token id back to
        text; the edit path needs both directions and they are not the same
        function.

        The returned tokens and text always describe the same reply.  When a
        recorded reply is longer than the caller's budget it is cut to the budget
        and the text is rebuilt from the surviving tokens -- and re-trimmed at a
        stop marker, because the token stream of a reply that ended on its own
        carries the stop token that the stored text had already had trimmed off.
        Serving the tokens truncated while returning the full stored text is the
        easy version of this and it desynchronises the two.

        Without ``CC_QEDIT`` no trace is consulted; the reply is returned as
        stored, possibly cut to the budget as above.
        """
        entry = match.entry
        cut = min(max_new, len(entry.toks))
        shortened = cut < len(entry.toks)
        toks = list(entry.toks[:cut])
        text = _trim("".join(decode(t) for t in toks), entry.stopk) if shortened else entry.text
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
            # Always rebuild from the token stream once an edit has been applied.
            # A refused substitution falls back to the model's argmax inside
            # _validate, so the edited string is no longer what the tokens say,
            # and only the tokens are authoritative.
            text = _trim("".join(decode(t) for t in toks), entry.stopk)
        toks = toks[:max_new]
        info["saved"] = len(toks)
        self.st["saved_tok"] += len(toks)
        info["text"] = text
        return toks, text, info

    def _validate(self, trace: Trace, base: list, candidate: list):
        """Align an edited token stream against the trace; keep only in-distribution edits.

        Only *substitutions* are checked against the model's cached top-k.  Three
        kinds of token are written without a check, and the caller's reports say
        so rather than implying otherwise:

        * a deletion -- removing text needs no distribution to justify;
        * an insertion, and the tail of a replacement that grew -- there is no
          cached row governing a position that did not exist;
        * the fallback for a refused substitution, which is the model's own
          argmax at that position.

        Returns ``(tokens, accepted, rejected)``.
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
                # Only the candidate tail is carried over.  A base tail must NOT
                # be re-appended: the candidate replaced the whole block, and
                # keeping the base remainder splices tokens the edit deleted back
                # into the reply -- measured as a word from the old reply
                # reappearing in the middle of the edited one.
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
