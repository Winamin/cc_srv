# -*- coding: utf-8 -*-
"""Stream text to the client while the model is still generating.

``feed`` returns only text guaranteed to be a prefix of what the buffered path
would have sent; ``finish`` returns the remainder once the real text is known.
Text from a marker onward is structured tool_use, trimmed, or a think block, so
it is cut there rather than streamed.
"""
from __future__ import annotations

# Markers whose arrival ends the text: everything from a <tool_call> becomes
# structured tool_use, and everything from a stop string is trimmed by the engine.
CUTS = ("<tool_call>",)

# Markers that must not be sent half-formed.  <think> is here and not in CUTS
# on purpose: a closed think block is removed and the text after it continues.
PARTS = ("<think>", "</think>", "<tool_call>", "</tool_call>")


def strip_think(region: str) -> str:
    """drop_think over a region that is already known to be text; an unclosed
    <think> drops the whole remainder."""
    out, k = [], 0
    while True:
        a = region.find("<think>", k)
        if a < 0:
            out.append(region[k:])
            return "".join(out)
        out.append(region[k:a])
        b = region.find("</think>", a)
        if b < 0:
            return "".join(out)      # unclosed: the remainder never becomes text
        k = b + len("</think>")


def partial_tail(s: str, markers) -> int:
    """Length of the longest suffix of ``s`` that could still grow into a marker."""
    best = 0
    for m in markers:
        for n in range(min(len(m) - 1, len(s)), best, -1):
            if s.endswith(m[:n]):
                best = n
                break
    return best


class Streamer:
    """Feed raw generated chunks in, take safe ``(kind, text)`` pairs out.

    ``holdback`` is the caller's stop strings, which a partial one must not pass
    either.  ``think=True`` returns "thinking" before ``</think>`` and "text"
    after; ``opened`` says the prompt already emitted the opening tag.
    """

    def __init__(self, holdback=(), think=False, opened=False):
        stops = tuple(holdback or ())
        self.cuts = CUTS + stops
        self.parts = PARTS + stops
        self.buf = ""
        self.sent = 0
        self.mismatch = False
        self.think = think
        self.tbuf = ""
        self.tsent = 0
        # ``opened``: the prompt already ended with the opening tag, so the model's
        # output carries only the closing one.  "probe" would otherwise wait for
        # an opener that is never coming and stream the whole answer as text.
        self.tstate = "in" if (think and opened) else ("probe" if think else "done")

    def feed(self, chunk):
        """Newly safe ``(kind, text)`` pairs, or [] if nothing settled."""
        if not chunk:
            return []
        self.buf += chunk
        out = []
        if self.think and self.tstate != "done":
            t = self._think_step()
            if t:
                out.append(("thinking", t))
            if self.tstate != "done":
                return out          # still inside the reasoning
        safe = self._safe_prefix()
        if len(safe) > self.sent:
            out.append(("text", safe[self.sent:]))
            self.sent = len(safe)
        return out

    def _think_step(self) -> str:
        """Drain what is safe of the think region.  "" if nothing new.

        "probe" holds the first bytes while they could still be the opener, then
        commits to no reasoning.
        """
        op, end = "<think>", "</think>"
        if self.tstate == "probe":
            if len(self.buf) < len(op) and op.startswith(self.buf):
                return ""               # still could turn into the opener
            if not self.buf.startswith(op):
                self.tstate = "done"    # no reasoning here; the rest is the answer
                return ""
            self.buf = self.buf[len(op):]
            self.tstate = "in"
        i = self.buf.find(end)
        if i >= 0:
            body, self.buf = self.buf[:i], self.buf[i + len(end):]
            self.tstate = "done"
        else:
            # Hold back a tail that could still be the closing tag.
            body = self.buf[:len(self.buf) - partial_tail(self.buf, (end,))]
        if len(body) <= self.tsent:
            return ""
        new = body[self.tsent:]
        self.tsent = len(body)
        return new

    def _safe_prefix(self) -> str:
        # Cut at the earliest marker whose arrival ends the text.
        cut = len(self.buf)
        for m in self.cuts:
            i = self.buf.find(m)
            if 0 <= i < cut:
                cut = i
        region = self.buf[:cut]
        # Hold back a tail that might still turn into a marker.
        region = region[:len(region) - partial_tail(region, self.parts)]
        # Then the think blocks, then the strip the buffered path applies.  In
        # think mode the block is already behind us, so nothing is left to drop.
        return (region if self.think else strip_think(region)).strip()

    def finish(self, final_text: str) -> str:
        """The remainder, so that everything sent adds up to ``final_text``.

        Sets ``mismatch`` and sends nothing if the streamed text is not a prefix.
        """
        if not final_text.startswith(self._emitted()):
            self.mismatch = True
            return ""
        return final_text[self.sent:]

    def _emitted(self) -> str:
        return self._safe_prefix()[:self.sent]
