# -*- coding: utf-8 -*-
"""Stream text to the client while the model is still generating.

Why
---
The buffered path waits for the whole reply and then sends it in one delta, so
the first thing a client sees arrives after the ENTIRE generation.  Measured on
a live fan-out request that generated 626 tokens: 18.4 s before any text at all.
Streaming moves that to prefill + one token.

It does not make the agent loop faster -- Claude Code still waits for
message_stop before it acts on a tool call -- so this is about what a person
watching sees, not about throughput.

The hard part
-------------
What may be sent early is constrained by what the buffered path would have sent:

* ``calls_of`` turns everything from ``<tool_call>`` onward into structured
  tool_use blocks.  None of it is text, so none of it may be streamed as text.
  Emitting half a tool call is not a display glitch -- the client would execute
  the wrong thing.
* ``drop_think`` removes ``<think>...</think>`` (and an UNCLOSED ``<think>``
  takes the rest of the reply with it), so think content must never be sent.
* The final text is ``.strip()``-ed, so leading and trailing whitespace never
  appears in it.
* A tail that could still grow into any of those markers -- or into one of the
  caller's stop strings -- has to be held back until the next chunk decides it.

``Streamer.feed`` returns only text that is guaranteed to be a prefix of what the
buffered path would have sent, and ``finish`` returns whatever is left once the
real text is known.  The two together are exactly the buffered text; that is the
property ``test_stream.py`` checks, on adversarial inputs and on randomized ones.
"""
from __future__ import annotations

# Markers whose ARRIVAL ends the text: everything from a <tool_call> becomes
# structured tool_use, and everything from a stop string is trimmed off by the
# engine.  Neither is text, so the stream is cut at whichever comes first.
CUTS = ("<tool_call>",)

# Markers that do not end the text but must not be sent half-formed.  <think> is
# here and not in CUTS on purpose: a CLOSED think block is removed and the text
# after it continues, so cutting at it would throw away good text.
PARTS = ("<think>", "</think>", "<tool_call>", "</tool_call>")


def strip_think(region: str) -> str:
    """drop_think over a region that is already known to be text.

    Mirrors Eng/srv drop_think exactly, including its treatment of an unclosed
    <think>: everything from there on is dropped, not just up to the next tag.
    """
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
    """Length of the longest suffix of ``s`` that could still grow into a marker.

    A marker that is already complete is not a partial tail; that case is handled
    by cutting at it.  This is what holds back "<tool_c" until the next chunk
    says whether it is a tool call.
    """
    best = 0
    for m in markers:
        for n in range(min(len(m) - 1, len(s)), best, -1):
            if s.endswith(m[:n]):
                best = n
                break
    return best


class Streamer:
    """Feed raw generated chunks in, take safe text out.

    ``holdback`` is the caller's stop strings: a partial one must not be sent
    either, since the buffered path trims the reply at it.

    ``think=True`` surfaces the reasoning instead of dropping it; ``opened``
    says the prompt already emitted the opening tag, so only ``</think>`` will
    appear in the model's output.  ``feed`` then
    returns ``(kind, text)`` pairs -- "thinking" for the region before the first
    ``</think>``, "text" for everything after -- and the caller puts them in
    separate content blocks.  The text half runs through exactly the same rules
    as before, so turning thinking on does not change what the answer looks like.
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
        # ``opened``: the prompt already ended with the opening tag, so the
        # model's own output carries only the closing one.  Starting in "probe"
        # would look for an opening tag that is never coming, decide there is no
        # reasoning, and stream the whole answer as text.
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

        Three states, because a reply that never opens with <think> has no
        reasoning at all and must not be mistaken for one: "probe" holds the
        first bytes while they could still be the opener, then commits.
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
        # Then the think blocks, then the strip the buffered path applies.
        # In think mode the block is already behind us, so there is nothing left
        # to remove -- the text half is identical either way.
        return (region if self.think else strip_think(region)).strip()

    def finish(self, final_text: str) -> str:
        """The remainder, so that everything sent adds up to ``final_text``.

        The whole point of the safety rules is that this is a plain suffix.  If
        it is not, something upstream sent text that the buffered path would not
        have, and sending more would compound it -- so nothing more is sent and
        ``mismatch`` is set for the caller to log.
        """
        if not final_text.startswith(self._emitted()):
            self.mismatch = True
            return ""
        return final_text[self.sent:]

    def _emitted(self) -> str:
        return self._safe_prefix()[:self.sent]
