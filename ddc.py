"""Margin-aware draft cache. Every proposed token still needs target verification.

Margin at offset i predicts token i, not token i+1. Token zero of a draft
already agrees with the live argmax; truncation starts at offset one.
"""
from array import array
from collections import OrderedDict, deque
from dataclasses import dataclass
import math
import numpy as np

QUANTUM = 1 / 16


def features(logits):
    """Top-five IDs and margins; a CPU vocabulary scan, not zero overhead."""
    ids = np.argpartition(logits, -5)[-5:]
    ids = ids[np.lexsort((ids, -logits[ids]))]
    # Ties at the partition boundary need not include the lowest vocabulary ID.
    # The live argmax is authoritative, and a disagreeing draft is discarded.
    return tuple(map(int, ids)), tuple(map(float, logits[ids[:-1]] - logits[ids[1:]]))


def quantize(margin):
    if not math.isfinite(margin):
        raise ValueError("Nonfinite margin")
    return min(255, max(0, math.floor(margin / QUANTUM)))


@dataclass
class Trace:
    tokens: array
    margins: bytearray
    keys: list


@dataclass(frozen=True)
class Proposal:
    tokens: tuple
    trace_id: int
    offset: int
    cross_request: bool


class DraftCache:
    def __init__(self, k=2, max_draft=16, threshold=0., gate=0., capacity=65536,
                 reset_per_request=False):
        if k not in (2, 4) or max_draft < 2 or capacity < max_draft:
            raise ValueError("Unsupported cache configuration")
        if not all(math.isfinite(x) and x >= 0 for x in (threshold, gate)):
            raise ValueError("Thresholds must be finite and nonnegative")
        self.k, self.max_draft, self.gate = k, max_draft, gate
        self.cutoff = math.ceil(threshold / QUANTUM)
        self.capacity = capacity
        self.reset_per_request = reset_per_request
        self.clear()

    def clear(self):
        """Discard all drafts and indexes, preserving policy configuration.

        Rebuilding uses only subsequently committed tokens. This does not alter
        model KV, prefill logits, or the engine's exact-response caches.
        """
        self.traces = OrderedDict()
        self.index = {}
        self.next_id = 0
        self.size = 0
        self.current = None
        self.request_start = 0

    def begin(self):
        if self.reset_per_request:
            self.clear()
        self.request_start = self.next_id
        self._segment()

    def _segment(self):
        if self.current is not None and not self.traces[self.current].tokens:
            del self.traces[self.current]
        self.current = self.next_id
        self.next_id += 1
        self.traces[self.current] = Trace(array("i"), bytearray(), [])

    def append(self, token, ids, gaps):
        if self.current is None:
            raise RuntimeError("begin() must precede append()")
        trace = self.traces[self.current]
        offset = len(trace.tokens)
        trace.tokens.append(int(token))
        trace.margins.append(quantize(gaps[0]))
        key = tuple(ids[:self.k])
        # For a top-k key, use the k versus k+1 membership boundary.
        # This does not certify the stability of ordering INSIDE the top-k.
        admitted = gaps[self.k-1] >= self.gate and ids[0] == token
        trace.keys.append(key if admitted else None)
        if admitted:
            bucket = self.index.setdefault(key, deque(maxlen=8))
            bucket.append((self.current, offset))
        self.size += 1
        self._evict()

    def _evict(self):
        while self.size > self.capacity:
            tid, trace = self.traces.popitem(last=False)
            self.size -= len(trace.tokens)
            for key in set(x for x in trace.keys if x is not None):
                bucket = deque((x for x in self.index[key] if x[0] != tid), maxlen=8)
                if bucket:
                    self.index[key] = bucket
                else:
                    del self.index[key]
            if tid == self.current:
                # A single request may exceed capacity. Keep a new bounded
                # segment; this loses proposals, never changes target tokens.
                self.current = None
                self._segment()

    def draft(self, ids, next_token, room):
        proposal = self.propose(ids, next_token, room)
        return list(proposal.tokens) if proposal is not None else None

    def propose(self, ids, next_token, room):
        """Longest usable draft for this position, or None.

        The truncation loop is what decides draft LENGTH: it stops at the first
        recorded position whose margin is under the cutoff, so ``threshold``
        controls length far more than ``max_draft`` does.  Measured on real CC
        traffic, the round's rate rises monotonically with draft length --
        27 tok/s at 2-3 tokens, 84 at 4-7, 96 at 8-15, 160 at 16+ -- because a
        verification pass costs the same whatever it carries.  So the default is
        set aggressive (short cutoff, long cap) and the per-round gate in spec.py
        drops whatever bucket turns out not to pay.
        """
        if room < 2:
            return None
        for tid, offset in reversed(self.index.get(tuple(ids[:self.k]), ())):
            trace = self.traces[tid]
            length = min(self.max_draft, room, len(trace.tokens)-offset)
            if length < 2 or trace.tokens[offset] != next_token:
                continue
            # Exclude the first uncertain token, not its predecessor. If this
            # leaves only the already-known token, bypass speculation entirely.
            for i in range(1, length):
                if trace.margins[offset+i] < self.cutoff:
                    length = i
                    break
            if length < 2:
                return None
            return Proposal(tuple(trace.tokens[offset:offset+length]), tid, offset,
                            tid < self.request_start)
        return None
