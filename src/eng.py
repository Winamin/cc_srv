# -*- coding: utf-8 -*-
"""Engine layer: KV, APC, logits cache, trajectory recall, turn-keyed reuse.

Cache layers, cheapest first:
  1)  logits cache   : prompt identical token for token -> replay with no forward
                      pass and no KV work
  1b) trajectory recall: prompt is a PREFIX of a recorded trajectory -> hand back
                      the recorded continuation, again with no forward pass
  1c) turn-keyed reuse: the trailing user turn was answered before in this same
                      dialog -> replay (or edit) that reply; opt-in, see qcache
  2)  APC            : longest common prefix -> only the new part is forwarded
  3)  cold start     : full prefill

This model is hybrid (full attention plus recurrent layers).  The recurrent
state is a running summary over 0..p and cannot be rewound, so a PARTIAL
truncation via llama_memory_seq_rm is a no-op in practice (measured: truncating
to 2500 still leaves kvlen at 3000).  trunc() therefore verifies after
truncating and, when the truncation did not take, clears everything and makes
the caller restart from position zero.  Correctness first: the price is paying
for the whole context again.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import os
import time
from array import array

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np

import spec
from lib import Mp, Cp, Msg, Batch, GGUF, Q4_0, bind


def ddc_key():
    """The CC_DDC setting in force, or None when the draft cache is off.

    Off by default.  It was on by default until the archive became the default
    instead: the archive is the layer that helps the shape this server actually
    sees (agent fan-out), while a draft cache is a throughput knob that pays
    only where the content repeats and changes the numerical path when it does.
    See the draft-cache block in Eng.__init__.

    This stays the place the default is decided, so that eng.plan_sequences and
    Eng.__init__ cannot disagree about whether DDC is on -- they both call it.
    """
    v = os.environ.get("CC_DDC")
    return v if (v and v != "0") else None


def arc_slot_count() -> int:
    """How many sequences the prefix archive asks to keep.  0 when it is off.

    On by default; ``CC_ARCHIVE=0`` turns it off.
    """
    v = os.environ.get("CC_ARCHIVE", "1").strip().lower()
    if v in ("0", "false", "off", "no"):
        return 0
    return int(os.environ.get("CC_ARCHIVE_SLOTS", "1"))


def ddc_batch_pool() -> int:
    """Scratch sequences for DDC inside the batched scheduler.

    0 (off) unless CC_DDC_BATCH=1.  Unverified -- see Batcher's speculation
    block -- so it stays opt-in rather than riding on CC_DDC's default.
    """
    if os.environ.get("CC_DDC_BATCH") != "1":
        return 0
    return max(0, int(os.environ.get("CC_DDC_BATCH_SCRATCH", "1")))


def ddc_scratch_count(workers: int) -> int:
    """How many sequences DDC needs of its own.

    One, on the serial path.  Under CC_BATCH it is a small pool instead: the
    scheduler serves several requests at once, and a verification needs its
    scratch for the whole round, so a single one would let only one of them
    speculate.  The pool is a bound on the KV reservation, not on correctness --
    a job that cannot get a scratch decodes normally.
    """
    if ddc_key() is None:
        return 0
    if workers == 1:
        return 1
    return ddc_batch_pool()


def plan_sequences(n_ctx: int, workers: int) -> tuple[int, int, int | None]:
    """Decide the sequence budget, and with it the KV reservation.

    Returns ``(nseq, n_ctx, ddc_seq)``: the sequence count, the cells to reserve,
    and the first index DDC's verification scratch lives on (None when none is
    reserved).

    One layout rule, in index order::

        0 .. workers-1                 work sequences
        workers .. workers+scratch-1   DDC's verification scratch (or pool)
        the last CC_ARCHIVE_SLOTS      archived whole states

    The scratch is reserved only when DDC can actually run: on the serial path,
    or under CC_BATCH when CC_DDC_BATCH turns the speculative scheduler on.

    srv.py calls this before the context exists and Eng calls it again while
    constructing; its outputs are pure functions of (workers, environment) -- the
    n_ctx argument only scales the returned reservation -- so the two calls
    cannot drift.  They used to be sized separately -- srv.py owned the archive,
    Eng owned DDC's doubling -- which is exactly how DDC's scratch ended up on
    the sequence the archive had already claimed.

    n_ctx is multiplied so that every sequence keeps the caller's full window:
    llama.cpp divides its cells across n_seq_max, so this is the whole reason the
    combination costs memory rather than nothing.
    """
    nseq = max(1, workers + ddc_scratch_count(workers) + arc_slot_count())
    ddc_seq = workers if ddc_scratch_count(workers) else None
    return nseq, (n_ctx * nseq if nseq > 1 else n_ctx), ddc_seq


def probe(e, pids: list[int], max_new: int, stop: list[str], stopk: tuple,
          turn=None, on_token=None, on_serve=None):
    """Try every layer that needs no KV work.  Returns gen()'s triple, or None.

    These are the exact-prompt logits cache, trajectory recall, and the
    turn-keyed reuse layer.  None of them forwards anything or touches the KV,
    so they are cheap enough to run in front of a batched scheduler -- and doing
    so is the point: a request served here never occupies a sequence, which
    leaves the slots free for the requests that really do need a forward pass.

    A module-level function rather than a method because the engine is duck-typed
    in places (test_ddc drives Eng.gen with a stand-in object), and a stand-in
    cannot be expected to grow a method for every refactor.

    The order is deliberate: the two exact layers are sound, and the turn-keyed
    layer is a semantic choice, so it goes last.

    ``on_serve(reused, hit)`` is reported BEFORE ``on_token`` on every hit.  A
    hit here reuses the whole prompt and then hands over the text in one call,
    so reporting afterwards would have the text -- and with it message_start --
    reach the client before the reuse was known, and the start frame is exactly
    where the client reads the cache fields from.
    """
    st = e.st
    # 1) Whole prompt identical -> logits cache.  Entries store the TEXT and not
    #    a re-decode of the tokens: stop-string trimming happens at the text
    #    level, so rebuilding from tokens would emit the stop token again.  The
    #    stop set has to be part of the key.
    entry = e.lgc.get(e.key(pids))
    if entry is not None and entry[3] == stopk:
        toks_, text_, trunc = entry[0], entry[1], entry[2]
        if (not trunc) or len(toks_) >= max_new:
            st["lgc"] += 1
            if trunc and len(toks_) > max_new:
                toks = toks_[:max_new]
                text = "".join(e.piece(t) for t in toks)
            else:
                toks, text = toks_, text_
            if on_serve:
                on_serve(len(pids), "lgc")
            if on_token:
                on_token(text)
            return toks, text, {"hit": "lgc", "pre": 0, "reuse": len(pids)}

    # 2) Trajectory recall (a weaker requirement than 1: only that the prompt is
    #    a prefix of some recorded trajectory).
    r = e.recall(pids, max_new, stop)
    if r is not None:
        if on_serve:
            on_serve(r[2].get("reuse") or len(pids), r[2].get("hit") or "rcl")
        if on_token:
            on_token(r[1])
        return r

    # 3) Turn-keyed reuse.
    qc = getattr(e, "qc", None)
    if qc is not None:
        # A qreuse hit also answers with the whole prompt reused, but the layer
        # decides that itself and calls on_token from inside -- so the report is
        # wrapped onto that call rather than made after it returns.
        served = []

        def _tok(chunk, _t=on_token):
            if on_serve and not served:
                served.append(1)
                on_serve(len(pids), "qreuse")
            if _t:
                _t(chunk)

        r = e.qreuse(pids, max_new, stop, stopk, turn,
                     _tok if (on_token or on_serve) else None)
        if r is not None:
            if on_serve and not served:
                served.append(1)
                on_serve(r[2].get("reuse") or len(pids), "qreuse")
            return r
    return None


class Eng:
    def __init__(self, n_ctx: int = 131072, bs: int = 512, n_rs: int = 0,
                 nseq: int = 1, workers: int = 1, log=print):
        self.log = log
        L = self.L = bind()

        # The sequence layout is settled once, here, from the caller's numbers
        # and the environment; every index below is read off it instead of being
        # re-derived.  srv.py has already applied the same plan, so nseq normally
        # arrives at the planned value and nothing below moves.
        plan_nseq, _, self.ddc_seq = plan_sequences(n_ctx, workers)
        if nseq < plan_nseq:
            # A direct caller (this module's own smoke test, a benchmark harness)
            # asked for fewer sequences than the configuration needs.  The window
            # it asked for is n_ctx // nseq, and every sequence keeps it: that
            # reservation is the price of the layout, and paying it beats the old
            # behaviour of quietly halving every sequence's window instead.
            n_ctx = max(1, n_ctx // max(1, nseq)) * plan_nseq
            nseq = plan_nseq
        # The sequences DDC may verify on.  On the serial path this is the single
        # scratch ddc_decode copies into (and ddc_seq is its index); under
        # CC_DDC_BATCH the scheduler claims one per speculative round from the
        # same list, so the two paths can never disagree about which sequences
        # are DDC's.
        self.ddc_pool: list = list(range(workers,
                                          workers + ddc_scratch_count(workers)))

        # DDC draft cache (experimental, off by default; data in DDC_EXPERIMENT.md).
        # Key = top-k token ranking of the current logits -- what is read out is
        # decided by the direction alone (oscillation_subspace.tex: cell
        # membership decides the readout, a cosine test does not work), so equal
        # top-1 means the same readout cell; that is a discrete predicate, not an
        # approximate cosine.  Value = the continuation tokens of a past
        # trajectory plus a per-position margin (the margin only decides where a
        # draft is cut, never whether it is right).  Every draft still has to be
        # verified by the target model in a batch: the approximate key is a
        # throughput knob, correctness does not rest on it.
        # Verification needs a second sequence, so KV reservation doubles with
        # nseq.
        self.ddc = None
        self.ddc_max_ctx = 0          # set below, once the window is known
        # Off by default.  Turned on it is the draft cache that makes a long
        # agentic turn cheap, and what keeps it honest is the per-round gate in
        # spec.py: it drops any draft length that stops paying, and stands the
        # whole thing down on content that is not repeating.
        #   unset       off
        #   CC_DDC=0    off
        #   CC_DDC=2|4  on, with that n-gram key width
        #
        # k=4 is the recommended width because that is the configuration the
        # draft lengths were measured in: k4 + M=32 + T=0.5 gave 15-22 tokens per
        # round against 6-13 for the older k2/M=16/T=1.0.  k=2 was what the live
        # CC session had been run with, and the two were never separated in a
        # clean sweep.
        #
        # The prefix archive used to keep its states in the same spare
        # sequences, and the pair was refused rather than letting the two
        # silently overwrite each other.  They no longer overlap: DDC's
        # verification scratch gets a sequence of its own and the archive sits
        # above it (see plan_sequences), so naming both serves both.
        _k = ddc_key()
        if _k is not None:
            from ddc import DraftCache
            _k = int(_k)
            if _k not in (2, 4):
                raise RuntimeError(f"CC_DDC supports only 0, 2 or 4, got {_k!r}")
            # Aggressive draft settings on purpose.  Measured on real CC
            # traffic the round's rate rises monotonically with draft length
            # (27 tok/s at 2-3 tokens, 84 at 4-7, 96 at 8-15, 160 at 16+), so
            # long drafts are what pays -- and the per-round gate in spec.py
            # drops any bucket that turns out not to, which is what makes it
            # safe to aim long by default instead of guessing a threshold.
            self.ddc = DraftCache(k=_k,
                                  max_draft=int(os.environ.get("CC_DDC_M", "32")),
                                  threshold=float(os.environ.get("CC_DDC_T", "0.5")),
                                  gate=float(os.environ.get("CC_DDC_GATE", "0.")),
                                  reset_per_request=os.environ.get("CC_DDC_RESET") == "1")
            if self.ddc_seq is None:
                # DDC is on but no scratch sequence was reserved, which under
                # this layout means a batcher is serving every request.  Saying
                # so once is worth more than the silence: the DDC counters and
                # the spec.py governor would otherwise just never move.
                log("CC_DDC is on but the batched scheduler serves every request "
                    "and DDC runs on the serial path only -- no scratch sequence "
                    "reserved, so no speculation will happen (unset CC_BATCH to "
                    "use DDC)")
            # Default the cutoff to this sequence's own window, so it never
            # stands DDC down early; it remains only as the clamp that stops a
            # draft running past the end of the context.
            #
            # What decides whether speculation pays is the accept rate, not the
            # length.  A round that lands its whole draft is ~3x faster than
            # decoding one token at a time; a round that lands a fraction of it
            # pays two forward passes for about one token.  The gates in spec.py
            # measure that directly, so a static length threshold would be both a
            # poor proxy and redundant.  The whole-state copy does scale with
            # length, but it is second-order: ~129 MB at 14k tokens, so ~1.3 ms a
            # copy at 64k against ~15 ms for a decoded token.
            _mctx = os.environ.get("CC_DDC_MAX_CTX")
            self.ddc_max_ctx = (n_ctx // nseq) if _mctx is None else int(_mctx)
            if self.ddc_max_ctx < 0:
                raise ValueError("CC_DDC_MAX_CTX must be >= 0 (0 means unlimited)")
            log(f"DDC on k={_k} M={self.ddc.max_draft} "
                f"T={os.environ.get('CC_DDC_T', '0.5')} gate={self.ddc.gate} "
                f"reset_per_request={self.ddc.reset_per_request} "
                f"max_context={self.ddc_max_ctx} "
                f"(scratch={self.ddc_seq if self.ddc_seq is not None else 'none'}, "
                f"nseq={nseq}, n_ctx={n_ctx})")

        # Turn-keyed reuse (experimental, off by default; see qcache.py and
        # USAGE.md).  Unlike the exact-prompt cache this one fires when the same
        # user turn comes back inside a running dialog, which is the only shape
        # in which the exact cache can never fire: the prompt grows every turn,
        # so it is never byte-identical to an earlier one.
        self.qc = None
        _q = int(os.environ.get("CC_QREUSE", "0") or 0)
        if _q:
            from qcache import QCache
            self.qc = QCache(level=_q,
                             edit=os.environ.get("CC_QEDIT") == "1",
                             topk=int(os.environ.get("CC_QEDIT_TOPK", "16")),
                             sim=float(os.environ.get("CC_QEDIT_SIM", "0.9")),
                             capacity=int(os.environ.get("CC_QREUSE_CAP", "512")),
                             skip_tool=os.environ.get("CC_QREUSE_TOOL") != "1")
            log(f"turn-keyed reuse on level={_q} edit={self.qc.edit} "
                f"topk={self.qc.topk} sim={self.qc.sim} skip_tool={self.qc.skip_tool}")

        # Prefix archive (on by default; CC_ARCHIVE=0 turns it off.  See USAGE.md
        # 6.4).
        # The engine keeps one KV, and a request that is shorter than it -- or
        # that diverges before its end -- needs a truncation, which this hybrid
        # memory cannot do.  That is what drops prefix capture from ~0.85 in a
        # linear dialog to ~0.06 under agent fan-out: siblings of the same parent
        # share a long prefix but are not extensions of each other.
        #
        # The archive stores whole states at the points where requests have
        # actually diverged, in spare sequences, and restores one wholesale with
        # seq_cp when a later request asks for it.  Whole-sequence copies are the
        # operation that does work here (measured; see snap/fork below).
        self.arc = None
        self.arc_slots: list = []
        self.arc_min = int(os.environ.get("CC_ARCHIVE_MIN", "512"))
        # Which sequences a batched scheduler may use for work: the first
        # ``workers`` of them and no others.  Derived here rather than left as
        # range(nseq), because under the current layout nseq counts the archive
        # slots and DDC's scratch as well, and handing a scheduler one of those
        # would overwrite an archived state or DDC's verification copy mid-round.
        self.work_seqs: list = list(range(workers))
        want = arc_slot_count()
        if want:
            self.arc = {}
            self.arc_clock = 1
            # The top of the range, above the workers and above DDC's scratch.
            # That is what makes the arithmetic safe: the layout reserves the
            # scratch BELOW the archive, so "nseq - want" can only ever land on
            # an archive slot.  Reading it off the layout instead of recomputing
            # workers+scratch here is deliberate -- recomputing is how the index
            # that used to be DDC's scratch came to be handed out as an archive
            # slot, which is the overlap this pairing was refused for.
            self.arc_slots = list(range(nseq - want, nseq))
            log(f"prefix archive on, slots {self.arc_slots}, "
                f"worker sequences {self.work_seqs}, "
                f"min prefix {self.arc_min} tokens (nseq={nseq}, n_ctx={n_ctx})")
        if self.ddc is not None and self.ddc_seq is not None:
            log(f"DDC verification scratch {self.ddc_pool}, "
                f"archive slots {self.arc_slots or 'none'} "
                f"(disjoint; nseq={nseq})")

        # The KV cells are n_ctx in total and llama.cpp divides them across
        # n_seq_max sequences, so this is how many tokens ONE sequence can hold.
        # Past it llama_decode fails, and a failed decode leaves the sequences in
        # that batch holding tokens the ledger does not know about, so the next
        # admission computes its reusable prefix from stale data and decodes from
        # the wrong position.  Everything that has to stay inside the window
        # reads it from here rather than recomputing the division.
        self.window = n_ctx // nseq

        t0 = time.time()
        mp = L.llama_model_default_params(); mp.ngl = 999
        self.mdl = L.llama_model_load_from_file(GGUF.encode(), mp)
        cp = L.llama_context_default_params()
        cp.n_ctx = n_ctx; cp.n_batch = bs; cp.n_ubatch = 128
        cp.n_seq_max = nseq; cp.n_rs_seq = n_rs
        cp.tk = Q4_0; cp.tv = Q4_0          # quantized KV; 9216 B/position x n_ctx, paid once at context creation
        self.ctx = L.llama_init_from_model(self.mdl, cp)
        if not self.ctx:
            raise RuntimeError(f"could not create context n_ctx={n_ctx}")
        self.vcb = L.llama_model_get_vocab(self.mdl)
        self.nv = L.llama_vocab_n_tokens(self.vcb)
        self.kmem = L.llama_get_memory(self.ctx)   # not named mem: mem() below is the memory report and would collide
        self.bs = bs

        self.cur: list[int] = []                   # the token sequence the KV currently holds
        self.lgc: dict = {}                        # logits cache: key -> (toks, text, trunc, stopk)
        self.lgc_max = 512
        # Trajectory recall index.  The test is a hard condition given by causal
        # attention, exact and needing no verification:
        #   the running sequence R is a prefix of a recorded trajectory T
        #   <=>  T[len(R)] is the token the model would produce next
        # (the token at position i is decided by 0..i-1 alone, so equal prefixes
        # imply equal outputs).  This covers rewind, fork, cutting in at any
        # position, and converging back onto a recorded trajectory mid-generation.
        self.ridx: dict = {}                       # (prefix length, digest) -> (trajectory id, position)
        self.rgen: dict = {}                       # trajectory id -> (gen_ids, text, stopk)
        self.ridx_max = 200_000                    # FIFO eviction; a broken chain only loses a recall, it never recalls wrongly
        # Kept for reference only.  reg() used to index just the first rcap
        # positions of a trajectory while rgen held the whole reply, so a recall
        # starting early in a long reply ran out of index and handed back a
        # fragment as if it were the finished answer -- a cache hit that ended
        # the dialog mid-sentence.  Everything generated is indexed now; the
        # bound that matters is ridx_max.
        self.rcap = 256
        self.tid = 0
        self.ng = {}; self.ng_done = 0; self.ng_keep = None
        # Fork archives.  Nothing calls snap()/fork() today, and they are NOT
        # layout-aware: they cycle 1..nseq-1, which now includes DDC's scratch
        # and the archive slots.  Whoever wires them up has to start the cycle
        # above plan_sequences' reservations first.
        self.nseq = nseq; self.snaps = {}; self.slot = 1
        self.seq = []; self.ng_k = 6                        # used by prompt lookup
        # What each sequence's KV holds, as far as the scheduler's ledger knows.
        # Only the batched path uses this; the single-sequence path keeps the
        # one sequence it owns in self.cur.
        self.seq_tokens: dict = {}

        self.st = {"req": 0, "apc": 0, "lgc": 0, "rcl": 0, "pre": 0, "reuse": 0,
                   "gen": 0, "t_pre": 0.0, "t_gen": 0.0,
                   # rcl_short counts recalls that were DECLINED because the chain
                   # broke before the end of the trajectory.  Declining is correct
                   # (the alternative is a fragment served as a finished reply),
                   # but it means the layer stopped paying, so it is counted
                   # rather than silent.
                   "rcl_short": 0,
                   "ddc_req": 0, "ddc_spec": 0, "ddc_full": 0, "ddc_acc1": 0,
                   "ddc_tok": 0, "ddc_bypass": 0, "ddc_cutoff": 0,
                   "ddc_gated": 0, "ddc_stood_down": 0, "ddc_feature_s": 0.0,
                   "qhit": 0, "qsaved": 0, "qedit": 0, "qtopk_s": 0.0, "qstore": 0,
                   "arc_put": 0, "arc_hit": 0, "arc_restore": 0,
                   "arc_s": 0.0, "arc_tok": 0}

        self._turn_mark = self.tok("<|im_start|>user")
        tb = ct.create_string_buffer(1 << 17)
        n = L.llama_model_meta_val_str(self.mdl, b"tokenizer.chat_template", tb, len(tb))
        self.tpl = tb.raw[:n].decode("utf-8") if n > 0 else None
        log(f"n_ctx={n_ctx} nv={self.nv} {time.time()-t0:.1f}s "
            f"tpl={'yes, ' + str(len(self.tpl)) + ' chars' if self.tpl else 'none'}")

    # ---------------- basics ----------------
    def tmpl(self, msgs: list, add_ass: bool = True) -> str:
        """[(role, content)] -> one string, flattened by the model's own template."""
        if not self.tpl:
            return "\n".join(f"<|{r}|>\n{c}" for r, c in msgs) + "\n<|assistant|>\n"
        arr = (Msg * len(msgs))()
        for i, (r, c) in enumerate(msgs):
            arr[i].role = r.encode(); arr[i].content = c.encode()
        buf = ct.create_string_buffer(1 << 22)
        n = self.L.llama_chat_apply_template(self.tpl.encode(), arr, len(msgs),
                                             add_ass, buf, len(buf))
        if n < 0:
            raise RuntimeError(f"template render failed {n}")
        return buf.raw[:n].decode()

    def tok(self, s: str) -> list[int]:
        L, b = self.L, s.encode()
        n = L.llama_tokenize(self.vcb, b, len(b), None, 0, True, True)
        n = -n if n < 0 else n
        if n == 0:
            return []
        buf = (ct.c_int32 * n)()
        got = L.llama_tokenize(self.vcb, b, len(b), buf, n, True, True)
        return list(buf[:got])

    def piece(self, t: int) -> str:
        b = ct.create_string_buffer(64)
        n = self.L.llama_token_to_piece(self.vcb, t, b, 64, 0, True)
        return b.raw[:n].decode("utf-8", "replace")

    def feed(self, ids: list[int], chunk: int | None = None):
        """Push ids into the KV.  Chunked because of the n_batch limit; the batch
        is built from a memory view rather than copying element by element."""
        L = self.L
        chunk = chunk or self.bs
        raw = bytes((ct.c_int32 * len(ids))(*ids))
        i = 0
        while i < len(ids):
            m = min(chunk, len(ids) - i)
            sub = (ct.c_int32 * m).from_buffer_copy(raw[i * 4:(i + m) * 4])
            if L.llama_decode(self.ctx, L.llama_batch_get_one(sub, m)) != 0:
                raise RuntimeError(f"decode failed at {i}/{len(ids)}")
            i += m
        self.cur.extend(ids)

    def kvlen(self) -> int:
        """Real KV length on the llama.cpp side (last position + 1).  This is how
        a truncation is checked for actually having taken effect."""
        p = self.L.llama_memory_seq_pos_max(self.kmem, 0)
        return p + 1 if p >= 0 else 0

    def trunc(self, n: int) -> bool:
        """Truncate to the first n tokens; returns whether it REALLY happened.

        Measured: seq_rm(mem,0,n,-1) for 0 < n < current length is a complete
        no-op on this hybrid memory (the recurrent state cannot be rewound), not
        an intermittent failure.  So the truncation is verified afterwards, and
        when it did not take we clear everything -- the full clear (seq_rm to 0)
        is reliable and verified.  Returning False tells the caller to restart
        from position zero.
        """
        if n >= len(self.cur):
            return True
        self.L.llama_memory_seq_rm(self.kmem, 0, n, -1)
        if self.kvlen() <= n:
            del self.cur[n:]
            return True
        self.L.llama_memory_seq_rm(self.kmem, 0, 0, -1)
        self.cur = []
        self.st["tfb"] = self.st.get("tfb", 0) + 1
        return False

    def clear_seq(self, seq: int):
        """Empty a sequence's KV and forget what it held."""
        self.L.llama_memory_seq_rm(self.kmem, seq, 0, -1)
        self.seq_tokens[seq] = []

    def turn_start(self, pids: list[int]):
        """Token index where the trailing user turn begins, or None.

        This is the boundary siblings share: five subagents answering the same
        parent differ only in their own trailing turn, so everything before it
        is common and the state there is what they all want.  Divergence from
        whatever a worker happens to hold is NOT that boundary -- in a fan-out
        the workers hold unrelated branches, so that measure is small and the
        archive never learns where the shared part ends.  Measured: with the
        wrong boundary the archive stayed empty and capture fell below the plain
        single-sequence server.

        Found by searching for the template's own turn marker rather than
        assuming a layout; this file is already specific to one template (the
        stop strings, the assistant header), and a sentinel here would have to
        reproduce the template's tokenization to be worth anything.
        """
        if not pids:
            return None
        mark = self._turn_mark
        if not mark:
            return None
        for i in range(len(pids) - len(mark), -1, -1):
            if list(pids[i:i + len(mark)]) == mark:
                return i
        return None

    def lcp(self, ids: list[int]) -> int:
        c, m, i = self.cur, min(len(self.cur), len(ids)), 0
        while i < m and c[i] == ids[i]:
            i += 1
        return i

    def top(self) -> int:
        # A DDC batch can leave several outputs before a threshold switches the
        # next request to sequential decoding.  Only the LAST row is current.
        return int(np.ctypeslib.as_array(
            self.L.llama_get_logits_ith(self.ctx, -1), shape=(self.nv,)).argmax())

    def topk(self, k: int):
        """Top-k ids and logits at the current position, plus the baseline cost.

        Timing this honestly takes care, and the same subtlety applies to every
        measurement taken around the logits buffer:

        * Reading the logits is what forces the wait for the forward pass that
          produced them, and the wait lands in whichever call first touches the
          buffer.  np.ctypeslib.as_array looks free and is a view, but it
          dereferences the pointer, so it is usually the call that pays.  Measured
          inside the real loop: 13.9 ms in as_array, 0.02 ms in argmax, 1.3 ms in
          argpartition.  A stopwatch around topk alone would therefore report the
          decode as trace overhead.
        * top() already scans the same buffer with argmax.

        So as_array and the argmax are timed together and returned as ``base`` --
        exactly what the trace-free path pays anyway -- and the caller subtracts
        it.  What is left is the real marginal cost of keeping a trace: measured
        on this box at 1.37 ms per token, against 20 ms per token of decode, so
        about +8% and not +60%.  An A/B on one engine, alternating the two paths,
        agrees: 20.2 ms/token without the trace, 21.9 with it.
        """
        t = time.perf_counter()
        lg = np.ctypeslib.as_array(self.L.llama_get_logits_ith(self.ctx, -1),
                                   shape=(self.nv,))
        lg.argmax()
        base = time.perf_counter() - t
        idx = np.argpartition(lg, -k)[-k:]
        idx = idx[np.argsort(-lg[idx], kind="stable")]
        return idx.astype(np.int32), lg[idx].astype(np.float32), base

    # ---------------- inference-time speculation over the context ----------------
    # Mechanism: at some generation step, take the last k tokens of the current
    # sequence, look up their previous occurrence in the context already seen,
    # and use the m tokens that followed it as a draft.  The context itself is
    # the cache -- no extra model, no training.  The longer the context and the
    # more self-similar it is (an agent repeating the same tool calls, quoting
    # the same code), the higher the hit rate.
    #
    # The hard part is verification: every draft has to be checked position by
    # position, and this hybrid model cannot rewind (the recurrent state is a
    # summary over 0..p).  So the draft never enters the real sequence; it is
    # copied into seq1 and verified there:
    #   seq_cp(0->1) copies the prefix  ->  feed the m draft tokens into seq1 in
    #   one batch, read per-position logits  ->  compare position by position to
    #   get the accepted count p  ->  seq_cp(1->0, [L, L+p)) moves the accepted
    #   ones back into the real sequence (a memcpy, not a forward pass)
    #   ->  seq_rm(seq1, 0, -1) clears the scratch sequence outright (this does
    #   not touch seq0's last position, which is allowed)
    # The real sequence only ever moves forward and never needs rewinding.
    # Measured: all three primitives work, and batch per-position logits agree
    # with step-by-step decode position for position (12/12).
    def batch_parts(self, parts):
        """One batch built from tokens belonging to several sequences.

        ``parts`` is ``[(seq_id, pos0, tokens[, all_rows]), ...]``.  Every
        sequence keeps its own position, so the batch can carry a prefill chunk
        of one request next to a single decode step of another.

        Logits are requested only for the LAST token of each part, unless the
        part sets ``all_rows``.  A caller needs one row per sequence -- the
        next-token distribution -- and asking for every position would cost a
        vocabulary-sized row per token for nothing.  Speculative verification is
        the exception and the reason the flag exists.

        Returns ``(batch, keep, idx, first)`` where ``idx[k]`` and ``first[k]``
        are the BATCH POSITIONS of part k's last and first tokens.
        llama_get_logits_ith is indexed by batch position and answers only for a
        position whose batch.logits flag is set -- passing the part number
        instead is wrong the moment the parts have different lengths, and it
        fails loudly ("invalid logits id") rather than subtly.
        """
        total = sum(len(p[2]) for p in parts)
        if total == 0:
            raise ValueError("empty batch")
        tk = (ct.c_int32 * total)()
        ps = (ct.c_int32 * total)()
        ns = (ct.c_int32 * total)()
        lg = (ct.c_int8 * total)()
        sid = (ct.POINTER(ct.c_int32) * total)()
        holds = []
        idx = []
        first = []
        i = 0
        for part in parts:
            seq, pos0, toks = part[0], part[1], part[2]
            all_rows = len(part) > 3 and part[3]
            if not toks:
                continue
            hold = (ct.c_int32 * 1)(seq)
            holds.append(hold)
            ptr = ct.cast(hold, ct.POINTER(ct.c_int32))
            last = len(toks) - 1
            first.append(i)
            for j, t in enumerate(toks):
                tk[i] = t
                ps[i] = pos0 + j
                ns[i] = 1
                sid[i] = ptr
                # Speculative verification is the one caller that wants every
                # position: it compares the model's own prediction at position j
                # against the draft's token j+1, so a row per draft token is the
                # whole point.  A plain part wants only the next-token
                # distribution, and asking for the rest would be a
                # vocabulary-sized row per token for nothing.
                lg[i] = 1 if (all_rows or j == last) else 0
                i += 1
            idx.append(i - 1)
        b = Batch()
        b.n_tokens = total; b.token = tk; b.embd = None; b.pos = ps
        b.n_seq_id = ns; b.seq_id = sid; b.logits = lg
        # ``first`` is the batch position of part k's FIRST token, which is what
        # a verification scan walks forward from; ``idx`` stays the last one.
        return b, (tk, ps, ns, sid, lg, holds), idx, first

    def logits_row(self, i: int):
        """Logits at BATCH POSITION ``i`` (see batch_parts), as a numpy view."""
        return np.ctypeslib.as_array(
            self.L.llama_get_logits_ith(self.ctx, i), shape=(self.nv,))

    def batch(self, toks: list[int], seq: int, pos0: int):
        """Build a batch by hand: explicit seq_id, explicit start position, and
        logits requested for EVERY position.

        llama_batch_get_one can only feed seq0 and only returns logits for the
        last position; verifying a draft requires building the batch yourself.
        """
        m = len(toks)
        tk = (ct.c_int32 * m)(*toks)
        ps = (ct.c_int32 * m)(*[pos0 + i for i in range(m)])
        ns = (ct.c_int32 * m)(*[1] * m)
        hold = (ct.c_int32 * 1)(seq)
        sid = (ct.POINTER(ct.c_int32) * m)()
        for i in range(m):
            sid[i] = ct.cast(hold, ct.POINTER(ct.c_int32))
        lg = (ct.c_int8 * m)(*([1] * m))
        b = Batch()
        b.n_tokens = m; b.token = tk; b.embd = None; b.pos = ps
        b.n_seq_id = ns; b.seq_id = sid; b.logits = lg
        return b, (tk, ps, ns, hold, sid, lg)      # keep the references alive; do not let the GC free them

    def argmax_at(self, i: int) -> int:
        t = self.L.llama_get_logits_ith(self.ctx, i)
        return int(np.ctypeslib.as_array(t, shape=(self.nv,)).argmax())

    def ngram_feed(self, seq: list[int], k: int, upto: int):
        """Register the k-grams of seq[upto-k .. upto) into the index (new ones only)."""
        i = self.ng_done
        while i < upto:
            if i >= k:
                key = tuple(seq[i - k:i])
                lst = self.ng.get(key)
                if lst is None:
                    self.ng[key] = [i]
                else:
                    lst.append(i)
                    if len(lst) > 8:            # keep only the most recent occurrences of a common n-gram
                        del lst[0]
            i += 1
        self.ng_done = max(self.ng_done, upto)

    def draft(self, seq: list[int], k: int, m: int):
        """Look at the preceding context: where did the last k tokens occur
        before, and what were the m tokens after that occurrence?"""
        if len(seq) < k:
            return None
        lst = self.ng.get(tuple(seq[-k:]))
        if not lst:
            return None
        j = lst[-1]                              # most recent occurrence
        if j + 1 > len(seq):                     # must not see itself
            return None
        d = seq[j:j + m]
        return d if len(d) == m else None

    def gen_spec(self, pids: list[int], max_new: int = 512, m: int = 6, k: int = 3,
                 stop: list[str] | None = None):
        """Generation with inference-time context speculation.  Output must be
        token-for-token identical to plain gen().

        Each round:
          batch = [nxt] + draft[:m-1]     (nxt is the free prediction the model
                                           itself produced last round)
          feed it into seq1 (a copy of the real sequence made by seq_cp), take
          per-position logits; the logits at position i predict token i+1, so
          compare them against batch[i+1]; the accepted length p is at least 1,
          because nxt is the model's own output
          seq_cp(1->0, [pos, pos+p)) moves the accepted tokens into the real
          sequence and clears seq1
          the next nxt comes for free as argmax(logits at p-1)
        One batch forward pass buys p tokens (normally p passes).
        """
        stop = stop or []
        st = self.st
        st["req"] += 1
        stopk = tuple(stop)
        key = self.key(pids)
        e = self.lgc.get(key)
        if e is not None and e[3] == stopk and not e[2]:
            st["lgc"] += 1
            return e[0], e[1], {"hit": "lgc", "pre": 0, "reuse": len(pids)}

        # The prefix goes through the normal APC path -- speculation only takes
        # over the generation part.
        p_raw = self.lcp(pids)
        p = p_raw
        if p < len(self.cur) and not self.trunc(p):
            p = 0
        new = pids[p:]
        st["reuse"] += p; st["pre"] += len(new)
        if p > 0:
            st["apc"] += 1
        if new:
            self.feed(new)

        seq = list(pids)                     # the full sequence = prompt + generated
        self.ng = {}; self.ng_done = 0
        self.ngram_feed(seq, k, len(seq))    # load the preceding context into the n-gram index

        toks: list[int] = []
        text = ""
        ns = max((len(x) for x in stop), default=0)
        hit = None
        nxt = self.top()                     # prediction at the current position
        rounds = 0
        while len(toks) < max_new:
            rounds += 1
            pos = len(seq)
            d = self.draft(seq, k, m)        # look at the context -> draft
            if d is None:
                toks.append(nxt); self.feed([nxt]); seq.append(nxt)
                self.ngram_feed(seq, k, len(seq))
                text += self.piece(nxt); st["gen"] += 1
                nxt = self.top()
            else:
                bat = [nxt] + d[:m - 1]      # the free one goes first
                acc, nxt = self.verify(bat, pos)
                if not acc:
                    toks.append(nxt); self.feed([nxt]); seq.append(nxt)
                    self.ngram_feed(seq, k, len(seq))
                    text += self.piece(nxt); st["gen"] += 1
                    nxt = self.top()
                else:
                    self.L.llama_memory_seq_cp(self.kmem, 1, 0, pos, pos + len(acc))
                    self.L.llama_memory_seq_rm(self.kmem, 1, 0, -1)
                    self.cur.extend(acc)
                    toks.extend(acc); seq.extend(acc)
                    self.ngram_feed(seq, k, len(seq))
                    for t in acc:
                        text += self.piece(t)
                    st["gen"] += len(acc)
                    st["spec"] = st.get("spec", 0) + 1
                    st["spec_tok"] = st.get("spec_tok", 0) + len(acc)
            # stop-string trimming
            cut = len(text)
            if ns:
                for sp in stop:
                    j = text.find(sp)
                    if 0 <= j < cut:
                        cut, hit = j, sp
            if hit:
                text = text[:cut]
                break
        st["spec_r"] = st.get("spec_r", 0) + rounds
        if len(self.lgc) >= self.lgc_max:
            self.lgc.pop(next(iter(self.lgc)))
        tr = (hit is None) and (len(toks) >= max_new)
        self.lgc[key] = (list(toks), text, tr, stopk)
        if toks:
            self.reg(pids, toks, text, stopk)
        return toks, text, {"hit": "apc", "stop": hit, "trunc": tr,
                            "pre": len(new), "reuse": p, "spec": st.get("spec", 0)}

    def verify(self, bat: list[int], pos: int):
        """Check a batch on seq1.  Returns (accepted tokens, the free next token).

        Batch element 0 is nxt -- the model's own output, always accepted;
        element i+1 is accepted iff the logits at position i rank it first.
        """
        L0 = self.L
        L0.llama_memory_seq_rm(self.kmem, 1, 0, -1)
        L0.llama_memory_seq_cp(self.kmem, 0, 1, 0, -1)
        b, keep = self.batch(bat, 1, pos)
        if L0.llama_decode(self.ctx, b) != 0:
            L0.llama_memory_seq_rm(self.kmem, 1, 0, -1)
            return [], 0
        self.ng_keep = keep
        p = 1
        while p < len(bat) and self.argmax_at(p - 1) == bat[p]:
            p += 1
        return bat[:p], self.argmax_at(p - 1)

    def lookup(self, kk: tuple, ent):
        """Cache hit -> the next token straight from the table.  Returns
        (token, source) or (None, None).

        An lgc entry is (toks, text, trunc, stopk): the running sequence equals
        that prompt, so the next token is toks[0].  A ridx entry is (trajectory
        id, position): the running sequence is a prefix of that trajectory, so
        the next token is gen[k].
        """
        if len(ent) == 4:
            return (ent[0][0], "lgc") if ent[0] else (None, None)
        tid, k = ent
        g = self.rgen.get(tid)
        if g is None or k >= len(g[0]):
            return None, None
        return g[0][k], "ridx"

    # ---------------- forking: archive + restore ----------------
    # A hybrid model cannot rewind (the recurrent state is a running summary over
    # 0..p), so partial truncation via seq_rm is a no-op, and forking back means
    # clearing everything and recomputing (measured: 6178 tokens ~ 2.5 s).
    #
    # The way around it is NOT to rewind but to archive just before each fork
    # point: at the moment a trajectory passes the fork point, the KV is exactly
    # [0, fork point) -- copy the whole sequence aside at that instant.  To fork:
    # clear the working sequence, restore the archive wholesale, and go a
    # different way.
    #
    # All four primitives measured (functional check, not a position ledger):
    #   seq_cp over a partial range -> no, llama.cpp asserts GGML_ASSERT(is_full) and aborts
    #   seq_cp over the whole thing -> faithful (identical to a clean run, position by position)
    #   snapshot after the source sequence advanced -> still valid
    #   archive -> clear -> restore -> equivalent to a clean run
    def snap(self, tag: str) -> int:
        """Store the whole current KV in a slot.  Returns the token length at the archive point."""
        if self.nseq < 2:
            raise RuntimeError("archiving needs nseq >= 2")
        slot = self.slot
        self.slot = 1 + (self.slot % (self.nseq - 1))       # cycle through 1..nseq-1
        for t, (s, _) in list(self.snaps.items()):
            if s == slot:
                del self.snaps[t]                            # the slot is being reused, the old archive is void
        self.L.llama_memory_seq_rm(self.kmem, slot, 0, -1)   # clear it first
        self.L.llama_memory_seq_cp(self.kmem, 0, slot, 0, -1)
        self.snaps[tag] = (slot, len(self.cur))
        self.st["snap"] = self.st.get("snap", 0) + 1
        return len(self.cur)

    def fork(self, tag: str) -> bool:
        """Go back to an archive point, then take a different direction."""
        rec = self.snaps.get(tag)
        if rec is None:
            return False
        slot, n = rec
        self.L.llama_memory_seq_rm(self.kmem, 0, 0, -1)      # clear the working sequence (the reliable path)
        self.L.llama_memory_seq_cp(self.kmem, slot, 0, 0, -1)  # restore it wholesale
        self.cur = self.cur[:n]
        self.st["fork"] = self.st.get("fork", 0) + 1
        return True

    # ---------------- prefix archive ----------------
    # One KV, many branches.  A request that is shorter than the KV, or that
    # diverges before its end, needs a truncation -- and a partial truncation on
    # this hybrid memory is a no-op, so the whole context is rebuilt.  Under
    # agent fan-out that is the normal case rather than the exception: sibling
    # subagents share a long prefix but are not extensions of one another, so
    # lcp(prompt, cur) lands short of the KV's end every time and every sibling
    # re-prefills from zero.
    #
    # The state a sibling needs is the one at the divergence point, and the only
    # moment that state exists is while prefilling through it.  So arc_put is
    # called on the way past that boundary, and a later sibling calls arc_restore
    # to get it back whole.  Both use the whole-sequence copies that were
    # measured to work here; no partial operation is involved.
    def arc_ready(self, upto: int) -> bool:
        """Whether the boundary at ``upto`` is worth archiving."""
        return self.arc is not None and upto >= self.arc_min

    def arc_put(self, tokens, src: int = 0) -> None:
        """Store sequence ``src``'s KV, which holds exactly ``tokens``, in a spare slot.

        ``src`` is a parameter because the batched scheduler does not keep its
        work on sequence 0 -- each request has its own sequence -- so "the
        current state" is whatever sequence the caller is holding, not a fixed
        one.

        A slot is reused by least-recent use when they are all taken.  The host
        keeps the token sequence alongside the state because that sequence is the
        only thing a later request can be matched against.
        """
        if not self.arc_slots:
            return
        free = [s for s in self.arc_slots if s not in self.arc]
        if free:
            slot = free[0]
        else:
            slot = min(self.arc, key=lambda s: self.arc[s][2])
        L = self.L
        L.llama_memory_seq_rm(self.kmem, slot, 0, -1)        # clear it first
        L.llama_memory_seq_cp(self.kmem, src, slot, 0, -1)   # whole state, verified faithful
        self.arc[slot] = (array("i", tokens), len(tokens), self.arc_clock)
        self.arc_clock += 1
        self.st["arc_put"] = self.st.get("arc_put", 0) + 1

    def arc_find(self, pids: list):
        """Longest archived state whose token sequence is a prefix of ``pids``.

        Returns ``(slot, length)`` or ``None``.  The stored sequence must be a
        prefix in full: a state at length n can only be restored and extended, it
        cannot be shortened, because shortening is the truncation that does not
        work.  Matching is an exact token comparison, not a heuristic -- a slot
        whose tokens differ from the prompt at any position is not a candidate at
        all.
        """
        if not self.arc:
            return None
        best = None
        for slot, (toks, n, _) in self.arc.items():
            if n > len(pids) or (best is not None and n <= best[1]):
                continue
            if list(pids[:n]) == toks.tolist():
                best = (slot, n)
        return best

    def arc_restore(self, slot: int, dst: int = 0) -> list:
        """Move an archived state onto sequence ``dst``.  Returns its tokens.

        ``dst`` is a parameter for the same reason ``src`` is on arc_put: the
        batched scheduler restores onto whichever worker sequence it just handed
        the request, and only the single-sequence path uses sequence 0.
        """
        toks, n, clock = self.arc[slot]
        # Timed because a restore is the one cost this layer adds, and the whole
        # question of whether a low-reuse hit is worth taking turns on it.  It is
        # a whole-state seq_cp, so it scales with the state: n x 9216 B.  Measured
        # so the answer comes from the counter rather than from an estimate.
        t0 = time.perf_counter()
        self.L.llama_memory_seq_rm(self.kmem, dst, 0, -1)      # clear the target
        self.L.llama_memory_seq_cp(self.kmem, slot, dst, 0, -1)  # restore wholesale
        self.arc[slot] = (toks, n, self.arc_clock)
        self.arc_clock += 1
        self.seq_tokens[dst] = toks.tolist()
        self.st["arc_s"] = self.st.get("arc_s", 0.0) + time.perf_counter() - t0
        self.st["arc_tok"] = self.st.get("arc_tok", 0) + n
        if dst == 0:
            self.cur = self.seq_tokens[0]
        self.st["arc_restore"] = self.st.get("arc_restore", 0) + 1
        return self.seq_tokens[dst]

    def draft_ng(self, k: int, m: int):
        """Draft from the context itself: find the last k tokens in seq and take
        the m tokens that followed that occurrence.

        Free -- no extra model, no training.  Measured: with k=6 m=16, 13.83
        tokens are right in a row on average.
        """
        if os.environ.get("PL_NODRAFT"):
            return None
        seq = self.seq
        i = len(seq)
        if i < k:
            return None
        key = tuple(seq[i - k:i])
        lst = self.ng.get(key)
        if not lst:
            return None
        # Must try backwards one by one: lst[-1] may be too close (j+m past the
        # current position), in which case fall back to an earlier occurrence.
        # Taking only lst[-1] and giving up throws away most usable drafts
        # (measured: the number of draft rounds halves).
        n = len(seq)
        for j in reversed(lst):
            if j + m <= n:
                return seq[j:j + m]
        return None

    def ng_add(self, upto: int):
        """Register the newly added k-grams of seq[:upto] into the index."""
        k = self.ng_k
        i = self.ng_done
        while i < upto:
            if i >= k:
                key = tuple(self.seq[i - k:i])
                l = self.ng.get(key)
                if l is None:
                    self.ng[key] = [i]
                else:
                    l.append(i)
                    if len(l) > 8:
                        del l[0]
            i += 1
        self.ng_done = max(self.ng_done, upto)

    def gen_pl(self, pids: list[int], max_new: int = 512, m: int = 16, k: int = 6,
               stop: list[str] | None = None, on_token=None):
        """Prompt-lookup speculative generation, verified position by position
        against the target model's batch logits.

        The batch and per-token paths of this quantized hybrid differ
        numerically, so this cannot be guaranteed to match gen() token for token.

        Each round:
          batch = [the free prediction from last round] + [m-1 tokens of the draft]
          seq_cp(0->1) copies the prefix -> feed the batch into the scratch area,
          read per-position logits; the logits at position i predict token i+1,
          so compare them against batch[i+1] to get the accepted length p
          ALL accepted -> clear the real sequence and seq_cp(1->0) the whole
          thing back (a memcpy, zero forward passes)
          PARTIALLY accepted -> clear the scratch area and feed batch[:p] into
          the real sequence (one batch forward pass)
        Each round commits p tokens with 1-2 forward passes, and the logits after
        committing predict the next token.
        """
        stop = stop or []
        st = self.st
        st["req"] += 1
        stopk = tuple(stop)

        key = self.key(pids)
        e = self.lgc.get(key)
        if e is not None and e[3] == stopk and not e[2]:
            st["lgc"] += 1
            if on_token:
                on_token(e[1])
            return e[0], e[1], {"hit": "lgc", "pre": 0, "reuse": len(pids)}
        r = self.recall(pids, max_new, stop)
        if r is not None:
            if on_token:
                on_token(r[1])
            return r

        p_raw = self.lcp(pids); p = p_raw
        if p < len(self.cur) and not self.trunc(p):
            p = 0
        new = pids[p:]
        st["reuse"] += p; st["pre"] += len(new)
        if p > 0:
            st["apc"] += 1
        t0 = time.time()
        if new:
            self.feed(new)
        st["t_pre"] += time.time() - t0

        # Build the n-gram index; seq is the running full token sequence.
        self.seq = list(pids)
        self.dbg = []
        self.ng = {}; self.ng_done = 0; self.ng_k = k
        n0 = len(self.seq)
        self.ng_add(n0)

        toks: list[int] = []
        text = ""
        ns = max((len(x) for x in stop), default=0)
        hit = None
        nxt = self.top()
        rounds = spl = 0
        t0 = time.time()
        while len(toks) < max_new:
            rounds += 1
            L = len(self.seq)
            d = self.draft_ng(k, m)
            if os.environ.get("PL_DBG"):
                self.dbg.append((L, nxt, None if d is None else d[0],
                                 None if d is None else len(d)))
            # The draft's key is "the last k tokens of the sequence", so d[0] is
            # already supposed to equal nxt; if it does not, the draft starts in
            # the wrong place and is treated as absent.  (An earlier version
            # pushed another nxt on top, duplicating it; verification then broke
            # at position 1 and acceptance was always exactly 1.)
            if d is None or d[0] != nxt:
                self.feed([nxt]); self.seq.append(nxt)
                self.ng_add(len(self.seq))
                toks.append(nxt); text += self.piece(nxt); st["gen"] += 1
                nxt = self.top()
            else:
                acc, free = self.check(d, L)
                if len(acc) == len(d):
                    # Fully accepted: the scratch area holds the correct state,
                    # so move it back wholesale (a memcpy, no forward pass).
                    self.L.llama_memory_seq_rm(self.kmem, 0, 0, -1)
                    self.L.llama_memory_seq_cp(self.kmem, 1, 0, 0, -1)
                    self.L.llama_memory_seq_rm(self.kmem, 1, 0, -1)
                    self.cur.extend(acc)        # the Python-side KV ledger must stay in sync
                    spl += 1
                else:
                    self.L.llama_memory_seq_rm(self.kmem, 1, 0, -1)
                    self.feed(list(acc))                 # one batch forward pass
                self.seq.extend(acc)
                self.ng_add(len(self.seq))
                toks.extend(acc)
                for t in acc:
                    text += self.piece(t)
                st["gen"] += len(acc)
                st["spec"] = st.get("spec", 0) + 1
                st["spec_tok"] = st.get("spec_tok", 0) + len(acc)
                # Refeeding changes the live numerical path.  A prediction taken
                # from seq1 is valid only when that whole state was adopted.
                nxt = free if len(acc) == len(d) else self.top()
            cut = len(text)
            if ns:
                for sp in stop:
                    j = text.find(sp)
                    if 0 <= j < cut:
                        cut, hit = j, sp
            if hit:
                text = text[:cut]
                break
        st["t_gen"] += time.time() - t0
        st["pl_round"] = st.get("pl_round", 0) + rounds
        st["pl_swap"] = st.get("pl_swap", 0) + spl
        if len(self.lgc) >= self.lgc_max:
            self.lgc.pop(next(iter(self.lgc)))
        tr = (hit is None) and (len(toks) >= max_new)
        self.lgc[key] = (list(toks), text, tr, stopk)
        if toks:
            self.reg(pids, toks, text, stopk)
        if os.environ.get("PL_DBG"):
            import json as _j
            with open(os.environ["PL_DBG"], "w", encoding="utf-8") as f:
                _j.dump(self.dbg, f)
        return toks, text, {"hit": "pl", "stop": hit, "trunc": tr,
                            "pre": len(new), "reuse": p, "rounds": rounds}

    def check(self, bat: list[int], pos: int):
        """Feed bat into the scratch area and check it position by position.
        Returns (accepted tokens, the free next token).

        Position 0 is the prediction won for free last round and is always
        accepted; position i+1 is accepted iff the logits at position i rank it
        first.
        """
        L0 = self.L
        L0.llama_memory_seq_rm(self.kmem, 1, 0, -1)
        L0.llama_memory_seq_cp(self.kmem, 0, 1, 0, -1)
        b, keep = self.batch(bat, 1, pos)
        if L0.llama_decode(self.ctx, b) != 0:
            L0.llama_memory_seq_rm(self.kmem, 1, 0, -1)
            return [bat[0]], bat[0]
        self.ng_keep = keep
        p = 1
        while p < len(bat) and self.argmax_at(p - 1) == bat[p]:
            p += 1
        return bat[:p], self.argmax_at(p - 1)

    # ---------------- keys / memory ----------------
    def key(self, ids: list[int]) -> tuple:
        """(length, 128-bit digest).

        The key used to be tuple(ids) -- a tuple of the whole prompt's tokens.
        tracemalloc measured 2.11 MB for a single key at a 61k context (0.47 for
        the tuple plus 1.63 for the int objects), while the generation result it
        guards is only ~0.2 KB -- the key was 99.99% of the entry, and 512
        entries came to 1.06 GB.  That is why keeping logits resident did not
        previously pay (it was not much cheaper than the KV itself).  With a
        digest it is 112 B per entry.  Collisions: at 128 bits, a noticeable
        probability needs 2^64 entries resident at once, and the length is part
        of the key, so different lengths never collide.
        """
        h = hashlib.blake2b(digest_size=16)
        h.update(np.asarray(ids, dtype=np.int32).tobytes())
        return (len(ids), h.digest())

    def hnew(self, ids: list[int]):
        """Incremental hasher on the same footing as key(), so a prefix index can
        keep feeding it token by token."""
        h = hashlib.blake2b(digest_size=16)
        if ids:
            h.update(np.asarray(ids, dtype=np.int32).tobytes())
        return h

    @staticmethod
    def tbytes(t: int) -> bytes:
        """Must match the byte layout of np.int32, or the incremental hash and
        key() will disagree."""
        return int(t).to_bytes(4, "little", signed=True)

    def drop(self) -> int:
        """Unload the KV (keeping the logits cache).  The costs are asymmetric,
        which is why this is an explicit call and not the default: after
        unloading a hit is still free, but a miss costs the full prefill of the
        whole context again (~30 s).  Only worth calling once a trajectory has
        settled and what follows is likely a replay or a rewind.  Returns the
        number of positions freed.
        """
        n = self.kvlen()
        self.L.llama_memory_seq_rm(self.kmem, 0, 0, -1)
        self.cur = []
        self.st["kev"] = self.st.get("kev", 0) + 1
        return n

    def mem(self) -> dict:
        """Resident memory (MB).  KV is converted from the position count; logits
        entries are estimated from their contents."""
        b = 0
        for k, v in self.lgc.items():
            b += 72 + len(v[0]) * 4 + len(v[1].encode("utf-8", "ignore")) \
                 + sum(len(x) for x in v[3])
        q = 0
        if self.qc is not None:
            for entries in self.qc.entries.values():
                for e in entries:
                    q += len(e.toks) * 4 + len(e.text)
                    if e.trace is not None:
                        q += len(e.trace) * self.qc.topk * 8
        return {"kv": len(self.cur) * 9216 / 2**20, "lgc": b / 2**20,
                "qc": q / 2**20, "n": len(self.lgc)}

    # ---------------- trajectory recall ----------------
    def reg(self, pids: list[int], toks: list[int], text: str, stopk: tuple):
        """Register every prefix of (prompt + generation) into the index -- any
        later request starting with one of those prefixes then gets the tokens
        that follow it with no forward pass at all.

        Cover the WHOLE reply, not the first ``rcap`` tokens of it.  Indexing
        only a prefix left the tail of a long reply unreachable, and recall --
        which reads the continuation straight out of ``rgen`` -- then produced a
        fragment and reported it as a complete answer, ending the client's turn
        mid-thought.  ``rgen`` already holds every token, so indexing the whole
        thing costs one dict entry per generated token, and the FIFO bound
        ``ridx_max`` is what actually limits resident memory.  ``rcap`` is
        therefore no longer consulted here.

        Correctness does not depend on how much is indexed: ``walk`` follows the
        chain position by position and stops the moment it points anywhere other
        than the next position of this same trajectory, so a partial index only
        ever loses a recall, never invents one.
        """
        tid = self.tid
        self.tid += 1
        self.rgen[tid] = (list(toks), text, stopk)
        h = self.hnew(pids)
        n = len(pids)
        self.ridx[(n, h.digest())] = (tid, 0)
        for k, t in enumerate(toks):
            h.update(self.tbytes(t))
            self.ridx[(n + k + 1, h.digest())] = (tid, k + 1)
        while len(self.ridx) > self.ridx_max:
            self.ridx.pop(next(iter(self.ridx)))

    def walk(self, tid: int, k: int, h, n: int, room: int, gen: list[int]):
        """Walk down a trajectory, requiring at each step that the index points
        at exactly the next position of the SAME trajectory.

        A broken chain only loses a recall, it never recalls the wrong thing.
        Returns (tokens taken, position reached).
        """
        out: list[int] = []
        while k < len(gen) and len(out) < room:
            t = gen[k]
            out.append(t); k += 1
            h.update(self.tbytes(t))
            nx = self.ridx.get((n + len(out), h.digest()))
            if nx is None or nx[0] != tid or nx[1] != k:
                break
        return out, k

    def recall(self, pids: list[int], max_new: int, stop: list[str]):
        """The prompt lands inside a recorded trajectory -> take the continuation
        back wholesale (no forward pass, no KV work).

        Only a continuation that reaches the END of the recorded trajectory is a
        complete reply, and only a complete reply may be handed back.  The walk
        can stop early for two different reasons, and they are not the same
        thing:

        * the caller's own budget, ``max_new``, ran out.  That is a legitimate
          truncation -- the reply really did run into the token cap, exactly as
          it would have on the generating path -- so it is served and reported
          with ``trunc``.
        * the chain broke before the end of the trajectory.  The index entry the
          walk needed next is gone (FIFO eviction) or now points at a different
          trajectory, so what was recovered is a PREFIX of the answer and not the
          answer.  Serving that is what made a cache hit end a dialog: the client
          got a mid-sentence fragment, the engine reported no truncation, and the
          reply looked like a normal ``end_turn``, so nothing ever produced the
          rest.  This used to be reached every time by a reply longer than
          ``rcap``, because reg() indexed only its first 256 tokens; reg() now
          indexes all of them, so the case is genuinely rare -- but it is still
          the case that must not be answered short.

        Breaking off declines and lets the normal path generate.  That is
        strictly better than answering short: the cost is one prefill, and the
        alternative is a silently wrong reply.
        """
        e = self.ridx.get(self.key(pids))
        if e is None:
            return None
        tid, k = e
        gen, gtxt, gstop = self.rgen.get(tid, (None, None, None))
        if gen is None or k >= len(gen) or tuple(stop) != gstop:
            return None          # past the end of the trajectory / different stop set -> hand back to the normal path
        out, k2 = self.walk(tid, k, self.hnew(pids), len(pids), max_new, gen)
        # Did the walk reach the end of the trajectory, or did the chain break?
        # Only a walk that arrived at the end recovered the whole reply.
        if k2 < len(gen) and len(out) < max_new:
            self.st["rcl_short"] = self.st.get("rcl_short", 0) + 1
            return None
        # A walk stopped by max_new is a genuine budget truncation; one stopped
        # at the end of the trajectory ended on its own.
        trunc = len(out) >= max_new
        self.st["rcl"] += 1
        self.st["gen"] += len(out)
        txt = gtxt if k2 >= len(gen) else "".join(self.piece(t) for t in out)
        return out, txt, {"hit": "rcl", "pre": 0, "reuse": len(pids),
                          "n": len(out), "trunc": trunc}

    # ---------------- turn-keyed reuse ----------------
    def qreuse(self, pids: list[int], max_new: int, stop: list[str], stopk: tuple,
               turn, on_token):
        """Answer from a reply to the same user turn, or return None.

        Only ever called after the exact-prompt cache and trajectory recall have
        both declined, so the sound layers keep priority.  Nothing is forwarded:
        the KV is left exactly as it was, and the next request re-establishes
        its own prefix through the normal path.
        """
        if self.qc is None or turn is None:
            return None
        m = self.qc.match(turn, pids, stopk, max_new)
        if m is None:
            return None
        toks, text, info = self.qc.realize(m, turn[2], max_new, self.tok, self.piece)
        if not toks:
            return None
        self.st["qhit"] += 1
        self.st["qsaved"] += len(toks)
        self.st["qedit"] += int(bool(info.get("edited")))
        # A reused reply is deliberately NOT registered into the recall index.
        #
        # recall's guarantee is that a registered trajectory is one the model
        # actually produced, so that equal prefixes imply equal continuations.  A
        # reused reply was produced under `entry.prompt`, which is a strict prefix
        # of this prompt -- so by construction it is not the continuation of this
        # prompt.  Registering it would put a semantically chosen reply into the
        # layer that is supposed to be exact, and because recall is consulted
        # before this one, every later request served from that entry would bypass
        # all the gates that live here (budget, recency, the tool-result refusal).
        # The cost of not registering is nil in a normal dialog: the next turn's
        # prompt is longer than this trajectory, so recall could not have fired
        # for it anyway.
        if on_token:
            on_token(text)
        out = {"hit": "qreuse", "pre": 0, "reuse": len(pids), "stop": None,
               "trunc": False, "qpreuse": m.level, "qwhy": m.reason}
        out.update({k: info[k] for k in ("replayed", "edited", "edit_applied",
                                         "edit_rejected")})
        return toks, text, out

    # ---------------- prefill, shared by the plain and DDC serving paths ----------------
    # These two are the APC/archive half of gen(), lifted out so that gen_ddc()
    # can run the same prefill.  The pair has to stay in step with itself: which
    # prefix is used (plan_prefix) decides what feed_prefix may skip, and the
    # archive put in feed_prefix depends on the divergence point plan_prefix
    # reported.  It also has to be the SAME code in both serving paths, because
    # the archive is only ever populated on the way past a divergence point --
    # with DDC on, a prefill that skipped the put would leave the archive
    # permanently empty and every restore a miss.
    def plan_prefix(self, pids: list[int], arc) -> tuple[int, int]:
        """Decide how much of ``pids`` the KV can hold.  -> (p, p_raw).

        ``p`` is the position the KV is left at, ``p_raw`` the common prefix
        before any truncation.  No forward pass happens here, which is what lets
        the caller report on_serve before the prefill rather than after it.
        """
        p_raw = self.lcp(pids)
        p = p_raw
        if p < len(self.cur):
            # Anything short of the KV's end needs a truncation, and a partial
            # truncation is a no-op on this hybrid memory -- so on its own this
            # is a full rebuild.  An archived state, if one holds a prefix of
            # this prompt, is restored whole instead, and seq_cp does do that.
            hit = self.arc_find(pids) if arc is not None else None
            if hit is not None and hit[1] < len(pids):
                # arc_restore returns the tokens it restored, because the batched
                # scheduler needs them for its ledger; what this path wants is how
                # many there are, since p is a position.
                #
                # The hit has to leave something to forward.  A state exactly as
                # long as the prompt would mean no forward pass at all, and both a
                # restore and a truncation leave the logits buffer holding
                # whatever the LAST decode produced -- some previous request's
                # token, not this prompt's next one.  Those hits are dropped and
                # the prompt rebuilt, which is the only case that pays full price.
                p = len(self.arc_restore(hit[0]))
                self.st["arc_hit"] = self.st.get("arc_hit", 0) + 1
            elif p == len(pids) or not self.trunc(p):
                p = 0
        return p, p_raw

    def feed_prefix(self, pids: list[int], p: int, p_raw: int, arc) -> None:
        """Forward ``pids[p:]``, archiving the divergence point on the way past.

        The boundary at ``p_raw`` -- the last position this prompt shares with
        whatever was in the KV -- is where a sibling branch diverges too, so it is
        the state a fan-out will ask for next, and this is the only moment it
        exists.
        """
        if p >= len(pids):
            return
        if arc is not None and self.arc_ready(p_raw) and p < p_raw <= len(pids):
            self.feed(pids[p:p_raw])
            self.arc_put(pids[:p_raw])
            self.feed(pids[p_raw:])
        else:
            self.feed(pids[p:])

    # ---------------- main entry point ----------------
    def gen(self, pids: list[int], max_new: int = 512, stop: list[str] | None = None,
            on_token=None, turn=None, on_serve=None):
        """-> (toks, text, info).  on_token(piece) is used for streaming.

        ``turn`` is the qcache.turn_key() of the request this prompt came from.
        The engine only sees tokens, so the caller has to pass it; without it the
        turn-keyed layer is inert.

        ``on_serve(reused, hit)`` fires as soon as the serving layer is decided,
        which is before any token is produced.  The HTTP layer needs it that
        early: the usage it puts in message_start is what Claude Code records,
        and a hit that is only reported at the end would reach the client after
        it had already written down zero.
        """
        stop = stop or []
        st = self.st
        st["req"] += 1
        stopk = tuple(stop)
        # Duck-typed engines reach gen() too (see test_ddc.FakeEngine, which calls
        # Eng.gen unbound on an object that is not an Eng), so the optional layers
        # are looked up rather than assumed.
        qc = getattr(self, "qc", None)
        arc = getattr(self, "arc", None)

        # 1) The layers that need no KV work at all.  A batched server runs this
        #    in front of its scheduler, because a hit here never has to occupy a
        #    sequence.
        key = self.key(pids)          # gen() files its own result under this later
        r = probe(self, pids, max_new, stop, stopk, turn, on_token, on_serve)
        if r is not None:
            return r

        # 1d) DDC (when CC_DDC is on): rank fingerprint looks up past
        #     trajectories -> draft -> target-model batch verification.
        #
        #     Gated on the governor, because speculation is not always a win: on
        #     long-context CC traffic it measured 0.712x, slower than decoding one
        #     token at a time.  A static length threshold cannot see that -- the
        #     acceptance rate is what decides it.  Requests that reach here and
        #     skip DDC are also where the governor measures the plain decoding
        #     rate, so both sides of the comparison are measured, not assumed.
        #     The decision of whether any GIVEN round is worth taking belongs to
        #     the governor inside ddc_decode, per draft length: speculation is
        #     ~3x faster when a draft lands and a net loss when it does not, so
        #     switching it off wholesale would throw away the wins with the
        #     losses.  See spec.py.
        if self.ddc is not None:
            if not self.ddc_max_ctx or len(pids) < self.ddc_max_ctx:
                return self.gen_ddc(pids, max_new, stop, on_token, stopk, on_serve)
            st["ddc_bypass"] += 1

        # 2) APC.  p_raw = the common prefix BEFORE truncation, which is what
        #    separates the two kinds of reuse=0: genuinely no prefix (p_raw=0)
        #    versus a prefix whose truncation failed and cleared everything
        #    (p_raw>0).
        p, p_raw = self.plan_prefix(pids, arc)
        new = pids[p:]
        st["reuse"] += p; st["pre"] += len(new)
        if on_serve is not None:
            # p is final here and nothing has been generated yet, so this is the
            # last moment the HTTP layer can still get it into message_start.
            on_serve(p, "apc" if p else "cold")
        if p > 0:
            st["apc"] += 1
        t0 = time.time()
        self.feed_prefix(pids, p, p_raw, arc)
        st["t_pre"] += time.time() - t0

        # 3) Token-by-token generation.  A stop string has to be judged BEFORE it
        #    is emitted: hold back the last n_stop-1 characters so a stop string
        #    cannot be split across tokens, and flush them at a normal end, or
        #    the tail gets eaten.
        t0 = time.time()
        trace = None
        # The logits trace exists only for the edit path, which only level 2 can
        # reach.  Level 1 replays a verbatim reply and never consults a trace, so
        # recording one there would be pure overhead on the cheap feature.
        if qc is not None and qc.edit and qc.level >= 2:
            from qcache import Trace
            trace = Trace()
        toks: list[int] = []
        text = ""
        emit = 0
        hit = None
        ns = max((len(x) for x in stop), default=0)
        h = self.hnew(pids)                  # incremental digest of the running sequence, one step at a time
        n_pre = len(pids)
        while len(toks) < max_new:
            # Inference-time cache lookup: if the current running sequence
            # (the preceding context) is already cached, the next token comes
            # straight from the table.
            kk = (n_pre + len(toks), h.digest())
            ent = self.lgc.get(kk)
            if ent is None:
                ent = self.ridx.get(kk)
            if ent is not None:
                nxt, how = self.lookup(kk, ent)
                if nxt is not None:
                    toks.append(nxt); self.feed([nxt]); text += self.piece(nxt)
                    h.update(self.tbytes(nxt)); st["gen"] += 1
                    st["hit_tok"] = st.get("hit_tok", 0) + 1
                    st["hit_" + how] = st.get("hit_" + how, 0) + 1
                    if trace is not None:
                        trace.append(None, None)   # served from a table, no logits to record
                    continue
            if trace is not None:
                t0k = time.perf_counter()
                ids_, lg_, base = self.topk(qc.topk)
                # Subtract the argmax we would have paid for anyway, so this
                # counter is the marginal cost of the trace and not the decode.
                st["qtopk_s"] += (time.perf_counter() - t0k) - base
                t = int(ids_[0])
                trace.append(ids_, lg_)
            else:
                t = self.top()
            toks.append(t); self.feed([t]); text += self.piece(t)
            h.update(self.tbytes(t))
            st["gen"] += 1
            cut = len(text)
            if ns:
                for sp in stop:
                    k = text.find(sp)
                    if 0 <= k < cut:
                        cut, hit = k, sp
            safe = cut if hit else max(emit, len(text) - (ns - 1) if ns else len(text))
            if safe > emit:
                if on_token:
                    on_token(text[emit:safe])
                emit = safe
            if hit:
                text = text[:cut]
                break
        dt = time.time() - t0
        st["t_gen"] += dt
        # Plain decoding, one token at a time: the yardstick the governor
        # measures speculation against.
        spec.note_sequential(len(toks), dt)
        why = spec.consumed()
        if why:
            self.log("speculation governor: " + why)
        if on_token and emit < len(text):
            on_token(text[emit:])

        # 4) Enter the caches and register the trajectory (the recall-hit branch
        #    returned above; everything reaching here was really computed).
        tr = (hit is None) and (len(toks) >= max_new)
        if len(self.lgc) >= self.lgc_max:
            self.lgc.pop(next(iter(self.lgc)))
        self.lgc[key] = (list(toks), text, tr, stopk)
        if toks:
            self.reg(pids, toks, text, stopk)
        # The turn-keyed cache wants the same reply under the user turn that
        # produced it, plus the logits trace when the edit path is enabled.
        if qc is not None and turn is not None and toks:
            # has_call is what makes reusing a reply to a TOOL RESULT safe: a
            # reply that itself calls a tool would re-issue that call, the tool
            # returns the same result, the same trailing turn arrives again --
            # and the cache answers again, forever.
            qc.store(turn, pids, toks, text, stopk, tr, trace,
                     has_call="<tool_call>" in text)
            st["qstore"] += 1
        return toks, text, {"hit": "apc", "stop": hit, "trunc": tr,
                            "pre": len(new), "reuse": p,
                            "p_raw": p_raw, "cur": len(self.cur)}

    def gen_ddc(self, pids: list[int], max_new: int, stop: list[str],
                on_token, stopk: tuple, on_serve=None):
        """DDC serving path (the 1d branch of gen()).  The exact layers already
        returned upstream, so this is verified draft decoding.

        This path owns the prefill -- APC and, when it is on, the prefix archive
        (see plan_prefix/feed_prefix) -- and then hands ddc_decode the position
        it left the KV at.  ddc_decode brings the stop handling and the UTF-8
        handling, and guarantees: only committed tokens are registered (a
        rejected tail never enters the index), and after a partial acceptance the
        logits are re-read from the committed sequence (a verification copy's
        prediction goes stale after refeeding -- the same trap gen_pl had fixed).
        Output is verified position by position by the target model, but the
        batch path differs numerically from the token-by-token path, so this is
        not promised to match gen() position by position.  Experimental.
        """
        from ddc_decode import generate as _ddc_gen
        st = self.st
        # The prefill runs HERE rather than inside ddc_decode, so that the archive
        # takes part in it.  A restore is only available at this level -- it is a
        # whole-state copy onto the sequence the decoder is about to use, which is
        # the one thing ddc_decode does not own -- and the put on the way past a
        # divergence point is the only thing that ever fills the archive at all.
        # Since DDC is the default serving path, a prefill that skipped the put
        # would leave the archive permanently empty and make every restore miss.
        # The decoder is told what the KV already holds (``reuse``) instead of
        # working it out for itself from lcp/trunc.
        arc = getattr(self, "arc", None)
        p, p_raw = self.plan_prefix(pids, arc)
        if on_serve is not None:
            # p is final and nothing has been generated yet, so this is where the
            # HTTP layer gets its reuse figure into message_start.  Without it a
            # request that restored an archived state reported a cache read of
            # zero, which is what the hit-rate tooling reads off the transcript.
            on_serve(p, "ddc")
        t0 = time.time()
        self.feed_prefix(pids, p, p_raw, arc)
        pre_s = time.time() - t0
        toks, text, ds = _ddc_gen(self, list(pids), max_new,
                                  cache=self.ddc, stop=tuple(stop),
                                  max_context=self.ddc_max_ctx, reuse=p)
        st["t_pre"] += pre_s + ds["prefill_seconds"]
        st["t_gen"] += ds["generation_seconds"]
        # The governor is fed per ROUND inside ddc_decode, which is finer and
        # sees the state-copy cost of each speculative round.  Reporting the
        # whole request here as well would count the same tokens twice.
        st["reuse"] += ds["reused"]
        st["pre"] += len(pids) - ds["reused"]
        st["gen"] += ds["tokens"]
        st["ddc_req"] += 1
        st["ddc_spec"] += ds["spec"]
        st["ddc_full"] += ds["full"]
        st["ddc_acc1"] += ds["acc1"]
        st["ddc_tok"] += ds["accepted"]
        st["ddc_cutoff"] += ds["ddc_disabled_at"] is not None
        # Both of these were already being computed and then dropped here, which
        # is why /stats could not show either the gate's refusals or the fixed
        # per-token feature-scan cost.
        st["ddc_gated"] += ds.get("gated", 0)
        st["ddc_stood_down"] += ds.get("stood_down", 0)
        st["ddc_feature_s"] += ds["feature_seconds"]
        tr = (ds["stop"] is None) and (len(toks) >= max_new)
        # Enter lgc and register the trajectory on the same footing as gen(): the
        # next request with the same prompt still replays from the logits cache
        # with no forward pass, and prefix requests can still be recalled.  Every
        # token went through model verification, so registering does not break
        # the correctness premise of recall.
        if len(self.lgc) >= self.lgc_max:
            self.lgc.pop(next(iter(self.lgc)))
        self.lgc[self.key(pids)] = (list(toks), text, tr, stopk)
        if toks:
            self.reg(pids, toks, text, stopk)
        if on_token:
            on_token(text)
        return toks, text, {"hit": "ddc", "stop": ds["stop"], "trunc": tr,
                            "pre": len(pids) - ds["reused"], "reuse": ds["reused"],
                            "spec": ds["spec"], "full": ds["full"],
                            "acc1": ds["acc1"], "cur": len(self.cur),
                            "ddc_max_context": self.ddc_max_ctx,
                            "ddc_disabled_at": ds["ddc_disabled_at"]}

    def line(self) -> str:
        s, m = self.st, self.mem()
        out = (f"req={s['req']} apc={s['apc']} lgc={s['lgc']} rcl={s['rcl']} "
               f"rcl_short={s.get('rcl_short', 0)} "
               f"pre={s['pre']} reuse={s['reuse']} gen={s['gen']} "
               f"t_pre={s['t_pre']:.1f}s t_gen={s['t_gen']:.1f}s "
               f"kv={m['kv']:.0f}MB lgt={m['lgc']:.3f}MB/{m['n']}entries "
               f"kev={s.get('kev',0)} tfb={s.get('tfb',0)} "
               f"hit_tok={s.get('hit_tok',0)} "
               f"ddc={s['ddc_req']}/{s['ddc_spec']}spec/{s['ddc_full']}full/"
               f"acc1={s['ddc_acc1']}/tok={s['ddc_tok']} "
               f"bypass={s['ddc_bypass']}/cutoff={s['ddc_cutoff']}/"
               f"gated={s['ddc_gated']}/stood_down={s['ddc_stood_down']} "
               f"dropped={s.get('ddc_dropped', 0)} "
               f"feat={s['ddc_feature_s']:.1f}s")
        if self.ddc is not None:
            out += " | " + spec.line()
        if self.arc is not None:
            out += (f" | arc_put={s['arc_put']} arc_hit={s['arc_hit']} "
                    f"arc_restore={s['arc_restore']} "
                    f"arc_s={s['arc_s']:.2f}s/{s['arc_tok']}tok "
                    f"slots={len(self.arc)}")
        if self.qc is not None:
            out += (f" | qhit={s['qhit']} qsaved={s['qsaved']} qedit={s['qedit']} "
                    f"qstore={s['qstore']} qmem={m['qc']:.3f}MB "
                    f"qtopk={s['qtopk_s']:.1f}s {self.qc.line()}")
        return out


if __name__ == "__main__":
    e = Eng()
    ids = e.tok("The capital of France is")
    print("tok:", ids)
    print(e.gen(ids, max_new=8)[1:])
    print(e.line())
