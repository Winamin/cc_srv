# -*- coding: utf-8 -*-
"""Engine layer: KV, APC, logits cache, trajectory recall, turn-keyed reuse.

Cache layers, cheapest first: logits cache, trajectory recall, turn-keyed reuse
(CC_QREUSE), then APC (longest common prefix, only the new part is forwarded),
then a cold full prefill.

The model is hybrid, so a partial llama_memory_seq_rm truncation is a no-op;
trunc() verifies it took and otherwise clears everything.
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
import utext
from chat_template import ChatTemplate, template_mode
from lib import Mp, Cp, Msg, Batch, GGUF, Q4_0, bind


def ddc_key():
    """The CC_DDC setting in force, or None when the draft cache is off."""
    v = os.environ.get("CC_DDC")
    return v if (v and v != "0") else None


# Tokens one archive slot is charged by default (see arc_cap_for);
# CC_ARCHIVE_CAP=0 charges it a whole window instead.
ARC_CAP_DIV = max(1, int(os.environ.get("CC_ARCHIVE_CAP_DIV", "2") or 2))


def arc_slot_count() -> int:
    """Archive sequences to keep; 0 when off.  CC_ARCHIVE=0 turns it off."""
    v = os.environ.get("CC_ARCHIVE", "1").strip().lower()
    if v in ("0", "false", "off", "no"):
        return 0
    return int(os.environ.get("CC_ARCHIVE_SLOTS", "1"))


def arc_cap_for(window: int) -> int:
    """The most tokens ONE archive slot may hold; CC_ARCHIVE_CAP=0 means no cap."""
    raw = os.environ.get("CC_ARCHIVE_CAP")
    if raw is None or not str(raw).strip():
        return max(1, max(1, window) // ARC_CAP_DIV)
    return _int_env("CC_ARCHIVE_CAP", 0, minimum=0)


def arc_capped(window: int) -> bool:
    """Whether archive slots are charged a capped budget instead of a window."""
    if not kv_unified_default() or not arc_slot_count():
        return False
    cap = arc_cap_for(window)
    return 0 < cap < max(1, window)


def ddc_batch_pool() -> int:
    """Scratch sequences for DDC in the batched scheduler; CC_DDC_BATCH=1 on."""
    if os.environ.get("CC_DDC_BATCH") != "1":
        return 0
    return max(0, int(os.environ.get("CC_DDC_BATCH_SCRATCH", "1")))


def ddc_scratch_count(workers: int) -> int:
    """How many sequences DDC needs of its own."""
    if ddc_key() is None:
        return 0
    if workers == 1:
        return 1
    return ddc_batch_pool()


def pool_for(window: int, workers: int) -> int:
    """The KV cells to reserve so ``workers`` sequences each keep ``window``.

    Capped: ``window x (workers + DDC scratch) + cap x archive slots``.
    """
    work = max(1, workers + ddc_scratch_count(workers))
    arc = arc_slot_count()
    if arc_capped(window):
        return max(1, window) * work + arc_cap_for(window) * arc
    return max(1, window) * max(1, work + arc)


def window_of(pool: int, workers: int) -> int:
    """How many tokens ONE sequence may hold, given a pool from pool_for.

    The exact inverse of pool_for, found by search rather than by rearranging it.
    """
    lo, hi = 1, max(1, pool)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if pool_for(mid, workers) <= pool:
            lo = mid
        else:
            hi = mid - 1
    return lo


def pool_terms(window: int, workers: int) -> str:
    """pool_for in words, for the startup line."""
    work = max(1, workers + ddc_scratch_count(workers))
    arc = arc_slot_count()
    if arc_capped(window):
        return f"window {window} x {work} + cap {arc_cap_for(window)} x {arc}"
    return f"window {window} x {work + arc}"


def plan_sequences(n_ctx: int, workers: int) -> tuple[int, int, int | None]:
    """Decide the sequence budget, and with it the KV reservation.

    -> ``(nseq, n_ctx, ddc_seq)``: sequence count, cells to reserve, and the
    first index DDC's scratch lives on (None when none is reserved).
    """
    scratch = ddc_scratch_count(workers)
    nseq = max(1, workers + scratch + arc_slot_count())
    ddc_seq = workers if scratch else None
    return nseq, pool_for(n_ctx, workers), ddc_seq


# Prompt-processing progress, reported per request rather than per
# llama_decode call: the C-level progress line is emitted once per n_batch pass
# with no notion of the request it belongs to.  `total` is the forwarded part --
# the reused prefix is already in the KV and costs nothing.
PROGRESS_STEP = float(os.environ.get("CC_PROGRESS_STEP", "10") or 10)


def progress_enabled() -> bool:
    """Whether to report prefill progress.  CC_PROGRESS=0 turns it off."""
    v = os.environ.get("CC_PROGRESS", "1").strip().lower()
    if v in ("0", "false", "off", "no"):
        return False
    try:
        return float(os.environ.get("CC_PROGRESS_STEP", "10") or 10) > 0
    except ValueError:
        return True


# Batch sizes, from the environment so the pair can be A/B'd without editing
# code.  They are not interchangeable:
#   n_batch   the LOGICAL batch: most tokens one llama_decode may carry.  Also
#             load-bearing in Python (Batcher sizes each step from it), and it
#             costs almost no VRAM.
#   n_ubatch  the PHYSICAL batch: what one GPU forward pass computes, so it is
#             the knob that moves prefill throughput.  It costs VRAM LINEARLY
#             (n_ubatch x n_embd) and is the one to watch for an OOM.
# n_ubatch > n_batch is rejected by llama.cpp and n_ubatch <= 0 is clamped, so
# both are validated here where the failure can name the setting.
def n_batch_default() -> int:
    """CC_N_BATCH, default 2048 (LM Studio's evaluation batch size)."""
    return _int_env("CC_N_BATCH", 2048, minimum=1)


def n_ubatch_default() -> int:
    """CC_N_UBATCH, default 512 (LM Studio's physical batch size)."""
    return _int_env("CC_N_UBATCH", 512, minimum=1)


def n_rs_seq_default() -> int:
    """CC_N_RS_SEQ, default 0.  Recurrent-state snapshots per sequence (llama.h
    calls it "for rollback, 0 = no rollback"); cc_srv's equivalent of LM Studio's
    Context Checkpoints is the prefix archive."""
    return _int_env("CC_N_RS_SEQ", 0, minimum=0)


def kv_unified_default() -> bool:
    """CC_KV_UNIFIED, default on.  One KV pool shared by every sequence, so a
    spare sequence costs only what it holds; with it off llama.cpp splits n_ctx
    into a fixed slice per sequence."""
    v = os.environ.get("CC_KV_UNIFIED", "1").strip().lower()
    return v not in ("0", "false", "off", "no")


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = int(str(raw).strip())
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}")
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {v}")
    return v


class Progress:
    """Per-request prefill progress, reported in whole steps.  The percentage is
    of the tokens THIS request forwards, not of the prompt."""

    __slots__ = ("total", "done", "next_at", "label", "log", "on", "started")

    def __init__(self, total: int, log, label=None):
        self.total = max(0, int(total))
        self.done = 0
        self.label = label
        self.log = log
        self.on = bool(self.total) and progress_enabled() and PROGRESS_STEP > 0
        # The threshold the NEXT line needs.  Only ever compared against the live
        # percentage: the number printed is recomputed from self.done, so the
        # percentage and the token counts on a line cannot disagree.
        self.next_at = PROGRESS_STEP
        # No 0% line here: a job is admitted well before it is first advanced.
        # The first line goes out on the first forward pass instead (see add).
        self.started = False

    def _say(self):
        if self.log is None:
            return
        pct = 100.0 * self.done / self.total if self.total else 100.0
        where = "" if self.label is None else f" seq={self.label}"
        self.log(f"prefill {pct:5.1f}% ({self.done}/{self.total} tok{where})")

    def add(self, n: int):
        """Account for ``n`` forwarded tokens, emitting at most one line."""
        if not self.on:
            return
        self.done += n
        pct = 100.0 * self.done / self.total
        if not self.started:
            # First real work: a cache-served prefill may never reach the first
            # whole step, so this is the line that makes it visible at all.
            self.started = True
            self._say()
            while self.next_at < 100.0 and pct >= self.next_at:
                self.next_at += PROGRESS_STEP
        elif pct >= self.next_at or self.done >= self.total:
            self._say()
            # Advance past every threshold this chunk cleared: one chunk can
            # clear several, and a stale next_at would repeat a line.
            while self.next_at < 100.0 and pct >= self.next_at:
                self.next_at += PROGRESS_STEP
        if self.done >= self.total:
            # Clear `on` so a later add() stays silent instead of printing 100%
            # a second time.
            self.on = False


def probe(e, pids: list[int], max_new: int, stop: list[str], stopk: tuple,
          turn=None, on_token=None, on_serve=None):
    """Try every layer that needs no KV work.  Returns gen()'s triple, or None.

    on_serve(reused, hit) fires before on_token: the text carries message_start,
    which is where the client reads the cache fields from.
    """
    st = e.st
    # 1) Whole prompt identical -> logits cache.  Entries store the TEXT and not
    #    a re-decode of the tokens: stop trimming happens at the text level, so
    #    the stop set has to be part of the key.
    entry = e.lgc.get(e.key(pids))
    if entry is not None and entry[3] == stopk:
        toks_, text_, trunc = entry[0], entry[1], entry[2]
        if (not trunc) or len(toks_) >= max_new:
            st["lgc"] += 1
            if trunc and len(toks_) > max_new:
                toks = toks_[:max_new]
                text = e.text_of(toks)
            else:
                toks, text = toks_, text_
            if on_serve:
                on_serve(len(pids), "lgc")
            if on_token:
                on_token(text)
            return toks, text, {"hit": "lgc", "pre": 0, "reuse": len(pids)}

    # 2) Trajectory recall (weaker than 1: only that the prompt is a PREFIX of
    #    some recorded trajectory).
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
        # The layer calls on_token from inside, so the report is wrapped onto
        # that call rather than made after it returns.
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
    def __init__(self, n_ctx: int = 131072, bs: int = None, n_rs: int = None,
                 nseq: int = 1, workers: int = 1, log=print):
        self.log = log
        L = self.L = bind()
        if bs is None:
            bs = n_batch_default()
        # None means "take the environment's answer", the convention bs uses
        # above; an explicit 0 still means 0.
        if n_rs is None:
            n_rs = n_rs_seq_default()

        # The sequence layout is settled once, here, from the caller's numbers
        # and the environment; srv.py has already applied the same plan, so
        # every index below reads off it instead of being re-derived.
        plan_nseq, _, self.ddc_seq = plan_sequences(n_ctx, workers)
        if nseq < plan_nseq:
            # A direct caller asked for fewer sequences than the layout needs;
            # it sized n_ctx for the ones it named, so charge the layout that
            # window.
            n_ctx = pool_for(max(1, n_ctx // max(1, nseq)), workers)
            nseq = plan_nseq

        # The tokens ONE sequence can hold; llama_decode fails past it and a
        # failed decode leaves the ledger stale.  Read from window_of rather
        # than n_ctx // nseq: under a capped archive the pool is not nseq equal
        # slices, and the division under-reports what a worker can hold.
        self.window = window_of(n_ctx, workers)

        # The sequences DDC may verify on (see plan_sequences).
        self.ddc_pool: list = list(range(workers,
                                          workers + ddc_scratch_count(workers)))

        # DDC draft cache (off by default).  Key = top-k token ranking of the
        # current logits; equal top-1 means the same readout cell.  Value = a
        # past trajectory's continuation plus a per-position margin.  Every
        # draft is still verified by the target model, so the key is a
        # throughput knob and correctness does not rest on it.
        # Verification needs a second sequence, so KV reservation doubles with
        # nseq.
        self.ddc = None
        self.ddc_max_ctx = 0          # set below, once the window is known
        # Off by default; CC_DDC=2|4 turns it on with that n-gram key width.
        # What keeps it honest is the per-round gate in spec.py, which drops any
        # draft length that stops paying.
        _k = ddc_key()
        if _k is not None:
            from ddc import DraftCache
            _k = int(_k)
            if _k not in (2, 4):
                raise RuntimeError(f"CC_DDC supports only 0, 2 or 4, got {_k!r}")
            # Aggressive draft settings on purpose: the round's rate rises with
            # draft length, and the per-round gate in spec.py drops any bucket
            # that turns out not to pay.
            self.ddc = DraftCache(k=_k,
                                  max_draft=int(os.environ.get("CC_DDC_M", "32")),
                                  threshold=float(os.environ.get("CC_DDC_T", "0.5")),
                                  gate=float(os.environ.get("CC_DDC_GATE", "0.")),
                                  reset_per_request=os.environ.get("CC_DDC_RESET") == "1")
            if self.ddc_seq is None:
                # DDC is on but no scratch sequence was reserved, so the batched
                # scheduler is serving every request and nothing will speculate.
                log("CC_DDC is on but the batched scheduler serves every request "
                    "and DDC runs on the serial path only -- no scratch sequence "
                    "reserved, so no speculation will happen (unset CC_BATCH to "
                    "use DDC)")
            # Default the cutoff to this sequence's own window, so it never stands DDC
            # down early; it remains the clamp that stops a draft running past the
            # end of the context.
            _mctx = os.environ.get("CC_DDC_MAX_CTX")
            # self.window, not n_ctx // nseq: DDC's limit has to be a length a
            # sequence can actually hold either way.
            self.ddc_max_ctx = self.window if _mctx is None else int(_mctx)
            if self.ddc_max_ctx < 0:
                raise ValueError("CC_DDC_MAX_CTX must be >= 0 (0 means unlimited)")
            log(f"DDC on k={_k} M={self.ddc.max_draft} "
                f"T={os.environ.get('CC_DDC_T', '0.5')} gate={self.ddc.gate} "
                f"reset_per_request={self.ddc.reset_per_request} "
                f"max_context={self.ddc_max_ctx} "
                f"(scratch={self.ddc_seq if self.ddc_seq is not None else 'none'}, "
                f"nseq={nseq}, n_ctx={n_ctx})")

        # Turn-keyed reuse (off by default; see qcache.py).  Unlike the exact-prompt
        # cache it fires when the same user turn comes back inside a running
        # dialog, the only shape in which the exact cache can never fire.
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

        # Prefix archive (on by default; CC_ARCHIVE=0 turns it off).
        # Whole states are copied aside with seq_cp at the divergence points,
        # because the truncation a shorter or diverging request would need is a
        # no-op on this memory.
        self.arc = None
        self.arc_slots: list = []
        self.arc_min = int(os.environ.get("CC_ARCHIVE_MIN", "512"))
        # The longest state an archive slot may hold, in tokens; 0 means the slot
        # is charged a whole window.  Read once here so arc_ready and arc_put
        # cannot disagree while a request is in flight.
        self.arc_cap = arc_cap_for(self.window)
        # Sequences a batched scheduler may use for work: the first ``workers``
        # and no others.  Derived rather than range(nseq), because nseq counts
        # the archive slots and DDC's scratch too.
        self.work_seqs: list = list(range(workers))
        want = arc_slot_count()
        if want:
            self.arc = {}
            self.arc_clock = 1
            # The top of the range, above the workers and DDC's scratch, so
            # "nseq - want" can only ever land on an archive slot.
            self.arc_slots = list(range(nseq - want, nseq))
            log(f"prefix archive on, slots {self.arc_slots}, "
                f"worker sequences {self.work_seqs}, "
                f"min prefix {self.arc_min} tokens"
                f"{f', cap {self.arc_cap}' if self.arc_cap else ''} "
                f"(nseq={nseq}, n_ctx={n_ctx}, window={self.window})")
        if self.ddc is not None and self.ddc_seq is not None:
            log(f"DDC verification scratch {self.ddc_pool}, "
                f"archive slots {self.arc_slots or 'none'} "
                f"(disjoint; nseq={nseq})")

        t0 = time.time()
        mp = L.llama_model_default_params(); mp.ngl = 999
        self.mdl = L.llama_model_load_from_file(GGUF.encode(), mp)
        cp = L.llama_context_default_params()
        ub = n_ubatch_default()
        if ub > bs:
            # llama.cpp aborts on this rather than recovering, so it is caught
            # here where the settings are known.
            raise ValueError(
                f"CC_N_UBATCH ({ub}) must not exceed the logical batch "
                f"CC_N_BATCH ({bs}); the physical batch is a subdivision of the "
                f"logical one, so a larger value has nothing to subdivide")
        cp.n_ctx = n_ctx; cp.n_batch = bs; cp.n_ubatch = ub
        cp.n_seq_max = nseq; cp.n_rs_seq = n_rs
        # One KV pool for every sequence instead of a fixed slice each; see
        # kv_unified_default.
        cp.kvu = kv_unified_default()
        cp.tk = Q4_0; cp.tv = Q4_0          # quantized KV; 9216 B/position x n_ctx, paid once at context creation
        self.ctx = L.llama_init_from_model(self.mdl, cp)
        if not self.ctx:
            raise RuntimeError(f"could not create context n_ctx={n_ctx}")
        self.vcb = L.llama_model_get_vocab(self.mdl)
        self.nv = L.llama_vocab_n_tokens(self.vcb)
        self.kmem = L.llama_get_memory(self.ctx)   # not named mem: mem() below is the memory report and would collide
        self.bs = bs
        # Kept so the startup line can report the pair that was actually applied.
        self.ub = ub
        self.kvu = bool(cp.kvu)

        # The prefill progress reporter in force, or None.  Only ever set around
        # a prefill: feed() is shared with the decode path, where a prompt
        # percentage would be meaningless.
        self.prog = None

        self.cur: list[int] = []                   # the token sequence the KV currently holds
        self.lgc: dict = {}                        # logits cache: key -> (toks, text, trunc, stopk)
        self.lgc_max = 512
        # Trajectory recall index.  The test is a hard condition given by causal
        # attention, exact and needing no verification:
        #   the running sequence R is a prefix of a recorded trajectory T
        #   <=>  T[len(R)] is the token the model would produce next
        # (the token at position i is decided by 0..i-1 alone).
        self.ridx: dict = {}                       # (prefix length, digest) -> (trajectory id, position)
        self.rgen: dict = {}                       # trajectory id -> (gen_ids, text, stopk)
        self.ridx_max = 200_000                    # FIFO eviction; a broken chain only loses a recall, it never recalls wrongly
        # Kept for reference only: reg() indexes the whole reply, so the bound
        # that matters is ridx_max.
        self.rcap = 256
        self.tid = 0
        self.ng = {}; self.ng_done = 0; self.ng_keep = None
        # Fork archives.  Nothing calls snap()/fork() today, and they are NOT
        # layout-aware: they cycle 1..nseq-1, so whoever wires them up has to
        # start the cycle above plan_sequences' reservations.
        self.nseq = nseq; self.snaps = {}; self.slot = 1
        self.seq = []; self.ng_k = 6                        # used by prompt lookup
        # What each sequence's KV holds, as far as the scheduler's ledger knows.
        self.seq_tokens: dict = {}

        self.st = {"req": 0, "apc": 0, "lgc": 0, "rcl": 0, "pre": 0, "reuse": 0,
                   "gen": 0, "t_pre": 0.0, "t_gen": 0.0,
                   # rcl_short counts recalls DECLINED because the
                   # chain broke before the end of the trajectory.
                   "rcl_short": 0,
                   "ddc_req": 0, "ddc_spec": 0, "ddc_full": 0, "ddc_acc1": 0,
                   "ddc_tok": 0, "ddc_bypass": 0, "ddc_cutoff": 0,
                   "ddc_gated": 0, "ddc_stood_down": 0, "ddc_feature_s": 0.0,
                   "qhit": 0, "qsaved": 0, "qedit": 0, "qtopk_s": 0.0, "qstore": 0,
                   "arc_put": 0, "arc_hit": 0, "arc_restore": 0,
                   # Boundaries the cap refused to archive: the one way
                   # CC_ARCHIVE_CAP costs reuse -- see arc_ready.
                   "arc_skip_cap": 0,
                   "arc_s": 0.0, "arc_tok": 0}

        self._turn_mark = self.tok("<|im_start|>user")
        # What must precede a REAL turn marker.  Derived, not hardcoded: the last
        # token of a lone newline is the newline token whatever this vocabulary
        # calls it.  An empty result disables the anchor, which puts the search
        # back to its unanchored form.
        _nl = self.tok("\n")
        self._turn_pre = {_nl[-1]} if _nl else set()
        tb = ct.create_string_buffer(1 << 17)
        n = L.llama_model_meta_val_str(self.mdl, b"tokenizer.chat_template", tb, len(tb))
        self.tpl = tb.raw[:n].decode("utf-8") if n > 0 else None
        # The model's own Jinja template, compiled; see chat_template.py for why
        # llama_chat_apply_template is not used instead.
        _tmpl_mode = template_mode()
        self.ct = ChatTemplate(self.tpl if _tmpl_mode else None, log,
                               why=None if _tmpl_mode else
                               "off (CC_TMPL=0); llama_chat_apply_template is in use",
                               mode=_tmpl_mode)
        self.ct_mode = _tmpl_mode
        log(f"n_ctx={n_ctx} nv={self.nv} {time.time()-t0:.1f}s "
            f"tpl={'yes, ' + str(len(self.tpl)) + ' chars' if self.tpl else 'none'} "
            f"n_batch={bs} n_ubatch={ub} kv_unified={self.kvu}")
        log(f"chat template: {self.ct.line()}")
        # The same settings in LM Studio's load-panel terms, read off the params
        # struct actually handed to the context.  Context Checkpoints is
        # --ctx-checkpoints, a llama-server SLOT option with no field in
        # llama_context_params; the nearest thing here is the prefix archive.
        log(f"load settings: n_batch={bs} n_ubatch={ub} n_seq_max={nseq} "
            f"kv_unified={self.kvu} n_rs_seq={n_rs} "
            f"offload_kqv={bool(cp.okq)} op_offload={bool(cp.oo)} "
            f"swa_full={bool(cp.swaf)} threads={cp.nt}/{cp.ntb} "
            f"load_mode={mp.lm} lazy_mode={mp.lz} flash_attn={cp.fat} "
            f"rope={cp.rfb}/{cp.rfs}")

    def tmpl_jinja(self, msgs: list, add_ass: bool = True,
                   thinking: bool | None = None):
        """The template's own render, or None if it would not produce one.

        Separate from tmpl(): a caller that cannot tell which renderer it got
        cannot tell what it is generating from.
        """
        if self.ct_mode == 0:
            return None
        if self.ct_mode == 1:
            return None            # mode 1 is spliced below; see tmpl()
        return self.ct.render(msgs, add_ass, thinking)

    # ---------------- basics ----------------
    def tmpl(self, msgs: list, add_ass: bool = True,
             thinking: bool | None = None) -> str:
        """[(role, content)] -> one string, flattened by the model's own template.

        Three modes, from CC_TMPL: 0 llama_chat_apply_template, 1 the template's
        leading system block plus this engine's rendering of the conversation,
        2 the template alone.  See chat_template.py.
        """
        if self.ct_mode == 2:
            got = self.ct.render(msgs, add_ass, thinking)
            return got if got is not None else self.tmpl_lib(msgs, add_ass)
        if self.ct_mode == 1:
            # The first message that is not a system message.  msgs_of emits at
            # most one system message and always first, so this is 0 or 1 --
            # computed rather than assumed, or a system block placed elsewhere
            # would be rendered twice.
            k = next((i for i, (r, _) in enumerate(msgs) if r != "system"), len(msgs))
            head = self.ct.system_block(msgs, thinking)
            if head is not None:
                body = self.tmpl_lib(msgs[k:], add_ass=False)
                if not add_ass:
                    return head + body
                # The generation header comes from the template too:
                # enable_thinking is what separates the closed empty reasoning
                # block from an open one, and the library renderer cannot be
                # told about a template variable at all.
                tail = self.ct.generation_head(msgs, thinking)
                if tail is not None:
                    return head + body + tail
        return self.tmpl_lib(msgs, add_ass)

    def tmpl_lib(self, msgs: list, add_ass: bool = True) -> str:
        """The pre-Jinja path, kept as the fallback and for the A/B."""
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

    def piece(self, t: int) -> bytes:
        """One token's BYTES.  Not text, and deliberately not named like it: a token is
        a byte string, so decoding one on its own would corrupt a character that
        spans several of them.  The size is a retry because
        llama_token_to_piece returns the NEGATIVE of the length it needed.
        """
        b = ct.create_string_buffer(64)
        n = self.L.llama_token_to_piece(self.vcb, t, b, 64, 0, True)
        if n < 0:
            b = ct.create_string_buffer(-n)
            n = self.L.llama_token_to_piece(self.vcb, t, b, len(b), 0, True)
        return b.raw[:n]

    def text_of(self, toks) -> str:
        """A whole token run as text.  Decodes ONCE, over all the bytes."""
        return utext.whole(b"".join(self.piece(t) for t in toks))

    def feed(self, ids: list[int], chunk: int | None = None):
        """Push ids into the KV.  Chunked because of the n_batch limit; the batch
        is built from a memory view rather than copying element by element."""
        L = self.L
        chunk = chunk or self.bs
        raw = bytes((ct.c_int32 * len(ids))(*ids))
        # Prefill progress, when a prefill established one.  The hook lives
        # here because this loop is the only place that knows a chunk was
        # actually forwarded; during generation `prog` is None and nothing is
        # emitted.
        prog = self.prog
        i = 0
        while i < len(ids):
            m = min(chunk, len(ids) - i)
            sub = (ct.c_int32 * m).from_buffer_copy(raw[i * 4:(i + m) * 4])
            if L.llama_decode(self.ctx, L.llama_batch_get_one(sub, m)) != 0:
                raise RuntimeError(f"decode failed at {i}/{len(ids)}")
            i += m
            if prog is not None:
                prog.add(m)
        self.cur.extend(ids)

    def kvlen(self) -> int:
        """Real KV length on the llama.cpp side (last position + 1).  This is how
        a truncation is checked for actually having taken effect."""
        p = self.L.llama_memory_seq_pos_max(self.kmem, 0)
        return p + 1 if p >= 0 else 0

    def trunc(self, n: int) -> bool:
        """Truncate to the first n tokens; returns whether it REALLY happened.

        Partial truncation is a no-op on this hybrid memory; verify it took, and
        when it did not clear everything (the full clear is reliable) and tell the
        caller to restart from position zero.
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
        the workers hold unrelated branches, so that measure is small.

        Found by searching for the template's own turn marker rather than
        assuming a layout; a sentinel here would have to reproduce the
        template's tokenization to be worth anything.

        The marker has to be ANCHORED, and that is not a refinement: eng.py
        contains the line ``self._turn_mark = self.tok("user")``, so a prompt
        that includes eng.py contains the marker as ordinary text and an
        unanchored search returns that occurrence instead of the turn.

        A real turn marker follows a newline.  The occurrence inside source code
        follows a quote.  That is the whole test, and it is derived from the
        tokenizer rather than written down as a number: ``_turn_pre`` is whatever
        token the newline really is on this vocabulary.
        """
        if not pids:
            return None
        mark = self._turn_mark
        if not mark:
            return None
        pre = self._turn_pre
        for i in range(len(pids) - len(mark), -1, -1):
            if list(pids[i:i + len(mark)]) != mark:
                continue
            # i == 0: a turn at the very start has nothing before it to check.
            if i == 0 or not pre or pids[i - 1] in pre:
                return i
            self.st["turn_skip"] = self.st.get("turn_skip", 0) + 1
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

        Reading the logits is what forces the wait for the forward pass that
        produced them, and the wait lands in whichever call first touches the
        buffer: np.ctypeslib.as_array looks free and is a view, but it
        dereferences the pointer, so it is usually the call that pays.

        as_array and the argmax are therefore timed together and returned as
        ``base`` -- exactly what the trace-free path pays anyway -- and the
        caller subtracts it.  What is left is the marginal cost of the trace.
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
    # Take the last k tokens of the current sequence, look up their previous
    # occurrence in the context already seen, and draft the m tokens that
    # followed it.  The context is the cache: no extra model, no training.
    #
    # Verification is the hard part, and this model cannot rewind, so the draft
    # never enters the real sequence -- it is copied into seq1 and verified
    # there, then the accepted prefix is seq_cp'd back.  The real sequence only
    # ever moves forward.
    def batch_parts(self, parts):
        """One batch built from tokens belonging to several sequences.

        ``parts`` is ``[(seq_id, pos0, tokens[, all_rows]), ...]``.  Every
        sequence keeps its own position, so the batch can carry a prefill chunk
        of one request next to a single decode step of another.

        Logits are requested only for the LAST token of each part, unless the
        part sets ``all_rows`` (speculative verification is the reason that
        flag exists; a plain part wants only the next-token distribution).

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
            self.prog = Progress(len(new), getattr(self, "log", None))
            try:
                self.feed(new)
            finally:
                self.prog = None

        seq = list(pids)                     # the full sequence = prompt + generated
        self.ng = {}; self.ng_done = 0
        self.ngram_feed(seq, k, len(seq))    # load the preceding context into the n-gram index

        toks: list[int] = []
        text = ""
        # Token bytes go in here and only whole characters come out, so a
        # character split across tokens survives.  NOT flushed on a stop hit:
        # whatever the decoder is still holding sits past the cut and is meant to
        # be dropped with the rest of the tail.
        _dec = utext.decoder()
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
                text += _dec.decode(self.piece(nxt)); st["gen"] += 1
                nxt = self.top()
            else:
                bat = [nxt] + d[:m - 1]      # the free one goes first
                acc, nxt = self.verify(bat, pos)
                if not acc:
                    toks.append(nxt); self.feed([nxt]); seq.append(nxt)
                    self.ngram_feed(seq, k, len(seq))
                    text += _dec.decode(self.piece(nxt)); st["gen"] += 1
                    nxt = self.top()
                else:
                    self.L.llama_memory_seq_cp(self.kmem, 1, 0, pos, pos + len(acc))
                    self.L.llama_memory_seq_rm(self.kmem, 1, 0, -1)
                    self.cur.extend(acc)
                    toks.extend(acc); seq.extend(acc)
                    self.ngram_feed(seq, k, len(seq))
                    for t in acc:
                        text += _dec.decode(self.piece(t))
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
        if hit is None:
            # Nothing is pending a stop hit has discarded, so finish the decoder.
            text += _dec.decode(b"", True)
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
    # A hybrid model cannot rewind, so partial truncation via seq_rm is a no-op
    # and forking back means clearing and recomputing.  Instead, archive just
    # before each fork point -- at the moment a trajectory passes it the KV is
    # exactly [0, fork point) -- then to fork, clear the working sequence and
    # restore the archive wholesale.  Whole-sequence copies are the only
    # operation that works here; a partial seq_cp aborts on GGML_ASSERT(is_full).
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
    # One KV, many branches.  A request shorter than the KV, or one that diverges
    # before its end, needs a truncation -- which is a no-op on this hybrid
    # memory, so the whole context is rebuilt.  Under agent fan-out that is the
    # normal case: sibling subagents share a long prefix but are not extensions
    # of one another.
    #
    # The state a sibling needs is the one at the divergence point, and it only
    # exists while prefilling through it.  So arc_put is called on the way past
    # that boundary and a later sibling calls arc_restore to get it back whole.
    def arc_ready(self, upto: int) -> bool:
        """Whether the boundary at ``upto`` is worth archiving."""
        if self.arc is None or upto < self.arc_min:
            return False
        # A state longer than the slot's budget must not be copied into it: with a
        # capped archive the slot's cells are only CC_ARCHIVE_CAP long, and the
        # seq_cp would take cells the pool promised to a worker.  Counted, not
        # just declined: a rising arc_skip_cap against a low arc_put means raise
        # CC_ARCHIVE_CAP; arc_put high with reuse low means the archive's
        # matching, not its size, is the problem.
        if 0 < self.arc_cap < upto:
            self.st["arc_skip_cap"] = self.st.get("arc_skip_cap", 0) + 1
            return False
        return True

    def arc_put(self, tokens, src: int = 0) -> None:
        """Store sequence ``src``'s KV, which holds exactly ``tokens``, in a spare slot.

        ``src`` is a parameter because the batched scheduler keeps each request
        on its own sequence.  A slot is reused by least-recent use when they are
        all taken; the host keeps the token sequence alongside the state because
        that is what a later request is matched against.
        """
        if not self.arc_slots:
            return
        if 0 < self.arc_cap < len(tokens):
            # arc_ready cleared this boundary, but the state that exists to store is
            # the one the sequence holds NOW, and a chunk that straddled it left
            # the state up to a batch past it.  It cannot be shortened to fit --
            # the copy is whole-sequence -- so the boundary is dropped rather
            # than the slot overrunning its budget.
            self.st["arc_skip_cap"] = self.st.get("arc_skip_cap", 0) + 1
            return
        free = [s for s in self.arc_slots if s not in self.arc]
        if free:
            slot = free[0]
        else:
            slot = min(self.arc, key=lambda s: self.arc[s][2])
        L = self.L
        L.llama_memory_seq_rm(self.kmem, slot, 0, -1)        # clear it first
        L.llama_memory_seq_cp(self.kmem, src, slot, 0, -1)   # whole state
        self.arc[slot] = (array("i", tokens), len(tokens), self.arc_clock)
        self.arc_clock += 1
        self.st["arc_put"] = self.st.get("arc_put", 0) + 1

    def arc_find(self, pids: list):
        """Longest archived state whose token sequence is a prefix of ``pids``.

        Returns ``(slot, length)`` or ``None``.  The match must be a prefix in full:
        a state at length n can only be restored and extended, not shortened.
        Matching is an exact token comparison, not a heuristic.
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
        # Timed because a restore is the one cost this layer adds: a whole-state
        # seq_cp, so it scales with the state (n x 9216 B).
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

        Free -- no extra model, no training.
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
        # Walk the occurrences backwards: lst[-1] may be too close (j+m past the
        # current position), in which case fall back to an earlier one.
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
            self.prog = Progress(len(new), getattr(self, "log", None))
            try:
                self.feed(new)
            finally:
                self.prog = None
        st["t_pre"] += time.time() - t0

        # Build the n-gram index; seq is the running full token sequence.
        self.seq = list(pids)
        self.dbg = []
        self.ng = {}; self.ng_done = 0; self.ng_k = k
        n0 = len(self.seq)
        self.ng_add(n0)

        toks: list[int] = []
        text = ""
        _dec = utext.decoder()          # see gen_pl: bytes in, whole characters out
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
            # the wrong place and is treated as absent.
            if d is None or d[0] != nxt:
                self.feed([nxt]); self.seq.append(nxt)
                self.ng_add(len(self.seq))
                toks.append(nxt); text += _dec.decode(self.piece(nxt)); st["gen"] += 1
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
                    text += _dec.decode(self.piece(t))
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
        if hit is None:
            text += _dec.decode(b"", True)
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

        The key is a digest rather than tuple(ids): a tuple of the whole prompt's
        tokens is ~99 % of the entry it guards, and 512 of them come to about a
        GB.  With a digest it is 112 B per entry.  At 128 bits a noticeable
        collision needs 2^64 entries resident at once, and the length is part
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
        txt = gtxt if k2 >= len(gen) else self.text_of(out)
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
        # recall's guarantee is that a registered trajectory is one the model
        # actually produced, and a reused reply was produced under
        # ``entry.prompt``, a strict prefix of this one -- registering it would
        # put a semantically chosen reply into the layer that is meant to be
        # exact, and later requests served from it would bypass the gates here.
        if on_token:
            on_token(text)
        out = {"hit": "qreuse", "pre": 0, "reuse": len(pids), "stop": None,
               "trunc": False, "qpreuse": m.level, "qwhy": m.reason}
        out.update({k: info[k] for k in ("replayed", "edited", "edit_applied",
                                         "edit_rejected")})
        return toks, text, out

    # ---------------- prefill, shared by the plain and DDC serving paths ----------------
    # The APC/archive half of gen(), lifted out so gen_ddc() can run the same
    # prefill.  The pair must stay in step: the prefix plan_prefix chose decides
    # what feed_prefix may skip, and the archive put in feed_prefix depends on
    # the divergence point plan_prefix reported.
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
        #     Whether a given round is worth taking is decided by the governor
        #     inside ddc_decode, per draft length: a round that lands is several
        #     times faster than decoding one token at a time and one that does
        #     not is a net loss, so it cannot be switched off wholesale.
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
        # Prefill progress for this request alone.  The total is what this
        # request forwards, not the prompt: the reused prefix is already in the
        # KV, so a cache hit that forwards 300 of 50k tokens is nearly done at
        # its first line rather than 1% through work it is not doing.
        self.prog = (Progress(len(new), getattr(self, "log", None))
                     if new else None)
        try:
            self.feed_prefix(pids, p, p_raw, arc)
        finally:
            self.prog = None
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
        _dec = utext.decoder()          # see gen_pl: bytes in, whole characters out
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
                    toks.append(nxt); self.feed([nxt]); text += _dec.decode(self.piece(nxt))
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
            toks.append(t); self.feed([t]); text += _dec.decode(self.piece(t))
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
        if hit is None:
            text += _dec.decode(b"", True)
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
        # takes part in it: the put on the way past a divergence point is the
        # only thing that ever fills the archive.  The decoder is told what the
        # KV already holds (``reuse``) instead of working it out from lcp/trunc.
        arc = getattr(self, "arc", None)
        p, p_raw = self.plan_prefix(pids, arc)
        if on_serve is not None:
            # p is final and nothing has been generated yet, so this is where the
            # HTTP layer gets its reuse figure into message_start.  Without it a
            # request that restored an archived state reported a cache read of
            # zero, which is what the hit-rate tooling reads off the transcript.
            on_serve(p, "ddc")
        t0 = time.time()
        self.prog = (Progress(len(pids) - p, getattr(self, "log", None))
                     if p < len(pids) else None)
        try:
            self.feed_prefix(pids, p, p_raw, arc)
        finally:
            self.prog = None
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
                    f"cap={self.arc_cap or 'none'} "
                    f"skip_cap={s.get('arc_skip_cap', 0)} "
                    f"slots={len(self.arc)}")
        if self.qc is not None:
            out += (f" | qhit={s['qhit']} qsaved={s['qsaved']} qedit={s['qedit']} "
                    f"qstore={s['qstore']} qmem={m['qc']:.3f}MB "
                    f"qtopk={s['qtopk_s']:.1f}s {self.qc.line()}")
        out += f" | {self.ct.line()}"
        return out


if __name__ == "__main__":
    e = Eng()
    ids = e.tok("The capital of France is")
    print("tok:", ids)
    print(e.gen(ids, max_new=8)[1:])
    print(e.line())
