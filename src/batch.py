# -*- coding: utf-8 -*-
"""Batched generation: many requests, one forward pass per step.

A scheduler thread owns the context: it gives every active request a sequence,
advances each by one step, decodes all of them in a single llama_decode, and
hands each sequence back the logits row that belongs to it. Batching is sound
only if a sequence's output does not depend on its batchmates. The KV-free cache
layers (Eng.probe) answer what they can before a request ever reaches a
sequence, keeping slots free for the rest.
"""
from __future__ import annotations

import os
import threading
import time

import spec
import utext
from eng import Progress

# A step batches only with the steps either side of it, so the scheduler sees
# nothing beyond what a job carries.
PREFILL = "prefill"
DECODE = "decode"
DONE = "done"


class Job:
    """One request in flight."""

    __slots__ = ("prompt", "max_new", "stop", "turn", "seq", "cur", "pos",
                 "out", "raw", "text", "hit", "state", "cursor", "stopped",
                 "done", "result", "error", "t0", "prefilled", "reused",
                 "arc_at", "arc_done", "on_token", "pf_s", "gn_s", "on_serve",
                 "ddc", "scratch", "draft", "verifying", "shared", "arc_s",
                 "prog", "dec")

    def __init__(self, prompt, max_new, stop, turn=None, on_token=None,
                 on_serve=None):
        # Each generated token's text as it is produced; None means the reply is
        # only available at the end.
        self.on_token = on_token
        # Called once at admission with (reused, hit), long before the reply
        # exists, because the HTTP layer needs it for message_start usage.
        self.on_serve = on_serve
        self.prompt = list(prompt)
        self.max_new = max_new
        self.stop = list(stop or ())
        self.turn = turn
        self.seq = None
        self.cur = []          # tokens this sequence's KV holds
        self.pos = 0           # next position to write
        self.out = []          # generated tokens
        self.raw = []          # text pieces, joined on demand
        self.text = ""
        self.hit = None
        self.state = "queued"
        self.cursor = 0        # prefill progress
        self.prefilled = 0     # reused prefix + tokens forwarded
        self.reused = 0        # just the reused prefix
        # Where this prompt diverges from what its sequence held: the state there
        # is what a sibling branch asks for, and only prefill can capture it.
        self.arc_at = None
        self.arc_done = False
        # Speculative decoding, when CC_DDC_BATCH puts DDC in this scheduler.
        # ``ddc`` is the draft cache this job may use (None = decode normally),
        # ``scratch`` the sequence reserved for its verification copies, and
        # ``draft`` the tokens it is holding for the next step to verify.  A job
        # without a scratch still decodes normally: speculation here is
        # opportunistic, never required for correctness.
        self.ddc = None
        self.scratch = None
        self.draft = None
        # Set by _plan on the part it emits for this job, read by _advance: the
        # part is a verification pass and carries logits for every position.
        self.verifying = False
        self.stopped = None
        # Wall time spent prefilling vs decoding, allocated by token share across the step.
        self.pf_s = 0.0
        self.gn_s = 0.0
        # This job's own archive restore: the engine's arc_s cannot be split
        # across concurrent jobs, so it is timed at the restore site in _admit.
        self.arc_s = 0.0
        # Whether a step of this job also carried another job's part: that makes
        # pf_s/gn_s an allocation rather than a measurement, and the HTTP layer
        # then leaves the prefill/decode split out of the request log.
        self.shared = False
        # This job's own prefill progress reporter, or None (see eng.Progress).
        # A shared one would attribute each job's chunks to whoever reported last.
        self.prog = None
        # Incremental UTF-8 for this job's own token bytes (see utext). A shared
        # decoder would splice one reply's partial character onto another's.
        self.dec = utext.decoder()
        self.done = threading.Event()
        self.result = None
        self.error = None
        self.t0 = time.time()

    def seal(self) -> str:
        """Flush the decoder (a trailing partial character becomes U+FFFD) and return the text."""
        self.text = "".join(self.raw) + self.dec.decode(b"", True)
        return self.text

    def finish(self):
        self.state = DONE
        self.done.set()


class Batcher:
    """Runs several requests' forward passes as one.

    A single scheduler thread owns the context; callers submit a job and wait
    on its event, never touching the context themselves.
    """

    def __init__(self, eng, chunk=None, log=print, start=True):
        self.e = eng
        self.log = log
        # Prefill is chunked so that a long prompt does not monopolise a step: a
        # request still prefilling then shares the batch with decoding ones.
        # The share one prefill may take is CC_BATCH_CHUNK, deliberately not
        # tied to eng.bs: a chunk that scales with n_batch starves the
        # sequences already decoding.
        self.chunk = chunk or int(os.environ.get("CC_BATCH_CHUNK", "0") or 0) \
            or max(64, min(eng.bs // 4, 128))
        # CC_BATCH_WAIT: how much better a busy home worker's prefix must be
        # before blocking for it beats taking a free worker.  Zero would mean
        # "always wait for the longest"; a huge value "never wait".
        self.wait_margin = int(os.environ.get("CC_BATCH_WAIT", "512"))
        # The archive keeps its states in sequences of its own, so the scheduler
        # only uses the sequences the engine says are free (eng.work_seqs).
        self.seqs = list(getattr(eng, "work_seqs", None) or range(eng.nseq))
        # DDC's verification sequences (eng.ddc_pool), kept out of work_seqs so
        # a scheduler cannot hand one out as a main sequence.  A job holds one
        # for its whole reply, so the pool bounds how many may speculate at once.
        self.ddc_pool = list(getattr(eng, "ddc_pool", None) or [])
        self.ddc_used = set()
        self.jobs: list[Job] = []
        self.queued: list[Job] = []
        self.lock = threading.Lock()
        # Guards the KV-free caches (lgc / ridx / qcache), which request threads
        # read through Eng.probe while the scheduler thread writes them.
        # Eng.gen writes those same caches WITHOUT this lock, so srv.py picks one
        # serving path at startup and never both; mixing them would race on ridx.
        self.state_lock = threading.Lock()
        self.wake = threading.Event()
        self.st = {"steps": 0, "batched_tokens": 0, "jobs": 0, "admitted": 0,
                   "peak_seq": 0}
        self._stop = False
        # start=False lets the pure-Python scheduling logic be tested without a GPU.
        self.thread = None
        if start:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    # ---------------- public ----------------
    def submit(self, prompt, max_new, stop, turn=None, on_token=None,
               on_serve=None) -> Job:
        job = Job(prompt, max_new, stop, turn, on_token, on_serve)
        with self.lock:
            self.queued.append(job)
            self.st["jobs"] += 1
        self.wake.set()
        return job

    def shutdown(self):
        self._stop = True
        self.wake.set()

    # ---------------- scheduling ----------------
    def _free_seqs(self):
        used = {j.seq for j in self.jobs if j.seq is not None}
        return [s for s in self.seqs if s not in used]

    def _admit(self):
        """Move queued jobs onto free sequences, reusing a prefix if one fits.

        ``usable`` is the strict prefix a sequence already holds; ``diverged``
        is how far the prompt agrees with it, which is the boundary worth
        archiving.
        """
        with self.lock:
            pending, self.queued = self.queued, []
        for job in pending:
            # A prompt too long for one sequence must fail here: a failed decode does
            # not retire its job, and only _retire writes seq_tokens, so the
            # sequence keeps tokens the ledger does not know about and the next
            # request placed on it decodes from the wrong position.
            if len(job.prompt) > self.e.window:
                job.error = ValueError(
                    f"prompt is {len(job.prompt)} tokens but a sequence holds "
                    f"{self.e.window} (n_ctx / nseq); raise --n-ctx")
                job.finish()
                continue
            busy = {j.seq for j in self.jobs if j.seq is not None}
            free = [s for s in self.seqs if s not in busy]

            # Search EVERY worker, not just the free ones: the one holding this
            # prompt's history is usually busy, and taking it beats a cold
            # prefill.
            home, usable, diverged = None, -1, 0
            alt, alt_usable = None, -1
            for s in self.seqs:
                cur = self._seq_tokens(s)
                n = 0
                m = min(len(cur), len(job.prompt))
                while n < m and cur[n] == job.prompt[n]:
                    n += 1
                diverged = max(diverged, n)
                if n == len(cur):                # a strict prefix: reusable as is
                    if n > usable:
                        home, usable = s, n
                    if s in free and n > alt_usable:
                        alt, alt_usable = s, n

            if home is not None and home not in busy:
                best = home                     # free AND the longest: no choice
            elif alt is not None and (home is None
                                      or usable - alt_usable <= self.wait_margin):
                # A free worker holds nearly as much of this prompt as the busy one
                # does, so take it rather than block.
                best, usable = alt, alt_usable
            elif home is not None:
                # Waiting for it is worth it: the alternative gives up a lot.
                self.st["waited"] = self.st.get("waited", 0) + 1
                with self.lock:
                    self.queued.append(job)
                continue
            elif free:
                best, usable = free[0], 0
            else:
                with self.lock:
                    self.queued.append(job)      # no room yet, keep waiting
                continue
            job.seq = best
            cur = self._seq_tokens(best)
            self._ddc_claim(job)

            # Nothing on a sequence fits.  An archived state might.
            if self.e.arc is not None and usable < self.e.arc_min:
                hit = self.e.arc_find(job.prompt)
                if hit is not None and hit[1] > usable:
                    # Timed here rather than read off the engine's global arc_s:
                    # a concurrent admission's restore would be credited to this
                    # request.
                    _t0 = time.perf_counter()
                    self.e.arc_restore(hit[0], best)
                    job.arc_s = time.perf_counter() - _t0
                    self.e.st["arc_hit"] = self.e.st.get("arc_hit", 0) + 1
                    self._start(job, hit[1], "arc")
                    continue

            # A sequence must be cleared before decoding into it, even a fresh one:
            # one that never went through seq_rm fails llama_decode with
            # "failed to initialize batch".
            had = len(cur)
            if usable == 0 or usable < had:
                self.e.clear_seq(best)
            if self.e.arc is not None:
                # Archive at the start of the trailing user turn: the boundary every
                # sibling of this request shares.  Divergence from the worker's
                # own contents is not that boundary.
                at = self.e.turn_start(job.prompt)
                if at is None or at <= usable:
                    at = diverged if diverged > usable else None
                if at is not None and self.e.arc_ready(at):
                    job.arc_at = at
            self._start(job, usable, "apc")

    def _start(self, job, reused, how):
        """Put a job into prefill with ``reused`` tokens already in its sequence."""
        if reused >= len(job.prompt):
            # A decode step needs a fresh logits row, so the last token is always re-forwarded.
            reused = max(0, len(job.prompt) - 1)
        job.cur = list(job.prompt[:reused])
        job.cursor = reused
        job.prefilled = reused
        job.reused = reused
        job.pos = reused
        job.state = PREFILL
        self.e.st["reuse"] += reused
        self.e.st["pre"] += len(job.prompt) - reused
        if reused:
            self.e.st["apc"] += 1
        # Tell an archive restore from an in-sequence reuse from a cold start.
        job.hit = ("batch_arc" if how == "arc"
                   else "batch_apc" if reused else "batch_cold")
        # Progress sized by what this job will still forward (cursor reused ->
        # len(prompt)); labelled with the sequence because concurrent prefills
        # interleave line by line in the log.
        job.prog = Progress(len(job.prompt) - reused, self.log, job.seq)
        # Report the serving decision now: it is known at admission, before a
        # single token of this request exists.
        if job.on_serve is not None:
            job.on_serve(job.reused, job.hit)
        self.jobs.append(job)
        self.st["admitted"] += 1
        self.st[how] = self.st.get(how, 0) + 1

    def _seq_tokens(self, seq):
        """Tokens the KV for ``seq`` holds, as far as the engine's ledger knows."""
        for j in self.jobs:
            if j.seq == seq:
                return j.cur
        return self.e.seq_tokens.get(seq, [])

    # ---------------- speculative decoding (CC_DDC_BATCH) ----------------
    # DDC's round, as the scheduler sees it: copy the committed state aside,
    # write the whole draft into the copy in one pass, and commit only a fully
    # accepted draft.  A partial round is dropped and reported to the spec.py
    # governor as zero tokens returned, and its scratch is never committed.
    def _ddc_claim(self, job):
        """Give this job a scratch sequence to verify drafts on, if one is free.

        A job that gets none is not a failure -- it decodes one token at a time.
        """
        e = self.e
        cache = getattr(e, "ddc", None)
        if cache is None or not self.ddc_pool:
            return
        # Past its cutoff the cache stands down for this request anyway, so the
        # sequence would be held for nothing.
        if e.ddc_max_ctx and len(job.prompt) >= e.ddc_max_ctx:
            return
        free = [s for s in self.ddc_pool if s not in self.ddc_used]
        if not free:
            return
        job.ddc = cache
        job.scratch = free[0]
        self.ddc_used.add(free[0])
        cache.begin()
        spec.begin_request()
        e.st["ddc_req"] = e.st.get("ddc_req", 0) + 1

    def _ddc_release(self, job):
        """Hand a job's scratch sequence back to the pool."""
        if job.scratch is not None:
            self.ddc_used.discard(job.scratch)
            job.scratch = None
        job.draft = None

    def _index_and_offer(self, job, row, token):
        """Index a produced token against the row that predicted it, then offer a draft.

        Both halves share one full-vocabulary scan; spec.note_token runs even
        while DDC is stood down, being the repetition detector.
        """
        spec.note_token(token)
        if job.ddc is None:
            return
        # The scan is the fixed cost the gate exists to avoid: standing down
        # after paying it would save nothing, so it is checked first.
        if not spec.paying():
            return
        from ddc import features
        e = self.e
        # Timed into the same counter the serial path (ddc_decode.py) reports
        # it under, so /stats does not claim the scan is free.
        t0 = time.perf_counter()
        first = features(row)
        e.st["ddc_feature_s"] = e.st.get("ddc_feature_s", 0.0) + (
            time.perf_counter() - t0)
        job.ddc.append(token, first[0], first[1])
        self._offer(job, first, token)

    def _offer(self, job, first, nxt):
        """Look a draft up for the position this row opens, and hold it for the next step."""
        if job.draft is not None:
            return
        e = self.e
        room = min(job.max_new - len(job.out), e.window - job.pos)
        if e.ddc_max_ctx:
            room = min(room, e.ddc_max_ctx - job.pos)
        if room < 2:                      # the cache wants two tokens to be worth it
            return
        proposal = job.ddc.propose(first[0], nxt, room)
        spec.note_offer(len(proposal.tokens) if proposal is not None else 0)
        if proposal is None or not spec.allow(len(proposal.tokens)):
            return
        job.draft = list(proposal.tokens)

    def _finish_round(self, job, first_idx):
        """Turn a verification pass into committed tokens.  -> (draft, committed).

        Read every logits row first: they are views into the buffer the next
        decode overwrites.  The scan starts at one, because a proposal's first
        token is the model's own next token (ddc.propose refuses any other).
        """
        from ddc import features
        e, L = self.e, self.e.L
        draft = job.draft
        k = len(draft)
        count = 1
        while count < k:
            if int(e.logits_row(first_idx + count - 1).argmax()) != draft[count]:
                break
            count += 1
        accepted = draft[:count]
        pending = None
        if count == k:
            # The copy IS the committed state now: one seq_cp moves it back,
            # and no forward pass is needed.
            L.llama_memory_seq_rm(e.kmem, job.seq, 0, -1)
            L.llama_memory_seq_cp(e.kmem, job.scratch, job.seq, 0, -1)
            pending = int(e.logits_row(first_idx + k - 1).argmax())
        # Cleared whether or not the round was kept: a rejected tail must not be
        # left where the next round would copy from it.
        L.llama_memory_seq_rm(e.kmem, job.scratch, 0, -1)
        job.draft = None
        e.st["ddc_spec"] = e.st.get("ddc_spec", 0) + 1
        e.st["ddc_full"] = e.st.get("ddc_full", 0) + (count == k)
        e.st["ddc_acc1"] = e.st.get("ddc_acc1", 0) + (count == 1)
        e.st["ddc_tok"] = e.st.get("ddc_tok", 0) + count
        if count < k:
            # Partial: given up, so nothing was committed, indexed or noted -- only
            # draft[0] is real, and the offering step already indexed it.  No
            # per-position features either: a discarded round must not pay that
            # scan, the largest term in a round's real cost.
            e.st["ddc_dropped"] = e.st.get("ddc_dropped", 0) + 1
            return k, 0
        # Kept: features for the committed positions, read while the rows are
        # still live; draft[0] was indexed by the offering step, so this list
        # pairs with accepted[1:].
        t_feat = time.perf_counter()
        descs = [features(e.logits_row(first_idx + i)) for i in range(count - 1)]
        e.st["ddc_feature_s"] = e.st.get("ddc_feature_s", 0.0) + (
            time.perf_counter() - t_feat)
        # cur takes the whole draft, because that is what the KV holds; out takes
        # it one token at a time so _emit applies max_new and the stop strings.
        job.cur.extend(accepted)
        job.pos += k
        # Index the committed tokens against the features that preceded them: a
        # rejected tail would poison the index with tokens this model never chose.
        for t, (ids, gaps) in zip(accepted[1:], descs):
            job.ddc.append(t, ids, gaps)
            spec.note_token(t)
        # draft[0] is committed but not emitted: the ledger is cur == prompt + out[:-1].
        for t in accepted[1:]:
            job.out.append(t)
            self._emit(job, t)
            if job.state == DONE:
                return k, count
        if job.pos >= e.window:
            # The main sequence is full: decoding one position past the cells would
            # fail the whole batch and every other request sharing it.
            job.stopped = None
            job.text = "".join(job.raw)
            job.finish()
            return k, count
        job.out.append(pending)
        self._emit(job, pending)
        if job.state != DONE:
            # A produced token like any other: indexed against the row that
            # predicted it, which is also where the next round's draft is looked up.
            self._index_and_offer(job, e.logits_row(first_idx + k - 1), pending)
        return k, count

    def _retire(self):
        """Hand finished jobs back and free their sequences.

        Registering the trajectory happens here, on the scheduler thread, which
        is why the KV-free caches are guarded against Eng.probe readers.
        """
        for j in list(self.jobs):
            if j.state != DONE:
                continue
            self.jobs.remove(j)
            self._ddc_release(j)
            self.e.seq_tokens[j.seq] = list(j.cur)
            with self.state_lock:
                if j.out:
                    self.e.reg(j.prompt, j.out, j.text, tuple(j.stop))
            self.e.st["gen"] += len(j.out)

    def _plan(self):
        """One step's worth of work: a list of (seq, pos0, tokens, job) parts.

        The budget is shared out in two passes: every sequence that is DECODING
        gets its one token, then what is left is split between the sequences
        that are PREFILLING.
        """
        live = [j for j in self.jobs if j.state != DONE]
        if not live:
            return []
        budget = self.e.bs
        parts = []

        # Decode first: a decode needs exactly one token and a caller is blocked on it, so prefill yields.
        decoding = [j for j in live if j.state == DECODE]
        for job in decoding:
            if budget < 1:
                break
            if job.draft is not None and job.scratch is not None and len(job.draft) <= budget:
                # The part is the whole draft written into the job's scratch
                # sequence, with a logits row per position.
                job.verifying = True
                parts.append((job.scratch, job.pos, job.draft, job))
                budget -= len(job.draft)
                continue
            if job.draft is not None:
                # Not verified this step, so the draft goes and the decode stays:
                # a stale draft describes the wrong position, and its first token
                # is never verified (a proposal opens with the model's own next
                # token), so it would commit that token unverified.
                job.draft = None
            job.verifying = False
            parts.append((job.seq, job.pos, [job.out[-1]], job))
            budget -= 1

        prefilling = [j for j in live if j.state == PREFILL]
        if prefilling and budget > 0:
            share = max(1, budget // len(prefilling))
            for job in prefilling:
                if budget <= 0:
                    break
                job.verifying = False
                take = min(self.chunk, share, budget, len(job.prompt) - job.cursor)
                if job.arc_at is not None and not job.arc_done:
                    # Land the chunk exactly on arc_at so the state there can be
                    # copied out; overshooting loses the boundary for good.
                    take = min(take, max(1, job.arc_at - job.cursor))
                take = max(1, take)
                toks = job.prompt[job.cursor:job.cursor + take]
                if not toks:
                    # Nothing left to forward: only a job with a token to decode
                    # FROM may end prefill, or the next step indexes out[-1] on
                    # an empty list and takes the whole batch down.
                    if job.out:
                        job.state = DECODE
                    else:
                        job.error = RuntimeError(
                            "prefill ended with nothing forwarded and no first token")
                        job.finish()
                    continue
                parts.append((job.seq, job.pos, toks, job))
                budget -= len(toks)
        return parts

    def _advance(self, parts):
        """Decode one step for every part and give each job its own logits row."""
        e, L = self.e, self.e.L
        t_step = time.perf_counter()
        # A verification needs the committed state copied aside first: the draft
        # is written into the copy, never into the real sequence, because a
        # rejected tail cannot be truncated away on this hybrid memory.  The copy
        # cannot be made later -- it has to be the state as it stands before this
        # step's decode touches anything.
        for seq, pos0, toks, job in parts:
            if job.verifying:
                L.llama_memory_seq_rm(e.kmem, job.scratch, 0, -1)
                L.llama_memory_seq_cp(e.kmem, job.seq, job.scratch, 0, -1)
        b, keep, idx, first = e.batch_parts(
            [(s, p, t, j.verifying) for s, p, t, j in parts])
        if L.llama_decode(e.ctx, b) != 0:
            import os as _os
            if _os.environ.get("BATCH_DBG"):
                print("FAILED BATCH:", [(s, p, len(t), t[:4]) for s, p, t, _ in parts],
                      "n_tokens=", b.n_tokens, "nseq=", e.nseq, flush=True)
                print("  arrays: pos=", list(b.pos[:b.n_tokens]), flush=True)
                print("          nseqid=", list(b.n_seq_id[:b.n_tokens]), flush=True)
                print("          logits=", list(b.logits[:b.n_tokens]), flush=True)
                # Retry the same work one sequence at a time, on this same
                # engine state.  If the parts decode individually, the batch is
                # what is wrong; if not, the state is.
                for s, p, t, _ in parts:
                    bb, kk, ii, ff = e.batch_parts([(s, p, t)])
                    print(f"  alone seq={s} pos={p} n={len(t)} -> "
                          f"rc={L.llama_decode(e.ctx, bb)}", flush=True)
            raise RuntimeError("batched decode failed")
        _ = keep
        self.st["steps"] += 1
        self.st["batched_tokens"] += b.n_tokens
        self.st["peak_seq"] = max(self.st["peak_seq"], len(parts))
        if len(parts) > 1:
            # Every part in this step shares one llama_decode, so from here on
            # each job's pf_s/gn_s split is an allocation by token count, not a
            # clock it owns.  Flagged for the HTTP layer, which suppresses the
            # prefill/decode display for such a request instead of showing a
            # number that reads more precise than it is.
            for _, _, _, job in parts:
                job.shared = True
        # Captured before the loop below, which flips a finished prefill to
        # DECODE -- reading state afterwards would file the last chunk of every
        # prompt as decode time.
        was_prefill = [job.state == PREFILL for _, _, _, job in parts]
        n_tok = sum(len(t) for _, _, t, _ in parts)
        # Rounds are reported to the governor after the step's cost is known,
        # because what it compares is tokens returned against time spent and the
        # step's wall time is only available here.
        rounds = []
        for k, (seq, pos0, toks, job) in enumerate(parts):
            row = e.logits_row(idx[k])
            if job.verifying:
                rounds.append(self._finish_round(job, first[k]))
                continue
            token = int(row.argmax())
            if job.state == PREFILL:
                job.cursor += len(toks)
                job.pos = pos0 + len(toks)
                job.cur.extend(toks)
                job.prefilled += len(toks)
                # Charged here, where the tokens have actually been forwarded,
                # rather than in _plan where they were merely scheduled: a part
                # that never reaches this loop (a failed decode, a job dropped
                # from the batch) must not report progress it did not make.
                if job.prog is not None:
                    job.prog.add(len(toks))
                if (job.arc_at is not None and not job.arc_done
                        and job.cursor >= job.arc_at):
                    self.e.arc_put(list(job.cur), job.seq)
                    job.arc_done = True
                if job.cursor >= len(job.prompt):
                    job.state = DECODE
                    job.out.append(token)
                    self._emit(job, token)
                    if job.state != DONE:
                        # The first row is a decode row like any other, so this
                        # is where a reply starts feeding the draft index.
                        self._index_and_offer(job, row, token)
            else:
                job.pos = pos0 + 1
                job.cur.append(toks[0])
                job.out.append(token)
                self._emit(job, token)
                if job.state != DONE and job.pos >= self.e.window:
                    # The sequence is full.  Stop this reply here: the next step
                    # would decode at a position past the cells, which fails the
                    # whole batch and every other request sharing it, rather than
                    # just truncating this one.
                    job.stopped = None
                    job.seal()
                    job.finish()
                elif job.state != DONE:
                    self._index_and_offer(job, row, token)

        # Split the step's wall time across its parts by token count.  A prefill
        # chunk is up to a hundred tokens and a decode is one, so token share is
        # what the step's cost tracks; charging the whole step to each part would
        # report a decode pinned behind a prefill as prefill time.
        dt = time.perf_counter() - t_step
        if n_tok:
            n_pre = 0
            for pre, (_, _, toks, job) in zip(was_prefill, parts):
                share = dt * len(toks) / n_tok
                if pre:
                    job.pf_s += share
                    n_pre += len(toks)
                else:
                    job.gn_s += share
            # Keep the engine-level counters meaningful too: the batched path
            # never moved them, so every batched request logged a split of
            # "prefill 0.0s + decode 0.0s".
            self.e.st["t_pre"] += dt * n_pre / n_tok
            self.e.st["t_gen"] += dt * (n_tok - n_pre) / n_tok
            # The governor's plain-decoding yardstick.  A batched decode step is
            # exactly one token decoded one at a time, which is what it measures
            # -- charged its share of the step, the same way the job's own
            # counters are.
            for pre, (_, _, toks, job) in zip(was_prefill, parts):
                if not pre and not job.verifying:
                    spec.note_sequential(1, dt * len(toks) / n_tok)
            for draft_len, committed in rounds:
                spec.note_round(draft_len, committed,
                                dt * draft_len / n_tok)

    def _emit(self, job, token):
        """Append a token's text and apply the stop strings.

        Stopping is decided on TEXT, exactly as Eng.gen does it: a stop string
        can straddle tokens, and trimming at the token level would either emit
        half a marker or swallow a token that is not part of one.

        The token's BYTES go through an incremental decoder rather than being
        decoded on their own.  A token is a byte string, not a character -- a
        4-byte emoji is three tokens here, and decoding each one separately
        yields three U+FFFD that no amount of joining can undo.  The decoder
        also buffers a trailing partial character, so what reaches on_token is
        always whole characters and the stream cannot carry a replacement
        character either.
        """
        piece = self.e.piece(token)
        chunk = job.dec.decode(piece)
        job.raw.append(chunk)
        if job.on_token is not None and chunk:
            # Hand over the increment and let the callback decide what is safe to
            # send: it knows about think blocks, tool calls and stop strings, and
            # this function deliberately does not.
            job.on_token(chunk)
        if len(job.out) >= job.max_new:
            job.stopped = None
            job.seal()
            job.finish()
            return
        # Rebuilt from the piece list each time, which is O(n^2) in the reply
        # length -- but a reply is tens to hundreds of tokens, and this keeps the
        # text exactly what the pieces say.
        job.text = "".join(job.raw)
        cut = len(job.text)
        for s in job.stop:
            i = job.text.find(s)
            if 0 <= i < cut:
                cut, job.stopped = i, s
        if job.stopped is not None:
            job.text = job.text[:cut]
            job.finish()

    def _loop(self):
        while not self._stop:
            self._retire()
            self._admit()
            live = [j for j in self.jobs if j.state != DONE]
            if not live:
                self.jobs = []
                self.wake.clear()
                self.wake.wait(0.05)
                continue
            parts = []
            try:
                parts = self._plan()
                if parts:
                    self._advance(parts)
            except BaseException as ex:           # never take the scheduler down
                # Drop the KV of every sequence in the failed batch.  A decode
                # that fails does not retire its job, and only _retire writes
                # seq_tokens, so those sequences hold tokens the ledger has never
                # heard of.  Keeping them is what turns one bad batch into a
                # permanent failure: the next admission reads the stale ledger,
                # believes a prefix is reusable, and decodes from the wrong
                # position.  Re-prefilling costs a request; not clearing costs
                # every request after it.
                for seq in {s for s, _, _, _ in parts}:
                    try:
                        self.e.clear_seq(seq)
                    except Exception:
                        pass
                # The scratches in that set were cleared with everything else, so
                # all that is left is to hand them back -- dropping the job list
                # below would otherwise strand every one of them.
                for j in live:
                    self._ddc_release(j)
                self.st["failed"] = self.st.get("failed", 0) + 1
                self.log(f"batch failed ({ex!r}); cleared sequences "
                         f"{sorted({s for s, _, _, _ in parts})}, jobs lost: "
                         f"{len(parts)}")
                for j in live:
                    j.error = ex
                    j.finish()
                self.jobs = []
                continue


    def line(self) -> str:
        s = self.st
        return (f"steps={s['steps']} batched_tok={s['batched_tokens']} "
                f"jobs={s['jobs']} admitted={s['admitted']} peak={s['peak_seq']}")


def run_job(engine, batcher, prompt, max_new, stop, turn=None, on_token=None,
            on_serve=None):
    """Serve one request: the KV-free layers first, then the batcher.

    This is the whole serving path.  A request that any exact layer can answer
    never reaches a sequence, which is what makes the batcher's slots go to the
    requests that need them.
    """
    from eng import probe

    e = engine
    st = e.st
    st["req"] += 1
    stopk = tuple(stop or ())
    with batcher.state_lock:
        # on_token is passed through: the serial path already does (eng.py, in
        # gen), and without it a hit on any of these three layers emits nothing
        # while "generating", so the whole reply lands in one block at the end
        # instead of streaming.  Same reply either way, different experience.
        hit = probe(e, prompt, max_new, stop or [], stopk, turn, on_token, on_serve)
    if hit is not None:
        return hit
    job = batcher.submit(prompt, max_new, stop or [], turn, on_token, on_serve)
    if not job.done.wait(timeout=1800):
        raise TimeoutError("batcher did not finish the job")
    if job.error is not None:
        raise job.error
    job.hit = job.hit or "batch"
    # pre is what was actually forwarded; prefilled counts the reused prefix
    # too, so subtracting it here would report zero for every request.
    info = {"hit": job.hit, "pre": job.prefilled - job.reused,
            "reuse": job.reused, "cur": len(job.cur),
            "stop": job.stopped, "trunc": job.stopped is None,
            "seq": job.seq, "pf": job.pf_s, "gn": job.gn_s,
            "arc": job.arc_s, "shared": job.shared}
    return job.out, job.text, info
