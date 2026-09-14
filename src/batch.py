# -*- coding: utf-8 -*-
"""Batched generation: many requests, one forward pass per step.

Why this exists
---------------
Measured on the agent workload, the server was 97 % busy and the dialog was
waiting on generation, not on caching: solving the time budget out of two runs
gives ~3 380 tok/s of prefill against ~38.5 tok/s of decode, with decode 64 % of
the server's time.  Five subagents issuing requests at once did not overlap --
they queued behind one lock on one sequence, and each step read the whole 5.5 GB
of weights to produce a single token.

A 9B Q4 model's decode step is memory-bandwidth bound: the weights are read once
per step no matter how many tokens the step contains.  So the same read can
advance several sequences at once.  Measured here with three sequences in one
batch: 15.9 ms/token each when run one after another, 6.7 ms/token each when run
together -- and the probe verifies each sequence still produces exactly the
tokens it would have produced alone.

What this module is
-------------------
A scheduler.  It owns the context, hands each active request its own sequence,
advances every sequence that has work by one step, decodes all of them in a
single llama_decode, and hands each sequence back the logits row that belongs to
it.

The KV-free cache layers stay outside it: ``Eng.probe`` answers the requests that
need no forward pass at all before they ever reach a sequence, which is what
keeps slots free for the requests that do.

Correctness
-----------
Batching is only sound if a sequence's output does not depend on which other
sequences shared its batches.  That is a property of the engine, not a hope, and
``batch_probe.py`` measures it: sequences run in lockstep must produce exactly
the tokens they produce alone.  Everything here assumes that has been checked.
"""
from __future__ import annotations

import os
import threading
import time

# A step is only batched together with the steps either side of it, so the
# scheduler needs to know nothing about prompts or stops beyond what a job
# carries.
PREFILL = "prefill"
DECODE = "decode"
DONE = "done"


class Job:
    """One request in flight."""

    __slots__ = ("prompt", "max_new", "stop", "turn", "seq", "cur", "pos",
                 "out", "raw", "text", "hit", "state", "cursor", "stopped",
                 "done", "result", "error", "t0", "prefilled", "reused",
                 "arc_at", "arc_done", "on_token", "pf_s", "gn_s")

    def __init__(self, prompt, max_new, stop, turn=None, on_token=None):
        # Called with each generated token's text as it is produced, so a caller
        # can stream.  None means the reply is only available at the end.
        self.on_token = on_token
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
        # The position where this prompt diverged from what its sequence held.
        # The state there is the one a sibling branch will ask for, and the only
        # moment it exists is while prefilling through it.
        self.arc_at = None
        self.arc_done = False
        self.stopped = None
        # Wall time this request spent prefilling vs decoding.  Accumulated per
        # job because a batch serves several at once: one llama_decode covers
        # every part, so there is no separate clock to read and the engine-level
        # counters cannot tell one request's prefill from its neighbours'.
        self.pf_s = 0.0
        self.gn_s = 0.0
        self.done = threading.Event()
        self.result = None
        self.error = None
        self.t0 = time.time()

    def finish(self):
        self.state = DONE
        self.done.set()


class Batcher:
    """Runs several requests' forward passes as one.

    A single scheduler thread owns the context.  Callers submit a job and wait on
    its event; they never touch the context themselves.
    """

    def __init__(self, eng, chunk=None, log=print, start=True):
        self.e = eng
        self.log = log
        # Prefill is chunked so that a long prompt does not monopolise a step:
        # chunking is what lets a request that is still prefilling share the
        # batch with requests that are already decoding.
        self.chunk = chunk or max(64, eng.bs // 4)
        # How much better the busy home worker's prefix must be before it is
        # worth waiting for.  Waiting is not free: in a fan-out the home worker
        # is often mid-generation, and a free worker holding almost as long a
        # prefix should be taken instead.  Zero would mean "always wait for the
        # longest"; a huge value would mean "never wait".
        self.wait_margin = int(os.environ.get("CC_BATCH_WAIT", "512"))
        # The archive keeps its states in sequences of its own.  Handing work to
        # one of those would overwrite an archived state mid-flight, so the
        # scheduler only ever uses the sequences the engine says are free.
        self.seqs = list(getattr(eng, "work_seqs", None) or range(eng.nseq))
        self.jobs: list[Job] = []
        self.queued: list[Job] = []
        self.lock = threading.Lock()
        # Guards the KV-free caches (lgc / ridx / qcache), which request threads
        # read through Eng.probe while the scheduler thread writes them when a
        # job is retired.  They are cheap and hit rarely enough that one lock
        # around them costs nothing measurable.
        #
        # Eng.gen writes those same caches WITHOUT this lock, so the two serving
        # paths must not be mixed.  srv.py picks one at startup -- run_job when a
        # scheduler exists, Eng.gen otherwise -- and never both, which is what
        # makes that safe.  Anything that starts calling Eng.gen while the
        # scheduler is live would race on ridx.
        self.state_lock = threading.Lock()
        self.wake = threading.Event()
        self.st = {"steps": 0, "batched_tokens": 0, "jobs": 0, "admitted": 0,
                   "peak_seq": 0}
        self._stop = False
        # start=False lets the pure-Python scheduling logic be tested without a
        # GPU: the state machine is where the interesting mistakes live, and it
        # needs no model to exercise.
        self.thread = None
        if start:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    # ---------------- public ----------------
    def submit(self, prompt, max_new, stop, turn=None, on_token=None) -> Job:
        job = Job(prompt, max_new, stop, turn, on_token)
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

        Three different positions matter here and they are easy to conflate:

        * ``usable`` -- how much of this prompt a sequence already holds as a
          strict prefix.  That is the only part that can be reused in place: a
          sequence holding MORE than the prompt would need its tail truncated
          away, and a partial truncation is a no-op on this hybrid memory, so
          such a sequence is cleared instead.
        * ``diverged`` -- how far this prompt AGREES with what the sequence
          holds, even when that cannot be reused.  That boundary is exactly
          where a sibling branch will diverge too, so it is the state worth
          archiving, and the prefill is asked to stop there on the way past.
          It is not the same number as ``usable``, and using one for the other
          is why an earlier version archived nothing at all.
        * whatever the archive can offer, which is the only source of reuse
          between two sibling subagents that share a prefix but are not
          extensions of one another.
        """
        with self.lock:
            pending, self.queued = self.queued, []
        for job in pending:
            # A prompt that cannot fit one sequence must fail here, with a
            # reason, rather than reach llama_decode.  A failed decode is not
            # contained: it does not retire its job, only _retire writes
            # seq_tokens, so the sequence is left holding tokens the ledger does
            # not know about -- and the next request placed on it computes its
            # reusable prefix from that stale ledger and decodes from the wrong
            # position.  One over-long prompt would take down every request
            # after it.
            if len(job.prompt) > self.e.window:
                job.error = ValueError(
                    f"prompt is {len(job.prompt)} tokens but a sequence holds "
                    f"{self.e.window} (n_ctx / nseq); raise --n-ctx")
                job.finish()
                continue
            busy = {j.seq for j in self.jobs if j.seq is not None}
            free = [s for s in self.seqs if s not in busy]

            # Search EVERY worker, not just the free ones.
            #
            # The worker holding this prompt's history is the one worth using,
            # and in a fan-out dialog it is usually busy -- a subagent is
            # generating on it.  An earlier version looked only at free workers,
            # so a request whose history sat on a busy one took a cold worker
            # and paid a full prefill.  Measured: capture fell to 0.160, below
            # the 0.241 of the single-sequence server it was meant to beat, and
            # the archive could not cover it because it only had one slot.
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
                # A free worker holds nearly as much of this prompt as the busy
                # one does, so take it rather than block.  Always waiting for the
                # longest prefix looks principled and is not: the home worker can
                # be mid-generation for a long time, and a cold-ish worker that
                # saves all but the last chunk beats an idle caller.
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

            # Nothing on a sequence fits.  An archived state might: that is what
            # the archive is for, and it is what a sibling subagent needs.
            if self.e.arc is not None and usable < self.e.arc_min:
                hit = self.e.arc_find(job.prompt)
                if hit is not None and hit[1] > usable:
                    self.e.arc_restore(hit[0], best)
                    self.e.st["arc_hit"] = self.e.st.get("arc_hit", 0) + 1
                    self._start(job, hit[1], "arc")
                    continue

            # A sequence has to be cleared before it can be decoded into, even a
            # fresh one: one that has never been through seq_rm makes
            # llama_decode fail with "failed to initialize batch", which reads
            # like a malformed batch rather than an uninitialised sequence.
            had = len(cur)
            if usable == 0 or usable < had:
                self.e.clear_seq(best)
            if self.e.arc is not None:
                # Archive at the start of the trailing user turn: that is the
                # boundary every sibling of this request shares, and the state
                # there is what they will all ask for.  Divergence from the
                # worker's own contents is not that boundary -- in a fan-out the
                # workers hold unrelated branches, so it is small and the archive
                # would never learn where the shared part ends.
                at = self.e.turn_start(job.prompt)
                if at is None or at <= usable:
                    at = diverged if diverged > usable else None
                if at is not None and self.e.arc_ready(at):
                    job.arc_at = at
            self._start(job, usable, "apc")

    def _start(self, job, reused, how):
        """Put a job into prefill with ``reused`` tokens already in its sequence."""
        if reused >= len(job.prompt):
            # The whole prompt is already in the sequence -- a worker happened to
            # hold it, or an archived state restored to exactly its length.
            # A decode step still needs a FRESH logits row for the last
            # position, and an inherited or restored KV does not come with one
            # (the logits buffer belongs to whatever decoded last), so the final
            # token is forwarded again.  Rewriting a position with the token that
            # is already there is safe, and one duplicated token costs nothing
            # beside a full prefill.
            #
            # Leaving it at len(prompt) was a crash: the cursor already sat at
            # the end, the prefill had nothing to forward, and the job was
            # flipped to DECODE without a first token -- so the next step indexed
            # out[-1] on an empty list.  The handler for that catches
            # BaseException and clears every job in flight, so one narrow case
            # would silently poison a whole run.
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
        # Report which layer actually served the prefix.  Every job used to say
        # "batch", which made the hit histogram useless for telling an archive
        # restore from an in-sequence reuse from a cold start -- and those are
        # exactly the three outcomes the archive work is trying to move between.
        job.hit = ("batch_arc" if how == "arc"
                   else "batch_apc" if reused else "batch_cold")
        self.jobs.append(job)
        self.st["admitted"] += 1
        self.st[how] = self.st.get(how, 0) + 1

    def _seq_tokens(self, seq):
        """Tokens the KV for ``seq`` holds, as far as the engine's ledger knows."""
        for j in self.jobs:
            if j.seq == seq:
                return j.cur
        return self.e.seq_tokens.get(seq, [])

    def _retire(self):
        """Hand finished jobs back and free their sequences.

        Registering the trajectory happens here, on the scheduler thread, which
        is why the KV-free caches are guarded: a request thread may be reading
        them through Eng.probe at the same moment.
        """
        for j in list(self.jobs):
            if j.state != DONE:
                continue
            self.jobs.remove(j)
            self.e.seq_tokens[j.seq] = list(j.cur)
            with self.state_lock:
                if j.out:
                    self.e.reg(j.prompt, j.out, j.text, tuple(j.stop))
            self.e.st["gen"] += len(j.out)

    def _plan(self):
        """One step's worth of work: a list of (seq, pos0, tokens, job) parts.

        The budget is shared out in two passes, and the order is the whole point:

        1. every sequence that is DECODING gets its one token, then
        2. what is left is split between the sequences that are PREFILLING.

        Draining the budget in one pass -- prefill first, in job order -- looked
        reasonable and was badly wrong.  A request arriving with a 20 000-token
        prompt would spend the entire step budget on its own prefill chunk, step
        after step, and the sequences already generating got no token at all
        until it finished.  Measured on a live dialog: p90 latency 199 s, and a
        request that should have taken seconds took three minutes.  A decode
        needs exactly one token and a caller is blocked on it; a prefill chunk is
        arbitrary and can always be smaller, so prefill yields.
        """
        live = [j for j in self.jobs if j.state != DONE]
        if not live:
            return []
        budget = self.e.bs
        parts = []

        decoding = [j for j in live if j.state == DECODE]
        for job in decoding:
            if budget < 1:
                break
            parts.append((job.seq, job.pos, [job.out[-1]], job))
            budget -= 1

        prefilling = [j for j in live if j.state == PREFILL]
        if prefilling and budget > 0:
            share = max(1, budget // len(prefilling))
            for job in prefilling:
                if budget <= 0:
                    break
                take = min(self.chunk, share, budget, len(job.prompt) - job.cursor)
                if job.arc_at is not None and not job.arc_done:
                    # Land the chunk exactly on the boundary so the state there
                    # can be copied out; overshooting it loses the boundary for
                    # good, because the KV can never come back.
                    take = min(take, max(1, job.arc_at - job.cursor))
                take = max(1, take)
                toks = job.prompt[job.cursor:job.cursor + take]
                if not toks:
                    # Nothing left to forward.  That is only a legitimate
                    # transition when there is a token to decode FROM; a job with
                    # an empty output would make the next step index out[-1] on
                    # nothing and take the whole batch down with it.
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
        b, keep, idx = e.batch_parts([(s, p, t) for s, p, t, _ in parts])
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
                    bb, kk, ii = e.batch_parts([(s, p, t)])
                    print(f"  alone seq={s} pos={p} n={len(t)} -> "
                          f"rc={L.llama_decode(e.ctx, bb)}", flush=True)
            raise RuntimeError("batched decode failed")
        _ = keep
        self.st["steps"] += 1
        self.st["batched_tokens"] += b.n_tokens
        self.st["peak_seq"] = max(self.st["peak_seq"], len(parts))
        # Captured before the loop below, which flips a finished prefill to
        # DECODE -- reading state afterwards would file the last chunk of every
        # prompt as decode time.
        was_prefill = [job.state == PREFILL for _, _, _, job in parts]
        n_tok = sum(len(t) for _, _, t, _ in parts)
        for k, (seq, pos0, toks, job) in enumerate(parts):
            row = e.logits_row(idx[k])
            token = int(row.argmax())
            if job.state == PREFILL:
                job.cursor += len(toks)
                job.pos = pos0 + len(toks)
                job.cur.extend(toks)
                job.prefilled += len(toks)
                if (job.arc_at is not None and not job.arc_done
                        and job.cursor >= job.arc_at):
                    self.e.arc_put(list(job.cur), job.seq)
                    job.arc_done = True
                if job.cursor >= len(job.prompt):
                    job.state = DECODE
                    job.out.append(token)
                    self._emit(job, token)
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
                    job.text = "".join(job.raw)
                    job.finish()

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

    def _emit(self, job, token):
        """Append a token's text and apply the stop strings.

        Stopping is decided on TEXT, exactly as Eng.gen does it: a stop string
        can straddle tokens, and trimming at the token level would either emit
        half a marker or swallow a token that is not part of one.
        """
        piece = self.e.piece(token)
        job.raw.append(piece)
        if job.on_token is not None:
            # Hand over the increment and let the callback decide what is safe to
            # send: it knows about think blocks, tool calls and stop strings, and
            # this function deliberately does not.
            job.on_token(piece)
        if len(job.out) >= job.max_new:
            job.stopped = None
            job.text = "".join(job.raw)
            job.finish()
            return
        # Rebuilding from the piece list each time is O(n^2) in the reply length,
        # but a reply is tens to hundreds of tokens and this keeps the text
        # exactly what the pieces say -- including a multi-byte character split
        # across two tokens, which decoding token by token would corrupt.
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


def run_job(engine, batcher, prompt, max_new, stop, turn=None, on_token=None):
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
        hit = probe(e, prompt, max_new, stop or [], stopk, turn)
    if hit is not None:
        return hit
    job = batcher.submit(prompt, max_new, stop or [], turn, on_token)
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
            "seq": job.seq, "pf": job.pf_s, "gn": job.gn_s}
    return job.out, job.text, info
