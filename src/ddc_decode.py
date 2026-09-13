"""Experimental target-verified DDC decoder; no HTTP/default-engine changes.

Batch and sequential decoding can differ numerically on this quantized hybrid
model. Verification is against the target's batch logits, not a proof of bitwise
equivalence to sequential decoding. A partial acceptance always refreshes logits
from the committed sequence; speculative logits cannot survive a refeed.
"""
import ctypes as ct
import time

import numpy as np

import spec
from ddc import features


def token_bytes(e, token):
    buf = ct.create_string_buffer(256)
    n = e.L.llama_token_to_piece(e.vcb, token, buf, len(buf), 0, True)
    if n < 0:
        buf = ct.create_string_buffer(-n)
        n = e.L.llama_token_to_piece(e.vcb, token, buf, len(buf), 0, True)
    return buf.raw[:n]


def generate(e, prompt, max_new, cache=None, stop=("<|im_end|>",), prepared_logits=None,
             max_context=0):
    """Return tokens, UTF-8 text, and synchronized wall-time/call counters.

    A cache contains only committed past tokens. Sequence 1 is scratch space;
    whole-state copies are used because hybrid recurrence cannot be truncated.
    Timing includes lookups, feature extraction, copies, refeeding and decoding.
    max_context disables DDC at that many committed prompt+output tokens (0
    means unlimited). Drafts cannot cross the boundary. The caller's cache is
    retained for subsequent short requests; long requests do not update it.
    """
    if max_context < 0:
        raise ValueError("max_context must be nonnegative")
    disabled_at = None
    if cache is not None and max_context and len(prompt) >= max_context:
        cache = None
        disabled_at = len(prompt)
    if cache is not None and e.nseq < 2:
        raise ValueError("DDC verification requires nseq >= 2")
    if max_new < 0 or not prompt:
        raise ValueError("Need a nonempty prompt and nonnegative token budget")
    # The repetition detector's grace window is per generation, so it restarts
    # here rather than only when the cache does -- a request that DDC skips
    # entirely still has to not be judged on the previous request's content.
    spec.begin_request()
    lib, mem = e.L, e.kmem
    stats = dict(tokens=0, calls=0, spec=0, full=0, acc1=0, accepted=0,
                 verified_positions=0, refeed_positions=0, cross_spec=0,
                 copy_seconds=0., batch_seconds=0., refeed_seconds=0.,
                 plain_seconds=0., verify_scan_seconds=0.,
                 cross_accepted=0, feature_seconds=0., prefill_seconds=0.,
                 generation_seconds=0., reused=0, stop=None,
                 cache_tokens_discarded=0, cache_tokens_at_start=0,
                 ddc_max_context=max_context, ddc_disabled_at=disabled_at)
    if prepared_logits is None:
        started = time.perf_counter()
        p = e.lcp(prompt)
        if p < len(e.cur) and not e.trunc(p):
            p = 0
        e.feed(prompt[p:])
        # Read the last output, also when the prompt already equals the current
        # prefix and the preceding call left multiple verification outputs.
        # The getter synchronizes pending work before timing prefill ends.
        live = np.ctypeslib.as_array(lib.llama_get_logits_ith(e.ctx, -1), shape=(e.nv,))
        stats.update(prefill_seconds=time.perf_counter()-started, reused=p)
    else:
        # Benchmark-only: caller restored an identical whole-state snapshot.
        if e.cur != prompt or e.kvlen() != len(prompt):
            raise ValueError("Prepared logits require the matching restored prefix")
        live = prepared_logits
        stats["reused"] = len(prompt)
    tokens, raw = [], bytearray()
    markers = [(x, x.encode("utf-8")) for x in stop if x]
    stopped = None

    def describe(logits):
        begin = time.perf_counter()
        result = features(logits)
        stats["feature_seconds"] += time.perf_counter()-begin
        return result

    def row(i):
        return np.ctypeslib.as_array(lib.llama_get_logits_ith(e.ctx, i), shape=(e.nv,))

    def decode_many(ids, sequence, position):
        batch, keep = e.batch(ids, sequence, position)
        if lib.llama_decode(e.ctx, batch) != 0:
            raise RuntimeError("DDC target decode failed")
        stats["calls"] += 1

    started = time.perf_counter()
    if cache is not None:
        stats["cache_tokens_discarded"] = cache.size if cache.reset_per_request else 0
        cache.begin()  # Include deleting/rebuilding draft bookkeeping in timing.
        stats["cache_tokens_at_start"] = cache.size
    while len(tokens) < max_new and stopped is None:
        # Per-round timing, so the governor has a plain-decoding reference even
        # when no request ever bypasses DDC.  Without this the only sequential
        # sample came from requests long enough to skip the cache, and a session
        # whose prompts all sit below that threshold would never be judged at
        # all -- leaving speculation on at 0.712x, which is the case this exists
        # to catch.
        round_started = time.perf_counter()
        pos = len(e.cur)
        if cache is not None and max_context and pos >= max_context:
            cache = None
            stats["ddc_disabled_at"] = pos
        # Repetition detector.  If recent rounds were offered no draft -- or one
        # or two tokens -- the content is not repeating, and every further round
        # would pay the full-vocabulary feature scan for nothing.  Stand down
        # BEFORE the scan rather than refusing the proposal after it: refusing
        # later saves the verification but not the scan, which is the fixed cost.
        # The statistics are global, so a workload that turns repetitive comes
        # back on its own through the cooldown re-probe.
        if cache is not None and not spec.paying():
            cache = None
            stats["stood_down"] = stats.get("stood_down", 0) + 1
        nxt = int(live.argmax())
        # Time the DDC-only work so the plain-decoding yardstick below can be
        # charged for it.  Without this the no-draft round -- which ran describe()
        # and propose() and then returned one token -- is reported as a clean
        # plain sample and drags _seq down, making every bucket look better than
        # it is and the gate more lenient than intended.
        overhead_started = time.perf_counter()
        first = describe(live) if cache is not None else None
        room = max_new-len(tokens)
        if cache is not None and max_context:
            room = min(room, max_context-pos)
        if cache is not None:
            proposal = cache.propose(first[0], nxt, room)
            # Score the round by what it was offered, 0 included.  A round that
            # found nothing is the one measurement the bucket gate never sees.
            spec.note_offer(len(proposal.tokens) if proposal is not None else 0)
            overhead = time.perf_counter() - overhead_started
        else:
            proposal = None
            overhead = 0.0
        # Per-round gate.  A round pays a verification pass over the whole draft
        # and, unless it is accepted in full, a second pass to refeed -- so a
        # short draft that is only partly accepted costs two forwards and
        # returns about one token.  The gate refuses the draft lengths that have
        # measured that way, and only those: the lengths that land keep their
        # ~3x.  Refusing here turns the round into an ordinary decode step,
        # which is also what supplies the plain rate the gate compares against.
        if proposal is not None and not spec.allow(len(proposal.tokens)):
            stats["gated"] = stats.get("gated", 0) + 1
            proposal = None
        if proposal is None:
            accepted = [nxt]
        else:
            draft = list(proposal.tokens)
            t_copy = time.perf_counter()
            lib.llama_memory_seq_rm(mem, 1, 0, -1)
            lib.llama_memory_seq_cp(mem, 0, 1, 0, -1)
            stats["copy_seconds"] += time.perf_counter() - t_copy
            try:
                t_batch = time.perf_counter()
                decode_many(draft, 1, pos)
                stats["batch_seconds"] += time.perf_counter() - t_batch
                t_scan = time.perf_counter()
                count = 1
                while count < len(draft) and int(row(count-1).argmax()) == draft[count]:
                    count += 1
                stats["verify_scan_seconds"] += time.perf_counter() - t_scan
                accepted = draft[:count]
            except BaseException:
                lib.llama_memory_seq_rm(mem, 1, 0, -1)
                raise

        # Apply stopping before committing state or indexing a speculative tail.
        pieces = []
        trial = bytes(raw)
        for i, token in enumerate(accepted):
            piece = token_bytes(e, token)
            pieces.append(piece)
            trial += piece
            matches = [(trial.find(marker), name) for name, marker in markers if marker in trial]
            if matches:
                cut, stopped = min(matches)
                accepted = accepted[:i+1]
                break

        if proposal is None:
            t_plain = time.perf_counter()
            e.feed(accepted)
            stats["plain_seconds"] += time.perf_counter() - t_plain
            stats["calls"] += 1
            live = np.ctypeslib.as_array(lib.llama_get_logits(e.ctx), shape=(e.nv,))
            descriptions = [first] if cache is not None else []
        else:
            count = len(accepted)
            full = count == len(draft)
            stats["spec"] += 1
            stats["full"] += full
            stats["acc1"] += count == 1
            stats["accepted"] += count
            stats["verified_positions"] += len(draft)
            stats["cross_spec"] += proposal.cross_request
            stats["cross_accepted"] += count if proposal.cross_request else 0
            if full:
                # Only a full prefix can be moved with the hybrid memory API.
                # This is the whole-state copy the long-context case pays for.
                t_swap = time.perf_counter()
                lib.llama_memory_seq_rm(mem, 0, 0, -1)
                lib.llama_memory_seq_cp(mem, 1, 0, 0, -1)
                lib.llama_memory_seq_rm(mem, 1, 0, -1)
                stats["copy_seconds"] += time.perf_counter() - t_swap
            else:
                t_rm = time.perf_counter()
                lib.llama_memory_seq_rm(mem, 1, 0, -1)
                stats["copy_seconds"] += time.perf_counter() - t_rm
                t_re = time.perf_counter()
                decode_many(accepted, 0, pos)
                stats["refeed_seconds"] += time.perf_counter() - t_re
                stats["refeed_positions"] += count
            e.cur.extend(accepted)
            # Read committed-path features only, before another decode can
            # overwrite the shared logits buffer. No rejected tail is indexed.
            descriptions = [first] + [describe(row(i)) for i in range(count-1)]
            live = row(count-1)
            if lib.llama_memory_seq_pos_max(mem, 1) != -1:
                raise RuntimeError("Scratch sequence was not cleared")

        round_seconds = time.perf_counter() - round_started - overhead
        if proposal is None:
            spec.note_sequential(1, round_seconds)
        else:
            # Bucket by the length we took, credit what it returned.  That is the
            # question the gate asks -- "when I take a draft this long, what do I
            # get back" -- so a long draft that a stop string truncated to one
            # token belongs in the long bucket, where it reads as the bad round
            # it was.
            spec.note_round(len(draft), len(accepted), round_seconds)
        if e.kvlen() != len(e.cur):
            raise RuntimeError("Committed KV length disagrees with token history")
        # Feed the repetition detector every committed token, indexed or not.
        # It has to keep running while DDC is stood down, or it could never see
        # the content turn repetitive and switch back on.
        for token in accepted:
            spec.note_token(token)
        if cache is not None:
            for token, (ids, gaps) in zip(accepted, descriptions):
                cache.append(token, ids, gaps)
        tokens.extend(accepted)
        raw.extend(b"".join(pieces))

    stats.update(tokens=len(tokens), stop=stopped,
                 generation_seconds=time.perf_counter()-started)
    text = bytes(raw)
    if stopped is not None:
        text = text[:text.find(stopped.encode("utf-8"))]
    return tokens, text.decode("utf-8", "replace"), stats
