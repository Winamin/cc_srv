# -*- coding: utf-8 -*-
"""Entry point. Usage:

    python run.py --port 8788 --n-ctx 131072

Environment variables for pointing Claude Code at it (note: the env block in
~/.claude/settings.json outranks process environment variables, so if the command
line doesn't work, pass an override config via --settings):

    ANTHROPIC_BASE_URL = http://127.0.0.1:8788
    ANTHROPIC_AUTH_TOKEN = local
    ANTHROPIC_MODEL = qwythos-9b

Debug switches:
    CC_LOGDIR=<dir> where the request log and server log go (default ./logs)
    CC_DUMP=<path>  append each request's prompt/gen to this file
    CC_KEEP_CNT=1   turn off counter normalization (for A/B comparison)
    CC_CNT_N=N      the constant the token counter is pinned to (default 15000000)
    CC_KEEP_AH=1    turn off assistant-header normalization (for A/B comparison)
    CC_TOOLS_HINT=en  use the English tool-call hint instead of the original
                    Chinese one.  That hint is prompt content, not a comment, so
                    changing it changes the token ids of every tool-bearing
                    request and invalidates the recorded fixtures.

Turn-keyed reuse (experimental, off by default; see USAGE.md section 3):
    CC_QREUSE=0      off
    CC_QREUSE=1      reuse the reply to the same user turn, same dialog (the
                     recorded prompt must be a strict prefix of the incoming one)
    CC_QREUSE=2      also reuse a similar turn from any dialog; requires CC_QEDIT.
                     Measured to be dangerous at a loose similarity floor: Claude
                     Code questions share so much boilerplate that two different
                     questions score ~0.65, and at the original 0.6 floor this
                     fired on 10 of 18 requests of a live dialog and degenerated
                     it.  Level 2 also refuses a difference that admits no
                     substitution at all.
    CC_QEDIT=1       keep a top-k logits trace per generated token, replay it
                     through argmax, and apply a server-side edit whose rewritten
                     substitutions must appear in the model's own cached top-k.
                     Deletions, insertions and a grown replacement's tail have no
                     cached row and are written unchecked.  Costs about 8% of
                     decode, and is only paid at level 2.
    CC_QEDIT_TOPK=16 top-k width stored per position
    CC_QEDIT_SIM=0.9 similarity floor for a level-2 turn
    CC_QREUSE_CAP=512  how many replies the turn-keyed cache holds
    CC_QREUSE_TOOL=1   also reuse the reply to a tool result.  Refused by default:
                     such a reply usually contains the tool call that produced the
                     result, so replaying it re-issues the call and the dialog
                     loops.

Batched serving (experimental, off by default; see USAGE.md section 6.5):
    CC_BATCH=1      run concurrent requests through one scheduler, advancing
                    every active sequence by one step inside a single
                    llama_decode.  Measured 3.3x the decode throughput of the
                    single-sequence path.  It is opt-in for two reasons: it
                    multiplies KV reservation by CC_BATCH_N, and sharing a batch
                    changes the graph shape, so a reply can differ from what the
                    single-sequence path would have produced (deterministically
                    so -- measured at about half of prompts diverging somewhere,
                    which is the engine's documented batch-vs-sequential
                    numerics, not a race).
    CC_BATCH_N=4    sequences, i.e. how many requests can be in flight at once

Prefix archive (experimental, off by default; see USAGE.md section 6.4):
    CC_ARCHIVE=1    keep whole KV states at the points where requests have
                    actually diverged, in spare sequences, and restore one
                    wholesale when a later request asks for it.  This is what
                    makes a fan-out workload reusable: sibling subagents share a
                    long prefix but are not extensions of each other, so without
                    it every sibling rebuilds its whole context.  Measured at 65x
                    less prefill on siblings, with an identical reply.  It needs
                    a spare sequence, so n_ctx doubles (KV 1.2 GB -> 2.4 GB at
                    131072) -- that is the cost, and it is why this is opt-in.
    CC_ARCHIVE_MIN=512  shortest prefix worth archiving

Speculation gate (see spec.py): a speculative round costs a verification pass
over the whole draft plus, unless it is accepted in full, a second pass to
refeed -- so short drafts that are only partly accepted cost two forwards and
return about one token.  The gate buckets rounds by draft length and measures
each bucket against plain decoding, dropping the lengths that do not pay and
re-probing them after a cooldown.  It is what makes the aggressive draft
settings above safe.
    CC_SPEC_GOV=0        disable the gate (speculate whenever a draft exists)
    CC_SPEC_MIN=12       rounds of evidence before a bucket is judged
    CC_SPEC_COOLDOWN=64  rounds before a dropped bucket is re-probed (doubles)

The gate has a second half, which decides whether to look for a draft at all.
Indexing one costs a full-vocabulary scan per committed token -- measured at 7.9%
of generation time -- and pays off only where drafts land, so DDC is stood down
on content that is not repeating and switched back on the moment it is.  The
signal is a rolling n-gram over the generated token stream: a tuple hash per
token, and it keeps running while DDC is stood down.
    CC_SPEC_REP_N=32         n-gram length scored per token
    CC_SPEC_REP_WINDOW=4096  how far back an n-gram may match
    CC_SPEC_REP_MIN=0.15     repeat-rate floor; below this DDC stands down
    CC_SPEC_REP_TAU=32       EWMA time constant for the rate, in tokens
    CC_SPEC_REP_GRACE=128    tokens of every generation during which DDC is
                    never stood down.  A repetitive request always BEGINS novel
                    -- DDC drafts the second copy from the first, so the first
                    copy has to be indexed before the repetition is visible at
                    all -- and judging inside that window would stand down
                    exactly the requests that pay.  This is the bootstrap cost,
                    and it is why a short generation saves nothing.

    Measured on three prompts (two diverse, one repetitive), tokens/sec:
        DDC off                 69.2  69.8   69.6
        DDC on, no detector     58.1  59.0  128.8
        DDC on, detector        66.6  67.9  141.5
    The ~13% diverse penalty becomes ~3% (the grace window is the rest), and the
    repetitive win survives at 2.0x.  A detector measuring DDC's own output was
    tried first and failed outright: it reported 0 speculative rounds everywhere
    and destroyed the win, because the scan it was trying to save is also what
    builds the index, so standing down froze the index that produced the
    evidence for standing down.

DDC draft cache (on by default; data in DDC_EXPERIMENT.md):
    CC_DDC=0        turn it off
    CC_DDC=2|4      on, with that n-gram key width (default 4)
                    Look up a historical trajectory by ranking fingerprint to
                    produce a draft, batch-verify it against the target model;
                    automatically raises nseq>=2. ~1.5x on repeated code edits in
                    a small context, switches to per-token once the context
                    threshold is reached, no promise that short segments match the
                    default path bit for bit.  It yields to CC_ARCHIVE=1, which
                    needs the same spare sequences; setting both by name is
                    refused.
    CC_DDC_M=32     max draft length (the cap; T is what usually binds)
    CC_DDC_T=0.5    margin truncation threshold -- LOWER means LONGER drafts
                    (0 = fixed-length drafts).  The round's rate rises
                    monotonically with draft length, so this is set aggressive
                    on purpose; spec.py drops the lengths that stop paying.
                    At M=32/T=0.5 drafts land at 15-22 tokens per round.
    CC_DDC_GATE=0   admission gate: positions whose top-k boundary margin is
                    insufficient don't enter the index
    CC_DDC_RESET=1  clear the DDC draft before each actual generation, rebuild
                    only from this request's committed tokens
    CC_DDC_MAX_CTX=32768  disable DDC once prompt+generated tokens reach this
                    length, 0 means unlimited
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srv import main

if __name__ == "__main__":
    main()
