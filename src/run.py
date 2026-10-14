# -*- coding: utf-8 -*-
"""Entry point. Usage:

    python run.py --port 8788 --n-ctx 131072

Environment variables for pointing Claude Code at it (the env block in
~/.claude/settings.json outranks the process environment, so if the command
line doesn't work, pass an override config via --settings):

    ANTHROPIC_BASE_URL = http://127.0.0.1:8788
    ANTHROPIC_AUTH_TOKEN = local
    ANTHROPIC_MODEL = Your_Model

Debug switches:
    CC_LOGDIR=<dir>        where the request log and server log go (default ./logs)
    CC_DUMP=<path>         append each request's prompt/gen to this file
    CC_KEEP_CNT=1          turn off counter normalization (for A/B comparison)
    CC_CNT_N=N             the constant the token counter is pinned to (default 15000000)
    CC_KEEP_AH=1           turn off assistant-header normalization (for A/B comparison)
    CC_TOOLS_HINT=en       use the English tool-call hint instead of the Chinese one

Turn-keyed reuse (off by default; see qcache.py):
    CC_QREUSE=0            off
    CC_QREUSE=1            reuse the reply to the same user turn, same dialog
    CC_QREUSE=2            also reuse a similar turn from any dialog; requires CC_QEDIT
    CC_QEDIT=1             keep a top-k logits trace per generated token, replay it through argmax, and apply a server-side edit
    CC_QEDIT_TOPK=16       top-k width stored per position
    CC_QEDIT_SIM=0.9       similarity floor for a level-2 turn
    CC_QREUSE_CAP=512      how many replies the turn-keyed cache holds
    CC_QREUSE_TOOL=1       also reuse the reply to a tool result (off by default)

Batched serving (off by default; see batch.py):
    CC_BATCH=1             run concurrent requests through one scheduler, one step each per llama_decode
    CC_BATCH_N=4           sequences, i.e. how many requests can be in flight at once

Chat template (CC_TMPL=2, the default; see chat_template.py):
    CC_TMPL=2              render the model's own tokenizer.chat_template with Jinja, as llama-server does
    CC_TMPL=1              take only the template's system block and render assistant turns with this engine's own renderer
    CC_TMPL=0              library only: llama_chat_apply_template.  No identity message

Thinking (CC_THINK=1, the default):
    CC_THINK=1             the model reasons; the reasoning is delivered as Anthropic thinking blocks
    CC_THINK=0             HARD LOCK: refuses /think?on=1 and the client's thinking field, strips reasoning from history
    GET /think?on=1        live switch (refused while locked)

Prefix archive (on by default; CC_ARCHIVE=0 turns it off):
    CC_ARCHIVE=1           keep whole KV states at the points where requests diverged, in spare sequences, and restore one wholesale
    CC_ARCHIVE=0           turn it off, back to a plain single-sequence server
    CC_ARCHIVE_MIN=512     shortest prefix worth archiving
    CC_ARCHIVE_CAP=N       most tokens one archive slot is charged.  0 charges a whole window

Speculation gate (see spec.py):
    CC_SPEC_GOV=0            disable the gate (speculate whenever a draft exists)
    CC_SPEC_MIN=12           rounds of evidence before a bucket is judged
    CC_SPEC_COOLDOWN=64      rounds before a dropped bucket is re-probed (doubles)
    CC_SPEC_REP_N=32         n-gram length scored per token
    CC_SPEC_REP_WINDOW=4096  how far back an n-gram may match
    CC_SPEC_REP_MIN=0.15     repeat-rate floor; below this DDC stands down
    CC_SPEC_REP_TAU=32       EWMA time constant for the rate, in tokens
    CC_SPEC_REP_GRACE=128    tokens of every generation during which DDC is never stood down

DDC draft cache (off by default; see ddc.py):
    CC_DDC=0               off (the default)
    CC_DDC=2|4             on, with that n-gram key width (4 recommended); drafts from a historical trajectory by ranking fingerprint and batch-verifies them
    CC_DDC_M=32            max draft length
    CC_DDC_T=0.5           margin truncation threshold; LOWER means LONGER drafts
    CC_DDC_GATE=0          admission gate: positions whose top-k boundary margin is insufficient don't enter the index
    CC_DDC_RESET=1         clear the DDC draft before each generation, rebuild only from this request's committed tokens
    CC_DDC_MAX_CTX=<window>  the length at which DDC stands down; 0 removes even that clamp
    CC_DDC_BATCH=1          under CC_BATCH, let the scheduler draft for a request
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srv import main

if __name__ == "__main__":
    main()
