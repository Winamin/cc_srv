# -*- coding: utf-8 -*-
"""HTTP layer: Anthropic Messages API compatibility plus three Claude Code adapters.

All three adapters exist to keep the KV prefix aligned.  A misaligned prefix
means APC collapses and every turn pays a full prefill:
  A. counter normalization: CC injects <total_tokens>N tokens left</total_tokens>
     on every request and N changes each time
  B. assistant header: the header used when generating and the header used when
     re-rendering history must be byte-identical -- obtain it from the template
     itself with a sentinel, do not hardcode newlines
  C. empty block filter: CC's assistant messages usually carry an empty text
     block, and "\n".join then emits two extra newlines
     (</think>\n\n\n\n<tool_call> instead of </think>\n\n<tool_call>).
     Two newlines are enough to zero out a thousand tokens of reuse.
A and B each have a kill switch for A/B comparisons.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch import Batcher, run_job
from eng import Eng
from log import C, log, setup as log_setup
from qcache import turn_key
from stream import Streamer

# Stream text as it is generated instead of buffering the whole reply.
# CC_STREAM=0 restores the buffered behaviour, which is what the server did
# before: it matters if a client is confused by early text, and it is the only
# difference between the two.
STREAM = os.environ.get("CC_STREAM", "1") != "0"

E: Eng | None = None
BATCHER: "Batcher | None" = None
LK = threading.Lock()
LOGK = threading.Lock()

LOGDIR = os.environ.get("CC_LOGDIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs")


def logit(msg: str, level: str = "info"):
    """Compatibility wrapper: the logger itself lives in log.py.

    Kept because every call site here and Eng's ``log=`` hook use this name, and
    they all pass a single argument.  The old body wrote its own timestamp and
    its own file; log.py does both, and adds the level and the tag that make a
    log line readable -- and it also owns llama.cpp's C-level output, which used
    to bypass Python entirely.
    """
    log(msg, level)


def record(obj: dict):
    """Append one structured record to logs/requests.jsonl."""
    try:
        os.makedirs(LOGDIR, exist_ok=True)
        obj = dict(obj, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with LOGK, open(os.path.join(LOGDIR, "requests.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


# A. counter normalization (CC_KEEP_CNT=1 disables it).
#
# Claude Code has a source-side fix that is strictly better than rewriting the
# text here: CLAUDE_CODE_TOTAL_TOKENS_REMINDER=infinite makes the block
# byte-stable, so the prompt no longer changes between requests at all.  See
# _cc_local.json.  This normalization stays as the fallback for clients that do
# not set it, and because it also covers the copies embedded in the environment
# block.
CNT = None if os.environ.get("CC_KEEP_CNT") else \
    re.compile(r"<total_tokens>\s*\d+\s*tokens left</total_tokens>")
CNT_FIX = "<total_tokens>" + os.environ.get("CC_CNT_N", "15000000") + " tokens left</total_tokens>"

# B/C. assistant header + empty blocks (CC_KEEP_AH=1 disables both).
AH = not os.environ.get("CC_KEEP_AH")
AH_HEAD = "<think>\n\n</think>\n\n"

# NOTE ON LANGUAGE
# This is the one Chinese string the server puts INTO the prompt rather than in a
# comment.  It is left exactly as it was: it is part of the bytes the model sees,
# so translating it would change the token ids of every tool-bearing request and
# invalidate the captured fixtures (_d1.jsonl, _r1.jsonl, _r2.jsonl) and every
# cached baseline recorded against them.  CC_TOOLS_HINT=en selects the English
# wording for a fresh release; the default keeps recorded runs reproducible.
TOOLS_HINT_ZH = """
你可以调用工具。可用工具如下 (JSON Schema):
{tools}

调用格式 (必须严格遵守, 一次只调用一个):
<tool_call>
{{"name": "<工具名>", "arguments": {{...}}}}
</tool_call>

不需要调用工具时, 直接回复文本。
""".strip()

TOOLS_HINT_EN = """
You can call tools. The available tools are (JSON Schema):
{tools}

Call format (follow this strictly, one call at a time):
<tool_call>
{{"name": "<tool name>", "arguments": {{...}}}}
</tool_call>

When no tool is needed, reply with text directly.
""".strip()

TOOLS_HINT = TOOLS_HINT_EN if os.environ.get("CC_TOOLS_HINT") == "en" else TOOLS_HINT_ZH


def sys_txt(system) -> str:
    """CC's system field is an array of content blocks (with cache_control); pull the text out."""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n".join(b if isinstance(b, str) else b.get("text", "")
                         for b in system if isinstance(b, (str, dict)))
    return ""


def msgs_of(system, msgs: list, tools: list | None) -> list:
    """Anthropic system/messages/tools -> [(role, content)] for the template to flatten."""
    out = []
    s = sys_txt(system)
    if tools:
        sch = json.dumps([{"name": t.get("name"), "description": t.get("description", ""),
                           "parameters": t.get("input_schema", {})} for t in tools],
                         ensure_ascii=False, indent=1)
        s = (s + "\n\n" + TOOLS_HINT.format(tools=sch)).strip()
    if s:
        out.append(("system", s))
    for m in msgs:
        role = m.get("role", "user")
        c = m.get("content")
        ch = []
        if isinstance(c, str):
            ch.append(c)
        else:
            for b in (c or []):
                bt = b.get("type")
                if bt == "text":
                    ch.append(b.get("text", ""))
                elif bt == "tool_use":
                    ch.append("<tool_call>\n" + json.dumps(
                        {"name": b.get("name"), "arguments": b.get("input", {})},
                        ensure_ascii=False) + "\n</tool_call>")
                elif bt == "tool_result":
                    inner = b.get("content")
                    if isinstance(inner, list):
                        inner = "".join(x.get("text", "") for x in inner
                                        if isinstance(x, dict))
                    ch.append("<tool_result>\n" + str(inner) + "\n</tool_result>")
        if role == "assistant" and AH:
            # C. drop the empty blocks -- do not let "\n".join invent newlines
            out.append((role, AH_HEAD + "\n".join(x for x in ch if x)))
        else:
            out.append((role, "\n".join(ch)))
    return out


def drop_think(t: str) -> str:
    """Strip <think>...</think> (an unclosed one too)."""
    while "<think>" in t:
        i = t.find("<think>")
        j = t.find("</think>", i)
        t = t[:i] + (t[j + 8:] if j >= 0 else "")
    return t.strip()


def calls_of(text: str):
    """Pull <tool_call>...</tool_call> out of generated text.  Lenient: if JSON
    parsing fails, fall back to brace matching.

    The standard library's json is awkward here -- the model often emits half a
    JSON object, so the braces are matched by hand.
    """
    pre, calls, rest = [], [], text
    while True:
        i = rest.find("<tool_call>")
        if i < 0:
            break
        j = rest.find("</tool_call>", i)
        body = rest[i + 11: j if j >= 0 else len(rest)]
        pre.append(rest[:i])
        rest = rest[j + 12:] if j >= 0 else ""
        o = None
        try:
            o = json.loads(body.strip())
        except Exception:
            k = body.find("{")
            if k >= 0:
                d = 0
                for e in range(k, len(body)):
                    d += (body[e] == "{") - (body[e] == "}")
                    if d == 0:
                        try:
                            o = json.loads(body[k:e + 1])
                        except Exception:
                            pass
                        break
        if isinstance(o, dict) and o.get("name"):
            calls.append({"name": o["name"], "arguments": o.get("arguments") or {}})
    pre.append(rest)
    return "".join(pre), calls


def sse(h, ev: str, data: dict):
    """Write one SSE frame.  False once the client is gone.

    A client that hangs up mid-stream raises ConnectionResetError on the next
    write, and there is no recovering from it -- but the generation loop keeps
    calling back for every token, so an unguarded write turns one disconnect into
    a traceback per token.  Latch it instead and go quiet.
    """
    if getattr(h, "_gone", False):
        return False
    try:
        h.wfile.write(f"event: {ev}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
        h.wfile.flush()
        return True
    except (ConnectionError, OSError):
        h._gone = True
        return False


def usage(i: int, o: int, cache_read: int = 0) -> dict:
    """Anthropic usage, with the cache fields filled in.

    Claude Code does not ask the server whether the cache hit -- it records
    whatever these fields say into its session transcript, and every tool that
    reports a hit rate (the statusline payload, cc-live, claude-stat) computes
    it from that transcript.  Reporting zero here means the hit rate reads as
    zero however much the local cache actually served.

    The three input fields have to add up to the prompt: ``input_tokens`` is
    what was billed as fresh input and ``cache_read_input_tokens`` is what the
    cache answered.  cc_srv has no separate write-to-cache step -- every
    forwarded token also lands in the KV -- so ``cache_creation_input_tokens``
    stays 0 and the forwarded count is reported as plain input rather than
    being counted twice.
    """
    return {"input_tokens": max(0, i - cache_read), "output_tokens": o,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": cache_read}


def prepare(req):
    """Anthropic request -> (prompt token ids, turn key).

    The turn key identifies the trailing user turn and is what the turn-keyed
    reuse layer keys on.  It is computed from the same flattened message list the
    prompt is rendered from, so the two can never disagree about what the
    question was.
    """
    m = msgs_of(req.get("system", ""), req.get("messages", []), req.get("tools") or [])
    turn = turn_key(m)
    if AH:
        # B. the assistant header cannot be hardcoded -- the layout (how many
        #    newlines, the separator before the body) belongs to the template.
        #    Render one sentinel message and take the span between the end of the
        #    base rendering and the sentinel: that is everything the template
        #    emits before an assistant body.
        #    Trap 1: do not write probe.endswith(SENT) -- the template renders a
        #            message ending in assistant completely (including
        #            <|im_end|>), so endswith never holds, tail comes back empty,
        #            and the generation prompt does not even contain
        #            '<|im_start|>assistant' (the measured divergence token was
        #            previous-turn <think> versus this-turn <|im_start|>).
        #    Trap 2: find() must start at len(base) -- that is what brings in the
        #            separator before the body, and only then are the two sides
        #            byte-identical.
        SENT = "SENTINELPROBE"
        base = E.tmpl(m, add_ass=False)
        probe = E.tmpl(m + [("assistant", SENT)], add_ass=False)
        i = probe.find(SENT, len(base))
        p = base + (probe[len(base):i] if i >= 0 else "") + AH_HEAD
    else:
        p = E.tmpl(m, add_ass=True)
    if CNT is not None:
        # A. pin the counter to a constant -> the prefix is stable -> the fatal
        #    partial truncation is no longer needed
        p = CNT.sub(CNT_FIX, p)
    return E.tok(p), turn


def render(req) -> list[int]:
    """Compatibility shim: prompt ids only."""
    return prepare(req)[0]


class Api(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._send(200, {"ok": True})
        if self.path.startswith("/stats"):
            return self._send(200, {"eng": E.line() if E else None})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads((self.rfile.read(n) if n else b"{}").decode())
        except Exception as ex:
            return self._send(400, {"type": "error", "error": {
                "type": "invalid_request_error", "message": str(ex)}})
        if self.path.startswith("/v1/messages/count_tokens"):
            return self._send(200, {"input_tokens": len(render(req)) if E else 0})
        if not self.path.startswith("/v1/messages"):
            return self._send(404, {"error": "not found"})
        try:
            self.msg(req)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            try:
                self._send(500, {"type": "error", "error":
                                 {"type": "api_error", "message": str(ex)}})
            except Exception:
                pass

    def msg(self, req):
        mdl = req.get("model", "qwythos-9b")
        mx = int(req.get("max_tokens") or 512)
        stream = bool(req.get("stream"))
        stops = list(req.get("stop_sequences") or [])
        ids, turn = prepare(req)
        mid = "msg_" + uuid.uuid4().hex[:24]
        stops += ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]   # template control tokens are natural stop strings

        def run(on_token=None, on_serve=None):
            t = time.time()
            if BATCHER is not None:
                # The scheduler already runs requests concurrently; a lock here
                # would put back the serialisation it exists to remove.  The
                # KV-free layers are reached inside run_job, which guards them.
                t0 = time.time()
                pre_before, gen_before = E.st["t_pre"], E.st["t_gen"]
                arc_before = E.st.get("arc_s", 0.0)
                r = run_job(E, BATCHER, ids, mx, stops, turn, on_token, on_serve)
            else:
                with LK:
                    w = time.time() - t
                    if w > 1.0:
                        logit(f"waited {w:.1f}s for the engine lock "
                              f"(single context, serial)", "warn")
                    t0 = time.time()
                    pre_before, gen_before = E.st["t_pre"], E.st["t_gen"]
                    arc_before = E.st.get("arc_s", 0.0)
                    r = E.gen(ids, max_new=mx, stop=stops, turn=turn,
                              on_token=on_token, on_serve=on_serve)
            if os.environ.get("CC_DUMP"):
                with open(os.environ["CC_DUMP"], "a", encoding="utf-8") as f:
                    f.write(json.dumps({"prompt": list(ids), "gen": list(r[0]),
                                        "hit": r[2]["hit"]}) + "\n")
            info = r[2]
            # Report the split, not just the total.  A request that reused 21k
            # tokens and generated 578 can take LONGER than one that prefilled
            # 25k and generated 76 -- measured: 12.3 s against 11.8 s, with the
            # reuse actually saving 8.2 s.  Without the split that reads as "the
            # cache made it slower", which is how it got read.
            # The batched path reports these per request.  A batch serves several
            # requests in one llama_decode, so the engine-level counters cannot
            # separate them and the deltas below would credit this request with
            # its neighbours' prefill time; the serial path has no per-job figure
            # and falls back to the deltas, which are exact there.
            pf = info.get("pf")
            gn = info.get("gn")
            if pf is None:
                pf = E.st["t_pre"] - pre_before
            if gn is None:
                gn = E.st["t_gen"] - gen_before
            arcd = E.st.get("arc_s", 0.0) - arc_before
            sec = time.time() - t0
            # Whatever the two counters did not claim.  A batched request is only
            # advanced on the steps it appears in, and _admit can put it straight
            # back on the queue to wait for the worker holding its history -- so
            # without this bucket prefill+decode do not add up to the wall clock
            # and the prefill rate below reads far better than the request felt.
            wait = max(0.0, sec - pf - gn - arcd)
            # Four different speeds, because they answer four different
            # questions and only the first is comparable to a solo prefill:
            #   pre/s  tokens actually forwarded, per second spent forwarding
            #   ctx/s  the whole prompt, per second spent forwarding -- the cache's
            #          contribution included, so a restored prefix shows up here
            #   gen/s  the model's own decode rate, over decode time alone
            #   out/s  what the caller waited for: output over the whole request
            pre_n = info.get("pre") or 0
            out_n = len(r[0])
            rates = []
            if pf > 0:
                if pre_n:
                    rates.append(f"{pre_n / pf:.0f}pre/s")
                if ids:
                    rates.append(f"{len(ids) / pf:.0f}ctx/s")
            if gn > 0 and out_n:
                rates.append(f"{out_n / gn:.0f}gen/s")
            if sec > 0 and out_n:
                rates.append(f"{out_n / sec:.0f}out/s")
            hit_c = "red" if info["hit"] == "batch_cold" else "green"
            msg = (f"{len(ids)}+{out_n} {C(info['hit'], hit_c)} "
                   f"pre={info.get('pre')} reuse={info.get('reuse')} "
                   f"p_raw={info.get('p_raw','-')} cur={info.get('cur','-')} "
                   f"{sec:.1f}s (prefill {pf:.1f}s + decode {gn:.1f}s")
            if wait > 0.05:
                msg += f" + {C('wait', 'yellow')} {wait:.1f}s"
            msg += ")"
            if rates:
                msg += " " + " ".join(rates)
            if arcd > 0.001:
                msg += f" arc_restore={arcd:.2f}s"
            if turn is not None:
                msg += f" turn={turn[0]}:{turn[1].hex()[:8]}"
            if info["hit"] == "qreuse":
                msg += (f" qlevel={info.get('qpreuse')} qwhy={info.get('qwhy')} "
                        f"replayed={info.get('replayed')} "
                        f"edit={info.get('edit_applied')}ok/"
                        f"{info.get('edit_rejected')}rej")
            if info.get("seq") is not None:
                msg += f" seq={info['seq']}"
            # The busiest line in the log; its own tag so it can be picked out.
            log("req " + msg, "info", "req")
            record({"prompt_tokens": len(ids), "gen_tokens": len(r[0]),
                    "hit": info["hit"], "pre": info.get("pre"),
                    "reuse": info.get("reuse"), "p_raw": info.get("p_raw"),
                    "cur": info.get("cur"), "seconds": round(time.time()-t0, 3),
                    "turn_kind": None if turn is None else turn[0],
                    "turn_digest": None if turn is None else turn[1].hex(),
                    "max_tokens": mx, "stream": stream,
                    "qpreuse": info.get("qpreuse"), "qwhy": info.get("qwhy"),
                    "replayed": info.get("replayed"), "edited": info.get("edited"),
                    "edit_applied": info.get("edit_applied"),
                    "edit_rejected": info.get("edit_rejected"),
                    # Deltas, not the engine's running totals: a consumer
                    # reading this file otherwise sees a monotonically growing
                    # "seconds" that is not this request's time.
                    "prefill_seconds": round(pf, 3),
                    "generation_seconds": round(gn, 3)})
            return r

        def parts(text):
            pre, calls = calls_of(text)
            c = []
            pre = drop_think(pre)
            if pre.strip():
                c.append({"type": "text", "text": pre})
            for x in calls:
                c.append({"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:20],
                          "name": x["name"], "input": x["arguments"]})
            return (c or [{"type": "text", "text": text}]), ("tool_use" if calls else "end_turn")

        if not stream:
            toks, text, info = run()
            c, why = parts(text)
            return self._send(200, {"id": mid, "type": "message", "role": "assistant",
                                    "model": mdl, "content": c, "stop_reason": why,
                                    "stop_sequence": None,
                                    "usage": usage(len(ids), len(toks),
                                                   info.get("reuse") or 0)})

        # ---- streaming ----
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        # message_start is NOT sent here.  Its usage is what Claude Code writes
        # into the session transcript, and the cache fields in it can only be
        # right if the serving layer is already known -- which it is not yet:
        # the probe and the scheduler both decide later, and both decide before
        # a single token is produced.  So it goes out at the first of those
        # moments, or at the latest before any content frame.
        served = {"reuse": 0, "sent": False}
        emit0 = threading.Lock()

        def ensure_start():
            """Write message_start and the opening text block once.  Caller holds emit0."""
            if served["sent"]:
                return
            served["sent"] = True
            sse(self, "message_start", {"type": "message_start", "message": {
                "id": mid, "type": "message", "role": "assistant", "model": mdl,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": usage(len(ids), 0, served["reuse"])}})
            # content_block_start travels with it: it used to be emitted up front,
            # which forced message_start out before the serving layer was known
            # and put a zero in the cache fields the client records.
            sse(self, "content_block_start", {"type": "content_block_start",
                "index": 0, "content_block": {"type": "text", "text": ""}})

        def on_serve(reuse, hit):
            served["reuse"] = reuse or 0
            with emit0:
                ensure_start()

        def emit(ev, data):
            # The generation thread and this one both write to the socket -- the
            # generation thread sends text deltas while this one sends pings --
            # and interleaving two SSE frames corrupts both.
            with emit0:
                ensure_start()        # never let content precede message_start
                return sse(self, ev, data)

        st = Streamer(stops) if STREAM else None

        def on_tok(chunk):
            """Text that is safe to send now, computed from the raw chunks.

            The safety rules live in Streamer: nothing from a tool call, nothing
            from inside a think block, nothing from a stop string, and nothing
            that is still a half-formed marker.  What comes out has to add up to
            exactly what the buffered path would have sent, which test_stream.py
            checks.
            """
            if st is None:
                return
            safe = st.feed(chunk)
            if safe:
                emit("content_block_delta", {"type": "content_block_delta",
                     "index": 0, "delta": {"type": "text_delta", "text": safe}})

        box, done = [None, None], threading.Event()

        def work():
            # try/finally, not a bare pair: without it a raise inside run() left
            # the flag unset forever, so the handler below pinged every 6 s and
            # never reached message_stop -- and those pings keep the client's
            # idle watchdog alive, so the client waited indefinitely instead of
            # erroring.  The exception is carried out and reported below.
            try:
                box[0] = run(on_tok, on_serve)
            except BaseException as ex:       # noqa: BLE001 - reported below
                box[1] = ex
            finally:
                done.set()

        threading.Thread(target=work, daemon=True).start()
        # Event.wait returns the moment the flag is set, so a reply goes out as
        # soon as it exists.  Polling with a bare time.sleep(6.0) instead -- the
        # earlier shape -- slept the full timeout BEFORE looking, which pushed
        # every reply to the next 6 s boundary: a cache hit answered in 1 ms
        # still reached the client 6 s later.
        while not done.wait(6.0):
            emit("ping", {"type": "ping"})    # keep an idle watchdog from cutting the stream
        if box[1] is not None:
            # The response headers are already sent, so the only honest report is
            # an SSE error event, then close.
            emit("error", {"type": "error",
                           "error": {"type": "api_error", "message": str(box[1])}})
            self.close_connection = True
            return
        if box[0] is None:
            # Unreachable while the loop above waits on done: work() sets
            # box[0] or box[1] in every path.  Asserted so that anything which
            # leaves that loop early fails here with a reason, instead of dying
            # on the unpack with "cannot unpack non-sequence NoneType".
            raise RuntimeError("generation finished without a result")
        toks, text, info = box[0]
        # Everything below writes frames directly, so make sure message_start has
        # gone out first.  It normally has by now (on_serve fired at admission,
        # or the first delta went through emit), but a reply that produced no
        # streamed text at all would otherwise reach message_delta without it.
        with emit0:
            ensure_start()

        # Everything is sent after generation completes, not while generating --
        # tool blocks cannot be streamed character by character or CC sees half a
        # text block first.
        pre, calls = calls_of(text)
        pre = drop_think(pre)
        if st is not None:
            # Whatever the safety rules held back, now that the real text is
            # known.  After this the client holds exactly `pre`.
            rest = st.finish(pre)
            if rest:
                emit("content_block_delta", {"type": "content_block_delta",
                     "index": 0, "delta": {"type": "text_delta", "text": rest}})
            if st.mismatch:
                logit("what was streamed is not a prefix of the buffered text; "
                      "no correction sent, see stream.py", "warn")
        elif pre:
            emit("content_block_delta", {"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": pre}})
        emit("content_block_stop", {"type": "content_block_stop", "index": 0})
        for i, c in enumerate(calls, start=1):
            sse(self, "content_block_start", {"type": "content_block_start", "index": i,
                "content_block": {"type": "tool_use",
                                  "id": "toolu_" + uuid.uuid4().hex[:20],
                                  "name": c["name"], "input": {}}})
            sse(self, "content_block_delta", {"type": "content_block_delta", "index": i,
                "delta": {"type": "input_json_delta",
                          "partial_json": json.dumps(c["arguments"], ensure_ascii=False)}})
            sse(self, "content_block_stop", {"type": "content_block_stop", "index": i})
        sse(self, "message_delta", {"type": "message_delta",
            "delta": {"stop_reason": "tool_use" if calls else "end_turn",
                      "stop_sequence": None},
            # Repeats the whole usage rather than just output_tokens.  Claude
            # Code is known to record the fields from message_start; this one
            # carries the same numbers so a reader that prefers the last usage
            # in the stream sees them too.
            "usage": usage(len(ids), len(toks), info.get("reuse") or 0)})
        sse(self, "message_stop", {"type": "message_stop"})
        # An SSE response has neither Content-Length nor chunked encoding, so
        # under HTTP/1.1 the client can only tell the response ended by the
        # connection CLOSING.  message_stop is only the SSE layer.  Without this,
        # keep-alive reuses the connection and the next request falls into a
        # black hole.
        self.close_connection = True


class Srv(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, req, addr):
        """CC reconnecting or resetting after a timeout is normal; do not pollute
        the log with a traceback for it."""
        if isinstance(sys.exc_info()[1], (ConnectionResetError, ConnectionAbortedError,
                                          BrokenPipeError, TimeoutError, OSError)):
            return
        super().handle_error(req, addr)

    def process_request_thread(self, req, addr):
        try:
            super().process_request_thread(req, addr)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError,
                TimeoutError, OSError):
            pass


def main():
    global E
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--n-ctx", type=int, default=131072)
    a = ap.parse_args()
    global BATCHER
    nseq, n_ctx = 1, a.n_ctx
    workers = 1
    if os.environ.get("CC_BATCH") == "1":
        workers = max(2, int(os.environ.get("CC_BATCH_N", "4")))
    # The archive keeps its own sequences, and the scheduler gets the rest.  They
    # are sized together because they share the context: sizing them separately
    # is how the archive ends up handing work to a sequence it is using.
    arc_slots = (int(os.environ.get("CC_ARCHIVE_SLOTS", "1"))
                 if os.environ.get("CC_ARCHIVE") == "1" else 0)
    nseq = workers + arc_slots
    if nseq > 1:
        # The KV cells are n_ctx in total and are split across sequences, so
        # n_ctx is multiplied to keep every sequence's window -- which is what
        # makes this cost memory rather than nothing.
        n_ctx = a.n_ctx * nseq
        logit(f"{workers} worker sequence(s) + {arc_slots} archive slot(s), "
              f"n_ctx {a.n_ctx} -> {n_ctx} (KV reservation x{nseq})")
    log_setup(LOGDIR)
    log_setup(LOGDIR)
    logit(f"loading engine n_ctx={n_ctx}")
    E = Eng(n_ctx=n_ctx, nseq=nseq, log=logit)
    if workers > 1:
        BATCHER = Batcher(E, log=logit)
        logit(f"scheduler up: {len(E.work_seqs)} worker sequences "
              f"{E.work_seqs}, requests run concurrently")
    try:
        os.makedirs(LOGDIR, exist_ok=True)
    except OSError as ex:
        # logit()/record() swallow the same failure, so aborting here after the
        # 5.5 GB model is loaded would contradict the module's own rule.
        logit(f"cannot create log dir {LOGDIR}: {ex} (continuing without it)")
    s = Srv((a.host, a.port), Api)
    logit(f"listening on http://{a.host}:{a.port}")
    logit(f"point Claude Code at: ANTHROPIC_BASE_URL=http://{a.host}:{a.port}")
    logit(f"logs -> {LOGDIR}")
    logit("single context, serial -- concurrent requests queue, that is the inherent bottleneck")
    try:
        s.serve_forever()
    except KeyboardInterrupt:
        logit("exit")
