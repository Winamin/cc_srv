# -*- coding: utf-8 -*-
"""Claude Code hook: append the user's typed prompt to a JSONL capture log.

Registered for ``UserPromptSubmit`` (and harmless for ``SessionStart`` /
``Stop``).  Claude Code 2.1.270 hands the hook a JSON object on stdin::

    session_id, transcript_path, cwd, prompt_id, permission_mode,
    hook_event_name, prompt

One JSON line is appended per event to ``<this dir>/logs/cc_prompts.jsonl``.

The ``turn_digest`` field is the turn key: it must be the *same* value the
server computes in ``qcache.turn_key`` for the request this prompt produces,
because that is how a captured prompt is matched to a server request.  The
normalization below is a line-for-line copy of that function -- the same
``<system-reminder>`` regex, the same ``.strip()`` after substitution, the same
utf-8 encode and ``digest_size=16`` -- so the two cannot drift apart.

Two rules govern this file, and both come from Claude Code's contract for
``UserPromptSubmit``: a non-empty but invalid stdout, or a timeout, is treated
as BLOCKING, and the user's prompt is refused.  So the success path writes
nothing to stdout and finishes in milliseconds, and every failure -- bad JSON,
unreadable stdin, unwritable log directory -- is swallowed.  The process always
exits 0.  There is deliberately no logging of its own failures: a hook that
reports a problem on stdout is worse than a hook that silently misses a line.

Stdlib only; numpy is not importable from a hook without a real interpreter
problem, and nothing here needs it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time

# --- copied verbatim from qcache.py; keep the two in step ---
# Claude Code injects volatile reminders into the user turn (attribution block,
# environment block).  They are not part of the question and they differ between
# a first ask and a repeat, so leaving them in the key would stop the same
# question from ever matching itself.
_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "logs", "cc_prompts.jsonl")


def turn_digest(prompt: str):
    """blake2b-128 of the prompt with reminder blocks stripped and trimmed.

    Hex form of exactly the digest ``qcache.turn_key`` puts in its returned
    ``(kind, digest, text)`` tuple, i.e. the server-side turn key.

    Returns ``None`` when nothing is left after stripping, because ``turn_key``
    skips such a turn and walks further back rather than keying on an empty
    string.  Emitting a digest there would create a capture row that can never
    match any server request.
    """
    text = _REMINDER.sub("", prompt).strip()
    if not text:
        return None
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def build_record(obj: dict) -> dict:
    """Project the hook payload down to the fields worth keeping."""
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "ts_epoch": round(time.time(), 3),
        "hook_event_name": obj.get("hook_event_name"),
        "session_id": obj.get("session_id"),
        "prompt_id": obj.get("prompt_id"),
        "cwd": obj.get("cwd"),
    }
    # SessionStart and Stop carry no prompt; they are recorded without one
    # rather than with an empty string, so a consumer can tell the difference.
    prompt = obj.get("prompt")
    if isinstance(prompt, str):
        rec["prompt"] = prompt
        digest = turn_digest(prompt)
        if digest is not None:
            rec["turn_digest"] = digest
    return rec


def append_line(path: str, line: str) -> None:
    """Append one complete line, serialised against other hook processes.

    Plain append is not safe here.  On this Windows host, forty simultaneous
    invocations lost about 5% of their records and occasionally produced a
    spliced, unparseable line -- the CRT seeks to end-of-file and then writes,
    and two processes can pick the same offset.  Opening with O_APPEND does not
    fix it (measured: losses either way), so the append is guarded by a lock
    file created with O_EXCL, which is atomic here.

    Retries are bounded and short: this runs inside a hook whose timeout counts
    as BLOCKING for UserPromptSubmit, so a stuck lock must degrade into a missed
    line, never into a refused prompt.
    """
    lock = path + ".lock"
    for _ in range(50):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            time.sleep(0.002)
            continue
        try:
            os.close(fd)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
            return
        finally:
            try:
                os.unlink(lock)
            except OSError:
                pass
    # Could not take the lock in ~100 ms.  Fall back to a plain append rather
    # than dropping the record outright; a rare interleaved line is better than
    # no line, and the consumer tolerates unparseable rows.
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def main() -> None:
    raw = sys.stdin.buffer.read()
    obj = json.loads(raw.decode("utf-8", "replace")) if raw.strip() else {}
    if not isinstance(obj, dict):
        obj = {}
    rec = build_record(obj)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    append_line(LOG, json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Never raise, never print, never block the prompt.
        pass
    sys.exit(0)
