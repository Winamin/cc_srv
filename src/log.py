# -*- coding: utf-8 -*-
"""One logger for the server, including llama.cpp's own C-level output.

The problem
-----------
llama.cpp logs through its own C callback and by default writes straight to
stderr, bypassing Python entirely.  On this model that means every decode prints
CUDA graph lines -- measured at 28 of them per 30 decode steps, so tens of
thousands over a session -- and the lines that actually matter get buried: a
context that ran out of cells, a decode that failed, a batch that could not be
initialised.  Those were the errors worth reading, and they were unreadable.

The fix
-------
``llama_log_set`` and ``ggml_log_set`` take a C callback, so both streams can be
routed here.  Measured: the CUDA graph flood arrives at level 1 (DEBUG), which
means a level threshold removes it without pattern-matching anything, and
everything at INFO and above still reaches the console.

Continuations (ggml level 5) are appended to the line before them rather than
given a new timestamp, because that is what they are -- llama.cpp splits a
single long message across several callbacks.

    CC_LOG=debug|info|warn|error|off     console and file threshold
    CC_LOG_CONSOLE=0                     file only
"""
from __future__ import annotations

import ctypes as ct
import os
import sys
import threading
import time

LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40, "off": 100}

# ggml_log_level: NONE=0 DEBUG=1 INFO=2 WARN=3 ERROR=4 CONT=5.  CONT is a
# continuation of the previous message, not a message of its own.
GGML_LEVEL = {0: "debug", 1: "debug", 2: "info", 3: "warn", 4: "error"}
CONT = 5

_lock = threading.Lock()
_threshold = LEVELS[os.environ.get("CC_LOG", "info").lower()]
_to_console = os.environ.get("CC_LOG_CONSOLE", "1") != "0"
_path = None
_cb = None            # the ctypes callback; the C side holds only a pointer, so
                      # this reference is what keeps it from being collected


def setup(logdir: str | None = None, tag: str = "cc_srv"):
    """Point the logger at a file, and return the tag used in every line."""
    global _path
    if logdir:
        _path = os.path.join(logdir, "server.log")
    return tag


def _emit(line: str, console: bool = True):
    """Write one finished line.  Never raises -- logging must not fail a request."""
    if console and _to_console:
        try:
            print(line, flush=True)
        except Exception:
            pass
    if _path:
        try:
            os.makedirs(os.path.dirname(_path), exist_ok=True)
            with _lock, open(_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def log(msg: str, level: str = "info", tag: str = "cc_srv"):
    """Format and write one message, if the level allows it."""
    if LEVELS.get(level, 20) < _threshold:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for line in str(msg).splitlines() or [""]:
        _emit(f"{stamp} [{level.upper():5s}] [{tag}] {line}")


# ---------------- llama.cpp / ggml ----------------
def _on_ggml(level: int, text, _user):
    """C callback for llama.cpp's own logging.

    Called from llama.cpp's threads, so everything here must be cheap and must
    not raise: an exception crossing back into C would take the process down.
    """
    try:
        if level == CONT:
            # A continuation belongs to the line before it.
            _emit("    " + (text or b"").decode("utf-8", "replace").rstrip())
            return
        name = GGML_LEVEL.get(level, "debug")
        if LEVELS[name] < _threshold:
            return
        body = (text or b"").decode("utf-8", "replace").rstrip()
        if not body:
            return
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        _emit(f"{stamp} [{name.upper():5s}] [llama] {body}")
    except Exception:
        pass


def capture_llama_log(log_fn=log):
    """Route llama.cpp's logging through this module.

    Installed as early as possible -- before the model is loaded -- because the
    backend and model load emit a lot, and a callback installed afterwards
    misses all of it.
    """
    global _cb
    if _cb is not None:
        return False
    _cb = ct.CFUNCTYPE(None, ct.c_int, ct.c_char_p, ct.c_void_p)(_on_ggml)
    ok = False
    for lib, name in ((_llama_lib(), "llama_log_set"),
                      (_ggml_base_lib(), "ggml_log_set")):
        if lib is None:
            continue
        try:
            f = getattr(lib, name)
            f.restype = None
            f.argtypes = [ct.CFUNCTYPE(None, ct.c_int, ct.c_char_p, ct.c_void_p),
                          ct.c_void_p]
            f(_cb, None)
            ok = True
        except (AttributeError, OSError):
            pass
    return ok


_LIBS: dict = {}


def register(name: str, lib):
    """Hand the loaded DLLs over, so this module does not load them itself."""
    _LIBS[name] = lib


def _llama_lib():
    return _LIBS.get("llama")


def _ggml_base_lib():
    return _LIBS.get("ggml-base")
