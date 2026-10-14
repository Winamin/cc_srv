# -*- coding: utf-8 -*-
"""Render the GGUF's own Jinja chat template, the way llama-server does.

``llama_chat_apply_template`` does not run Jinja: it falls back to plain ChatML
for any template outside its 54 built-ins, so this model's system (identity)
message and reasoning block never reached the prompt.

The template is internally inconsistent about the assistant turn: with
``enable_thinking`` false the generation prompt ends
``assistant\\n<think>\\n\\n</think>\\n\\n`` while the same turn rendered as
history is just ``assistant\\n``, because the assistant branch drops an empty
think block.  Mode 2 therefore round-trips only when the reasoning came back,
and on this hybrid memory a divergence anywhere costs the whole prefix.

Mode 1 is the way out of that: the template's system block, which the engine
cannot produce for itself, plus the engine's own renderer for the assistant
turns, which does reproduce the closed block.
"""
import os


def template_mode() -> int:
    """CC_TMPL.  0 = library, 1 = system block only, 2 = all (default).

    Mode 2 needs thinking: it rebuilds a finished assistant turn from its
    content, so the prompt matches the KV only if the client echoed the
    reasoning back.  With CC_THINK=0 it silently loses the cache, so srv.py
    warns at startup; use mode 1 when thinking is locked.
    """
    v = os.environ.get("CC_TMPL", "2").strip().lower()
    if v in ("0", "false", "off", "no"):
        return 0
    if v in ("1", "system", "block", "sys"):
        return 1
    return 2


class ChatTemplate:
    """A compiled Jinja template from the GGUF, or a reason it is not one.

    render() never raises: a refusal is counted, reported once, and the caller
    falls back to the library renderer.
    """

    def __init__(self, src: str | None, log=print, why: str | None = None,
                 mode: int = 1):
        self.src = src
        self.log = log
        self.mode = mode
        self.err = None            # the reason it is unusable, or None
        self._tmpl = None
        self.refused = 0           # renders the template itself rejected
        self.failed = 0            # renders that blew up for any other reason
        self.render_ms = 0.0
        self.renders = 0
        self._told = False
        if src is None and why:
            # Off rather than unavailable; the two must not read the same.
            self.err = why
            return
        if not src:
            self.err = "the model carries no tokenizer.chat_template"
            return
        try:
            import jinja2
        except ImportError:
            self.err = "jinja2 is not installed"
            return
        env = jinja2.Environment()
        # The one global the template calls that Jinja does not provide; its
        # message is the diagnostic.
        env.globals["raise_exception"] = _raise
        try:
            self._tmpl = env.from_string(src)
        except Exception as e:                       # noqa: BLE001
            self.err = f"the template does not compile: {type(e).__name__}: {e}"

    @property
    def active(self) -> bool:
        return self._tmpl is not None

    def render(self, msgs: list, add_ass: bool = True,
               thinking: bool | None = None, tools: list | None = None):
        """[(role, content)] -> prompt string.  None means it could not render.

        Plain dicts are enough for ``message.role``: Jinja's getattr falls back
        to __getitem__, so a missing key becomes Undefined -- which is what the
        template's ``is defined`` tests are written against.  ``thinking=None``
        leaves enable_thinking undefined, which the template reads as "on".
        """
        if self._tmpl is None:
            return None
        ctx = {"messages": [{"role": r, "content": c} for r, c in msgs],
               "add_generation_prompt": bool(add_ass)}
        if tools is not None:
            ctx["tools"] = tools
        if thinking is not None:
            ctx["enable_thinking"] = bool(thinking)
        import time
        t0 = time.perf_counter()
        try:
            out = self._tmpl.render(**ctx)
        except _Refused as e:
            # The template's own raise_exception: a statement about the
            # conversation, not a bug.
            self.refused += 1
            self._tell(f"the model's template refused a request: {e}")
            return None
        except Exception as e:                       # noqa: BLE001
            self.failed += 1
            self._tell(f"the model's template raised {type(e).__name__}: {e}")
            return None
        finally:
            self.render_ms += time.perf_counter() - t0
        self.renders += 1
        return out

    def _tell(self, msg):
        """Report the first occurrence in full, then only count it.

        A refusal is per-conversation, so logging every one would flood the log.
        """
        if not self._told:
            self._told = True
            self.log(f"{msg} (falling back to llama_chat_apply_template; "
                     f"further occurrences are counted, not logged)")
        else:
            self.log(f"{msg}")

    def system_block(self, msgs: list, thinking: bool | None = None):
        """Everything the template emits before the first non-system message.

        Found by rendering the leading system messages plus a sentinel user turn
        and cutting at the sentinel, rather than by searching for "user": a
        marker search would also match the string inside a tool result.  None
        means the caller should fall back whole.
        """
        if self._tmpl is None:
            return None
        k = next((i for i, (r, _) in enumerate(msgs) if r != "system"), len(msgs))
        probe = list(msgs[:k]) + [("user", _SENTINEL)]
        out = self.render(probe, add_ass=False, thinking=thinking)
        if out is None:
            return None
        i = out.find(_SENTINEL)
        return out[:i] if i >= 0 else None

    def generation_head(self, msgs: list, thinking: bool | None = None):
        """What the template appends after the last message to continue from.

        enable_thinking lives only here, and llama_chat_apply_template cannot be
        told about a template variable, so this span has to come from the
        template.  Found by rendering twice and taking the difference.  None
        means the caller should fall back whole.
        """
        if self._tmpl is None:
            return None
        without = self.render(msgs, add_ass=False, thinking=thinking)
        if without is None:
            return None
        with_ = self.render(msgs, add_ass=True, thinking=thinking)
        if with_ is None:
            return None
        # add_generation_prompt only ever appends; anything else means the
        # caller cannot account for this head.
        return with_[len(without):] if with_.startswith(without) else None

    def line(self) -> str:
        """One line for the stats report; the mode is named, not implied."""
        what = "system-block" if self.mode == 1 else "whole-prompt"
        if not self.active:
            return f"tmpl=library ({self.err})"
        return (f"tmpl=jinja/{what} {len(self.src)}c renders={self.renders} "
                f"refused={self.refused} failed={self.failed} "
                f"{self.render_ms * 1000:.0f}ms")


class _Refused(Exception):
    """Raised by the raise_exception global the template calls."""


# A private-use run: cannot occur in real message content, and the template's
# |trim leaves it intact.
_SENTINEL = "\ue000ccsrv-sys-probe\ue000"


def _raise(msg):
    raise _Refused(msg)
