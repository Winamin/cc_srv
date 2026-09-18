# -*- coding: utf-8 -*-
"""Render the GGUF's own Jinja chat template, the way llama-server does.

Why this exists
---------------
The engine used to hand the model's template string to llama_chat_apply_template.
llama.h says of that function:

    "NOTE: This function does not use a jinja parser. It only support a
     pre-defined list of template."

This model's template is 7950 characters of Jinja that do real work -- it injects
a system message naming the model, enforces that a system message is first,
refuses a conversation with no user query, and opens or closes the reasoning
block depending on a template variable.  None of that was happening: the string
is not one of the 54 built-ins llama.cpp knows, so it silently fell back to plain
ChatML.  Measured on this model's template, a bare user turn:

    llama_chat_apply_template   <|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n
    the real Jinja              <|im_start|>system\nYou are Qwythos, ...<|im_end|>\n
                                <|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n

The identity line was never reaching the model.  bench/_diag_template.py shows
both sides of that.

The trap, and why the fallback is not just belt-and-braces
----------------------------------------------------------
The template is internally inconsistent about the assistant turn, and this is
the thing to understand before switching it on.

With enable_thinking false, the GENERATION prompt it emits is

    <|im_start|>assistant\n<think>\n\n</think>\n\n

but the same turn in HISTORY renders as just

    <|im_start|>assistant\n

because the assistant branch strips a <think> wrapper and drops it entirely when
what is inside is empty.  The empty block the model was told to answer after is
therefore not reproduced when that answer comes back as history.

Measured (bench/_check_template.py), with the current turn's flag set both ways:

    thinking ON    the round trip is byte-exact.  The reasoning block comes back
                   as <think>\n{reasoning}\n</think>\n\n{answer}, which is what
                   the model generated -- the template's trim is a no-op because
                   the reasoning already ends in the newline it re-adds.
    thinking OFF   the round trip breaks at the assistant header.  The KV holds
                   "...assistant\n<think>\n\n</think>\n\n4", the re-render gives
                   "...assistant\n4".

And the cost of that break is not "one answer re-prefills", which is what it
looks like it should be.  It is the whole prompt.

The fix, and why it is an ordering argument rather than a compromise
--------------------------------------------------------------------
The system block is the part this engine cannot produce for itself, and it is
also the only part that does not touch the assistant turns -- which are exactly
where the template and this engine disagree.  So take the block from the
template and let the engine render everything from the first user turn on.
Nothing a previous turn ended on changes, so the prompt still EXTENDS the
sequence and reuse still happens.

Measured, same two-turn conversation, warm engine, 2005-token system preamble:

    CC_TMPL=0  library          turn 2:   20 forwarded, 2007 reused
    CC_TMPL=1  system block     turn 2:   20 forwarded, 2035 reused   <- default
    CC_TMPL=2  whole template   turn 2: 2048 forwarded,    0 reused

Turn 1 is 2033 tokens under mode 1 against 2005 under mode 0.  Those 28 tokens
are the identity message, and they are the entire difference.
"""
import os


def template_mode() -> int:
    """CC_TMPL.  0 = library, 1 = system block only, 2 = all (default).

    Mode 2 is the default because it is the template doing its job, and it hits
    the cache -- but ONLY alongside thinking, which is the coupling this file
    exists to make explicit.  See the measurement at the end of the docstring:

        CC_TMPL  CC_THINK   turn 2 forwarded   reused
           1        0              20           2035
           2        0            2048              0
           2        1              18           2083     <- default

    Mode 2 renders a finished assistant turn from its content, so the prompt it
    builds next turn only matches the KV if the reasoning came back.  Reasoning
    comes back only if the client was given it -- an Anthropic thinking block in
    the reply is what the client echoes in the next request.  With thinking
    locked off, cc_srv sends the closed empty reasoning block, the model answers
    without reasoning, the client has no reasoning to echo, and the template
    renders that turn without the block the KV is holding.  Divergence, and on
    this hybrid memory a divergence anywhere costs the whole prefix.

    So mode 2 and CC_THINK=0 are a configuration that silently loses the cache;
    srv.py says so at startup rather than leaving it to be discovered.

    Mode 1 is kept, and is the answer when thinking must stay locked: it renders
    the assistant turns with the engine's own renderer, which reproduces the
    closed block, so the bodies still match the KV.  Its system block still comes
    from the template, so the identity message is still there.
    """
    v = os.environ.get("CC_TMPL", "2").strip().lower()
    if v in ("0", "false", "off", "no"):
        return 0
    if v in ("1", "system", "block", "sys"):
        return 1
    return 2


class ChatTemplate:
    """A compiled Jinja template from the GGUF, or a reason it is not one.

    Never raises out of render(): a template that refuses a conversation (this
    one raises on a non-first system message, on no user query, on an unknown
    role) must not turn into a failed request.  A refusal is counted, reported
    once, and the caller falls back to the library renderer.
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
            # Turned off rather than unavailable, and the two must not read the
            # same in the log: "the model carries no template" when it plainly
            # does is worse than saying nothing.
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
        # The one global the template calls that Jinja does not provide.  Its
        # message is the diagnostic: it is how the template says WHY it refused.
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

        Plain dicts are enough for ``message.role``: Jinja's default getattr
        falls back to __getitem__ when the attribute is missing.  A missing key
        with no fallback becomes Undefined, which is what the template's
        ``is defined`` / ``is string`` tests are written against.

        ``thinking=None`` leaves enable_thinking UNDEFINED, which is what the
        template treats as "on"; False is the value that closes the block.
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
            # The template's own raise_exception.  It is a statement about the
            # conversation, not a bug in the template.
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
        """Report the first occurrence in full, then only count.

        A refusal is per-conversation, so a loop that logs each one would flood
        the log with the same sentence for every request of a shape the template
        does not accept.  The count keeps it visible without the flood.
        """
        if not self._told:
            self._told = True
            self.log(f"{msg} (falling back to llama_chat_apply_template; "
                     f"further occurrences are counted, not logged)")
        else:
            self.log(f"{msg}")

    def system_block(self, msgs: list, thinking: bool | None = None):
        """Everything the template emits BEFORE the first non-system message.

        Found by rendering the leading system messages plus a sentinel user turn
        and cutting at the sentinel, rather than by searching for
        "<|im_start|>user": a marker search would also match the string inside a
        tool result, and this file should not add a second place that assumes the
        template's layout.

        The sentinel is a private-use run, so the template's own ``|trim`` leaves
        it intact and no real content can contain it.  Returns None when the
        template would not render, which is the caller's cue to fall back whole.
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

        enable_thinking lives ONLY here, and this is why the engine cannot get
        this span from llama_chat_apply_template: that function has no way to be
        told about a template variable, so it never emits the closed empty
        reasoning block that means "your reasoning is already done".  Rendering
        twice and taking the difference is what recovers it without hardcoding
        the two forms.

        Returns "" when add_generation_prompt adds nothing, None when the
        template would not render.
        """
        if self._tmpl is None:
            return None
        without = self.render(msgs, add_ass=False, thinking=thinking)
        if without is None:
            return None
        with_ = self.render(msgs, add_ass=True, thinking=thinking)
        if with_ is None:
            return None
        # add_generation_prompt only ever appends, so anything else means the
        # assumption behind this shortcut is wrong and the caller should not use
        # a head it cannot account for.
        return with_[len(without):] if with_.startswith(without) else None

    def line(self) -> str:
        """One line for the stats report.

        The mode is named, not implied: "jinja" alone cannot say whether the
        template built the whole prompt or only the system block, and those two
        behave nothing alike.
        """
        what = "system-block" if self.mode == 1 else "whole-prompt"
        if not self.active:
            return f"tmpl=library ({self.err})"
        return (f"tmpl=jinja/{what} {len(self.src)}c renders={self.renders} "
                f"refused={self.refused} failed={self.failed} "
                f"{self.render_ms * 1000:.0f}ms")


class _Refused(Exception):
    """Raised by the raise_exception global the template calls."""


# A private-use run: cannot occur in real message content, and not whitespace,
# so the template's |trim does not touch it.
_SENTINEL = "\ue000ccsrv-sys-probe\ue000"


def _raise(msg):
    raise _Refused(msg)
