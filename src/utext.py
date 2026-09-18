# -*- coding: utf-8 -*-
"""Token bytes -> text, without ever splitting a character.

Why this is its own module
--------------------------
A token is a byte string, not a character.  On this model's vocabulary a 4-byte
emoji is THREE tokens whose bytes are individually invalid UTF-8:

    U+1F389  ->  b'\\xf0\\x9f'  b'\\x8e'  b'\\x89'

Decoding each of those on its own -- which is what ``bytes.decode("utf-8",
"replace")`` does, and what this codebase did at ten different places -- yields
three U+FFFD REPLACEMENT CHARACTERS.  The original character is gone; joining
the results afterwards cannot bring it back.

Measured (bench/_diag_utf8.py): 31 of the tokens in nine short samples are
partial UTF-8.  U+1F600 happens to be a single token in this vocabulary and
survives; U+1F389, U+1F680, U+1F4CA and every regional-indicator pair do not.
That is why the symptom looked intermittent.

The rule, and the only rule here: DECODE ONCE, over the accumulated bytes.  When
text is needed before the end of the stream -- which it is, because the stop
strings are matched as the reply grows -- use the incremental decoder, which
returns only the characters that are complete and holds the remainder until the
next call.

``ddc_decode.py`` already did this correctly (it accumulates bytes and decodes
once), which is why the draft-cache path never showed the corruption while the
serial and batched paths did.
"""
import codecs

# "replace" and not "strict": a genuinely invalid byte sequence should become a
# visible U+FFFD rather than raise somewhere far from the cause.  The point of
# this module is that a VALID character split across tokens no longer counts as
# invalid.
_ERRORS = "replace"


def decoder():
    """A fresh incremental UTF-8 decoder.

    Feed it token bytes with ``.decode(b)``; it returns the characters that are
    now complete and buffers a trailing partial one.  Call ``.decode(b"", True)``
    at the end to flush -- a decoder left unflushed drops its tail silently.
    """
    return codecs.getincrementaldecoder("utf-8")(_ERRORS)


def whole(bs) -> str:
    """Decode a COMPLETE byte string.  For a stream, use decoder()."""
    return bytes(bs).decode("utf-8", _ERRORS)


class Text:
    """Accumulated token bytes as text, for loops that need it as they go.

    Behaves like the string it is growing: read ``.s`` for the text so far, call
    ``add`` with one token's bytes.  ``.s`` only ever holds whole characters, so
    a byte count is not a character count and neither is ever used as the other
    here -- the stop-string cuts index into ``.s``, which is text throughout.

    ``flush`` must be called once when the stream ends, or a final partial
    character is lost instead of being replaced.
    """

    __slots__ = ("_d", "s")

    def __init__(self):
        self._d = decoder()
        self.s = ""

    def add(self, b: bytes) -> str:
        self.s += self._d.decode(b)
        return self.s

    def flush(self) -> str:
        """Finish the stream.  Safe to call twice; the second is a no-op."""
        self.s += self._d.decode(b"", True)
        return self.s
