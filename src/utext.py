# -*- coding: utf-8 -*-
"""Token bytes -> text, without ever splitting a character.

A token is bytes, not characters: decode once over the accumulated run.
Decoding each token on its own turns a valid character split across tokens into
U+FFFD replacements, and joining the results afterwards cannot bring it back.

Use ``decoder()`` when text is needed before the end of the stream; ``whole()``
is for a complete byte string.
"""
import codecs

# "replace" and not "strict": a genuinely invalid byte sequence becomes a visible
# U+FFFD instead of raising somewhere far from the cause.
_ERRORS = "replace"


def decoder():
    """A fresh incremental UTF-8 decoder: ``.decode(b)`` returns the characters
    that are complete and buffers a trailing partial one.  Call
    ``.decode(b"", True)`` to flush, or the tail is dropped silently.
    """
    return codecs.getincrementaldecoder("utf-8")(_ERRORS)


def whole(bs) -> str:
    """Decode a COMPLETE byte string.  For a stream, use decoder()."""
    return bytes(bs).decode("utf-8", _ERRORS)


class Text:
    """Accumulated token bytes as text; ``.s`` only ever holds whole characters.

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
