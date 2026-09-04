"""Text safety for anything the vault prints or raises."""

import re

# Unicode category Cc: C0 and C1 control characters. A bare CR or an ESC
# sequence inside a note title would otherwise rewrite the terminal line.
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


class VaultError(Exception):
    """An error the vault raised on purpose, with a message meant for a human."""


def safe(text: str) -> str:
    """Untrusted text, safe to echo: control characters print as `?`."""
    return _CONTROL.sub("?", text)


def printable(error: object) -> str:
    """An error as its message, scrubbed line by line so deliberate newlines survive."""
    return "\n".join(safe(line) for line in str(error).split("\n"))
