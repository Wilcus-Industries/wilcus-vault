"""Textual edits to notes the vault did not author.

A YAML round-trip is lossy against hand-written data, so existing notes are
patched line by line: every byte the edit does not mean stays identical.
"""

import json
import re

from .note import WIKILINK, link_of, usable_frontmatter
from .term import VaultError

_ISO_TIMESTAMP = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$")
_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def frontmatter_block(raw: str) -> tuple[int, int, int] | None:
    """(start, end, body_start) offsets of the frontmatter YAML inside `raw`.

    Offsets are into the raw text, so a BOM and CRLF endings are stepped over
    rather than normalized away. None when `parse_note` would not use the
    block either, so text edits and the parsed view agree on what is real.
    """
    bom = 1 if raw.startswith("\ufeff") else 0
    opening = re.match(r"^---\r?\n", raw[bom:])
    if opening is None:
        return None
    start = bom + opening.end()
    closing = re.search(r"^---[ \t]*\r?(\n|$)", raw[start:], re.MULTILINE)
    if closing is None:
        return None
    end = start + closing.start()
    body_start = start + closing.end()
    yaml_text = raw[start:end].replace("\r\n", "\n")
    return None if usable_frontmatter(yaml_text) is None else (start, end, body_start)


def patch_frontmatter(raw: str, key: str, value: str | None) -> str:
    """Set one top-level frontmatter key by editing the text, or unset it with None.

    Appends the key's line or replaces every line already defining it (YAML
    last-wins would otherwise resurrect a duplicate). A file with no usable
    block gets a fresh one prepended; unsetting never creates a block.
    """
    if not _KEY.match(key):
        raise VaultError(f"patch_frontmatter: unusable key {key}")
    at = frontmatter_block(raw)
    if value is None:
        if at is None:
            return raw
        start, end, _ = at
        gone = re.sub(rf"^{key}[ \t]*:[^\r\n]*\r?\n?", "", raw[start:end], flags=re.MULTILINE)
        return raw[:start] + gone + raw[end:]

    # A timestamp reads as a YAML scalar; anything else is quoted so a path or
    # a colon can never become syntax.
    line = f"{key}: {value if _ISO_TIMESTAMP.match(value) else json.dumps(value)}"
    if at is None:
        return f"---\n{line}\n---\n{raw}"

    start, end, _ = at
    eol = "\r\n" if raw[:start].endswith("\r\n") else "\n"
    block = raw[start:end]
    # Top-level only: an indented `key:` belongs to a nested mapping.
    existing = re.compile(rf"^{key}[ \t]*:[^\r\n]*", re.MULTILINE)
    seen = 0

    def swap(_match: re.Match[str]) -> str:
        nonlocal seen
        seen += 1
        return line if seen == 1 else ""

    swapped = existing.sub(swap, block)
    if seen > 0:
        patched = swapped
    else:
        glue = "" if block == "" or block.endswith("\n") else eol
        patched = block + glue + line + eol
    return raw[:start] + patched + raw[end:]


def qualify_links(raw: str, stem: str, target: str) -> str:
    """Rewrite every bare `[[stem]]` / `[[stem|alias]]` in the body to `[[target]]`.

    Only the body region is scanned, with the parser's own link regex, so what
    is rewritten and what counts as an edge cannot disagree. Aliases travel.
    """
    at = frontmatter_block(raw)
    start = at[2] if at is not None else 0

    def rewrite(match: re.Match[str]) -> str:
        inner = match.group(1)
        if link_of(inner) != stem:
            return match.group(0)
        pipe = inner.find("|")
        alias = "" if pipe == -1 else inner[pipe:]
        return f"[[{target}{alias}]]"

    return raw[:start] + WIKILINK.sub(rewrite, raw[start:])


def replace_body(raw: str, body: str) -> str:
    """Replace the body, keeping the frontmatter block byte-identical."""
    at = frontmatter_block(raw)
    return body if at is None else raw[: at[2]] + body
