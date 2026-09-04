"""Parse and render one note: `---` YAML frontmatter plus a markdown body.

Parsing never raises. A note with broken frontmatter still indexes with the whole
file as its body and `malformed_frontmatter` set for doctor to report.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

import yaml

# One scanner for both finding links and rewriting them. Single-line and
# length-capped so a stray `[[` cannot pair with a `]]` pages later.
WIKILINK = re.compile(r"\[\[([^\[\]\n]{1,256})\]\]")
_HEADING = re.compile(r"^#[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_CLOSE_FENCE = re.compile(r"^---[ \t]*$", re.MULTILINE)
_NODE_BUDGET = 4096


class _Loader(yaml.SafeLoader):
    """SafeLoader with the YAML 1.2 core schema for plain scalars.

    PyYAML speaks YAML 1.1, where `yes`, `on`, `12:30:00` and `2026-01-02` are
    not strings and `01234` is octal. Obsidian and the previous parser read 1.2,
    so only true/false, decimal/0o/0x integers, floats and null resolve.
    """


_CORE_SCHEMA = {
    "tag:yaml.org,2002:bool": (r"^(?:true|True|TRUE|false|False|FALSE)$", "tTfF"),
    "tag:yaml.org,2002:int": (r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$", "-+0123456789"),
    "tag:yaml.org,2002:float": (
        r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
        r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$",
        "-+.0123456789",
    ),
}
_Loader.yaml_implicit_resolvers = {
    key: [(tag, rx) for tag, rx in resolvers if tag.endswith((":null", ":merge"))]
    for key, resolvers in _Loader.yaml_implicit_resolvers.items()
}
for _tag, (_pattern, _first) in _CORE_SCHEMA.items():
    _Loader.add_implicit_resolver(_tag, re.compile(_pattern), list(_first))


def _construct_int(loader: yaml.SafeLoader, node: yaml.Node) -> int:
    text = loader.construct_scalar(node)  # type: ignore[arg-type]
    base = 16 if text[:2] == "0x" else 8 if text[:2] == "0o" else 10
    return int(text, base)


_Loader.add_constructor("tag:yaml.org,2002:int", _construct_int)


@dataclass(frozen=True)
class Note:
    path: str  # vault-relative, the note's identity
    slug: str  # filename stem, what a bare `[[stem]]` resolves against
    title: str
    type: str | None
    frontmatter: dict[str, Any]
    body: str
    links: list[str]  # deduped link targets as written, first-seen order
    hash: str  # sha256 hex of the raw text; a dirtiness check, not an identity
    malformed_frontmatter: bool = field(default=False)


def link_target(rel: str) -> str:
    """The path-qualified wikilink target of a vault-relative path: minus `.md`."""
    return rel[: -len(".md")]


def link_of(inner: str) -> str:
    """A match's link target: the text before any `|alias`, trimmed."""
    return inner.split("|")[0].strip()


def is_writable_target(target: str) -> bool:
    """Can this target survive a round trip through `[[...]]`?

    Brackets, pipes, newlines and surrounding whitespace all change what the
    scanner reads back, so a rewrite to such a target is never made.
    """
    return not re.search(r"[\[\]\r\n]", target) and link_of(target) == target


def within_node_budget(value: object, budget: int = _NODE_BUDGET) -> bool:
    """False when the parsed YAML has too many nodes to belong in an index row.

    YAML aliases re-expand on every reference, so a tiny billion-laughs block
    parses into a huge object. Every visit is counted, shared or not.
    """
    seen = 0
    stack = [value]
    while stack:
        seen += 1
        if seen > budget:
            return False
        current = stack.pop()
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return True


def usable_frontmatter(text: str) -> dict[str, Any] | None:
    """The block's YAML as a mapping, or None when it cannot be used as one.

    Every reader of a frontmatter block decides through this one function so
    the parser and the textual patcher can never disagree about what is real.
    """
    if text.strip() == "":
        return {}
    try:
        parsed = yaml.load(text, Loader=_Loader)
    # PyYAML composes and constructs recursively, so nesting deep enough to
    # exhaust the stack raises RecursionError rather than a YAMLError.
    except (yaml.YAMLError, RecursionError):
        return None
    if not isinstance(parsed, dict) or not within_node_budget(parsed):
        return None
    # An explicit `!!binary`, `!!set` or `!!timestamp` builds a value the index
    # cannot store as JSON: the block is malformed, never a crash in reindex.
    try:
        json.dumps(parsed)
    except (TypeError, ValueError):
        return None
    return parsed


def _split_frontmatter(text: str) -> tuple[str | None, str, bool]:
    """(yaml, body, unterminated): split a leading `---` block off the file."""
    if not text.startswith("---\n"):
        return None, text, False
    rest = text[4:]
    close = _CLOSE_FENCE.search(rest)
    if close is None:
        return None, text, True
    body = re.sub(r"^---[ \t]*\n?", "", rest[close.start() :], count=1)
    return rest[: close.start()], body, False


def parse_note(raw: str, rel_path: str) -> Note:
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    text = raw.replace("\r\n", "\n").removeprefix("\ufeff")
    slug = PurePosixPath(rel_path).stem

    yaml_text, split_body, unterminated = _split_frontmatter(text)
    frontmatter: dict[str, Any] = {}
    body = text
    malformed = unterminated
    if yaml_text is not None:
        parsed = usable_frontmatter(yaml_text)
        if parsed is None:
            malformed = True
        else:
            frontmatter, body = parsed, split_body

    fm_title = frontmatter.get("title")
    fm_type = frontmatter.get("type")
    for value in (fm_title, fm_type):
        if value is not None and not isinstance(value, str):
            malformed = True

    # A plain regex scan: links and headings inside code fences count too.
    links = list(dict.fromkeys(link_of(m) for m in WIKILINK.findall(body)))
    heading = _HEADING.search(body)
    title = slug
    if isinstance(fm_title, str) and fm_title.strip():
        title = fm_title
    elif heading is not None:
        title = heading.group(1)

    return Note(
        path=rel_path,
        slug=slug,
        title=title,
        type=fm_type if isinstance(fm_type, str) else None,
        frontmatter=frontmatter,
        body=body,
        links=[link for link in links if link],
        hash=digest,
        malformed_frontmatter=malformed,
    )


def serialize_note(frontmatter: dict[str, Any], body: str) -> str:
    """Render a note we authored. Never use it to rewrite a human's file: a YAML
    round-trip drops comments and rewrites `01234` and `1.0`."""
    if not frontmatter:
        return body
    # One line per key, however long the value: the textual patcher edits by line.
    dumped = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True, width=2**31)
    return f"---\n{dumped.rstrip('\n')}\n---\n{body}"
