"""The merger contract: what a cluster becomes, and the prompt/parser for an LLM merger."""

import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass

from .decision import fence
from .note import Note
from .term import VaultError, safe


@dataclass(frozen=True)
class MergedNote:
    """The one note a cluster becomes. The cluster decides where it lands, never the merger."""

    title: str
    body: str
    type: str | None = None


@dataclass(frozen=True)
class MergeInput:
    notes: list[Note]  # one cluster's notes, bodies as they are on disk right now


# The caller wires an LLM (with merge_prompt + parse_merged); tests wire a fake.
Merger = Callable[[MergeInput], Awaitable[MergedNote]]


def check_merged(value: object) -> MergedNote:
    """Validate a merger's answer. Malformed output raises, like the gate's
    check_decision: a wrong guess about what a model meant writes the wrong file."""
    raw = asdict(value) if isinstance(value, MergedNote) else value

    def bad(why: str) -> VaultError:
        return VaultError(
            f"consolidate: merger returned {why}: {json.dumps(raw, default=str)[:200]}"
        )

    if not isinstance(raw, dict):
        raise bad("a non-object")
    title, body, type_ = raw.get("title"), raw.get("body"), raw.get("type")
    # An empty title or body is a model that lost the text, not a note.
    if not isinstance(title, str) or title.strip() == "":
        raise bad("no usable title")
    if not isinstance(body, str) or body.strip() == "":
        raise bad("no usable body")
    if type_ is not None and not isinstance(type_, str):
        raise bad("a non-string type")
    return MergedNote(title, body, type_)  # unknown keys from a model do not travel


def parse_merged(text: str) -> MergedNote:
    """Parse an LLM's reply. Strictly one JSON object; a fenced or chatty answer is malformed."""
    try:
        value = json.loads(text)
    except ValueError:
        raise VaultError(f"consolidate: merger did not return JSON: {text[:200]}") from None
    return check_merged(value)


def merge_prompt(input: MergeInput) -> str:
    listed = "\n\n".join(
        f"[{i + 1}] path: {safe(n.path)}\n    title: {safe(n.title)}\n"
        f"--- begin note ---\n{fence(n.body)}\n--- end note ---"
        for i, n in enumerate(input.notes)
    )
    return f"""You are the consolidation pass of a markdown memory vault. The notes below are near-duplicates of each other. Write the single note that replaces all of them.

NOTES TO MERGE
{listed}

Rules:
- Keep every fact from every note. Drop only repetition — two notes saying the same thing become one sentence.
- Where they disagree, keep both readings and say which note each came from; do not decide.
- Invent nothing. If it is not in a note above, it does not belong in the merged note.

Reply with one JSON object and nothing else — no prose, no markdown fence:
{{"title": "<title for the merged note>", "type": "<optional type, as the notes above use it>", "body": "<full markdown body>"}}"""  # noqa: E501
