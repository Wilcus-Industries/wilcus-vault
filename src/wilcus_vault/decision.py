"""What the write gate asks a decider and how it reads the answer."""

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Literal

from .note import Note
from .term import VaultError

Action = Literal["update", "supersede", "create", "discard"]
ACTIONS: tuple[Action, ...] = ("update", "supersede", "create", "discard")


@dataclass(frozen=True)
class Candidate:
    """A note something wants to write. `namespace` is a directory under the root."""

    title: str
    body: str
    type: str | None = None
    namespace: str | None = None

    def to_json(self) -> dict[str, str]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class SimilarNote:
    """An existing note the candidate might belong in, as the decider sees it."""

    note: Note
    score: float  # fused RRF score
    hash: str  # the file's hash at search time; the apply step re-checks it
    read_only: bool = False  # readable but not writable by the calling agent


@dataclass(frozen=True)
class DeciderInput:
    candidate: Candidate
    similar: list[SimilarNote]


@dataclass(frozen=True)
class Decision:
    action: Action
    target: str | None = None  # note to update or supersede; those actions only
    body: str | None = None  # body to write instead of the candidate's


# The caller wires an LLM (with gate_prompt + parse_decision); tests wire a fake.
Decider = Callable[[DeciderInput], Awaitable[Decision]]


def check_decision(value: object) -> Decision:
    """Validate a decider's answer. Malformed output raises: the gate never
    guesses what a model meant, because a wrong guess writes to the wrong file."""
    raw = asdict(value) if isinstance(value, Decision) else value

    def bad(why: str) -> VaultError:
        return VaultError(
            f"write gate: decider returned {why}: {json.dumps(raw, default=str)[:200]}"
        )

    if not isinstance(raw, dict):
        raise bad("a non-object")
    action, target, body = raw.get("action"), raw.get("target"), raw.get("body")
    if action not in ACTIONS:
        raise bad("an unknown action")
    if target is not None and not isinstance(target, str):
        raise bad("a non-string target")
    if body is not None and not isinstance(body, str):
        raise bad("a non-string body")
    if body is not None and body.strip() == "":
        raise bad("an empty body")  # a model that lost the text, not an instruction to blank a note
    needs_target = action in ("update", "supersede")
    if needs_target and not target:
        raise bad(f"{action} without a target")
    if not needs_target and target is not None:
        raise bad(f"{action} with a target")
    # Rebuilt rather than passed through: unknown keys from a model do not travel.
    return Decision(action=action, target=target, body=body)


def parse_decision(text: str) -> Decision:
    """Parse an LLM's reply. Strictly one JSON object; a fenced or chatty answer is malformed."""
    try:
        value = json.loads(text)
    except ValueError:
        raise VaultError(f"write gate: decider did not return JSON: {text[:200]}") from None
    return check_decision(value)


def fence(text: str) -> str:
    """Note text is data, not instruction: indent anything that could pass for one
    of the prompt's delimiters so a note cannot close its own fence."""
    return re.sub(r"^---[ \t]*(begin|end)", r" \g<0>", text.strip(), flags=re.I | re.M)


def gate_prompt(input: DeciderInput) -> str:
    """The prompt an LLM decider gets. `parse_decision` reads what it asks for."""
    candidate, similar = input.candidate, input.similar
    if not similar:
        notes = "(none — the vault has nothing similar)"
    else:
        notes = "\n\n".join(
            f"[{i + 1}] path: {s.note.path}{' (read-only)' if s.read_only else ''}\n"
            f"    title: {s.note.title}\n"
            f"--- begin note ---\n{fence(s.note.body)}\n--- end note ---"
            for i, s in enumerate(similar)
        )
    # A hint, not the enforcement: the write check in propose is the rail.
    read_only = (
        "\nA note marked read-only is outside the calling agent's write scope: "
        "do not name it as a target — choose create instead.\n"
        if any(s.read_only for s in similar)
        else ""
    )
    return f"""You are the write gate of a markdown memory vault. Decide what should happen to one candidate note.

CANDIDATE
title: {candidate.title}
type: {candidate.type if candidate.type is not None else "(none)"}
--- begin candidate ---
{fence(candidate.body)}
--- end candidate ---

EXISTING NOTES THAT LOOK SIMILAR
{notes}

Choose exactly one action:
- update: the candidate is a correction or addition to one existing note. Give that note's path as "target" and the complete new body as "body".
- supersede: an existing note is now wrong or outdated and the candidate replaces it. Give the old note's path as "target"; it will be marked superseded and linked forward.
- create: nothing above covers this. Do not give a target.
- discard: the candidate adds nothing that is not already written down. Do not give a target.
{read_only}
Reply with one JSON object and nothing else — no prose, no markdown fence:
{{"action": "update" | "supersede" | "create" | "discard", "target": "<path exactly as listed above, update and supersede only>", "body": "<full markdown body, optional>"}}"""  # noqa: E501
