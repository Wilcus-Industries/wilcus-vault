"""The write side of the discard log: JSONL beside the notes, append-only, rotated.

It holds whole candidate bodies, so it is durable history that lives next to the
notes rather than in the disposable `.vault/` directory, and it is gitignored.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

from .decision import Candidate
from .paths import now

# Rotation cap: bounded growth, nothing deleted. At this size the live log
# moves aside to `.discarded.<n>.log` and a fresh one starts.
DISCARD_LOG_CAP = 5 * 1024 * 1024
ROTATED = re.compile(r"^\.discarded\.(\d+)\.log$")
_IGNORE_LINE = ".discarded.log*"  # covers the live log and every rotation of it


def discard_log(root: str | Path) -> Path:
    return Path(root) / ".discarded.log"


def _open_nofollow(path: Path, flags: int) -> int:
    """Open a fixed, user-visible path without following a planted symlink, which
    would redirect whole note bodies out of the vault. ELOOP is raised, not hidden."""
    return os.open(path, flags | os.O_NOFOLLOW)


def read_nofollow(path: Path) -> str | None:
    """The file's text, or None when it does not exist."""
    try:
        fd = _open_nofollow(path, os.O_RDONLY)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, encoding="utf-8") as f:
        return f.read()


def append_nofollow(path: Path, text: str) -> None:
    fd = _open_nofollow(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


def log_candidate(
    root: str | Path, candidate: Candidate, extra: dict[str, Any], cap: int = DISCARD_LOG_CAP
) -> None:
    """Append a refused candidate, whole: a wrong decision costs a line in a log,
    never the information. `cap` is a parameter for the tests only."""
    root = Path(root)
    path = discard_log(root)
    # lstat, not stat: a symlink's own tiny size must not trip a rotation that
    # would rename the link into history as if it were the log.
    try:
        size = os.lstat(path).st_size
    except FileNotFoundError:
        size = -1
    if size >= cap:
        highest = max(
            (int(m.group(1)) for f in os.listdir(root) if (m := ROTATED.match(f))), default=0
        )
        os.rename(path, root / f".discarded.{highest + 1}.log")
    # A fresh log (including one a rotation just started) gets gitignored once.
    # A user who later strips the line has decided; it is not re-added.
    if not path.exists():
        _ensure_gitignore(root)
    line = {"at": now(), "candidate": candidate.to_json(), **extra}
    append_nofollow(path, json.dumps(line) + "\n")


def _ensure_gitignore(root: Path) -> None:
    """Add the discard-log pattern to `.gitignore` exactly once, appending only."""
    path = root / ".gitignore"
    current = read_nofollow(path) or ""
    if any(line.strip() == _IGNORE_LINE for line in current.split("\n")):
        return
    glue = "" if current == "" or current.endswith("\n") else "\n"
    append_nofollow(path, f"{glue}{_IGNORE_LINE}\n")
