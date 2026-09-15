"""Path rails: every string that names a file goes through here before it is used."""

import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .scope import normalize_prefix
from .term import VaultError, safe

MAX_SLUG = 80  # readable as a filename, short enough for every filesystem


def slugify(title: str) -> str | None:
    """A single filename segment, `[a-z0-9-]+`. Separators, dots and punctuation
    collapse to hyphens, so a title can never smuggle a path in. None when nothing
    survives, which a CJK or Cyrillic title does legitimately."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())[:MAX_SLUG].strip("-")
    return slug or None


def canonical_path(root: str | Path, path: str) -> str:
    """`path` in the form the scan stores: relative to the root, forward slashes. So
    `./x.md`, `a//x.md` and an absolute path inside the vault all name one note."""
    return os.path.relpath(os.path.join(root, path), root).replace("\\", "/")


def confined_path(root: str | Path, rel: str) -> Path:
    """Resolve `rel` under `root` and prove it stays inside: no `..` escape, no
    absolute path, and no hidden or symlinked directory on the way down.

    Callers pass strings from an LLM, a caller's namespace or a stale index;
    none of them is trusted to be a safe path.
    """
    if "\0" in rel:
        raise VaultError(f"vault: {safe(rel)} contains a NUL byte")
    base = Path(os.path.abspath(root))
    abs_path = Path(os.path.normpath(base / rel))
    if abs_path != base and base not in abs_path.parents:
        raise VaultError(f"vault: {rel} resolves outside the vault")
    walk = base
    # The root itself is not checked: a vault opened through a symlinked root is a vault.
    for segment in abs_path.relative_to(base).parts:
        if segment.startswith("."):
            raise VaultError(f"vault: {rel} passes through a hidden directory")
        walk = walk / segment
        if walk.is_symlink():
            raise VaultError(f"vault: {rel} passes through a symlink")
    return abs_path


def canonical_namespace(root: str | Path, namespace: str | None) -> str:
    """A namespace in the one form it is checked and written in: through the
    confinement rail, as a prefix. `notes/../ledger` is `ledger/`; the root is `""`."""
    base = Path(os.path.abspath(root))
    return normalize_prefix(confined_path(base, namespace or "").relative_to(base).as_posix())


def _staged(abs_path: Path, text: str) -> Path:
    """`text` in a temp file beside its target, ready to be moved into place. The
    temp name is not `.md`, so a crash leaves nothing the scan indexes.
    ponytail: a failed write leaves the temp file behind; sweep them in doctor if
    a full disk ever turns that into more than clutter."""
    tmp = abs_path.with_name(f"{abs_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    tmp.write_text(text, encoding="utf-8", newline="")
    return tmp


def write_atomic(abs_path: Path, text: str) -> None:
    """Write through a temp file and rename it over the target: a reader never
    sees a half-written note, and rename replaces a symlink instead of following it."""
    os.replace(_staged(abs_path, text), abs_path)


def write_new(abs_path: Path, text: str) -> bool:
    """Write a file that is not there yet, or answer False because it is.

    The name is claimed by linking the finished temp file into place: `link`
    fails rather than overwriting, so of two writers racing for one filename
    exactly one wins and the loser still holds its note. `os.replace` would let
    the second silently destroy the first.
    """
    tmp = _staged(abs_path, text)
    try:
        os.link(tmp, abs_path)
    except FileExistsError:
        return False
    except OSError:
        # No hardlinks on this filesystem (FAT, some FUSE and network mounts).
        # `O_EXCL` claims the name just as exclusively; the note is written in
        # place rather than moved in, which is the lesser loss on such a mount.
        return _write_exclusive(abs_path, text)
    finally:
        tmp.unlink(missing_ok=True)
    return True


def _write_exclusive(abs_path: Path, text: str) -> bool:
    try:
        fd = os.open(abs_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return True


def now() -> str:
    """`2026-08-01T09:41:00.000Z`: a UTC timestamp with millisecond precision."""
    t = datetime.now(UTC)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"
