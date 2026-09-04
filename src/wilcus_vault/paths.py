"""Path rails: every string that names a file goes through here before it is used."""

import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .term import VaultError, safe

MAX_SLUG = 80  # readable as a filename, short enough for every filesystem


def slugify(title: str) -> str | None:
    """A single filename segment, `[a-z0-9-]+`. Separators, dots and punctuation
    collapse to hyphens, so a title can never smuggle a path in. None when nothing
    survives, which a CJK or Cyrillic title does legitimately."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())[:MAX_SLUG].strip("-")
    return slug or None


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


def write_atomic(abs_path: Path, text: str) -> None:
    """Write through a temp file and rename it over the target: a reader never
    sees a half-written note, and rename replaces a symlink instead of following
    it. The temp name is not `.md`, so a crash leaves nothing the scan indexes."""
    tmp = abs_path.with_name(f"{abs_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, abs_path)


def now() -> str:
    """`2026-08-01T09:41:00.000Z`: a UTC timestamp with millisecond precision."""
    t = datetime.now(UTC)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"
