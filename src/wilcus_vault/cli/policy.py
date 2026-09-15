"""The CLI's scope policy: `.vault-policy.json`, at the vault's root."""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from ..scope import ScopePolicy
from ..term import VaultError, printable

POLICY_FILE = ".vault-policy.json"


def outside_scoped_vaults(
    root: str | Path, command: str = "vault", advice: str | None = None
) -> Path:
    """`root` as an absolute path, refused when a directory above it holds a policy.
    Below a scoped vault's root that policy is out of sight, so every agent would run
    allow-all over its notes, and a policy written there would govern none of them.
    `command` and `advice` word the refusal; by default it points at the vault's root."""
    here = Path(os.path.abspath(root))  # as Vault does; the CLI has already resolved it
    for above in here.parents:
        if os.path.lexists(above / POLICY_FILE):
            raise VaultError(
                f"{command}: {here} is inside the scoped vault {above}; "
                f"{advice or f'use --vault {above}'}"
            )
    return here


def load_policy(root: str | Path) -> ScopePolicy | None:
    """The vault's policy, or None when there is no file: allow-all, the library's
    default. A file that is there but unusable raises rather than reading as absent,
    which would grant everything. The rules inside it are checked by `open()`."""
    path = outside_scoped_vaults(root) / POLICY_FILE
    if not os.path.lexists(path):  # a dangling symlink is there, and fails to read below
        return None
    try:
        policy = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    except (OSError, ValueError) as e:
        raise VaultError(f"vault: cannot load {POLICY_FILE}: {printable(e)}") from e
    if not isinstance(policy, dict):
        raise VaultError(f"vault: cannot load {POLICY_FILE}: not a JSON object of agent -> rules")
    return policy


def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """A `json.loads` hook that refuses a repeated key. JSON keeps the last value, so
    `"read": false, "read": true` would grant."""
    if repeated := [key for key, n in Counter(k for k, _ in pairs).items() if n > 1]:
        raise ValueError(f"duplicate key {json.dumps(repeated)}")
    return dict(pairs)
