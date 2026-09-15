"""The CLI's scope policy: `.vault-policy.json`, at the vault's root."""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from ..scope import ScopePolicy
from ..term import VaultError, printable

POLICY_FILE = ".vault-policy.json"


def load_policy(root: str | Path) -> ScopePolicy | None:
    """The vault's policy, or None when there is no file: allow-all, the library's
    default. A file that is there but unusable raises rather than reading as absent,
    which would grant everything. The rules inside it are checked by `open()`."""
    here = Path(os.path.abspath(root))  # as Vault does; the CLI has already resolved it
    for above in here.parents:
        # Below a scoped vault's root its policy is out of sight, and every agent
        # would run allow-all over that vault's notes.
        if os.path.lexists(above / POLICY_FILE):
            raise VaultError(
                f"vault: {here} is inside the scoped vault {above}; use --vault {above}"
            )
    path = here / POLICY_FILE
    if not os.path.lexists(path):  # a dangling symlink is there, and fails to read below
        return None
    try:
        policy = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except (OSError, ValueError) as e:
        raise VaultError(f"vault: cannot load {POLICY_FILE}: {printable(e)}") from e
    if not isinstance(policy, dict):
        raise VaultError(f"vault: cannot load {POLICY_FILE}: not a JSON object of agent -> rules")
    return policy


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """JSON keeps a repeated key's last value, so `"read": false, "read": true` would grant."""
    if repeated := [key for key, n in Counter(k for k, _ in pairs).items() if n > 1]:
        raise ValueError(f"duplicate key {json.dumps(repeated)}")
    return dict(pairs)
