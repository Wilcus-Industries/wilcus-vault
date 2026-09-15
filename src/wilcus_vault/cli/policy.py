"""The CLI's scope policy: `.vault-policy.json`, beside the notes."""

import json
import os
from pathlib import Path

from ..scope import ScopePolicy
from ..term import VaultError, printable

POLICY_FILE = ".vault-policy.json"


def load_policy(root: str | Path) -> ScopePolicy | None:
    """The vault's policy, or None when there is no file: allow-all, the library's
    default. A file that is there but unusable raises rather than reading as absent,
    which would grant everything. The rules inside it are checked by `open()`."""
    path = Path(root) / POLICY_FILE
    if not os.path.lexists(path):  # a dangling symlink is there, and fails to read below
        return None
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise VaultError(f"vault: cannot load {POLICY_FILE}: {printable(e)}") from e
    if not isinstance(policy, dict):
        raise VaultError(f"vault: cannot load {POLICY_FILE}: not a JSON object of agent -> rules")
    return policy
