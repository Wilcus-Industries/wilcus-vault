"""`vault init --layout swarm`: a roster becomes a swarm's tiered namespaces and their policy.

The library knows nothing about roles. What comes out is directories and a plain
ScopePolicy, which `open()` enforces like any other.
"""

import json
from pathlib import Path
from typing import Any

from ..paths import confined_path, write_atomic
from ..scope import ScopeRule, compile_scopes
from ..term import VaultError, printable, safe
from .commands import Args
from .policy import POLICY_FILE, _unique_keys
from .usage import USAGE

KINDS = ("orchestrator", "manager", "doer", "worker")
Roster = dict[str, dict[str, Any]]


def cmd_init(args: Args, rest: list[str]) -> int:
    if rest:
        raise VaultError(f"init takes no arguments\n\n{USAGE}")
    if args.layout != "swarm":
        raise VaultError(f"init needs --layout swarm\n\n{USAGE}")
    if args.roster is None:
        raise VaultError(f"init needs --roster <file>\n\n{USAGE}")
    roster = load_roster(args.roster)
    policy = {role: _rules(role, row) for role, row in roster.items()}
    compile_scopes(policy)  # so init never writes a policy open() would refuse
    owners = [role for role, row in roster.items() if row["kind"] in ("manager", "doer")]
    dirs = ["shared", *(f"{tier}/{role}" for role in owners for tier in ("roles", "proposals"))]
    # Every directory is confined before the first is made, so a refusal writes nothing.
    paths = [(rel, confined_path(args.root, rel)) for rel in dirs]
    for rel, path in paths:
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            print(f"created {rel}/")
    # Replaced whole: the roster is the source, so a row it drops loses its scope.
    write_atomic(Path(args.root) / POLICY_FILE, json.dumps(policy, indent=2) + "\n")
    print(f"wrote {POLICY_FILE}")
    return 0


def load_roster(path: str) -> Roster:
    """The roster, checked whole before anything is written. Every error names its row."""
    try:
        roster = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except (OSError, ValueError) as e:
        raise VaultError(f"init: cannot load roster {path}: {printable(e)}") from e
    if not isinstance(roster, dict):
        raise VaultError("init: the roster must be a JSON object of role -> {kind, manager?}")

    def bad(role: str, why: str) -> VaultError:
        return VaultError(f"init: roster row {json.dumps(safe(role))}: {why}")

    for role, row in roster.items():
        # The name is the --agent, the policy key and a directory, all verbatim: a name
        # that is not one path segment is refused, never transformed into one.
        if role == "" or role.startswith(".") or "/" in role or "\\" in role or safe(role) != role:
            raise bad(
                role,
                "a role name must be one path segment "
                "(not empty, no / or \\, not starting with ., no control characters)",
            )
        if not isinstance(row, dict) or row.get("kind") not in KINDS:
            raise bad(role, f"kind must be {', '.join(KINDS[:-1])} or {KINDS[-1]}")
    # A second pass, so a worker's manager row has already been checked.
    for role, row in roster.items():
        manager = roster.get(row["manager"]) if isinstance(row.get("manager"), str) else None
        if row["kind"] == "worker" and (manager is None or manager["kind"] != "manager"):
            raise bad(role, "a worker's manager must name a row whose kind is manager")
    return roster


def _rules(role: str, row: dict[str, Any]) -> list[ScopeRule]:
    """The orchestrator gets the whole vault. A manager or doer reads shared/ and owns its
    two directories. A worker reads shared/ and its manager's roles/, and has no write rule,
    so its writes fail closed."""
    if row["kind"] == "orchestrator":
        return [{"prefix": "", "read": True, "write": True}]
    if row["kind"] == "worker":
        return [
            {"prefix": "shared/", "read": True},
            {"prefix": f"roles/{row['manager']}/", "read": True},
        ]
    return [
        {"prefix": "shared/", "read": True},
        {"prefix": f"roles/{role}/", "read": True, "write": True},
        {"prefix": f"proposals/{role}/", "read": True, "write": True},
    ]
