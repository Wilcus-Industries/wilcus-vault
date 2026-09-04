"""Per-agent namespace scopes. Policy is fixed at open(), identity travels per call.

Scopes are advisory containment at the library API, not security: any process
with filesystem access can read or edit the notes directly.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

from .term import VaultError, safe

Permission = Literal["read", "write"]
PERMISSIONS: tuple[Permission, ...] = ("read", "write")


class ScopeRule(TypedDict):
    """A namespace subtree and what it grants. A permission left out defers to
    the next-shorter matching rule; `""` is the root rule."""

    prefix: str
    read: NotRequired[bool]
    write: NotRequired[bool]


ScopePolicy = Mapping[str, list[ScopeRule]]  # agent name -> its rules


@dataclass(frozen=True)
class VaultContext:
    """Who is calling, per call. `source` records what prompted the call."""

    agent: str
    source: str | None = None


@dataclass(frozen=True)
class Rule:
    prefix: str  # normalized: `ledger/`, or `""` for the root
    read: bool | None
    write: bool | None

    def get(self, permission: Permission) -> bool | None:
        return self.read if permission == "read" else self.write


CompiledPolicy = dict[str, list[Rule]]  # longest prefix first


def normalize_prefix(prefix: str | None) -> str:
    """`ledger`, `/ledger/` and `ledger/` all name `ledger/`; `""`, `/` and None name the root."""
    ns = (prefix or "").strip("/")
    return f"{ns}/" if ns else ""


def _resolve(rules: list[Rule], permission: Permission, path: str) -> bool:
    """Longest matching prefix that specifies this permission decides; none means denied."""
    for rule in rules:
        if path.startswith(rule.prefix):
            allowed = rule.get(permission)
            if allowed is not None:
                return allowed
    return False


@dataclass(frozen=True)
class Scope:
    """What one call may touch. `rules` None means no policy: everything allowed."""

    ctx: VaultContext | None = None
    rules: list[Rule] | None = None

    def may(self, permission: Permission, path: str) -> bool:
        return True if self.rules is None else _resolve(self.rules, permission, path)

    @property
    def read_sql(self) -> tuple[str, list[str]]:
        """The read check as a SQL boolean over `n.path`, for filters inside a query.

        A CASE over the rules in longest-first order. `substr` rather than
        `like`, whose wildcards would need escaping; SQLite measures the prefix
        itself so multi-byte characters count the same on both sides.
        """
        if self.rules is None:
            return "1", []
        params: list[str] = []
        whens = []
        for rule in self.rules:
            if rule.read is not None:
                params += [rule.prefix, rule.prefix]
                whens.append(f"when substr(n.path, 1, length(?)) = ? then {int(rule.read)}")
        if not whens:
            return "0", []
        return f"case {' '.join(whens)} else 0 end", params


ALLOW_ALL = Scope()


def compile_scopes(policy: ScopePolicy | None) -> CompiledPolicy | None:
    """Validate and normalize a policy at open(), or None when there is none.

    A policy is operator config, so it is checked rather than trusted: two
    rules answering one question two ways are refused, and so is a subtree an
    agent could write but not read (it would never see its own notes).
    """
    if policy is None:
        return None
    compiled: CompiledPolicy = {}
    for agent, given in policy.items():
        compiled[agent] = _compile_rules(agent, given)
    return compiled


def _compile_rules(agent: str, given: Any) -> list[Rule]:
    def bad(why: str) -> VaultError:
        return VaultError(f"vault: scope policy for {json.dumps(safe(agent))}: {why}")

    if not isinstance(given, list):
        raise bad("must be a list of rules")
    for raw in given:
        if not isinstance(raw, dict):
            raise bad(f"a rule must be {{prefix, read?, write?}}, got {json.dumps(raw)[:60]}")
        prefix = raw.get("prefix")
        if not isinstance(prefix, str):
            raise bad("a rule's prefix must be a string")
        for permission in PERMISSIONS:
            if permission in raw and not isinstance(raw[permission], bool):
                raise bad(
                    f"{permission} must be true, false or absent, "
                    f"in rule {json.dumps(safe(prefix))}"
                )
        # Stored paths are canonical, so a prefix that is not matches nothing,
        # and a deny that matches nothing never fires. Refused, not guessed at.
        segments = prefix.strip("/").split("/")
        canonical = all(
            s not in (".", "..") and (s != "" or i == 0) for i, s in enumerate(segments)
        )
        if prefix != "" and not canonical:
            raise bad(f"{json.dumps(safe(prefix))} is not a canonical namespace")

    rules = sorted(
        (Rule(normalize_prefix(r["prefix"]), r.get("read"), r.get("write")) for r in given),
        key=lambda r: len(r.prefix),
        reverse=True,
    )
    seen: dict[str, Rule] = {}
    for rule in rules:
        prior = seen.get(rule.prefix)
        for permission in PERMISSIONS:
            if rule.get(permission) is not None and prior and prior.get(permission) is not None:
                raise bad(
                    f"two rules for {json.dumps(safe(rule.prefix))} both specify {permission}"
                )
        seen[rule.prefix] = Rule(
            rule.prefix,
            rule.read if rule.read is not None else (prior.read if prior else None),
            rule.write if rule.write is not None else (prior.write if prior else None),
        )
    # Resolution only changes at a rule's own prefix, so checking each covers every path.
    for rule in rules:
        if _resolve(rules, "write", rule.prefix) and not _resolve(rules, "read", rule.prefix):
            raise bad(
                f"{json.dumps(safe(rule.prefix))} is writable but not readable — an agent that "
                "cannot see its own notes re-creates them on every propose"
            )
    return rules


def scope_for(policy: CompiledPolicy | None, ctx: VaultContext | None) -> Scope:
    """The scope in force for one call. A policy fails closed: no context, or an
    agent it never names, raises rather than answering silently empty."""
    if policy is None:
        return Scope(ctx=ctx)
    if ctx is None:
        raise VaultError(
            "vault: this vault has a scope policy, so every call needs a VaultContext — "
            "VaultContext(agent='<name>')"
        )
    rules = policy.get(ctx.agent)
    if rules is None:
        raise VaultError(f"vault: agent {json.dumps(safe(ctx.agent))} has no scope in this vault")
    return Scope(ctx=ctx, rules=rules)
