"""Per-agent scopes: the policy compiler and resolver on their own, without a vault."""

import sqlite3
from typing import Any

import pytest
from conftest import MakeVault

from wilcus_vault.embed import TokenOverlapEmbedder
from wilcus_vault.scope import Permission, ScopePolicy, VaultContext, compile_scopes, scope_for
from wilcus_vault.term import VaultError
from wilcus_vault.vault import open

# The policy DESIGN.md prints: reads everywhere, writes everywhere except
# `ledger/`, which stays readable.
DESIGN_EXAMPLE: ScopePolicy = {
    "core/scheduler": [
        {"prefix": "", "read": True, "write": True},
        {"prefix": "ledger/", "write": False},
    ],
}

# `notes/` is this agent's own namespace; `ledger/` it may only read.
POLICY: ScopePolicy = {
    **DESIGN_EXAMPLE,
    "core/notes": [
        {"prefix": "notes/", "read": True, "write": True},
        {"prefix": "ledger/", "read": True},
    ],
}


def checker(policy: ScopePolicy, agent: str) -> Any:
    """The resolver itself, without a vault around it."""
    scope = scope_for(compile_scopes(policy), VaultContext(agent))
    return lambda perm, path: scope.may(perm, path)


def refuse(root: Any, scopes: Any, embedder: TokenOverlapEmbedder) -> None:
    """Open with `scopes` and close again; a refused policy raises before a handle opens."""
    open(root, embedder, scopes=scopes).close()


def test_open_refuses_a_policy_that_contradicts_itself(
    make_vault: MakeVault, embedder: TokenOverlapEmbedder
) -> None:
    root = make_vault({})
    # `ledger` and `ledger/` are one prefix once normalized, so these two rules
    # are two answers to the same question.
    with pytest.raises(VaultError, match="both specify read"):
        refuse(
            root,
            {"a": [{"prefix": "ledger", "read": True}, {"prefix": "ledger/", "read": False}]},
            embedder,
        )
    with pytest.raises(VaultError, match="both specify write"):
        refuse(
            root,
            {"a": [{"prefix": "", "read": True, "write": True}, {"prefix": "/", "write": True}]},
            embedder,
        )
    # Same prefix, different permissions, is not a contradiction.
    refuse(
        root, {"a": [{"prefix": "x/", "read": True}, {"prefix": "x/", "write": False}]}, embedder
    )

    # A write-blind agent never sees its own notes as `similar`, so every propose
    # lands as `create`: a duplicate factory, not a scope.
    with pytest.raises(VaultError, match="writable but not readable"):
        refuse(root, {"a": [{"prefix": "notes/", "write": True}]}, embedder)
    # ...including when the write is inherited from a shorter rule.
    with pytest.raises(VaultError, match="writable but not readable"):
        refuse(
            root,
            {
                "a": [
                    {"prefix": "", "read": True, "write": True},
                    {"prefix": "ledger/", "read": False},
                ]
            },
            embedder,
        )
    # Read-only is fine in the other direction: `ledger/` above is exactly that.
    refuse(root, POLICY, embedder)


def test_open_refuses_a_rule_that_is_not_a_rule(
    make_vault: MakeVault, embedder: TokenOverlapEmbedder
) -> None:
    root = make_vault({})
    # A policy usually arrives as operator config. `read: "false"` is truthy, so a
    # policy that grants where it meant to deny must not get past `open()`.
    with pytest.raises(VaultError, match="read must be true"):
        refuse(root, {"a": [{"prefix": "notes/", "read": "false"}]}, embedder)
    with pytest.raises(VaultError, match="write must be true"):
        refuse(root, {"a": [{"prefix": "notes/", "read": True, "write": 1}]}, embedder)
    with pytest.raises(VaultError, match="prefix must be a string"):
        refuse(root, {"a": [{"prefix": 7, "read": True}]}, embedder)
    with pytest.raises(VaultError, match="rule must be"):
        refuse(root, {"a": [None]}, embedder)
    with pytest.raises(VaultError, match="must be a list of rules"):
        refuse(root, {"a": {"prefix": "notes/", "read": True}}, embedder)
    # A misspelt permission is no permission: `wirte: false` would be a deny that never fires.
    with pytest.raises(VaultError, match="unknown key"):
        refuse(root, {"a": [{"prefix": "", "read": True, "wirte": False}]}, embedder)

    # A prefix that is not the canonical form of a path matches nothing, so a
    # deny spelled that way is a deny that never fires. Refused, not normalized.
    for prefix in ["./ledger", "ledger//sub", "ledger/./sub", "ledger/../x", ".."]:
        with pytest.raises(VaultError, match="not a canonical namespace"):
            refuse(root, {"a": [{"prefix": prefix, "read": True}]}, embedder)


def test_a_prefix_is_a_namespace_matched_on_segment_boundaries_only() -> None:
    may = checker(DESIGN_EXAMPLE, "core/scheduler")
    # `ledger/` is the write-denied subtree; its sibling namespace is not.
    assert may("write", "ledger/q3.md") is False
    assert may("write", "ledger/2026/q3.md") is False
    assert may("write", "ledger-archive/q3.md") is True
    assert may("write", "ledgers/q3.md") is True
    # `""` is the root rule: it matches every note, and a root-level note (no `/`
    # in its path) matches only it.
    assert may("write", "root-note.md") is True
    only_ledger = checker({"a": [{"prefix": "ledger/", "read": True, "write": True}]}, "a")
    assert only_ledger("read", "q3.md") is False


def test_resolution_is_per_permission_longest_prefix_wins_unspecified_defers() -> None:
    may = checker(DESIGN_EXAMPLE, "core/scheduler")
    assert may("read", "ledger/q3.md") is True
    assert may("write", "ledger/q3.md") is False
    assert may("read", "secret/plans.md") is True
    assert may("write", "secret/plans.md") is True

    # Three deep, the middle rule specifying one half of the answer and
    # deferring the other to the root rule.
    deep = checker(
        {
            "a": [
                {"prefix": "", "read": True, "write": False},
                {"prefix": "a/", "write": True},
                {"prefix": "a/b/", "read": False, "write": False},
            ]
        },
        "a",
    )
    assert [deep("read", "elsewhere.md"), deep("write", "elsewhere.md")] == [True, False]
    assert [deep("read", "a/x.md"), deep("write", "a/x.md")] == [True, True]
    assert [deep("read", "a/b/x.md"), deep("write", "a/b/x.md")] == [False, False]

    # Nothing specifies ⇒ denied, both ways.
    thin = checker({"a": [{"prefix": "notes/", "read": True}]}, "a")
    assert thin("write", "notes/x.md") is False
    assert thin("read", "other/x.md") is False


def test_the_sql_read_filter_answers_exactly_what_may_answers() -> None:
    # `search` filters inside the query, so the rules exist twice: as `may` and
    # as SQL. This is the test that holds the two spellings together, and why the
    # prefix is measured by SQLite rather than by the host language.
    policies: list[ScopePolicy] = [
        {"a": []},
        {"a": [{"prefix": "", "read": True, "write": False}]},
        {
            "a": [
                {"prefix": "notes/", "read": True, "write": True},
                {"prefix": "ledger/", "read": True},
            ]
        },
        # deny inside an allow: the `then 0` arm nothing else in this suite reaches
        {"a": [{"prefix": "", "read": True, "write": False}, {"prefix": "secret/", "read": False}]},
        {
            "a": [
                {"prefix": "", "read": True, "write": False},
                {"prefix": "🔒secret/", "read": False},
            ]
        },
        {
            "a": [
                {"prefix": "", "read": True, "write": False},
                {"prefix": "a/", "read": False},
                {"prefix": "a/b/", "read": True},
            ]
        },
    ]
    probes = [
        "x.md",
        "notes/x.md",
        "notes/deep/x.md",
        "ledger/q3.md",
        "ledger-archive/q3.md",
        "secret/plans.md",
        "secretive/plans.md",
        "🔒secret/plans.md",
        "🔒secretive/plans.md",
        "a/x.md",
        "a/b/x.md",
    ]
    db = sqlite3.connect(":memory:")
    for policy in policies:
        scope = scope_for(compile_scopes(policy), VaultContext("a"))
        sql, params = scope.read_sql
        for path in probes:
            (ok,) = db.execute(
                f"select ({sql}) as ok from (select ? as path) as n", [*params, path]
            ).fetchone()
            # The policy and path ride along so a failure names which pair drifted.
            perm: Permission = "read"
            assert (policy, path, ok == 1) == (policy, path, scope.may(perm, path))
    db.close()
