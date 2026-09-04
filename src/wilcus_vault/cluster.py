"""Near-duplicate discovery: every live note pair under a distance ceiling,
agglomerated under complete linkage."""

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Cluster:
    members: list[str]  # vault-relative paths, sorted
    # The namespace every member shares (`""` is the root), or None when they
    # span namespaces: reported and never merged, since collapsing that boundary
    # is a human call.
    namespace: str | None
    distance: float  # the widest pair inside the cluster


def _key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def clusters(db: sqlite3.Connection, ceiling: float) -> list[Cluster]:
    """Complete linkage: a note joins a cluster only when it is under the ceiling
    from every note already in it. Single linkage would chain A~B~C and merge A
    with a C it does not resemble.

    All pairs, O(n²), accepted: notes embed whole, so n is the note count.
    Superseded notes are excluded and a note with no vector row never clusters.
    """
    pairs = db.execute(
        """select na.path as a, nb.path as b, vec_distance_cosine(va.emb, vb.emb) as d
           from vectors va join vectors vb on va.note_id < vb.note_id
           join notes na on na.id = va.note_id
           join notes nb on nb.id = vb.note_id
           where na.superseded_by is null and nb.superseded_by is null
             and vec_distance_cosine(va.emb, vb.emb) <= ?
           order by d, a, b""",
        (ceiling,),
    ).fetchall()
    under = {_key(p["a"], p["b"]): p["d"] for p in pairs}
    owner: dict[str, list[str]] = {}  # path -> the member list of its cluster, shared by identity
    # Ascending distance, so the closest notes cluster first and the answer does
    # not depend on which pair the query happened to return first.
    for p in pairs:
        left = owner.get(p["a"], [p["a"]])
        right = owner.get(p["b"], [p["b"]])
        if left is right:
            continue
        if not all(_key(x, y) in under for x in left for y in right):
            continue
        merged = sorted(left + right)
        for path in merged:
            owner[path] = merged

    seen: list[list[str]] = []
    for members in owner.values():
        if not any(members is s for s in seen):
            seen.append(members)
    out = []
    for members in seen:
        dirs = [p[: p.rfind("/") + 1] for p in members]
        widest = max(
            under[_key(members[i], members[j])]
            for i in range(len(members))
            for j in range(i + 1, len(members))
        )
        namespace = dirs[0] if all(d == dirs[0] for d in dirs) else None
        out.append(Cluster(members, namespace, widest))
    return sorted(out, key=lambda c: (c.distance, c.members[0]))
