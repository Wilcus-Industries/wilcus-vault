# wilcus-vault

Files-are-truth markdown memory vault for AI agents: atomic notes, wikilink graph,
hybrid semantic + keyword search (sqlite-vec + FTS5 + reciprocal rank fusion), and a
write gate so nothing is silently overwritten. Open the vault in Obsidian or any
editor — the SQLite index is derived and disposable (`vault doctor --rebuild`).

Status: MVP. Standalone (zero wilcus dependencies), MIT. See DESIGN.md for the
architecture contract.

## Files are truth

The `.md` files on disk are the only source of truth. Everything in
`<vault>/.vault/index.db` — note rows, wikilink edges, FTS5 text, embeddings — is
derived and rebuildable, and no code path may treat it as authoritative. So:

- edit, move, rename or delete notes with any editor, or `git checkout` the whole
  vault, and nothing is lost — `vault doctor` reconciles the index to the files;
- delete `index.db` whenever you like; `vault doctor --rebuild` recreates it;
- the vault is a plain directory of markdown. Nothing here owns it.

## Install

Python 3.12+ and [uv](https://docs.astral.sh/uv/). The system SQLite must be
3.39 or newer (`FULL OUTER JOIN`) and the Python's `sqlite3` must allow
`enable_load_extension` — most Linux distro builds do; a macOS system Python may
not, so use a uv-managed or Homebrew one there. Not published to PyPI yet:

```
git clone https://github.com/Wilcus-Industries/wilcus-vault && cd wilcus-vault
uv sync
uv run vault --help
```

`uv tool install .` puts a `vault` on your PATH. As a library, depend on the
directory (`uv add ../wilcus-vault`) and `from wilcus_vault import open,
FetchEmbedder`.

## CLI

```
vault reindex [--vault <dir>]            # index new and changed notes
vault doctor [--rebuild] [--vault <dir>] # check and repair the index
vault search <query> [--agent <a>]       # hybrid search, best first
vault watch [--vault <dir>]              # index changes as they are saved
vault propose --ceiling <d> [--agent <a>] [--namespace <ns>] < note.md
                                         # write a note through the gate
vault promote <path> --ceiling <d> [--agent <a>]
                                         # a proposals/ note, through the gate into shared/
vault get <path> [--agent <a>]           # print one note's file
vault list [prefix] [--agent <a>]        # note paths, one per line
vault consolidate --ceiling <d>          # report near-duplicate clusters
vault discards list                      # the discard log, newest first
vault discards show <n>                  # one refused candidate, in full
vault discards restore <n> --ceiling <d> # re-propose it through the gate
vault init --layout swarm --roster <file>
                                         # set up a swarm's tiered memory
vault --help                             # every command and flag
```

`--vault` defaults to the current directory, and is resolved through symlinks
before anything reads it, so a command's policy, index and notes all come from
one directory; `--` ends flag parsing, so a query may start with a dash. Exit code 0 on success, 1 on error — and 1 from `doctor`
when it found links only a human can fix: broken (nothing to point at) or
ambiguous (a bare `[[stem]]` several notes answer to — `doctor` prints the
candidate paths, and qualifying the link with one of them is the fix) — or when
`doctor`'s own reindex hit an error it could not absorb (a confinement or
permission failure, or a failed re-entry).

Every command embeds through the zero-config default described under
[Embedders](#embedders): a local Ollama (`all-minilm`, 384 dims), or whatever
`VAULT_EMBED_*` points at. Run `ollama pull all-minilm` once; with nothing
listening the command says so in one line — `no embedder configured: start
Ollama ...` — and exits 1, rather than quietly falling back to a cloud API.

`--lexical` swaps in the deterministic `TokenOverlapEmbedder` instead: no
daemon, no network, and no semantics either — for an offline machine, and what
the test suite runs on. The two are different vector spaces, so switching costs
a full re-embed: `reindex`, `doctor` and `watch` do it on their next pass, and
`search` refuses in the meantime (`... — run vault doctor to re-embed`) rather
than compare vectors that are not comparable.

```
$ vault search renewal terms
0.0328  customers/acme.md — Acme Corp
0.0161  customers/globex.md — Globex
```

One line per hit: fused RRF score, vault-relative path, title. The CLI sets no
relevance cutoffs — the ceiling that means "irrelevant" is a property of the
embedder, and it cannot know yours (under `--lexical` no fixed one is meaningful
at all) — so it shows the ranking and lets you judge it. Library callers pass
their own (`cutoffs`), and the write gate must.

`vault watch` reindexes once, then follows `watchfiles` (recursive) with a ~250ms
per-path debounce, re-embedding only notes whose content hash actually changed.
It logs the passes that changed something and stops on ctrl-c (finishing the
pass in flight first). It is a convenience, never a source of
truth: anything it misses — a directory rename, a pass that failed, a crash —
`vault doctor` finds and fixes.

```
$ vault consolidate --ceiling 0.15
indexed 0 new, 0 changed, 0 removed, 214 unchanged
0.0412  customers/acme.md customers/acme-corp.md
0.1180  cross-namespace  notes/pager.md support/rota.md
```

`vault consolidate` is report-only: one line per near-duplicate cluster —
widest distance inside it, then the note paths, with the clusters that span
namespaces flagged, because those are never merged. It reindexes first, so the
report describes the files rather than a stale index. Merging is a library
call — it needs a merger you inject, and the CLI wires none — so this is the
pass you run to find the ceiling your embedder calls a duplicate. See
[Consolidation](#consolidation).

`propose`, `promote`, `get`, `list`, `search` and `discards` act for an agent: `--agent
<name>` is the caller, and a `.vault-policy.json` at the vault's root is what it
is checked against — the [Scopes](#scopes) policy, as JSON:

```json
{
  "clerk": [
    {"prefix": "", "read": true, "write": true},
    {"prefix": "ledger", "write": false}
  ]
}
```

With no file, any agent may do anything and `--agent` is optional. With one,
every call needs an `--agent` the policy names, and answers only what that agent
may touch. `discards` runs only for an agent that may read the whole vault,
because the log holds refused candidates from every namespace, and `restore`
writes through the gate as that agent. A file that is there but cannot be used —
not JSON, a key given twice, not an object, a rule `open()` refuses, a dangling
symlink — is an error, never allow-all. So is a `--vault` inside a scoped vault,
where the file is out of sight: the error names the root to use. The file sits at
the root rather than in `.vault/` because `.vault/` is disposable, and a policy
deleted along with the index would fail open. `reindex`, `doctor`, `watch` and
`consolidate` never read it.

```
$ vault propose --agent clerk --namespace customers --ceiling 0.35 < renewal.md
create  customers/acme-renewal-2026.md
$ vault list customers --agent clerk
customers/acme-renewal-2026.md
customers/acme.md
```

`propose` reads a note's markdown on stdin and takes its title and type from the
frontmatter or the first `# heading`; no other frontmatter key is kept, since the
gate writes the frontmatter of the notes it authors. A note with no title, or
with malformed frontmatter, is refused. It reindexes, so the gate sees notes written by hand,
then puts the note through the write gate — which needs `--ceiling` and the same
chat model `discards restore` does — and prints the action and the path, plus
`(fell back)` when the gate had to create the note instead of applying its
decision. `--namespace` is where a created note goes: the vault root without it.
`get` prints the note's file, control characters scrubbed but for newlines and
tabs, or exits 1 with `no note at <path>` — one answer for a note that is not
there and a note the agent may not read. `list` reindexes first too, then prints
one path per line. It, `propose`, `promote` and `discards restore` send the reindex
summary to stderr, so stdout holds only the answer.

```
$ vault promote proposals/clerk/acme-renewal.md --agent orchestrator --ceiling 0.35
update  shared/acme.md
proposal removed
```

`promote` sends a note under `proposals/` through the same gate into `shared/` —
its title, type and body, as if proposed — then removes it. The decider judges it
against `shared/` alone, so a copy in `roles/` or another proposal can never
absorb it, and the note it writes records the proposal's path as `vault_source`.
The agent must be able to read and write the proposal and write `shared/`, and a
proposal with malformed frontmatter is refused, all before the chat model runs.
Just before the proposal is removed, its title, type and body are appended to
`.discarded.log` with reason `promoted` and the path they landed at, which
`discards show` prints (after a `discard` the gate's own entry is that record),
so a decider's merge that drops a fact loses nothing. A proposal edited while the gate ran is kept, and the second
line says `proposal kept: it changed during promote`. A path outside
`proposals/` is refused however it is spelled
(`proposals/../shared/x.md` is `shared/x.md`), and a path with no note exits 1.

`vault init --layout swarm --roster <file>` sets a vault up as a swarm's tiered
memory. The roster names each role and its kind (other keys are ignored):

```json
{
  "lead": {"kind": "orchestrator"},
  "planner": {"kind": "manager"},
  "coder": {"kind": "doer"},
  "helper": {"kind": "worker", "manager": "planner"}
}
```

```
$ vault init --layout swarm --roster roster.json --vault memory
created shared/
created roles/planner/
created proposals/planner/
created roles/coder/
created proposals/coder/
wrote .vault-policy.json
```

The policy it writes gives the orchestrator the whole vault. A manager or doer
reads `shared/` and writes its own `roles/<role>/` and `proposals/<role>/`. A
worker reads `shared/` and its manager's `roles/`, and writes nothing. A role
name is the `--agent` that role passes and its directory name, verbatim, so a
name that is not one path segment (blank, a `/` or `\`, a leading `.`, a control
character, or not valid UTF-8) is refused rather than rewritten. So is an unknown
`kind`, or a worker whose `manager` is not a manager row. Each error names its
row, and nothing is written until the whole roster checks out. Re-running init
keeps existing directories and replaces the policy, since the roster is its
source.

init runs only on a directory that is not there yet, is empty, or already holds
its own `.vault-policy.json`, and never inside a scoped vault. A policy scopes
everything below it, and `--vault` defaults to the current directory, so init
from a project root above a live swarm would otherwise lock that swarm's memory
out. As a result an existing unscoped vault is not converted in place: set the
swarm up in a fresh directory. This layout
narrows reads, which [Scopes](#scopes) otherwise advises against; DESIGN.md § One
shared memory says why it is acceptable here.

## Obsidian

Point Obsidian (or any editor) at the vault directory and work normally. Notes
are ordinary markdown with YAML frontmatter and `[[wikilink]]`s, one note per
file, subdirectories as namespaces.

Links resolve the way Obsidian resolves them, namespace-aware:

- `[[customers/acme]]` — a path-qualified link: the vault-relative path without
  `.md`. Always unambiguous, whatever else the vault holds. Prefer it; it is what
  the write gate writes for a note in a namespace (a note at the vault root has
  no qualified form — its path without `.md` *is* its stem — so keep notes you
  link to in namespaces);
- `[[acme]]` — a bare stem: resolves only while exactly one note in the vault is
  named `acme.md`. `customers/acme.md` and `vendors/acme.md` are two perfectly
  good notes, but a bare `[[acme]]` between them means nothing, so it stays
  unresolved rather than picking one. `vault doctor` lists it as ambiguous with
  both candidates, written as links, so the fix is a copy-paste:

```
$ vault doctor
ambiguous link: notes/deal.md -> [[acme]] (customers/acme, vendors/acme)
broken link:    notes/deal.md -> [[ghots]]
```

Most bare links never get that far: at the moment a second `acme.md` is first
indexed, every existing bare `[[acme]]` still unambiguously means the note that
was already there, so the pass rewrites them to its path-qualified form
(`[[acme|alias]]` keeps its alias) and says so. Only that moment can know the
incumbent — afterwards nothing records which note came first. The rule is
*never guesses*, not *never ambiguous*: an incumbent at the vault root (no
qualified form), one whose path a wikilink cannot carry (`[`, `]` or `|` in a
directory name), one whose file is already gone (a move, not a collision),
both notes appearing in one pass (no incumbent), or a linking note edited
mid-rewrite are left alone and fall through to `doctor`'s ambiguous report
above.

Obsidian hides dot-directories, so `.vault/` stays out of the way; the scan skips
it (and `.git/`, `.obsidian/`, …) for the same reason. If the vault is a git repo,
add `.vault/` to its `.gitignore` — the index is a build artifact, not content.
Run `vault watch` alongside your editing session to keep search current.

## Library

```python
from wilcus_vault import (
    Candidate,
    Cutoffs,
    GateOptions,
    SearchOptions,
    TokenOverlapEmbedder,
    VaultContext,
    gate_prompt,
    open,
    parse_decision,
)


# your LLM call; `gate_prompt` and `parse_decision` are the wiring, not the model
async def decider(input):
    return parse_decision(await ask_your_model(gate_prompt(input)))


vault = open(
    "/path/to/vault",
    TokenOverlapEmbedder(),
    # mandatory: without cutoffs "most similar" degrades into "least unrelated"
    gate=GateOptions(decider=decider, cutoffs=Cutoffs(distance_ceiling=0.35, bm25_ceiling=-1)),
    # optional: per-agent namespace rules. Omitted, every caller may do anything.
    scopes={"core/scheduler": [{"prefix": "", "read": True, "write": True}]},
)

ctx = VaultContext(agent="core/scheduler", source="task-42")  # who is calling, per call

await vault.reindex()
await vault.search(
    "acme renewal", SearchOptions(n=5, cutoffs=Cutoffs(distance_ceiling=0.35), ctx=ctx)
)
await vault.get("customers/acme.md", ctx)  # one note, parsed — read from the file
vault.list("customers", ctx)  # note paths under a namespace, sorted
await vault.propose(
    Candidate(title="Acme renewal 2026", type="customer", namespace="customers", body=body),
    ctx,  # optional — until a scope policy is in force, which needs it to decide
)
await vault.doctor()

watcher = vault.watch()  # keep the index warm while a human edits (needs a running loop)
await watcher.close()  # completes when the pass in flight is done
vault.close()  # ...so this cannot close the DB under a write
```

### Reading notes

`get` is a note's identity — its vault-relative path, `.md` and all — turned
into the parsed note: frontmatter, body, title, wikilinks, hash. It reads the
**file**, so it is never stale, whatever the index thinks; it returns `None`
when nothing is there, and a directory or a symlink at the path counts as
nothing. A path that leaves the vault, or runs through `.vault/` or a symlinked
directory, raises — that is a caller bug, not a missing note.

Spell the path however your code built it — `./customers/acme.md`, a doubled
slash, an absolute path inside the vault — and `note.path` still comes back as
the one canonical identity, which is what makes it safe to store.

`list` answers the cheap question from the index: which notes exist. Paths only,
sorted, optionally under one namespace — and a namespace means whole segments,
so `list("ledger")` never sweeps in `ledger-archive/`. Superseded notes are
listed; `list` is the note set, not the search set.

```python
note = await vault.get("customers/acme.md")  # Note | None
note.links  # ["support-rota", ...]
vault.list()  # every note path, sorted
vault.list("customers/")  # "customers" works too
```

Both take an optional trailing `VaultContext`, which decides what they answer
once the vault has a scope policy — see [Scopes](#scopes).

### The write gate

`propose` is the only way a program writes to the vault:

1. hybrid-search the candidate against the vault, with mandatory relevance
   cutoffs, and re-read each hit from disk;
2. ask your `decider` for one action — `update`, `supersede`, `create` or
   `discard` — over those notes;
3. apply it behind two rails. **Check-and-write:** a target is re-hashed
   immediately before it is touched, so a human edit mid-flight aborts the apply,
   re-runs the gate once, then falls back to `create`. **Path confinement:** every
   written path is `<namespace>/<slug>.md` with a single slugified segment,
   resolved under the vault root, never through a symlink or dot-directory.

Writes land through a temp file, so a reader never sees half a note: an update
renames it over the target, and a new note hardlinks it into place, which fails
rather than overwrite when another writer already claimed the name. A discarded candidate — or one the gate cannot place at all — is appended
whole to `<root>/.discarded.log`, along with the similar set the decider saw;
losing the note is never an outcome. That log is
history, not index, so it sits beside the notes rather than in the disposable
`.vault/` — a `doctor --rebuild` or an `rm -rf .vault` leaves it alone (a log left
in the old place is moved out by the next repairing `vault doctor`). It rotates at
5 MiB (`.discarded.N.log`, nothing deleted), and its first write adds
`.discarded.log*` to the vault's `.gitignore` so refused bodies stay out of git
history. `vault discards` reviews it — `list`, `show <n>`, and `restore <n>`,
which feeds the candidate back through the gate against the *current* vault
(`restore` needs `--ceiling` and a chat model via `VAULT_DECIDE_MODEL`, plus
`VAULT_DECIDE_ENDPOINT` / `VAULT_DECIDE_API_KEY` off the local Ollama default);
`vault doctor` reports the entry count. Notes the
gate did not author are patched textually, never re-serialized, so comments,
`01234` and `1.0` survive. Human edits bypass the gate by definition:
`vault watch` and `vault doctor` pick them up.

```python
result = await vault.propose(candidate, VaultContext(agent="core/scheduler"))
# GateResult(action="supersede", path="customers/acme-renewal-2026.md",
#            superseded="customers/acme.md", unmarked=None, fell_back=False)
```

The second argument is a `VaultContext` — `agent` plus an optional `source`, the caller's
identity for that one call. Given, the gate stamps `vault_agent` (and
`vault_source`) into the frontmatter of every note it *authors*: a `create`, and
a `supersede`'s successor, get them serialized in; an `update` gets them patched
in beside its `updated` bump. Marking the superseded note is bookkeeping rather
than authorship, so its own provenance is left alone. Omit the context and
nothing is stamped — on an `update` that also means the previous call's keys are
*removed*, so `vault_agent` never names an agent that did not write the note it
sits on. Omitting it stops being an option once the vault has a scope policy,
which has nothing to check the write against without it.

`promote` puts a note already in the vault through that same gate, then removes
it: what an orchestrator does with a peer's proposal.

```python
result = await vault.promote("proposals/clerk/acme.md", "shared", ctx)
# PromoteResult(action="update", path="shared/acme.md", superseded=None,
#               unmarked=None, fell_back=False, removed=True)
```

The candidate is the note's title, type and body, placed in the namespace, and
the gate runs confined to that namespace: the decider is shown only its notes and
can target nothing outside it. `ctx.source` defaults to the note's path. The
agent must be able to read the note and write both it and the namespace, and a
note with malformed frontmatter is refused, all before the decider runs; a note
it may not read is `no note at`, like an absent one. Whatever the gate decides,
the note is then removed only if its hash is unchanged, and just before that the
candidate is appended to the discard log with reason `promoted` and the landed
`path` (after a `discard`, the gate's own entry is that record). One a peer
edited meanwhile is kept, with `removed=False`. The library knows no layout;
`proposals/` and `shared/` are the CLI's.

### Consolidation

A vault accretes near-duplicates: the gate only sees the top-k similar notes at
write time, and humans add notes behind its back. `consolidate` is the
deliberate, occasional merge pass — manually triggered, never a daemon.

```python
from wilcus_vault import ConsolidateOptions, ConsolidateRun, merge_prompt, parse_merged


# your LLM call again; `merge_prompt` and `parse_merged` are the wiring
async def merger(input):
    return parse_merged(await ask_your_model(merge_prompt(input)))


vault = open("/path/to/vault", embedder, consolidate=ConsolidateOptions(merger=merger))

# dry run: what a merge pass *would* do, and it is the default
report = await vault.consolidate(ConsolidateRun(ceiling=0.15))
report.merges[0]  # Merge(cluster=Cluster(members, namespace, distance), candidate)
report.cross_namespace  # clusters spanning namespaces — reported, never merged
report.remaining  # clusters the cap did not reach
report.errors  # write runs: clusters whose merge raised — the run continues

await vault.consolidate(ConsolidateRun(ceiling=0.15, cap=3, write=True, ctx=ctx))
# merges[0] → Merge(..., path="customers/acme.md", superseded=[...], unmarked=[])
```

- **The ceiling is mandatory**, like the gate's cutoffs and for the same
  reason: there is no universal number for "duplicate", it is a property of
  your embedder. Find yours with `vault consolidate --ceiling <d>` before you
  let anything write.
- A cluster admits a note only if it is under the ceiling from **every** note
  already in it (complete linkage). Single linkage chains A~B~C and merges A
  with a C it does not resemble. The scan is all-pairs over the vectors table,
  O(n²) and accepted — notes embed whole, so n is the note count.
- Clusters that **span namespaces** are reported and never merged: namespaces
  are boundaries, and collapsing one is a human call. Superseded notes never
  cluster, and neither do notes the embedder had no tokens for (a CJK or
  emoji-only note has no vector row — it stays findable through FTS).
- **Dry-run is the default**; `write=True` is what makes a run act. A wrong
  ceiling found in a report costs nothing. A dry run still reindexes first (so
  discovery sees the files, not a stale index) and still calls your merger once
  per cluster it would merge, up to the cap.
- A write goes through the gate's own rails: the merged note is created with
  slug confinement, collision suffixing and `ctx`'s provenance, then **each**
  member is marked `superseded_by` it — a textual patch, check-and-write per
  member, and a member a human edited mid-flight comes back in `unmarked`
  rather than being clobbered. Nothing is ever deleted: the originals stay on
  disk, marked, out of search.
- On a **write run** a cluster whose merge raises (merger error, no free
  filename) lands in `errors` with its message — and with the merged note's
  `path` when it was created before the raise — and the run continues: merges
  that already landed are reported, not discarded behind one exception, and
  the closing reindex still runs so the index never lags them (a reindex
  failure comes back in `index_error` rather than discarding the report). A
  dry run still raises: nothing has landed that a report would need to
  account for.
- The **cap** (default 5) counts clusters whose merger ran — merged or errored
  after the call; an error before it burns no slot — and the rest come back in
  `remaining`. A pass that wants to rewrite half the vault
  is evidence the ceiling is wrong, and the cap turns that into a short
  report instead of a long mess.

Consolidation is an operator operation like `doctor` — unscoped, whole-vault —
so its `ctx` is provenance for the notes it writes, not a permission check.

### Scopes

Several agents usually share one vault. `scopes` says who may touch what —
namespace prefixes, per agent, for `read` and `write` independently:

```python
vault = open(
    "/path/to/vault",
    embedder,
    gate=gate,
    scopes={
        "core/scheduler": [
            {"prefix": "", "read": True, "write": True},
            {"prefix": "ledger/", "write": False},
        ],  # reads it, cannot rewrite it
        "core/support": [
            {"prefix": "support/", "read": True, "write": True},
            {"prefix": "customers/", "read": True},
        ],  # read-only next door
    },
)

support = VaultContext(agent="core/support")
await vault.search("acme renewal", SearchOptions(ctx=support))
await vault.get("ledger/q3.md", support)  # None: not readable
vault.list("customers", support)  # only what it may read
await vault.propose(candidate, support)
```

- **No `scopes` means allow-all**, so a single-agent caller changes nothing.
  With one, the vault fails closed: every call needs a `VaultContext`, an agent
  the policy does not name is refused with a `VaultError` rather than an empty result,
  and `{}` denies everyone.
- A prefix matches **whole segments** — `ledger/` never matches
  `ledger-archive/`; `""` is the root rule. For each of `read` and `write`
  separately, the longest matching prefix that *specifies* it wins; a rule that
  leaves one out defers to the next-shorter match, and nothing specifying it
  means denied. `open()` refuses a policy that answers one question twice, one
  with a subtree writable but not readable (an agent that cannot see its own
  notes re-creates them on every propose), a rule that is not
  `{prefix, read?, write?}` with booleans and no other key — a misspelt `wirte`
  would be ignored, and a JSON `"read": "false"` is truthy, so either would grant
  where it meant to deny — and a prefix that is not a canonical
  path (`./ledger`, `ledger//sub`), which would match nothing and deny nothing.
- `search` filters unreadable notes out of the over-fetched set before capping,
  so you get *up to* N readable hits (and `expand_links` neighbours are filtered
  too). `get` returns None for an unreadable note, exactly like an absent one.
  `propose` checks the candidate's namespace before your decider runs, shows the
  decider only notes the agent may read, marks the ones it may not write
  read-only in the prompt, and falls back to `create` if a decision targets one
  anyway — the candidate always lands somewhere.
- `doctor`, `reindex`, `watch` and `close` are operator operations on the whole
  vault and take no context.

Scopes are **advisory containment at the library API, not security**: any
process with filesystem access can read or edit the notes directly. That is the
files-are-truth contract, not a hole in it — the boundary that matters for
hostile code is the OS, not this policy object.

### Embedders

An `Embedder` is anything with `model`, `dims` and `async embed(texts)` and is always injected — the
vault never hardcodes a provider. Two ship:

- `FetchEmbedder` — any OpenAI-compatible `POST /v1/embeddings`. Unconfigured it
  is a local Ollama (`all-minilm`, 384 dims, no API key), so the zero-config
  default keeps every note on your machine. What the CLI uses.
- `TokenOverlapEmbedder` — deterministic bag-of-tokens, no network. What the test
  suite and `vault --lexical` use: it exercises the plumbing, not semantics.

```python
from wilcus_vault import FetchEmbedder, open

# zero config: http://localhost:11434/v1/embeddings, all-minilm, 384 dims.
# Run `ollama pull all-minilm` first; if nothing is listening the embed fails
# with "no embedder configured: start Ollama ... or configure a remote
# provider" — it never quietly falls back to a cloud API.
vault = open(root, FetchEmbedder())
await vault.doctor()  # first run with a new model: re-embeds everything
```

A `VAULT_EMBED_API_KEY` sitting in the environment is *not* sent to that default
endpoint — it belongs to whichever remote provider you configured it for, and
"whatever is listening on :11434" does not get to collect it. Configure an
endpoint, or pass `api_key` (a local gateway may want one), and it travels.

A remote provider is supported, but only as an explicit choice — whole note
bodies leave the machine on every embed. Each option falls back to its env var
(`VAULT_EMBED_ENDPOINT`, `VAULT_EMBED_MODEL`, `VAULT_EMBED_DIMS`,
`VAULT_EMBED_API_KEY`); a non-localhost endpoint requires the key, and it is
never persisted, logged, or echoed back in a provider's error message. A remote
endpoint must also name its `model` and `dims` — the defaults describe the local
model, not yours, and a wrong one would be filed as if it were right.

```python
vault = open(
    root,
    FetchEmbedder(
        endpoint="https://api.openai.com/v1/embeddings",
        model="text-embedding-3-small",
        dims=1536,  # api_key from VAULT_EMBED_API_KEY; timeout=30.0 seconds by default
    ),
)
```

`dims` must match what the model returns — it is part of the vec0 table's schema.
Changing either the model or the dims invalidates every stored vector, so the next
`doctor` (or `reindex`) drops the vector table and re-embeds every note; search
refuses to mix vector spaces until it has. Notes are embedded whole; there is no
chunking.

## Development

```
uv sync              # once; dev tools included
./scripts/check.sh   # ruff check, ruff format --check, mypy strict, pytest — green before any PR
```

Tests run on `TokenOverlapEmbedder`, so CI needs no Ollama; fixture vaults land
under `tmp-test/` (gitignored). The CLI's chat decider is `fetch_decider(model=...,
endpoint=..., api_key=...)`, configured by `VAULT_DECIDE_*`; library callers wire
their own `async def` — see [The write gate](#the-write-gate).
