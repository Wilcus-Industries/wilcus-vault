"""The write gate patches frontmatter textually: a YAML round-trip is lossy
against hand-written data."""

import pytest

from wilcus_vault.frontmatter import patch_frontmatter, qualify_links, replace_body
from wilcus_vault.note import parse_note
from wilcus_vault.term import VaultError


def test_patch_frontmatter_appends_a_key_leaving_every_other_byte_alone() -> None:
    raw = "---\ntype: customer\nid: 01234 # legacy\nrate: 1.0\n---\n# Acme\n\nbody\n"
    out = patch_frontmatter(raw, "superseded_by", "notes/new.md")
    assert out == (
        "---\ntype: customer\nid: 01234 # legacy\nrate: 1.0\n"
        'superseded_by: "notes/new.md"\n---\n# Acme\n\nbody\n'
    )
    assert parse_note(out, "x/a.md").frontmatter["superseded_by"] == "notes/new.md"


def test_patch_frontmatter_replaces_exactly_one_existing_top_level_key_line() -> None:
    raw = "---\nupdated: 2020-01-01\nmeta:\n  updated: nested\n---\nbody\n"
    out = patch_frontmatter(raw, "updated", "2026-08-01")
    assert out == '---\nupdated: "2026-08-01"\nmeta:\n  updated: nested\n---\nbody\n'


@pytest.mark.parametrize(
    "raw",
    [
        "# Acme\n\nbody\n",  # no frontmatter at all
        "---\ntype: customer\n\nunterminated\n",  # no closing fence
        # Fenced but not parseable: parse_note ignores the block and takes the whole
        # file as body, so patching *inside* it would write a key nothing reads.
        "---\ntype: [unclosed\n---\n# Acme\n\nbody\n",
        "---\n- a\n- b\n---\nbody\n",  # a sequence, not a mapping
        "",  # empty file
    ],
)
def test_patch_frontmatter_prepends_a_block_when_there_is_no_usable_one(raw: str) -> None:
    out = patch_frontmatter(raw, "superseded_by", "notes/new.md")
    assert out == f'---\nsuperseded_by: "notes/new.md"\n---\n{raw}'
    assert out.endswith(raw)  # the body is never touched
    assert parse_note(out, "x/a.md").frontmatter["superseded_by"] == "notes/new.md"


def test_a_duplicate_key_cannot_survive_the_patch_and_win_by_yaml_last_wins() -> None:
    out = patch_frontmatter(
        "---\nupdated: old\ntype: customer\nupdated: older\n---\nbody\n", "updated", "2026-08-01"
    )
    assert parse_note(out, "x/a.md").frontmatter["updated"] == "2026-08-01"
    assert "older" not in out
    assert "type: customer" in out


def test_patch_frontmatter_unsets_a_key_and_unsetting_an_absent_one_is_a_no_op() -> None:
    # A gate-owned key has to be removable, or it outlives the fact it records.
    raw = (
        '---\ntype: customer\nvault_agent: "a"\nid: 01234 # legacy\n'
        'vault_agent: "dupe"\n---\nbody\n'
    )
    assert patch_frontmatter(raw, "vault_agent", None) == (
        "---\ntype: customer\nid: 01234 # legacy\n---\nbody\n"
    )
    clean = "---\ntype: customer\n---\nbody\n"
    assert patch_frontmatter(clean, "vault_agent", None) == clean
    # no usable block to remove a key from: prepending one would be absurd
    assert patch_frontmatter("# Acme\n\nbody\n", "vault_agent", None) == "# Acme\n\nbody\n"
    assert patch_frontmatter(
        '---\r\nvault_agent: "a"\r\ntype: c\r\n---\r\nb\r\n', "vault_agent", None
    ) == ("---\r\ntype: c\r\n---\r\nb\r\n")


def test_patch_frontmatter_sees_through_a_bom_instead_of_prepending_past_it() -> None:
    out = patch_frontmatter("﻿---\ntype: customer\n---\nbody\n", "updated", "2026-08-01")
    assert out == '﻿---\ntype: customer\nupdated: "2026-08-01"\n---\nbody\n'
    assert replace_body("﻿---\ntype: customer\n---\nold\n", "new\n") == (
        "﻿---\ntype: customer\n---\nnew\n"
    )


def test_iso_timestamp_is_a_yaml_native_scalar_other_values_are_quoted() -> None:
    assert "updated: 2026-08-01T09:41:00.000Z\n" in patch_frontmatter(
        "---\na: b\n---\nx", "updated", "2026-08-01T09:41:00.000Z"
    )
    assert 'superseded_by: "notes/new.md"\n' in patch_frontmatter(
        "---\na: b\n---\nx", "superseded_by", "notes/new.md"
    )


def test_patch_frontmatter_keeps_crlf_line_endings_and_rejects_an_unusable_key() -> None:
    out = patch_frontmatter("---\r\ntype: customer\r\n---\r\nbody\r\n", "updated", "2026-08-01")
    assert out == '---\r\ntype: customer\r\nupdated: "2026-08-01"\r\n---\r\nbody\r\n'
    assert patch_frontmatter("---\r\nupdated: old\r\n---\r\nb\r\n", "updated", "x") == (
        '---\r\nupdated: "x"\r\n---\r\nb\r\n'
    )
    with pytest.raises(VaultError, match="key"):
        patch_frontmatter("body", "a: b\nevil", "x")


def test_replace_body_swaps_the_body_and_keeps_the_frontmatter_block_verbatim() -> None:
    raw = "---\ntype: customer\nid: 01234 # legacy\n---\nold body\n"
    assert (
        replace_body(raw, "new body\n")
        == "---\ntype: customer\nid: 01234 # legacy\n---\nnew body\n"
    )
    # no usable block: the whole file is the body, exactly as parse_note sees it
    assert replace_body("# Acme\n\nold\n", "new\n") == "new\n"
    assert replace_body("---\ntype: [unclosed\n---\nold\n", "new\n") == "new\n"


def test_qualify_links_rewrites_bare_links_to_one_stem_aliases_kept() -> None:
    body = (
        "See [[acme]] and [[acme|Acme Corp]] and [[ acme ]].\n"
        "Not these: [[acmena]], [[customers/acme]], [[other]], plain acme.\n"
        "```\ncode fence: [[acme]]\n```\n"
    )
    assert qualify_links(body, "acme", "customers/acme") == (
        "See [[customers/acme]] and [[customers/acme|Acme Corp]] and [[customers/acme]].\n"
        "Not these: [[acmena]], [[customers/acme]], [[other]], plain acme.\n"
        # the parser counts fenced links as edges, so the rewrite matches it:
        # the documented MVP simplification
        "```\ncode fence: [[customers/acme]]\n```\n"
    )
    # an alias holding a pipe travels whole
    assert qualify_links("[[acme|a|b]]", "acme", "customers/acme") == "[[customers/acme|a|b]]"
    # nothing to do leaves the body byte-identical
    untouched = "no links here, [[other]] only\n"
    assert qualify_links(untouched, "acme", "customers/acme") == untouched


def test_qualify_links_edits_raw_text_in_place_frontmatter_crlf_and_bom_survive() -> None:
    # only the parser's body region is scanned: a decoy in frontmatter is YAML,
    # not a link, and parse_note would never edge it either
    raw = '---\nnote: "[[acme]] is not a link here"\n---\nbody [[acme]]\n'
    assert qualify_links(raw, "acme", "customers/acme") == (
        '---\nnote: "[[acme]] is not a link here"\n---\nbody [[customers/acme]]\n'
    )
    # a CRLF file keeps every carriage return; a BOM stays where it was
    assert (
        qualify_links(
            "﻿---\r\ntype: x\r\n---\r\nsee [[acme]]\r\nplain\r\n", "acme", "customers/acme"
        )
        == "﻿---\r\ntype: x\r\n---\r\nsee [[customers/acme]]\r\nplain\r\n"
    )
    # malformed frontmatter is body, exactly as parse_note reads it
    assert qualify_links("---\ntype: [unclosed\n---\n[[acme]]\n", "acme", "customers/acme") == (
        "---\ntype: [unclosed\n---\n[[customers/acme]]\n"
    )
