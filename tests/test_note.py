import hashlib
import json
from dataclasses import asdict

from wilcus_vault.note import parse_note, serialize_note


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def test_parses_frontmatter_body_title_type_slug_hash() -> None:
    raw = (
        "---\ntype: customer\ncreated: 2026-07-31\ntitle: Acme Corp\n---\n"
        "# Ignored Heading\n\nBody text.\n"
    )
    n = parse_note(raw, "customers/acme.md")
    assert n.path == "customers/acme.md"
    assert n.slug == "acme"
    assert n.title == "Acme Corp"  # frontmatter wins over heading
    assert n.type == "customer"
    assert n.frontmatter == {"type": "customer", "created": "2026-07-31", "title": "Acme Corp"}
    assert n.body == "# Ignored Heading\n\nBody text.\n"
    assert n.hash == sha256(raw)
    assert n.malformed_frontmatter is False


def test_no_frontmatter_whole_file_is_body_title_from_first_heading() -> None:
    raw = "# Globex Renewal\n\nsome prose\n"
    n = parse_note(raw, "notes/globex-renewal.md")
    assert n.frontmatter == {}
    assert n.body == raw
    assert n.title == "Globex Renewal"
    assert n.type is None
    assert n.malformed_frontmatter is False


def test_title_falls_back_to_filename_stem() -> None:
    assert parse_note("just prose, no heading\n", "a/b/my-note.md").title == "my-note"
    # ## is not an h1; still falls through to the stem
    assert parse_note("## sub\n", "deep/sub-note.md").title == "sub-note"


def test_malformed_yaml_still_indexes() -> None:
    raw = "---\ntype: [unclosed\ntitle: nope\n---\n# Real Heading\n\nbody\n"
    n = parse_note(raw, "x/broken.md")
    assert n.malformed_frontmatter is True
    assert n.frontmatter == {}
    assert n.body == raw  # whole file, delimiters included
    assert n.title == "Real Heading"
    assert n.hash == sha256(raw)


def test_non_mapping_frontmatter_is_malformed() -> None:
    n = parse_note("---\njust a scalar\n---\nbody\n", "x/scalar.md")
    assert n.malformed_frontmatter is True
    assert n.frontmatter == {}


def test_empty_frontmatter_block_is_not_malformed() -> None:
    n = parse_note("---\n---\nbody\n", "x/empty.md")
    assert n.malformed_frontmatter is False
    assert n.frontmatter == {}
    assert n.body == "body\n"


def test_crlf_input_parses_and_normalizes_to_lf() -> None:
    raw = "---\r\ntype: customer\r\n---\r\n# Title\r\n\r\nsee [[acme]]\r\n"
    n = parse_note(raw, "x/crlf.md")
    assert n.frontmatter == {"type": "customer"}
    assert n.body == "# Title\n\nsee [[acme]]\n"
    assert n.title == "Title"
    assert n.links == ["acme"]
    assert n.hash == sha256(raw)  # hash is of the raw bytes, pre-normalization


def test_wikilinks_aliases_stripped_deduped_code_fences_counted() -> None:
    raw = (
        "# Links\n\nSee [[acme]] and [[globex|Globex Inc]] and [[acme]] again.\n\n"
        "```md\n[[in-a-fence]]\n```\n\n[[ spaced ]] and [[]] and [[a|b|c]]\n"
    )
    n = parse_note(raw, "x/links.md")
    # MVP does not parse markdown structure: fenced wikilinks count (documented)
    assert n.links == ["acme", "globex", "in-a-fence", "spaced", "a"]


def test_wikilinks_keep_a_path_qualified_target_whole() -> None:
    n = parse_note("[[customers/acme]] [[vendors/acme|the vendor]] [[acme]]\n", "x/q.md")
    assert n.links == ["customers/acme", "vendors/acme", "acme"]


def test_links_come_from_the_body_only() -> None:
    n = parse_note("---\nsuperseded_by: old/[[trap]]\n---\n[[real]]\n", "x/fm.md")
    assert n.links == ["real"]


def test_parse_serialize_parse_round_trips() -> None:
    raw = (
        "---\ntype: customer\ntags:\n  - a\n  - b\nsuperseded_by: null\n---\n"
        "# Acme\n\nbody with [[globex]]\n"
    )
    a = parse_note(raw, "customers/acme.md")
    out = serialize_note(a.frontmatter, a.body)
    b = parse_note(out, "customers/acme.md")
    assert b.frontmatter == a.frontmatter
    assert b.body == a.body
    assert b.title == a.title
    assert b.type == a.type
    assert b.links == a.links
    assert b.malformed_frontmatter is False
    assert serialize_note(b.frontmatter, b.body) == out  # serialization is stable


def test_serialize_without_frontmatter_emits_body_only() -> None:
    assert serialize_note({}, "# Bare\n") == "# Bare\n"


def test_alias_bomb_over_the_node_budget_is_malformed() -> None:
    # Billion-laughs: aliases re-expand on every reference, so a tiny file blows
    # up into a multi-megabyte object that would detonate on serialization.
    raw = (
        "---\n"
        'a: &a ["x","x","x","x","x","x","x","x","x"]\n'
        "b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a]\n"
        "c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b]\n"
        "d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c]\n"
        "e: [*d,*d,*d,*d,*d,*d,*d,*d,*d]\n"
        "---\nbody\n"
    )
    n = parse_note(raw, "x/bomb.md")
    assert n.malformed_frontmatter is True
    assert n.frontmatter == {}
    assert n.body == raw
    assert len(json.dumps(asdict(n))) < len(raw) * 20  # stays small


def test_ordinary_nested_frontmatter_stays_under_the_node_budget() -> None:
    n = parse_note("---\nmeta:\n  tags: [a, b, c]\n  nested:\n    k: v\n---\nbody\n", "x/nested.md")
    assert n.malformed_frontmatter is False
    assert n.frontmatter == {"meta": {"tags": ["a", "b", "c"], "nested": {"k": "v"}}}


def test_unterminated_frontmatter_is_flagged() -> None:
    # A mangled closing fence must not silently drop superseded_by from the index.
    raw = "---\ntype: customer\nsuperseded_by: new/acme.md\n\nbody\n"
    n = parse_note(raw, "x/unterminated.md")
    assert n.malformed_frontmatter is True
    assert n.frontmatter == {}
    assert n.body == raw


def test_closing_fence_with_trailing_junk_does_not_close_the_block() -> None:
    raw = "---\ntype: customer\n--- oops\nbody\n"
    n = parse_note(raw, "x/junk-fence.md")
    assert n.malformed_frontmatter is True
    assert n.body == raw


def test_wikilinks_never_span_newlines_and_are_length_capped() -> None:
    assert parse_note("[[foo\nbar]]\n\n[[ok]]\n", "x/nl.md").links == ["ok"]
    assert parse_note("[[\n\nlots of prose\n\n]]\n", "x/distant.md").links == []
    assert parse_note(f"[[{'a' * 300}]] and [[short]]\n", "x/long.md").links == ["short"]


def test_strips_a_leading_utf8_bom() -> None:
    n = parse_note("﻿---\ntype: customer\n---\n# Title\n", "x/bom.md")
    assert n.frontmatter == {"type": "customer"}
    assert n.malformed_frontmatter is False
    assert n.body == "# Title\n"

    no_fm = parse_note("﻿# Heading\n", "x/bom2.md")
    assert no_fm.title == "Heading"
    assert no_fm.body == "# Heading\n"


def test_non_string_title_type_are_ignored_and_flagged_for_doctor() -> None:
    n = parse_note("---\ntitle: 42\ntype: [a, b]\n---\n# Heading\n", "x/badtypes.md")
    assert n.title == "Heading"  # falls through to the heading
    assert n.type is None
    assert n.malformed_frontmatter is True
    assert n.frontmatter == {"title": 42, "type": ["a", "b"]}  # kept as written


def test_frontmatter_too_deeply_nested_to_parse_is_malformed_not_an_exception() -> None:
    """PyYAML's composer recurses per nesting level, so a deep enough block
    raises RecursionError rather than a YAMLError. Parsing never raises."""
    raw = "---\na: " + "[" * 3000 + "]" * 3000 + "\n---\n# Deep\n\nbody\n"
    note = parse_note(raw, "deep.md")
    assert note.malformed_frontmatter is True
    assert note.frontmatter == {}
    assert note.body == raw.replace("\r\n", "\n")
