"""Every source module stays under the line cap unless it has a written reason not to."""

from pathlib import Path

CAP = 200
SRC = Path(__file__).parent.parent / "src" / "wilcus_vault"
EXCEPTIONS: dict[str, str] = {}  # relative path -> why it may exceed the cap


def test_source_modules_under_cap() -> None:
    over = {
        str(p.relative_to(SRC)): n
        for p in sorted(SRC.rglob("*.py"))
        if (n := len(p.read_text().splitlines())) > CAP
        and str(p.relative_to(SRC)) not in EXCEPTIONS
    }
    assert not over, f"over {CAP} lines: {over}"
