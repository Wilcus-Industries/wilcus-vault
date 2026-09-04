"""Every source module stays under the line cap unless it has a written reason not to."""

from pathlib import Path

CAP = 200
SRC = Path(__file__).parent.parent / "src" / "wilcus_vault"
EXCEPTIONS: dict[str, str] = {}  # relative path -> why it may exceed the cap


def test_source_modules_under_cap() -> None:
    over = {
        p.name: n
        for p in sorted(SRC.glob("*.py"))
        if (n := len(p.read_text().splitlines())) > CAP and p.name not in EXCEPTIONS
    }
    assert not over, f"over {CAP} lines: {over}"
