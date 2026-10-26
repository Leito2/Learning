"""Validate vault notes: line count, balanced code fences, no ASCII box art, and wikilinks that resolve.

Usage: python scripts/check_note.py "path/to/note.md" [more notes...]
Links are checked against full vault paths under SW-ML-AI Engineering/ (or the note's own folder).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "SW-ML-AI Engineering"
LINK = re.compile(r"\[\[([^\]|#\\]+)")
BOX = re.compile("[┌└│├─]")

for f in sys.argv[1:]:
    p = Path(f)
    text = p.read_text(encoding="utf-8")
    body = re.sub(r"```.*?```", "", text, flags=re.S)
    broken = []
    for m in LINK.finditer(body):
        target = m.group(1).strip()
        if not (ROOT / (target + ".md")).exists() and not (p.parent / (target + ".md")).exists():
            broken.append(target)
    print(p.name[:50], "| lines:", len(text.splitlines()), "| fences even:", text.count("```") % 2 == 0,
          "| ascii:", bool(BOX.search(text)), "| broken:", broken)
