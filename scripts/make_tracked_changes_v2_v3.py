"""Record of the style revision (manuscript v2 -> v3, 2026-11-19).

v2 (archived in manuscript_archive/v2_2026-11-19_pre_style_revision/) consisted of pavenet_main.tex with four \\input
sections; v3 is the single file manuscript/pavenet_main.tex. The v2 sources are flattened in reading order and compared
with v3 as a unified diff and as a word-level HTML view (deletions struck through, insertions underlined).
"""
from __future__ import annotations

import difflib
import html
from pathlib import Path

from make_tracked_changes import word_diff

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "manuscript_archive" / "v2_2026-11-19_pre_style_revision"
NEW = ROOT / "manuscript" / "pavenet_main.tex"
OUT = ROOT / "manuscript_archive" / "revision_2026-11-19_style"


def flatten_v2() -> str:
    main = (OLD / "pavenet_main.tex").read_text(encoding="utf-8")
    for n in ("01_intro_related", "02_methods", "03_results", "04_discussion"):
        main = main.replace(f"\\input{{sections/{n}}}", (OLD / "sections" / f"{n}.tex").read_text(encoding="utf-8"))
    return main


def main():
    old, new = flatten_v2(), NEW.read_text(encoding="utf-8")
    diff = difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                fromfile="v2/pavenet_main.tex (flattened)", tofile="v3/pavenet_main.tex")
    (OUT / "manuscript_v2_to_v3.diff").write_text("".join(diff), encoding="utf-8")
    pa, pb = old.split("\n\n"), new.split("\n\n")
    blocks = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, pa, pb, autojunk=False).get_opcodes():
        if op == "equal":
            blocks += [f"<p>{html.escape(p)}</p>" for p in pa[i1:i2]]
        else:
            blocks.append(f"<p class='chg'>{word_diff(chr(10).join(pa[i1:i2]), chr(10).join(pb[j1:j2]))}</p>")
    page = ("<!doctype html><meta charset='utf-8'><title>Manuscript v2 to v3</title><style>body{font:14px/1.5 Arial,sans-serif;"
            "max-width:60em;margin:2em auto;padding:0 1em;white-space:pre-wrap}del{color:#b00020;text-decoration:line-through}"
            "ins{color:#0645ad;text-decoration:underline}p.chg{background:#f7f7f2}</style>"
            "<h1>Manuscript v2 to v3 (style revision, 2026-11-19)</h1>" + "".join(blocks))
    (OUT / "tracked_changes_v2_to_v3.html").write_text(page, encoding="utf-8")
    print("written", OUT)


if __name__ == "__main__":
    main()
