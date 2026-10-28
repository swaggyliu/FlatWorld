"""Render selected Markdown docs to static HTML for GitHub Pages."""
from __future__ import annotations

from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent

DOCS = [
    ("THEORY_AND_IMPLEMENTATION.md", "theory.html", "Theory & implementation"),
    ("THEORY_AND_IMPLEMENTATION.zh.md", "theory.zh.html", "理论与实现"),
    ("PARADIGMS.md", "paradigms.html", "Control paradigms"),
]

TEMPLATE = """<!DOCTYPE html>
<html lang="{lang}">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title} — FlatWorld</title>
  <link rel="stylesheet" href="doc.css" />
</head>
<body>
  <nav class="doc-nav">
    <a href="./">FlatWorld</a>
    <a href="theory.html">Theory</a>
    <a href="theory.zh.html">理论</a>
    <a href="paradigms.html">Paradigms</a>
    <a href="https://github.com/swaggyliu/FlatWorld">GitHub</a>
  </nav>
  <main class="doc-body">
{body}
  </main>
</body>
</html>
"""


def main() -> None:
    md = markdown.Markdown(
        extensions=["fenced_code", "tables", "toc", "sane_lists"],
        output_format="html5",
    )
    for src_name, out_name, title in DOCS:
        src = ROOT / src_name
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        md.reset()
        text = src.read_text(encoding="utf-8")
        # Point language switchers / sibling docs at HTML pages on the site.
        text = (
            text.replace("](THEORY_AND_IMPLEMENTATION.zh.md)", "](theory.zh.html)")
            .replace("](THEORY_AND_IMPLEMENTATION.md)", "](theory.html)")
            .replace("](PARADIGMS.md)", "](paradigms.html)")
        )
        body = md.convert(text)
        lang = "zh" if out_name.endswith(".zh.html") else "en"
        html = TEMPLATE.format(lang=lang, title=title, body=body)
        (ROOT / out_name).write_text(html, encoding="utf-8", newline="\n")
        print(f"wrote {out_name}")


if __name__ == "__main__":
    main()
