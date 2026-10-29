"""Turn the HTML some engines return into plain text."""

from __future__ import annotations

import re
from html.parser import HTMLParser

_BLOCK_END = {"p", "div", "tr", "li", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6", "section", "blockquote"}


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _BLOCK_END:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append("\t")

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(markup: str | None) -> str:
    """Plain text with one line per block, rows of a table on one line separated by tabs."""
    if not markup:
        return ""
    collector = _Collector()
    collector.feed(markup)
    collector.close()
    text = "".join(collector.parts)
    lines = [re.sub(r"[ \u00a0]+", " ", line).strip(" \t") for line in text.split("\n")]
    return "\n".join(line for line in lines if line)
